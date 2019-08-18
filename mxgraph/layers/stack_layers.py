import warnings
from mxnet.gluon import Block
from .aggregators import *


class BaseGraphMultiLayer(HybridBlock):
    def __init__(self, out_units, aggregator_args_list, dropout_rate_list, graph_type="homo", in_units=None, first_embed_units=256,
                 dense_connect=False, every_layer_l2_normalization=False, l2_normalization=False,
                 output_inner_result=False,
                 prefix=None, params=None):
        super(BaseGraphMultiLayer, self).__init__(prefix=prefix, params=params)
        self._aggregator_args_list = aggregator_args_list
        self._dropout_rate_list = dropout_rate_list
        self._dense_connect = dense_connect
        self._first_embed_units = first_embed_units
        self._every_layer_l2_normalization = every_layer_l2_normalization
        self._l2_normalization = l2_normalization
        self._graph_type = graph_type
        if self._l2_normalization:
            self._l2_normalization_layer = L2Normalization(axis=-1)
        self._output_inner_result = output_inner_result
        print("graph type", graph_type)
        with self.name_scope():
            if graph_type == "homo":
                self.embed = nn.Dense(self._first_embed_units, flatten=False)
            self.aggregators = nn.HybridSequential()
            self.dropout_layers = nn.HybridSequential()
            for args, dropout_rate in zip(aggregator_args_list, dropout_rate_list):
                self.aggregators.add(parse_aggregator_from_desc(args))
                self.dropout_layers.add(nn.Dropout(dropout_rate))
            self.out_layer = nn.Dense(units=out_units, flatten=False)

class GraphMultiLayerAllNodes(BaseGraphMultiLayer):
    """Generate the output for all nodes in the graph

    """
    def hybrid_forward(self, F, data, end_points, indptr, edge_data=None):
        """

        Parameters
        ----------
        F
        data : Symbol or NDArray
            Shape: (node_num, feat_dim)
        end_points: Symbol or NDArray
            Shape: (nnz,)
        indptr: Symbol or NDArray
            Shape: (node_num,)
        edge_data: Symbol or NDArray or None
            Shape: (nnz, edge_dim)
        Returns
        -------
        ret: Symbol or NDArray
        """
        ### DO NOT ADD THE DEGREE FEATURE
        # degrees = F.slice_axis(indptr, axis=-1, begin=1, end=None) \
        #           - F.slice_axis(indptr, axis=-1, begin=0, end=-1)
        # log_degrees = F.log(degrees.astype("float32") + 1.0)
        # data = F.concat(data, F.expand_dims(log_degrees, axis=1), dim=1)
        data = self.embed(data)
        layer_in_l = [data]
        # Forward through the aggregators
        gate_l = []
        sharpness_l = []
        attend_weights_wo_gate_l = []
        for aggregator, dropout_layer in zip(self.aggregators, self.dropout_layers):
            layer_in = layer_in_l[-1]
            # When the whole graph is used, neighbor_data will always be equal to data
            if isinstance(aggregator, MuGGA):
                out, gate, sharpness, attend_weights_wo_gate =\
                    aggregator(F.expand_dims(layer_in, axis=0),
                               F.expand_dims(layer_in, axis=0),
                               end_points, indptr, edge_data)
                gate_l.append(gate)
                sharpness_l.append(sharpness)
                attend_weights_wo_gate_l.append(attend_weights_wo_gate)
            else:
                out = aggregator(F.expand_dims(layer_in, axis=0),
                                 F.expand_dims(layer_in, axis=0),
                                 end_points, indptr, edge_data)
            out = F.reshape(out, shape=(0, 0), reverse=True)
            if self._every_layer_l2_normalization:
                out = self._l2_normalization_layer(out)
            else:
                out = dropout_layer(out)
            if self._dense_connect:
                # We are actually implementing skip connection
                out = F.concat(layer_in, out)
            layer_in_l.append(out)
        # Go through the output layer
        out = self.out_layer(layer_in_l[-1])
        if self._l2_normalization:
            out = self._l2_normalization_layer(out)
        if self._output_inner_result:
            return out, gate_l, sharpness_l, attend_weights_wo_gate_l
        else:
            return out


class GraphMultiLayerHierarchicalNodes(BaseGraphMultiLayer):
    """Generate the outputs for part of the nodes.

    The other nodes are appended to the list level by level (Level Set)

    """

    def hybrid_forward(self, F, data, end_points_l, indptr_l, indices_in_merged_l=None, edge_data_l=None):
        """
        We aggregate the lower layer information based on the end_points and indptr information

        Note that here the end_points, indptr, indices should from the highest layer to the lowest layer

        Remember the data should be from the 0th layer!

        Parameters
        ----------
        F
        data : Symbol or NDArray
            The basic features (Features in Layer0)
        end_points_l: list or Symbol or NDArray
            Stores the edge connectivity within a layer
            Shape: [(NNZ_1,), (NNZ_2,), ...]
        indptr_l: list
            Pointers to the relative end_points
            Shape: [(N_1,), (N_2,), ...]
        indices_in_merged_l: list
            indices_in_merged_l[i] Stores the relative ith layer position of the nodes that appear in the i+1th layer.
            To be more specific, indices_in_merged_l[1][j] stores the indices of the jth node in Layer 1 w.r.t Layer 0 (Basic Features Layer)
            We can select the features from the previous layer by calling `features_i[indices_in_merged_l[i], :]`
            Shape: [(N_1,), (N_2,), ...]
        edge_data_l: list
            Edge feature corresponds to the end_points
            Shape: [(NNZ_1, fdim1), (NNZ_2, fdim2), ...]
        Returns
        -------
        ret: Symbol or NDArray
        """
        layer_num = len(self.aggregators)
        if indices_in_merged_l is not None:
            assert len(end_points_l) == layer_num
            assert len(indptr_l) == layer_num
            assert len(indices_in_merged_l) == layer_num
        data = self.embed(data)
        layer_in_l = [data]
        gate_l = []
        sharpness_l = []
        attend_weights_wo_gate_l = []
        for i, (aggregator, dropout_layer) in enumerate(zip(self.aggregators, self.dropout_layers)):
            lower_layer_data = layer_in_l[-1]
            if indices_in_merged_l is None:
                end_points = end_points_l
                indptr = indptr_l
                upper_layer_base_data = lower_layer_data
            else:
                end_points = end_points_l[i]
                indptr = indptr_l[i]
                upper_layer_base_data = F.take(lower_layer_data,
                                               indices=indices_in_merged_l[i], axis=0)
            edge_data = None if edge_data_l is None else edge_data_l[i]
            if isinstance(aggregator, MuGGA):
                out, gate, sharpness, attend_weights_wo_gate =\
                    aggregator(F.expand_dims(upper_layer_base_data, axis=0),
                               F.expand_dims(lower_layer_data, axis=0),
                               end_points,
                               indptr,
                               edge_data)
                gate_l.append(gate)
                sharpness_l.append(sharpness)
                attend_weights_wo_gate_l.append(attend_weights_wo_gate)
            else:
                out = aggregator(F.expand_dims(upper_layer_base_data, axis=0),
                                 F.expand_dims(lower_layer_data, axis=0),
                                 end_points,
                                 indptr,
                                 edge_data)
            out = F.reshape(out, shape=(0, 0), reverse=True)
            if self._every_layer_l2_normalization:
                out = self._l2_normalization_layer(out)
            else:
                out = dropout_layer(out)
            if self._dense_connect:
                # We are actually implementing skip connection
                out = F.concat(lower_layer_data, out, dim=-1)
            layer_in_l.append(out)
        out = self.out_layer(layer_in_l[-1])
        if self._l2_normalization:
            out = self._l2_normalization_layer(out)
        if self._output_inner_result:
            return out, gate_l, sharpness_l, attend_weights_wo_gate_l
        else:
            return out


class HeterGraphMultiLayerHierarchicalNodes(BaseGraphMultiLayer):
    """Generate the outputs for part of the nodes.

    The other nodes are appended to the list level by level (Level Set)

    """

    def hybrid_forward(self, F, data, mask, end_points_l, indptr_l, indices_in_merged_l=None, edge_data_l=None):
        """
        We aggregate the lower layer information based on the end_points and indptr information

        Note that here the end_points, indptr, indices should from the highest layer to the lowest layer

        Remember the data should be from the 0th layer!

        Parameters
        ----------
        F
        data : Symbol or NDArray Shape (num_node, fea_dim)
            The basic features (Features in Layer0)
        end_points_l: list or Symbol or NDArray
            Stores the edge connectivity within a layer
            Shape: [(NNZ_1,), (NNZ_2,), ...]
        indptr_l: list
            Pointers to the relative end_points
            Shape: [(N_1,), (N_2,), ...]
        indices_in_merged_l: list
            indices_in_merged_l[i] Stores the relative ith layer position of the nodes that appear in the i+1th layer.
            To be more specific, indices_in_merged_l[1][j] stores the indices of the jth node in Layer 1 w.r.t Layer 0 (Basic Features Layer)
            We can select the features from the previous layer by calling `features_i[indices_in_merged_l[i], :]`
            Shape: [(N_1,), (N_2,), ...]
        mask: Symbol or NDArray Shape (num_node, num_set, 1)
            The node set mask  (nodes in Layer0)
        edge_data_l: list
            Edge feature corresponds to the end_points
            Shape: [(NNZ_1, fdim1), (NNZ_2, fdim2), ...]
        Returns
        -------
        ret: Symbol or NDArray
        """
        assert self._graph_type == "heter"
        layer_num = len(self.aggregators)
        if indices_in_merged_l is not None:
            assert len(end_points_l) == layer_num
            assert len(indptr_l) == layer_num
            assert len(indices_in_merged_l) == layer_num

        #data = self.embed(data)
        layer_in_l = [data]
        neighbor_mask_l = [mask]
        gate_l = []
        sharpness_l = []
        attend_weights_wo_gate_l = []
        for i, (aggregator, dropout_layer) in enumerate(zip(self.aggregators, self.dropout_layers)):
            assert isinstance(aggregator, HeterGraphPoolAggregator)

            lower_layer_data = layer_in_l[-1]
            neighbor_mask = neighbor_mask_l[-1]
            if indices_in_merged_l is None:
                end_points = end_points_l
                indptr = indptr_l
                upper_layer_base_data = lower_layer_data
                node_mask = neighbor_mask
            else:
                end_points = end_points_l[i]
                indptr = indptr_l[i]
                upper_layer_base_data = F.take(lower_layer_data,
                                               indices=indices_in_merged_l[i], axis=0)
                node_mask = F.take(neighbor_mask, indices=indices_in_merged_l[i], axis=0)

            edge_data = None if edge_data_l is None else edge_data_l[i]

            out = aggregator(F.expand_dims(upper_layer_base_data, axis=0),
                             F.expand_dims(lower_layer_data, axis=0),
                             F.expand_dims(node_mask, axis=0),
                             F.expand_dims(neighbor_mask, axis=0),
                             end_points,
                             indptr,
                             F.expand_dims(edge_data, axis=0))
            out = F.reshape(out, shape=(0, 0), reverse=True)
            if self._every_layer_l2_normalization:
                out = self._l2_normalization_layer(out)
            else:
                out = dropout_layer(out)
            if self._dense_connect:
                # We are actually implementing skip connection
                out = F.concat(lower_layer_data, out, dim=-1)
            layer_in_l.append(out)
            neighbor_mask_l.append(node_mask)
        out = self.out_layer(layer_in_l[-1])
        if self._l2_normalization:
            out = self._l2_normalization_layer(out)

        return out


class BaseHeterGraphMultiLayer(Block):
    def __init__(self, out_units, aggregator_args_list, dropout_rate_list,
                 dense_connect=False, every_layer_l2_normalization=False, l2_normalization=False,
                 output_inner_result=False,
                 graph_type="bi", num_node_set=None, num_edge_set=None,
                 prefix=None, params=None):
        super(BaseHeterGraphMultiLayer, self).__init__(prefix=prefix, params=params)
        self._aggregator_args_list = aggregator_args_list
        self._dropout_rate_list = dropout_rate_list
        self._dense_connect = dense_connect
        self._every_layer_l2_normalization = every_layer_l2_normalization
        self._l2_normalization = l2_normalization
        self._num_node_set = num_node_set
        self._num_edge_set = num_edge_set
        self._graph_type = graph_type
        if self._l2_normalization:
            self._l2_normalization_layer = L2Normalization(axis=-1)
        self._output_inner_result = output_inner_result
        with self.name_scope():
            self.aggregators = nn.Sequential()
            self.dropout_layers = nn.Sequential()
            for args, dropout_rate in zip(aggregator_args_list, dropout_rate_list):
                self.aggregators.add(parse_aggregator_from_desc(args))
                self.dropout_layers.add(nn.Dropout(dropout_rate))
            self.out_layer = nn.Dense(units=out_units, flatten=False, prefix="out_")

class BiGraphMultiLayerHierarchicalNodes(BaseHeterGraphMultiLayer):
    """Generate the outputs for part of the nodes.

    The other nodes are appended to the list level by level (Level Set)

    """

    def forward(self, data, end_points_l, indptr_l, indices_in_merged_l=None,
                       node_type_mask=None, edge_type_mask_l=None, seg_indices_l=None):
        """
        We aggregate the lower layer information based on the end_points and indptr information

        Note that here the end_points, indptr, indices should from the highest layer to the lowest layer

        Remember the data should be from the 0th layer!

        Parameters
        ----------
        F
        data : Symbol or NDArray Shape (num_node, fea_dim)
            The basic features (Features in Layer0)
        end_points_l: list or Symbol or NDArray
            Stores the edge connectivity within a layer
            Shape: [(NNZ_1,), (NNZ_2,), ...]
        indptr_l: list
            Pointers to the relative end_points
            Shape: [(N_1,), (N_2,), ...]
        indices_in_merged_l: list
            indices_in_merged_l[i] Stores the relative ith layer position of the nodes that appear in the i+1th layer.
            To be more specific, indices_in_merged_l[1][j] stores the indices of the jth node in Layer 1 w.r.t Layer 0 (Basic Features Layer)
            We can select the features from the previous layer by calling `features_i[indices_in_merged_l[i], :]`
            Shape: [(N_1,), (N_2,), ...]
        mask: Symbol or NDArray Shape (num_node, num_set, 1)
            The node set mask  (nodes in Layer0)
        edge_data_l: list
            Edge feature corresponds to the end_points
            Shape: [(NNZ_1, fdim1), (NNZ_2, fdim2), ...]
        seg_indices_l: list
            the edge indices of the end_points, which is arange(end_points_l[i].size)
            Shape: [(NNZ_1, ), (NNZ_2, ), ...]
        Returns
        -------
        ret: Symbol or NDArray
        """
        assert self._graph_type == "bi"
        layer_num = len(self.aggregators)
        if indices_in_merged_l is not None:
            assert len(end_points_l) == layer_num
            assert len(indptr_l) == layer_num
            assert len(indices_in_merged_l) == layer_num

        if self._num_edge_set is not None:
            assert len(edge_type_mask_l) == layer_num
            assert len(seg_indices_l) == layer_num
        else:
            edge_type_mask_l = [None for _ in range(layer_num)]
            seg_indices_l = [None for _ in range(layer_num)]

        if self._num_node_set is not None:
            assert node_type_mask is not None
            neighbor_node_type_mask_l = [node_type_mask]
            for i in range(layer_num):
                if indices_in_merged_l is None:
                    neighbor_node_type_mask_l.append(node_type_mask)
                else:
                    neighbor_node_type_mask_l.append(nd.take(neighbor_node_type_mask_l[-1],
                                                            indices=indices_in_merged_l[i], axis=0))
        else:
            neighbor_node_type_mask_l = [None for _ in range(layer_num+1)]

        # data = self.embed(data)
        layer_in_l = [data]
        for i, (aggregator, dropout_layer) in enumerate(zip(self.aggregators, self.dropout_layers)):
            assert isinstance(aggregator, BiGraphPoolAggregator)
            lower_layer_data = layer_in_l[-1]
            if indices_in_merged_l is None:
                end_points = end_points_l
                indptr = indptr_l
                upper_layer_base_data = lower_layer_data
            else:
                end_points = end_points_l[i]
                indptr = indptr_l[i]
                upper_layer_base_data = nd.take(lower_layer_data,
                                               indices=indices_in_merged_l[i], axis=0)
            lower_layer_data = dropout_layer(lower_layer_data)

            out = aggregator(nd.expand_dims(upper_layer_base_data, axis=0),
                             nd.expand_dims(lower_layer_data, axis=0),
                             end_points,
                             indptr,
                             nd.expand_dims(neighbor_node_type_mask_l[i+1], axis=0),
                             nd.expand_dims(neighbor_node_type_mask_l[i], axis=0),
                             nd.expand_dims(edge_type_mask_l[i], axis=0),
                             seg_indices_l[i])
            out = nd.reshape(out, shape=(0, 0), reverse=True)
            if self._every_layer_l2_normalization:
                out = self._l2_normalization_layer(out)
            # else:
            #     out = dropout_layer(out)
            if self._dense_connect:
                # We are actually implementing skip connection
                out = nd.concat(lower_layer_data, out, dim=-1)
            layer_in_l.append(out)
        out = self.out_layer(layer_in_l[-1])
        if self._l2_normalization:
            out = self._l2_normalization_layer(out)
        return out


class HeterEmbedLayer(Block):
    def __init__(self, embed_units, num_set, prefix=None, params=None):
        super(HeterEmbedLayer, self).__init__(prefix=prefix, params=params)
        self._num_set = num_set
        self._embed_units = embed_units
        with self.name_scope():
            self.embed_layer = nn.Sequential()
            for i in range(self._num_set):
                self.embed_layer.add(nn.Dense(units=self._embed_units, flatten=False, use_bias=False))

    def forward(self, sampled_node_order, layer0_features_nd_l):
        fea_embeds_l = []
        for i in range(len(self.embed_layer)):
            fea_embeds_l.append(self.embed_layer[i](layer0_features_nd_l[i]))
        fea_embeds = mx.nd.concat(*fea_embeds_l, dim=0)
        data = mx.nd.take(fea_embeds, indices=sampled_node_order)
        return data

class HeterPredLayer(HybridBlock):
    def __init__(self, input_node_dim, hidden_dim, out_units, num_pred_set, act="leaky", prefix=None, params=None):
        super(HeterPredLayer, self).__init__(prefix=prefix, params=params)
        self._input_node_dim = input_node_dim
        self._hidden_dim = hidden_dim
        self._out_units = out_units
        self._num_pred_set = num_pred_set
        with self.name_scope():
            self._act = get_activation(act)
            self.hidden_layer = nn.Dense(units=self._hidden_dim, flatten=False)
            self.out_layer = nn.Dense(units=self._out_units, flatten=False)

    def hybrid_forward(self, F, data):
        data = F.reshape(F.swapaxes(F.reshape(data,
                                              shape=(self._num_pred_set, -1, self._input_node_dim)), 0, 1),
                         shape=(-1, self._input_node_dim*self._num_pred_set))
        hidden_1 = self._act(self.hidden_layer(data))
        out = self.out_layer(hidden_1)
        return out



class HeterDecoder(HybridBlock):
    def __init__(self, input_node_dim, num_pred_set, out_units, act="leaky", prefix=None, params=None):
        super(HeterDecoder, self).__init__(prefix=prefix, params=params)
        self._input_node_dim = input_node_dim
        self._num_pred_set = num_pred_set
        self._out_units = out_units
        with self.name_scope():
            self._act = get_activation(act)
            self.bilinear_W = nn.HybridSequential()
            for i in range(self._out_units):
                self.bilinear_W.add(nn.Dense(units=self._input_node_dim, flatten=False, use_bias=False))


    def hybrid_forward(self, F, data, rates):
        data = F.reshape(data, shape=(self._num_pred_set, -1, self._input_node_dim))
        user_embeds = F.reshape(F.slice_axis(data, axis=0, begin=0, end=1),
                                 shape=(-1,self._input_node_dim))
        item_embeds = F.reshape(F.slice_axis(data, axis=0, begin=1, end=2),
                                 shape=(-1,self._input_node_dim))

        rate_weight_l = []
        for i in range(self._out_units):
            weight = F.sum(F.broadcast_mul(self.bilinear_W[i](user_embeds),
                                            item_embeds), axis=1, keepdims=True)
            #print("weight", weight.shape)
            rate_weight_l.append(weight)
        rate_weights = F.softmax(F.concat(*rate_weight_l, dim=1))
        #print("rate_weights", rate_weights.shape)
        out = F.sum(F.broadcast_mul(rate_weights, rates), axis=1)
        return out

class BiDecoder(Block):
    def __init__(self, input_node_dim, num_pred_set, out_units, act="leaky", prefix=None, params=None):
        super(BiDecoder, self).__init__(prefix=prefix, params=params)
        self._input_node_dim = input_node_dim
        self._num_pred_set = num_pred_set
        self._out_units = out_units
        with self.name_scope():
            # self._act = get_activation(act)
            self.bilinear_W = nn.Sequential()
            for i in range(self._out_units):
                self.bilinear_W.add(nn.Dense(units=self._input_node_dim, flatten=False, use_bias=False))


    def forward(self, data):
        data = nd.reshape(data, shape=(self._num_pred_set, -1, self._input_node_dim))
        user_embeds = nd.reshape(nd.slice_axis(data, axis=0, begin=0, end=1), # (user_num, input_node_dim)
                                 shape=(-1,self._input_node_dim))
        item_embeds = nd.reshape(nd.slice_axis(data, axis=0, begin=1, end=2), # (user_num, input_node_dim)
                                 shape=(-1,self._input_node_dim))

        rate_weight_l = []
        for i in range(self._out_units):
            weight = nd.sum(nd.broadcast_mul(self.bilinear_W[i](user_embeds), # (user_num, input_node_dim) *
                                            item_embeds), axis=1, keepdims=True)
            #print("weight", weight.shape)
            rate_weight_l.append(weight)
        out = nd.concat(*rate_weight_l, dim=1)
        return out
