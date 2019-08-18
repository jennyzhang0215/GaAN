import math
import logging
from collections import namedtuple
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import copy
from mxnet.gluon import nn, HybridBlock
from mxgraph.config import cfg
from mxgraph.utils import safe_eval
from mxgraph.layers import MuGGA
from mxgraph.layers.common import *
from mxgraph.layers.aggregators import parse_aggregator_from_desc, get_activation


def shortcut_aggregator_forward(aggregator, X, Z, end_points, indptr, edge_data):
    if isinstance(aggregator, MuGGA):
        data, gate, sharpness, attend_weights_wo_gate =\
            aggregator(X, Z, end_points, indptr, edge_data)
        mid_info = [gate, sharpness, attend_weights_wo_gate]
    else:
        data = aggregator(X, Z, end_points, indptr, edge_data)
        mid_info = []
    return data, mid_info


class GraphConvGRUCell(HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(GraphConvGRUCell, self).__init__(prefix=prefix, params=params)


class GraphRNNCell(HybridBlock):
    def __init__(self, aggregator_args, aggregation_type='all', typ='rnn', prefix=None, params=None):
        super(GraphRNNCell, self).__init__(prefix=prefix, params=params)
        assert len(aggregator_args) == 2
        self._aggregator_args = copy.deepcopy(aggregator_args)
        # In order to compute gates in GRU, we need to scale up the number of out_units
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._state_dim = aggregator_args[1][0]
        self._typ = typ.lower()
        assert self._typ in ['rnn', 'lstm']
        self._map_dim = 4 * self._state_dim if self._typ == 'lstm' else self._state_dim
        self._aggregation_type = aggregation_type
        with self.name_scope():
            if self._aggregation_type == 'all' or self._aggregation_type == 'in'\
                    or self._aggregation_type == 'state_only':
                self._aggregator_args[1][0] = self._map_dim
                self.aggregator = parse_aggregator_from_desc(
                    aggregator_desc=[self._aggregator_args[0],
                                     self._aggregator_args[1],
                                     'agg_', 'identity'])
            if self._aggregation_type == 'in' or self._aggregation_type == 'no_agg'\
                    or self._aggregation_type == 'state_only':
                self.direct_map = nn.Dense(units=self._map_dim, flatten=False, prefix='direct_')

    def summary(self):
        logging.info("Graph %s: State Dim=%d, Type=%s, Aggregator=%s"
                     % (self._typ.upper(), self._state_dim,
                        self._aggregation_type, str(self._aggregator_args)))

    def hybrid_forward(self, F, data, states,
                       end_points, indptr, edge_data=None):
        """

        Parameters
        ----------
        F
        data : Symbol or NDArray
            Shape (batch_size, node_num, feat_dim)
        states : list Symbol or NDArray
            Shape (batch_size, node_num, state_dim)
        endpoints : Symbol or NDArray
            Shape (nnz_x2h, )
        indptr : Symbol or NDArray
            Shape (node_num + 1, )
        edge_data : Symbol or NDArray or None
            Shape (nnz_x2h, )
        Returns
        -------
        next_hidden : Symbol or NDArray
            Shape (batch_size, node_num, feat_dim)
        mid_info : None or list
        """
        if states is None:
            prev_h = F.broadcast_axis(F.slice_axis(F.zeros_like(data), begin=0, end=1, axis=2),
                                      axis=2, size=self._state_dim)
            if self._typ == 'lstm':
                prev_c = prev_h
        else:
            if self._typ == "lstm":
                prev_h, prev_c = states
            else:
                prev_h = states[0]
        if self._aggregation_type == 'all':
            concat_data = F.concat(data, prev_h, dim=-1)
            h_data, mid_info = shortcut_aggregator_forward(self.aggregator, concat_data,
                                                           concat_data, end_points,
                                                           indptr, edge_data)
        elif self._aggregation_type == 'in':
            h_data, mid_info = shortcut_aggregator_forward(self.aggregator, data,
                                                           data, end_points,
                                                           indptr, edge_data)
            h_data = h_data + self.direct_map(prev_h)
        elif self._aggregation_type == 'state_only':
            h_data, mid_info = shortcut_aggregator_forward(self.aggregator, data,
                                                           prev_h, end_points,
                                                           indptr, edge_data)
            h_data = h_data + self.direct_map(prev_h)
        elif self._aggregation_type == 'no_agg':
            h_data = self.direct_map(F.concat(data, prev_h, dim=-1))
            mid_info = []
        else:
            raise NotImplementedError
        if self._typ == 'lstm':
            h_data_l = F.split(h_data, num_outputs=4, axis=2)
            F_t = F.Activation(h_data_l[0], act_type="sigmoid")  # Forget Gate
            I_t = self._act(h_data_l[1])  # Input Gate
            O_t = F.Activation(h_data_l[2], act_type="sigmoid")  # Output Gate
            H_t = F.Activation(h_data_l[3], act_type="sigmoid")  # Info
            new_c = F_t * prev_c + I_t * H_t
            new_h = O_t * self._act(new_c)
            return [new_h, new_c], mid_info
        else:
            new_h = self._act(h_data)
            return [new_h], mid_info


class GraphGRUCell(HybridBlock):
    def __init__(self, aggregator_args, aggregation_type='all', typ='rnn', prefix=None, params=None):
        super(GraphGRUCell, self).__init__(prefix=prefix, params=params)
        assert len(aggregator_args) == 2
        self._aggregator_args = copy.deepcopy(aggregator_args)
        # In order to compute gates in GRU, we need to scale up the number of out_units
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._state_dim = aggregator_args[1][0]
        self._typ = "gru"
        self._map_dim = 3 * self._state_dim
        self._aggregation_type = aggregation_type
        with self.name_scope():
            self._aggregator_args[1][0] = self._map_dim
            self.aggregator_x2h = parse_aggregator_from_desc(
                aggregator_desc=[self._aggregator_args[0],
                                 self._aggregator_args[1],
                                 'agg_x2h_', 'identity'])
            self.aggregator_h2h = parse_aggregator_from_desc(
                aggregator_desc=[self._aggregator_args[0],
                                 self._aggregator_args[1],
                                 'agg_h2h_', 'identity'])
            if self._aggregation_type != 'concat':
                self.direct_h2h = nn.Dense(units=self._map_dim, flatten=False, prefix='direct_h2h_')

    def summary(self):
        logging.info("Graph %s: State Dim=%d, Aggregation Type=%s, Aggregator=%s"
                     % (self._typ.upper(), self._state_dim, self._aggregation_type, str(self._aggregator_args)))

    def hybrid_forward(self, F, data, states,
                       end_points, indptr, edge_data=None):
        if states is None:
            prev_h = F.broadcast_axis(F.slice_axis(F.zeros_like(data), begin=0, end=1, axis=2),
                                      axis=2, size=self._state_dim)
        else:
            prev_h = states[0]
        x2h_data, x2h_mid_info = shortcut_aggregator_forward(self.aggregator_x2h, data,
                                                         data, end_points,
                                                         indptr, edge_data)
        if self._aggregation_type != 'concat':
            h2h_data, h2h_mid_info = shortcut_aggregator_forward(self.aggregator_h2h, data,
                                                                 prev_h, end_points, indptr, edge_data)
            h2h_data = h2h_data + self.direct_h2h(h2h_data)
        else:
            h2h_data, h2h_mid_info = shortcut_aggregator_forward(self.aggregator_h2h,
                                                                 F.concat(data, prev_h, dim=-1),
                                                                 prev_h, end_points, indptr, edge_data)
        mid_info = x2h_mid_info + h2h_mid_info

        x2h_data_l = F.split(x2h_data, num_outputs=3, axis=2)
        h2h_data_l = F.split(h2h_data, num_outputs=3, axis=2)
        U_t = F.Activation(x2h_data_l[0] + h2h_data_l[0], act_type='sigmoid')
        R_t = F.Activation(x2h_data_l[1] + h2h_data_l[1], act_type='sigmoid')
        H_prime_t = self._act(x2h_data_l[2] + R_t * h2h_data_l[2])
        H_t = (1 - U_t) * H_prime_t + U_t * prev_h
        return [H_t], mid_info


class StackGraphRNN(HybridBlock):
    def __init__(self, out_units, aggregator_args_list, aggregation_type, rnn_type, dropout,
                 in_length, out_length, prefix=None, params=None):
        super(StackGraphRNN, self).__init__(prefix=prefix, params=params)
        self._in_length = in_length
        self._out_length = out_length
        self._layer_num = len(aggregator_args_list)
        self._out_units = out_units
        self._dropout = dropout
        self._aggregation_type = aggregation_type
        self._rnn_type = rnn_type
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        with self.name_scope():
            self.enc_pre_embed = nn.HybridSequential(prefix='enc_pre_embed_')
            self.dec_pre_embed = nn.HybridSequential(prefix='dec_pre_embed_')
            with self.enc_pre_embed.name_scope():
                self.enc_pre_embed.add(nn.Dense(units=16, flatten=False))
                self.enc_pre_embed.add(self._act)
            with self.dec_pre_embed.name_scope():
                self.dec_pre_embed.add(nn.Dense(units=16, flatten=False))
                self.dec_pre_embed.add(self._act)
            self.dropout_layer = nn.Dropout(rate=dropout)
            self.encoder_graph_rnn_cells = nn.HybridSequential()
            for i, layer_args in enumerate(aggregator_args_list):
                self.encoder_graph_rnn_cells.add(GraphGRUCell(aggregator_args=layer_args,
                                                              aggregation_type=aggregation_type,
                                                              typ=self._rnn_type,
                                                              prefix="enc_graph_rnn%d_"%i))
            self.decoder_graph_rnn_cells = nn.HybridSequential()
            for i, layer_args in enumerate(aggregator_args_list):
                self.decoder_graph_rnn_cells.add(GraphGRUCell(aggregator_args=layer_args,
                                                              aggregation_type=aggregation_type,
                                                              typ=self._rnn_type,
                                                              prefix="dec_graph_rnn%d_"%i))
            self.out_layer = nn.Dense(units=self._out_units, flatten=False,
                                      prefix="out_")

    def summary(self):
        logging.info("Stack Graph RNN: in_length=%d, out_length=%d"
                     % (self._in_length, self._out_length))
        logging.info("Encoder:")
        for i in range(len(self.encoder_graph_rnn_cells)):
            self.encoder_graph_rnn_cells[i].summary()
        logging.info("Decoder:")
        for i in range(len(self.decoder_graph_rnn_cells)):
            self.decoder_graph_rnn_cells[i].summary()

    def hybrid_forward(self, F, data_in, data_out, gt_prob, out_additional_feature,
                       end_points, indptr, edge_data=None):
        """

        Parameters
        ----------
        F
        data_in :
            Shape (in_length, batch_size, node_num, feat_dim)
            Will be normalized!!!
        data_out:
            Shape (out_length, batch_size, node_num, feat_dim)
            Will be normalized!!!
        gt_prob:
            Shape (1,)
            The probability to use the groundtruth
        out_additional_feature:
            Shape (out_length, batch_size, node_num, time_feat_dim)
        endpoints
            Shape (nnz_x2h,)
        indptr
            Shape (node_num + 1,)
        edge_data
            Shape (nnz_x2h, edge_feat_dim)
        Returns
        -------
        pred:
            Shape (out_length, batch_size, node_num, out_dim)
        enc_mid_info: list
        dec_mid_info: list
        """
        states = [None for _ in range(self._layer_num)]
        data_in = self.enc_pre_embed(data_in)
        data_in_l = F.split(data_in, num_outputs=self._in_length, axis=0, squeeze_axis=True)
        data_out_l = F.split(data_out, num_outputs=self._out_length, axis=0, squeeze_axis=True)
        out_additional_feature_l = F.split(out_additional_feature, num_outputs=self._out_length,
                                           axis=0, squeeze_axis=True)
        enc_mid_info = [[] for _ in range(self._layer_num)]
        dec_mid_info = [[] for _ in range(self._layer_num)]
        pred_l = []
        curr_in = None
        for j in range(self._in_length):
            curr_in = data_in_l[j]
            for i in range(self._layer_num):
                new_states, mid_info = self.encoder_graph_rnn_cells[i](curr_in, states[i],
                                                                    end_points, indptr, edge_data)
                states[i] = new_states
                enc_mid_info[i].extend(mid_info)
                curr_in = self.dropout_layer(new_states[0])
        pred_l.append(self.out_layer(curr_in))
        # Begin forecaster
        for j in range(self._out_length - 1):
            use_gt = F.random_uniform(0, 1) < gt_prob
            data_in_after_ss = F.broadcast_mul(use_gt, data_out_l[j])\
                               + F.broadcast_mul(1 - use_gt, pred_l[-1])
            curr_in = F.concat(data_in_after_ss, out_additional_feature_l[j], dim=-1)
            curr_in = self.dec_pre_embed(curr_in)
            for i in range(self._layer_num):
                new_states, mid_info = self.decoder_graph_rnn_cells[i](curr_in, states[i],
                                                                    end_points, indptr, edge_data)
                states[i] = new_states
                dec_mid_info[i].extend(mid_info)
                curr_in = self.dropout_layer(new_states[0])
            pred_l.append(self.out_layer(curr_in))
        pred = F.concat(*[F.expand_dims(ele, axis=0) for ele in pred_l], dim=0)
        enc_gates = []
        enc_sharpness = []
        enc_attention_weights = []
        dec_gates = []
        dec_sharpness = []
        dec_attention_weights = []
        for i in range(self._layer_num):
            if len(enc_mid_info[i]) > 0:
                all_gates = F.concat(*[F.expand_dims(ele, 0) for ele in enc_mid_info[i][::3]],
                                     dim=0)
                all_sharpness = F.concat(*[F.expand_dims(ele, 0) for ele in enc_mid_info[i][1::3]],
                                         dim=0)
                all_attention_weights = F.concat(
                    *[F.expand_dims(ele, 0) for ele in enc_mid_info[i][2::3]],
                    dim=0)
                enc_gates.append(all_gates)
                enc_sharpness.append(all_sharpness)
                enc_attention_weights.append(all_attention_weights)
            if len(dec_mid_info[i]) > 0:
                all_gates = F.concat(*[F.expand_dims(ele, 0) for ele in dec_mid_info[i][::3]],
                                     dim=0)
                all_sharpness = F.concat(*[F.expand_dims(ele, 0) for ele in dec_mid_info[i][1::3]],
                                         dim=0)
                all_attention_weights = F.concat(
                    *[F.expand_dims(ele, 0) for ele in dec_mid_info[i][2::3]], dim=0)
                dec_gates.append(all_gates)
                dec_sharpness.append(all_sharpness)
                dec_attention_weights.append(all_attention_weights)
        return pred, enc_gates, enc_sharpness, enc_attention_weights, \
               dec_gates, dec_sharpness, dec_attention_weights


# class StackGraphRNN2(StackGraphRNN):
#     def hybrid_forward(self, F, data_in, out_additional_feature,
#                        end_points, indptr, edge_data=None):
#         """
#
#         Parameters
#         ----------
#         F
#         data_in :
#             Shape (in_length, batch_size, node_num, feat_dim)
#             Will be normalized!!!
#         out_additional_feature:
#             Shape (out_length, batch_size, node_num, time_feat_dim)
#         endpoints
#             Shape (nnz_x2h,)
#         indptr
#             Shape (node_num + 1,)
#         edge_data
#             Shape (nnz_x2h, edge_feat_dim)
#         Returns
#         -------
#         pred:
#             Shape (out_length, batch_size, node_num, out_dim)
#         enc_mid_info: list
#         dec_mid_info: list
#         """
#         states = [None for _ in range(self._layer_num)]
#         data_in = self.enc_pre_embed(data_in)
#         data_in_l = F.split(data_in, num_outputs=self._in_length, axis=0, squeeze_axis=True)
#         out_additional_feature_l = F.split(out_additional_feature, num_outputs=self._out_length,
#                                            axis=0, squeeze_axis=True)
#         enc_mid_info = [[] for _ in range(self._layer_num)]
#         dec_mid_info = [[] for _ in range(self._layer_num)]
#         pred_l = []
#         for j in range(self._in_length):
#             curr_in = data_in_l[j]
#             for i in range(self._layer_num):
#                 new_states, mid_info = self.encoder_graph_rnn_cells[i](curr_in, states[i],
#                                                                        end_points, indptr,
#                                                                        edge_data)
#                 states[i] = new_states
#                 enc_mid_info[i].extend(mid_info)
#                 curr_in = self.dropout_layer(new_states[0])
#         # Reverse the states
#         for j in range(self._out_length):
#             curr_in = self.dec_pre_embed(out_additional_feature_l[j])
#             for i in range(self._layer_num - 1, -1, -1):
#                 new_states, mid_info = self.decoder_graph_rnn_cells[i](curr_in, states[i],
#                                                                        end_points, indptr,
#                                                                        edge_data)
#                 states[i] = new_states
#                 dec_mid_info[i].extend(mid_info)
#                 curr_in = self.dropout_layer(new_states[0])
#             pred_l.append(self.out_layer(curr_in))
#         pred = F.concat(*[F.expand_dims(ele, axis=0) for ele in pred_l], dim=0)
#         enc_gates = []
#         enc_sharpness = []
#         enc_attention_weights = []
#         dec_gates = []
#         dec_sharpness = []
#         dec_attention_weights = []
#         for i in range(self._layer_num):
#             if len(enc_mid_info[i]) > 0:
#                 all_gates = F.concat(*[F.expand_dims(ele, 0) for ele in enc_mid_info[i][::3]],
#                                      dim=0)
#                 all_sharpness = F.concat(*[F.expand_dims(ele, 0) for ele in enc_mid_info[i][1::3]],
#                                      dim=0)
#                 all_attention_weights = F.concat(*[F.expand_dims(ele, 0) for ele in enc_mid_info[i][2::3]],
#                                                  dim=0)
#                 enc_gates.append(all_gates)
#                 enc_sharpness.append(all_sharpness)
#                 enc_attention_weights.append(all_attention_weights)
#             if len(dec_mid_info[i]) > 0:
#                 all_gates = F.concat(*[F.expand_dims(ele, 0) for ele in dec_mid_info[i][::3]],
#                                      dim=0)
#                 all_sharpness = F.concat(*[F.expand_dims(ele, 0) for ele in dec_mid_info[i][1::3]],
#                                          dim=0)
#                 all_attention_weights = F.concat(
#                     *[F.expand_dims(ele, 0) for ele in dec_mid_info[i][2::3]], dim=0)
#                 dec_gates.append(all_gates)
#                 dec_sharpness.append(all_sharpness)
#                 dec_attention_weights.append(all_attention_weights)
#         return pred, enc_gates, enc_sharpness, enc_attention_weights,\
#                dec_gates, dec_sharpness, dec_attention_weights
