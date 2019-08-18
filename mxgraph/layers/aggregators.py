import math
import logging
from collections import namedtuple
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import mxnet.gluon as gluon
from mxnet.gluon import nn, HybridBlock
from mxgraph.config import cfg
from mxgraph.utils import safe_eval
from mxgraph.layers.common import *

GraphPoolAggregatorParam = namedtuple('GraphPoolAggregatorParam', cfg.AGGREGATOR.GRAPHPOOL.ARGS)
HeterGraphPoolAggregatorParam = namedtuple('HeterGraphPoolAggregatorParam', cfg.AGGREGATOR.HETERGRAPHPOOL.ARGS)
GraphWeightedSumParam = namedtuple('GraphWeightedSumParam', cfg.AGGREGATOR.GRAPH_WEIGHTED_SUM.ARGS)
GraphMultiWeightedSumParam = namedtuple('GraphMultiWeightedSumParam',
                                        cfg.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.ARGS)
MuGGAParam = namedtuple('MuGGAParam', cfg.AGGREGATOR.MUGGA.ARGS)
BiGraphPoolAggregatorParam = namedtuple('BiGraphPoolAggregatorParam', cfg.AGGREGATOR.BIGRAPHPOOL.ARGS)


class BaseGraphHybridAggregator(HybridBlock):
    @staticmethod
    def from_args(prefix=None, out_act=None, *args):
        raise NotImplementedError

    def hybrid_forward(self, F, data, neighbor_data, neighbor_indices, neighbor_indptr,
                       edge_data=None):
        """The basic aggregator

        Parameters
        ----------
        F
        data: Symbol or NDArray
            The input data: Shape (batch_size, node_num, feat_dim1)
        neighbor_data: Symbol or NDArray
            Data related to the input: Shape (batch_size, neighbor_node_num, feat_dim2)
        neighbor_indices: Symbol or NDArray
            Ids of the neighboring nodes: Shape (nnz,)
        neighbor_indptr: Symbol or NDArray
            Shape (node_num + 1, )
        edge_data: Symbol or NDArray or None
            The edge data: Shape (batch_size, nnz, edge_feat_dim)
        Returns
        -------

        """
        raise NotImplementedError


class GraphPoolAggregator(BaseGraphHybridAggregator):
    def __init__(self, out_units, mid_units, mid_layer_num, out_act=None, prefix=None, params=None):
        """The graph pool aggregator in SAGE.

        out = g(pool(f(h_N(i))) | h_i)

        Parameters
        ----------
        out_units: int
        mid_units: int
        mid_layer_num: int
        dropout: float
            The dropout probability
        pool_type: str
        act: str
        out_act: str
        prefix: None or ...
        params: None or ...
        """
        super(GraphPoolAggregator, self).__init__(prefix=prefix, params=params)
        self._out_units = out_units
        self._mid_units = mid_units
        self._mid_layer_num = mid_layer_num
        self._pool_type = cfg.AGGREGATOR.GRAPHPOOL.POOL_TYPE
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._out_act = get_activation(out_act)\
            if out_act is not None else get_activation(cfg.AGGREGATOR.ACTIVATION)
        with self.name_scope():
            self.mid_map = DenseNetBlock(units=mid_units, layer_num=mid_layer_num, act=self._act,
                                         flatten=False, prefix='mid_map_')
            self.out_layer = nn.HybridSequential(prefix='out_')
            with self.out_layer.name_scope():
                self.out_layer.add(nn.Dense(self._out_units, flatten=False))
                self.out_layer.add(self._out_act)

    @staticmethod
    def from_args(prefix=None, out_act=None, *args):
        param = GraphPoolAggregatorParam(*args)
        logging.info("GraphPoolAggregator, Param=%s, Prefix=%s" % (str(param), str(prefix)))
        return GraphPoolAggregator(out_units=param.out_units,
                                   mid_units=param.mid_units,
                                   mid_layer_num=param.mid_layer_num,
                                   out_act=out_act,
                                   prefix=prefix)

    def hybrid_forward(self, F, data, neighbor_data, neighbor_indices, neighbor_indptr,
                       edge_data=None):
        """Map the input features to hidden states + apply pooling + apply FC

        Parameters
        ----------
        F
        data : Symbol or NDArray
            Shape (batch_size, node_num, feat_dim)
        neighbor_data : Symbol or NDArray
            Shape (batch_size, neighbor_node_num, feat_dim)
        neighbor_indices : Symbol or NDArray
            Shape (nnz, )
        neighbor_indptr : Symbol or NDArray
            Shape (node_num + 1, )
        edge_data : Symbol or NDArray or None
            Shape (nnz, edge_dim)

        Returns
        -------

        """
        neighbor_feats = self.mid_map(neighbor_data)
        if self._pool_type == "mixed":
            raise DeprecationWarning("mixed is deprecated!!!")
            mean_pool_data = F.contrib.seg_pool(data=neighbor_data,
                                                indices=neighbor_indices,
                                                indptr=neighbor_indptr,
                                                pool_type="avg")  # Shape(batch_size, node_num, feat_dim2)
            max_pool_data = F.contrib.seg_pool(data=neighbor_feats,
                                               indices=neighbor_indices,
                                               indptr=neighbor_indptr,
                                               pool_type="max")  # Shape(batch_size, node_num, mid_units)
            out = self.out_layer(F.concat(mean_pool_data, max_pool_data, data, dim=-1))
        else:
            pool_data = F.contrib.seg_pool(data=neighbor_feats,
                                           indices=neighbor_indices,
                                           indptr=neighbor_indptr,
                                           pool_type=self._pool_type)  # Shape(batch_size, node_num, mid_units)
            out = self.out_layer(F.concat(pool_data, data, dim=-1))
        return out


class HeterGraphPoolAggregator(BaseGraphHybridAggregator):
    def __init__(self, out_units, mid_units, mid_layer_num, num_set, out_act=None, prefix=None, params=None):
        """
        The graph pool aggregator in SAGE.

        out = g(pool(f(h_N(i))) | h_i)

        Parameters
        ----------
        out_units: int
        mid_units: int
        mid_layer_num: int
        num_set: int
        dropout: float
            The dropout probability
        pool_type: str
        act: str
        out_act: str
        prefix: None or ...
        params: None or ...
        """
        super(HeterGraphPoolAggregator, self).__init__(prefix=prefix, params=params)
        self._out_units = out_units
        self._mid_units = mid_units
        self._mid_layer_num = mid_layer_num
        self._num_set = num_set
        self._pool_type = cfg.AGGREGATOR.HETERGRAPHPOOL.POOL_TYPE
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._out_act = get_activation(out_act)\
            if out_act is not None else get_activation(cfg.AGGREGATOR.ACTIVATION)
        with self.name_scope():
            self.data_map = HeterDenseNetBlock(units=out_units, layer_num=1, act='identity', num_set=num_set,
                                               flatten=False, prefix='data_map_')
            self.neighbor_mid_map = HeterDenseNetBlock(units=mid_units, layer_num=mid_layer_num, act=self._act,
                                                       num_set=num_set, flatten=False, prefix='neighbor_mid_map_')

            self.out_layer = nn.HybridSequential(prefix='out_')
            with self.out_layer.name_scope():
                self.out_layer.add(nn.Dense(self._out_units, flatten=False))
                self.out_layer.add(self._out_act)

    @staticmethod
    def from_args(prefix=None, out_act=None, *args):
        param = HeterGraphPoolAggregatorParam(*args)
        logging.info("HeterGraphPoolAggregator, Param=%s, Prefix=%s" % (str(param), str(prefix)))
        return HeterGraphPoolAggregator(out_units=param.out_units,
                                        mid_units=param.mid_units,
                                        mid_layer_num=param.mid_layer_num,
                                        num_set=param.num_set,
                                        out_act=out_act,
                                        prefix=prefix)

    def hybrid_forward(self, F, data, neighbor_data, data_mask, neighbor_mask, neighbor_indices, neighbor_indptr,
                       edge_data=None):
        """Map the input features to hidden states + apply pooling + apply FC

        Parameters
        ----------
        F
        data : Symbol or NDArray
            Shape (batch_size, node_num, feat_dim)
        neighbor_data : Symbol or NDArray
            Shape (batch_size, neighbor_node_num, feat_dim)
        data_mask :  Symbol or NDArray
            Shape (batch_size, node_num, num_set, 1)
        neighbor_mask : Symbol or NDArray
            Shape (batch_size, neighbor_node_num, num_set, 1)
        neighbor_indices : Symbol or NDArray
            Shape (nnz, )
        neighbor_indptr : Symbol or NDArray
            Shape (node_num + 1, )
        edge_data : Symbol or NDArray or None
            Shape (nnz, edge_dim)

        Returns
        -------

        """
        data = self.data_map(data, data_mask)
        neighbor_feats = self.neighbor_mid_map(neighbor_data, neighbor_mask)
        if self._pool_type == "mixed":
            raise DeprecationWarning("mixed is deprecated!!!")
            mean_pool_data = F.contrib.seg_pool(data=neighbor_data,
                                                indices=neighbor_indices,
                                                indptr=neighbor_indptr,
                                                pool_type="avg")  # Shape(batch_size, node_num, feat_dim2)
            max_pool_data = F.contrib.seg_pool(data=neighbor_feats,
                                               indices=neighbor_indices,
                                               indptr=neighbor_indptr,
                                               pool_type="max")  # Shape(batch_size, node_num, mid_units)
            out = self.out_layer(F.concat(mean_pool_data, max_pool_data, data, dim=-1))
        else:
            pool_data = F.contrib.seg_pool(data=neighbor_feats,
                                           indices=neighbor_indices,
                                           indptr=neighbor_indptr,
                                           pool_type=self._pool_type)  # Shape(batch_size, node_num, mid_units)
            out = self.out_layer(F.concat(pool_data, data, dim=-1))
        return out


class GraphWeightedSumAggregator(BaseGraphHybridAggregator):
    def __init__(self, out_units, mid_units, attend_units, out_act=None, prefix=None, params=None):
        """Weighted sum of the

        Parameters
        ----------
        out_units
        mid_units
        gate_units
        out_act
        prefix
        params
        """
        super(GraphWeightedSumAggregator, self).__init__(prefix=prefix, params=params)
        self._out_units = out_units
        self._mid_units = mid_units
        self._attend_units = attend_units
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._out_act = get_activation(out_act) \
            if out_act is not None else get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._weight_act = get_activation(cfg.AGGREGATOR.GRAPH_WEIGHTED_SUM.WEIGHT_ACT)
        with self.name_scope():
            self.mid_map = DenseNetBlock(units=mid_units, layer_num=1, act=self._act,
                                         flatten=False, prefix='mid_map_')
            self.data_attend_embed = nn.Dense(self._attend_units, flatten=False,
                                              prefix='data_attend_embed_')
            self.neighbor_attend_embed = nn.Dense(self._attend_units, flatten=False,
                                                  prefix='neighbor_attend_embed_')
            self.out_layer = nn.HybridSequential(prefix='out_')
            self.attend_w_dropout = nn.Dropout(cfg.AGGREGATOR.GRAPH_WEIGHTED_SUM.ATTEND_W_DROPOUT)
            with self.out_layer.name_scope():
                self.out_layer.add(nn.Dense(self._out_units, flatten=False))
                self.out_layer.add(self._out_act)

    @staticmethod
    def from_args(prefix=None, out_act=None, *args):
        param = GraphWeightedSumParam(*args)
        logging.info("GraphWeightedSumAggregator, Param=%s, Prefix=%s" % (str(param), str(prefix)))
        return GraphWeightedSumAggregator(out_units=param.out_units,
                                          mid_units=param.mid_units,
                                          attend_units=param.attend_units,
                                          out_act=out_act,
                                          prefix=prefix)

    def hybrid_forward(self, F, data, neighbor_data, neighbor_indices, neighbor_indptr,
                       edge_data=None):
        """Map the input features to hidden states + apply weighted sum + apply FC

        Parameters
        ----------
        F
        data : Symbol or NDArray
            Shape (batch_size, node_num, feat_dim)
        neighbor_data : Symbol or NDArray
            Shape (batch_size, neighbor_node_num, feat_dim)
        neighbor_indices : Symbol or NDArray
            Shape (nnz, )
        neighbor_indptr : Symbol or NDArray
            Shape (node_num + 1, )
        edge_data : Symbol or NDArray or None
            Shape (nnz, edge_dim)

        Returns
        -------

        """
        neighbor_feats = self.mid_map(neighbor_data)  # Shape (batch_size, neighbor_node_num, mid_units)
        data_attend_embed = self.data_attend_embed(data)  # Shape (batch_size, node_num, attend_units)
        neighbor_attend_embed = self.neighbor_attend_embed(neighbor_data)  # Shape (batch_size, neighbor_node_num, attend_units)
        attend_score = \
            F.contrib.seg_take_k_corr(embed1=data_attend_embed,
                                      embed2=neighbor_attend_embed,
                                      neighbor_ids=neighbor_indices,
                                      neighbor_indptr=neighbor_indptr)  # Shape(batch_size, nnz)
        weights = self._weight_act(attend_score)  # Shape(batch_size, nnz)
        if cfg.AGGREGATOR.GRAPH_WEIGHTED_SUM.DIVIDE_SIZE:
            neighborhood_size = F.contrib.seg_sum(data=F.ones_like(weights), indptr=neighbor_indptr)
            neighborhood_size = F.maximum(neighborhood_size, F.ones_like(neighborhood_size))
            neighborhood_size = F.cast(neighborhood_size, dtype=np.float32)  # Shape (batch_size, node_num)
            weights = F.contrib.seg_broadcast_mul(weights, 1.0 / neighborhood_size, neighbor_indptr)

        weights = self.attend_w_dropout(weights)
        weighted_sum_feats = F.contrib.seg_weighted_pool(data=neighbor_feats,
                                                         weights=weights,
                                                         indices=neighbor_indices,
                                                         indptr=neighbor_indptr)  # Shape (batch_size, node_num, mid_units)
        out = self.out_layer(F.concat(weighted_sum_feats, data, dim=-1))
        return out


class GraphMultiWeightedSumAggregator(BaseGraphHybridAggregator):
    def __init__(self, out_units, mid_units, attend_units, K, out_act=None, prefix=None, params=None):
        """Weighted sum of the

        Parameters
        ----------
        out_units
        mid_units
        attend_units
        K
        out_act
        prefix
        params
        """
        super(GraphMultiWeightedSumAggregator, self).__init__(prefix=prefix, params=params)
        self._out_units = out_units
        self._mid_units = mid_units
        self._attend_units = attend_units
        self._K = K
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._out_act = get_activation(out_act) \
            if out_act is not None else get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._weight_act = get_activation(cfg.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.WEIGHT_ACT)
        with self.name_scope():
            self.mid_map = DenseNetBlock(units=self._K * mid_units, layer_num=1, act=self._act,
                                         flatten=False, prefix='mid_map_')
            self.data_attend_embed = nn.Dense(self._K * self._attend_units, flatten=False,
                                              prefix='data_attend_embed_')
            self.neighbor_attend_embed = nn.Dense(self._K * self._attend_units, flatten=False,
                                                  prefix='neighbor_attend_embed_')
            self.out_layer = nn.HybridSequential(prefix='out_')
            self.attend_w_dropout = nn.Dropout(cfg.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.ATTEND_W_DROPOUT)
            with self.out_layer.name_scope():
                self.out_layer.add(nn.Dense(self._out_units, flatten=False))
                self.out_layer.add(self._out_act)

    @staticmethod
    def from_args(prefix=None, out_act=None, *args):
        param = GraphMultiWeightedSumParam(*args)
        logging.info("GraphMultiWeightedSumAggregator, Param=%s, Prefix=%s" % (str(param), str(prefix)))
        return GraphMultiWeightedSumAggregator(out_units=param.out_units,
                                               mid_units=param.mid_units,
                                               attend_units=param.attend_units,
                                               K=param.K,
                                               out_act=out_act,
                                               prefix=prefix)

    def hybrid_forward(self, F, data, neighbor_data, neighbor_indices, neighbor_indptr,
                       edge_data=None):
        """Map the input features to hidden states + apply weighted sum + apply FC

        Parameters
        ----------
        F
        data : Symbol or NDArray
            Shape (batch_size, node_num, feat_dim)
        neighbor_data : Symbol or NDArray
            Shape (batch_size, neighbor_node_num, feat_dim)
        neighbor_indices : Symbol or NDArray
            Shape (nnz, )
        neighbor_indptr : Symbol or NDArray
            Shape (node_num + 1, )
        edge_data : Symbol or NDArray or None
            Shape (nnz, edge_dim)

        Returns
        -------

        """
        neighbor_feats = self.mid_map(neighbor_data)  # Shape (batch_size, neighbor_node_num, K * mid_units)
        neighbor_feats = F.reshape(F.transpose(F.reshape(neighbor_feats,
                                               shape=(0, 0, self._K, self._mid_units)),
                                               axes=(0, 2, 1, 3)),
                                   shape=(-1, 0, 0),
                                   reverse=True)  # Shape (batch_size * K, neighbor_node_num, mid_units)
        data_attend_embed = self.data_attend_embed(data)  # Shape (batch_size, node_num, K * attend_units)
        data_attend_embed = F.reshape(F.transpose(F.reshape(data_attend_embed,
                                                            shape=(
                                                            0, 0, self._K, self._attend_units)),
                                                  axes=(0, 2, 1, 3)),
                                      shape=(-1, 0, 0),
                                      reverse=True)  # Shape (batch_size * K, node_num, attend_units)
        neighbor_attend_embed = self.neighbor_attend_embed(neighbor_data)  # Shape (batch_size, neighbor_node_num, K * attend_units)
        neighbor_attend_embed = F.reshape(F.transpose(F.reshape(neighbor_attend_embed,
                                                                shape=(
                                                                0, 0, self._K, self._attend_units)),
                                                      axes=(0, 2, 1, 3)),
                                          shape=(-1, 0, 0),
                                          reverse=True)  # Shape (batch_size * K, neighbor_node_num, attend_units)
        attend_score = \
            F.contrib.seg_take_k_corr(embed1=data_attend_embed,
                                      embed2=neighbor_attend_embed,
                                      neighbor_ids=neighbor_indices,
                                      neighbor_indptr=neighbor_indptr)  # Shape(batch_size * K, nnz)
        weights = self._weight_act(attend_score)  # Shape(batch_size * K, nnz)
        if cfg.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.DIVIDE_SIZE:
            neighborhood_size = F.contrib.seg_sum(data=F.ones_like(weights), indptr=neighbor_indptr)
            neighborhood_size = F.maximum(neighborhood_size, F.ones_like(neighborhood_size))
            neighborhood_size = F.cast(neighborhood_size, dtype=np.float32)  # Shape (batch_size * K, node_num)
            weights = F.contrib.seg_broadcast_mul(weights, 1.0 / neighborhood_size, neighbor_indptr)

        weights = self.attend_w_dropout(weights)
        multi_weighted_sum_feats = F.contrib.seg_weighted_pool(data=neighbor_feats,
                                                         weights=weights,
                                                         indices=neighbor_indices,
                                                         indptr=neighbor_indptr)  # Shape (batch_size * K, node_num, mid_units)
        multi_weighted_sum_feats = F.reshape(F.transpose(F.reshape(multi_weighted_sum_feats,
                                                               shape=(-1, self._K, 0, 0),
                                                               reverse=True),
                                                     axes=(0, 2, 1, 3)),
                                         shape=(0, 0, -1))  # Shape (batch_size, node_num, K * value_units)
        out = self.out_layer(F.concat(multi_weighted_sum_feats, data, dim=-1))
        return out


class MuGGAContextNet(HybridBlock):
    def __init__(self, mid_units, mid_layer_num, K, prefix=None, params=None):
        super(MuGGAContextNet, self).__init__(prefix=prefix, params=params)
        # Parse parameters from cfg
        local_cfg = cfg.AGGREGATOR.MUGGA.CONTEXT
        self._mid_units = mid_units
        self._mid_layer_num = mid_layer_num
        self._K = K
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._use_max_pool = local_cfg.USE_MAX_POOL
        self._use_sum_pool = local_cfg.USE_SUM_POOL
        self._use_avg_pool = local_cfg.USE_AVG_POOL
        self._use_gate = local_cfg.USE_GATE
        self._use_sharpness = local_cfg.USE_SHARPNESS
        if not self._use_gate and not self._use_sharpness:
            print('Context Net is not used!')
            return
        assert not (self._use_max_pool is False and self._use_sum_pool is False
                    and self._use_avg_pool is False)
        # Define the layers
        with self.name_scope():
            if self._use_max_pool:
                self.max_pool_fc = nn.Dense(units=self._mid_units, flatten=False, prefix='max_pool_fc_')
            self.mid_mapping = DenseNetBlock(units=self._mid_units, layer_num=self._mid_layer_num,
                                             act=self._act, flatten=False, prefix='mid_map_')
            if self._use_gate:
                self.gate = nn.HybridSequential(prefix='gate_')
                with self.gate.name_scope():
                    self.gate.add(nn.Dense(units=self._K, flatten=False))
                    ### set lr_mult = 0.1
                    self.gate[0].weight.lr_mult = 0.1
                    self.gate[0].bias.lr_mult = 0.1
                    self.gate.add(nn.Activation('sigmoid'))

            if self._use_sharpness:
                self.sharpness = nn.HybridSequential(prefix='sharpness_')
                with self.sharpness.name_scope():
                    self.sharpness.add(nn.Dense(units=self._K, flatten=False))
                    ### set lr_mult = 0.1
                    self.sharpness[0].weight.lr_mult = 0.1
                    self.sharpness[0].bias.lr_mult = 0.1
                    self.sharpness.add(nn.Activation('softrelu'))

    def hybrid_forward(self, F, data, neighbor_data, neighbor_indices, neighbor_indptr, edge_data=None):
        """Generate the gate, sharpness and other parameters for the MuGGA layer

            TODO(sxjscience) Add edge_data: Shape (nnz, edge_feat)

        Parameters
        ----------
        F:
        data: Symbol or NDArray
            The input data: Shape (batch_size, node_num, data_feat_dim)
        neighbor_data: Symbol or NDArray
            Data related to the input: Shape (batch_size, neighbor_node_num, ndata_feat_dim)
        neighbor_indices: Symbol or NDArray
            Ids of the neighboring nodes: Shape (nnz,)
        neighbor_indptr: Symbol or NDArray
            Shape (node_num + 1, )
        edge_data: Symbol or NDArray or None
            Shape (nnz, edge_feat)

        Returns
        -------
        gate: Symbol or NDArray
            Shape (batch_size, node_num, self._K)
        sharpness: Symbol or NDArray
            Shape (batch_size, node_num, self._K)
        """
        if not self._use_gate and not self._use_sharpness:
            return None, None
        pool_data_l = []
        if self._use_max_pool:
            max_pool_data = self.max_pool_fc(neighbor_data)
            max_pool_data = F.contrib.seg_pool(data=max_pool_data,
                                               indices=neighbor_indices,
                                               indptr=neighbor_indptr,
                                               pool_type='max')  # Shape(batch_size, node_num, mid_units)
            pool_data_l.append(max_pool_data)
        if self._use_sum_pool:
            raise DeprecationWarning("Sum pooling is deprecated")
            sum_pool_data = F.contrib.seg_pool(data=neighbor_data,
                                               indices=neighbor_indices,
                                               indptr=neighbor_indptr,
                                               pool_type='sum')  # Shape(batch_size, node_num, mid_units)
            pool_data_l.append(sum_pool_data)
        if self._use_avg_pool:
            avg_pool_data = F.contrib.seg_pool(data=neighbor_data,
                                               indices=neighbor_indices,
                                               indptr=neighbor_indptr,
                                               pool_type='avg')  # Shape(batch_size, node_num, mid_units)
            pool_data_l.append(avg_pool_data)
        if len(pool_data_l) > 1:
            pool_data = F.concat(*pool_data_l, dim=-1)
        else:
            pool_data = pool_data_l[0]
        embed = self.mid_mapping(pool_data)
        if self._use_gate:
            gate = self.gate(embed)
        else:
            gate = None
        if self._use_sharpness:
            sharpness = self.sharpness(embed)
        else:
            sharpness = None
        return gate, sharpness


class MuGGA(BaseGraphHybridAggregator):
    def __init__(self, out_units, attend_units, value_units, K, context_units, context_layer_num, out_act=None,
                 prefix=None, params=None):
        """

        Parameters
        ----------
        out_units: int
            Size of the output channel
        attend_units: int
            The dimension we project the features to calculate the attention
        K:
            number of heads
        context_param: MuGGAContextParam
        act
        use_edge
        prefix
        params
        """
        super(MuGGA, self).__init__(prefix=prefix, params=params)
        self._out_units = out_units
        self._attend_units = attend_units
        self._value_units = value_units
        self._K = K
        self._context_units = context_units
        self._context_layer_num = context_layer_num
        self._out_act = get_activation(out_act)\
            if out_act is not None else get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._use_edge = cfg.AGGREGATOR.MUGGA.USE_EDGE
        with self.name_scope():
            self.context_net = MuGGAContextNet(mid_units=self._context_units,
                                               mid_layer_num=self._context_layer_num,
                                               K=K,
                                               prefix="context_")
            self.data_attend_embed = nn.Dense(self._K * self._attend_units, flatten=False,
                                              prefix='data_attend_embed_')
            self.neighbor_attend_embed = nn.Dense(self._K * self._attend_units, flatten=False,
                                                  prefix='neighbor_attend_embed_')
            self.neighbor_value = nn.HybridSequential(prefix='neighbor_value_')
            with self.neighbor_value.name_scope():
                self.neighbor_value.add(nn.Dense(self._K * self._value_units, flatten=False))
                self.neighbor_value.add(self._act)
            self.attend_w_dropout = nn.Dropout(cfg.AGGREGATOR.MUGGA.ATTEND_W_DROPOUT)
            if self._use_edge:
                self.edge_score = nn.HybridSequential(prefix='edge_score_')
                with self.edge_score.name_scope():
                    self.edge_score.add(nn.Dense(self._attend_units, flatten=False))
                    self.edge_score.add(self._act)
                    self.edge_score.add(nn.Dense(self._K, flatten=False))
            self.out_layer = nn.HybridSequential(prefix='out_')
            with self.out_layer.name_scope():
                self.out_layer.add(nn.Dense(self._out_units, flatten=False))
                self.out_layer.add(self._out_act)

    @staticmethod
    def from_args(prefix=None, out_act=None, *args):
        param = MuGGAParam(*args)
        logging.info("MuGGA Layer, Param=%s, Prefix=%s" % (str(param), str(prefix)))
        return MuGGA(out_units=param.out_units,
                     attend_units=param.attend_units,
                     value_units=param.value_units,
                     K=param.K,
                     context_units=param.context_units,
                     context_layer_num=param.context_layer_num,
                     out_act=out_act,
                     prefix=prefix)

    def hybrid_forward(self, F, data, neighbor_data, neighbor_indices, neighbor_indptr,
                       edge_data=None):
        """

        Parameters
        ----------
        F
        data: Symbol or NDArray
            The input data: Shape (batch_size, node_num, feat_dim1)
        neighbor_data: Symbol or NDArray
            Data related to the input: Shape (batch_size, neighbor_node_num, feat_dim2)
        neighbor_indices: Symbol or NDArray
            Ids of the neighboring nodes: Shape (nnz,)
        neighbor_indptr: Symbol or NDArray
            Shape (node_num + 1, )
        edge_data: Symbol or NDArray or None
            The edge data: Shape (nnz, edge_feat_dim)
        Returns
        -------
        out: Symbol or NDArray
            Shape (batch_size, node_num, out_dim)
        gate: Symbol or NDArray
            Shape (batch_size, node_num, K)
        sharpness: Symbol or NDArray
            Shape (batch_size, node_num, K)
        attend_weights_wo_gate: Symbol or NDArray
            Shape (batch_size, node_num, K)
        """
        if not self._use_edge:
            assert edge_data is None
        else:
            assert edge_data is not None
        gate, sharpness = self.context_net(data, neighbor_data, neighbor_indices, neighbor_indptr)  # Both have shape (batch_size, node_num, K)
        data_attend_embed = self.data_attend_embed(data)  # Shape (batch_size, node_num, K * attend_units)
        data_attend_embed = F.reshape(F.transpose(F.reshape(data_attend_embed,
                                                            shape=(0, 0, self._K, self._attend_units)),
                                                  axes=(0, 2, 1, 3)),
                                      shape=(-1, 0, 0), reverse=True)  # Shape (batch_size * K, node_num, attend_units)
        neighbor_attend_embed = self.neighbor_attend_embed(neighbor_data)    # Shape (batch_size, neighbor_node_num, K * attend_units)
        neighbor_attend_embed = F.reshape(F.transpose(F.reshape(neighbor_attend_embed,
                                                                shape=(0, 0, self._K, self._attend_units)),
                                                      axes=(0, 2, 1, 3)),
                                          shape=(-1, 0, 0),
                                          reverse=True)  # Shape (batch_size * K, neighbor_node_num, attend_units)
        if cfg.AGGREGATOR.MUGGA.RESCALE_INNERPRODUCT:
            data_attend_embed = data_attend_embed / math.sqrt(self._attend_units)
            neighbor_attend_embed = neighbor_attend_embed / math.sqrt(self._attend_units)
        attend_score =\
            F.contrib.seg_take_k_corr(embed1=data_attend_embed,
                                      embed2=neighbor_attend_embed,
                                      neighbor_ids=neighbor_indices,
                                      neighbor_indptr=neighbor_indptr)  # Shape(batch_size * K, nnz)

        if edge_data is not None:
            # Add edge information!
            edge_score = self.edge_score(edge_data)  # Shape(nnz, K)
            edge_score = F.expand_dims(F.transpose(edge_score, axes=(1, 0)), axis=0)  # Shape (1, K, nnz)
            attend_score = F.broadcast_add(F.reshape(attend_score, shape=(-1, self._K, 0), reverse=True),
                                           edge_score)
            attend_score = F.reshape(attend_score, shape=(-1, 0), reverse=True)
        if cfg.AGGREGATOR.MUGGA.CONTEXT.USE_SHARPNESS:
            attend_score = F.contrib.seg_broadcast_mul(attend_score,
                                                       F.reshape(F.transpose(sharpness, axes=(0, 2, 1)),
                                                                 shape=(-1, 0), reverse=True),
                                                       neighbor_indptr)
        attend_weights = F.contrib.seg_softmax(attend_score, neighbor_indptr)
        attend_weights_wo_gate = F.reshape(attend_weights, shape=(-1, self._K, 0), reverse=True)
        ### TODO add dropout for attend weight
        attend_weights = self.attend_w_dropout(attend_weights)

        if cfg.AGGREGATOR.MUGGA.CONTEXT.USE_GATE:
            attend_weights = F.contrib.seg_broadcast_mul(attend_weights,
                                                         F.reshape(F.transpose(gate, axes=(0, 2, 1)),
                                                                   shape=(-1, 0), reverse=True),
                                                         neighbor_indptr)  # Shape(batch_size * K, nnz)
        neighbor_value = self.neighbor_value(neighbor_data)  # Shape (batch_size, neighbor_node_num, K * value_units)
        neighbor_value = F.transpose(F.reshape(neighbor_value, shape=(0, 0, self._K, self._value_units)), axes=(0, 2, 1, 3))  # Shape(batch_size, K, neighbor_node_num, value_units)
        neighbor_value = F.reshape(neighbor_value, shape=(-1, 0, 0), reverse=True)  # Shape(batch_size * K, neighbor_node_num, value_units)
        multi_head_attention = F.contrib.seg_weighted_pool(data=neighbor_value,
                                                           weights=attend_weights,
                                                           indices=neighbor_indices,
                                                           indptr=neighbor_indptr)  # Shape (batch_size * K, node_num, value_units)
        multi_head_attention = F.reshape(F.transpose(F.reshape(multi_head_attention,
                                                               shape=(-1, self._K, 0, 0),
                                                               reverse=True),
                                                     axes=(0, 2, 1, 3)),
                                         shape=(0, 0, -1))  # Shape (batch_size, node_num, K * value_units)
        embed_all = F.concat(multi_head_attention, data, dim=-1)  # Shape (batch_size, node_num, K * value_units + data_feat_dim)
        out = self.out_layer(embed_all)
        if gate is None:
            gate = F.zeros(1)
        if sharpness is None:
            sharpness = F.zeros(1)
        return out, gate, sharpness, attend_weights_wo_gate





class BaseHeterGraphHybridAggregator(Block):
    @staticmethod
    def from_args(prefix=None, out_act=None, *args):
        raise NotImplementedError

    def forward(self, data, neighbor_data, neighbor_indices, neighbor_indptr,
                node_type_mask=None, neighbor_type_mask=None, edge_type_mask=None, seg_indices=None):
        """The basic aggregator

        Parameters
        ----------
        F
        data: Symbol or NDArray
            The input data: Shape (batch_size, node_num, feat_dim1)
        neighbor_data: Symbol or NDArray
            Data related to the input: Shape (batch_size, neighbor_node_num, feat_dim2)
        neighbor_indices: Symbol or NDArray
            Ids of the neighboring nodes: Shape (nnz,)
        neighbor_indptr: Symbol or NDArray
            Shape (node_num + 1, )
        edge_type_mask: Symbol or NDArray or None
            The edge data: Shape (batch_size, nnz, edge_type，1)
        seg_indices: Symbol or NDArray
            Ids of the edges as arange(nnz): Shape (nnz,)
        Returns
        -------

        """
        raise NotImplementedError


class BiGraphPoolAggregator(BaseHeterGraphHybridAggregator):
    def __init__(self, out_units, mid_units, num_node_set, num_edge_set,
                 out_act=None, prefix=None, params=None):
        super(BiGraphPoolAggregator, self).__init__(prefix=prefix, params=params)
        self._out_units = out_units
        self._mid_units = mid_units
        self._num_node_set = num_node_set
        self._num_edge_set = num_edge_set
        self._pool_type = cfg.AGGREGATOR.BIGRAPHPOOL.POOL_TYPE
        self._accum_type = cfg.AGGREGATOR.BIGRAPHPOOL.ACCUM_TYPE
        #self._act = get_activation(cfg.AGGREGATOR.ACTIVATION)
        self._out_act = get_activation(out_act)\
            if out_act is not None else get_activation(cfg.AGGREGATOR.ACTIVATION)
        with self.name_scope():
            if num_node_set is not None:
                self.data_map = HeterDenseNetBlock(units=out_units, layer_num=1, act='identity',
                                                   num_set=num_node_set, flatten=False, prefix='data_node_map_')
                self.neighbor_mid_map = HeterDenseNetBlock(units=mid_units, layer_num=1, act='identity',
                                                           num_set=num_node_set, flatten=False, prefix='neighbor_node_map_')
            if num_edge_set is not None:
                self.relation_W = nn.Dense(self._mid_units*self._num_edge_set, flatten=False, use_bias=False, prefix='relation_W_')
            # self.out_layer = nn.Sequential(prefix='out_')
            # with self.out_layer.name_scope():
            #     self.out_layer.add(nn.Dense(self._out_units, flatten=False))
            #     self.out_layer.add(self._out_act)

    @staticmethod
    def from_args(prefix=None, out_act=None, *args):
        param = BiGraphPoolAggregatorParam(*args)
        logging.info("BiGraphPoolAggregator, Param=%s, Prefix=%s" % (str(param), str(prefix)))
        return BiGraphPoolAggregator(out_units=param.out_units,
                                     mid_units=param.mid_units,
                                     num_node_set=param.num_node_set,
                                     num_edge_set=param.num_edge_set,
                                     out_act=out_act,
                                     prefix=prefix)

    def forward(self, data, neighbor_data, neighbor_indices, neighbor_indptr,
                       node_type_mask=None, neighbor_type_mask=None, edge_type_mask=None, seg_indices=None):
        """Map the input features to hidden states + apply pooling + apply FC

        Parameters
        ----------
        F
        data : Symbol or NDArray
            Shape (batch_size, node_num, feat_dim)
        neighbor_data : Symbol or NDArray
            Shape (batch_size, neighbor_node_num, feat_dim)
        data_mask :  Symbol or NDArray
            Shape (batch_size, node_num, num_set, 1)
        neighbor_mask : Symbol or NDArray
            Shape (batch_size, neighbor_node_num, num_set, 1)
        neighbor_indices : Symbol or NDArray
            Shape (nnz, )
        neighbor_indptr : Symbol or NDArray
            Shape (node_num + 1, )
        edge_data : Symbol or NDArray or None
            Shape (batch_size, nnz, num_edge_num, 1)

        Returns
        -------

        """
        ## TODO does not consider node type
        if self._num_node_set is not None:
            #print("data", data.shape)
            #print("node_type_mask", node_type_mask.shape)
            data = self.data_map(data, node_type_mask)
            neighbor_data = self.neighbor_mid_map(neighbor_data, neighbor_type_mask)
        if self._num_edge_set is not None:
            neighbor_data = self.relation_W(neighbor_data)  ### (batch_size, neighbor_node_num, mid_units*num_edge_set)
            neighbor_data = nd.take(neighbor_data, indices=neighbor_indices, axis=-2) ## (batch_size, nnz, mid_units*num_edge_set)
            #print("neighbor_data", neighbor_data.shape)
            neighbor_data = nd.reshape(neighbor_data,
                                       shape=(0, 0, self._num_edge_set, self._mid_units)) ## (batch_size, nnz, mid_units*num_edge_set)
            #print("neighbor_data", neighbor_data.shape)
            #print("edge_data", edge_data.shape)
            neighbor_data = nd.reshape(nd.broadcast_mul(neighbor_data, edge_type_mask),
                                       shape=(0, 0, -1))
            #print("neighbor_data", neighbor_data.shape)


        pool_data = nd.contrib.seg_pool(data=neighbor_data,
                                       indices=seg_indices,
                                       indptr=neighbor_indptr,
                                       pool_type=self._pool_type)  # Shape(batch_size, node_num, mid_units*num_edge_set)
        if self._num_edge_set is not None:
            if self._accum_type == "stack":
                pool_data = self._out_act(pool_data)
            elif self._accum_type == "sum":
                pool_data = self._out_act(nd.sum(nd.reshape(pool_data, shape=(0, 0, self._num_edge_set, self._mid_units )), axis=2))

        #out = self.out_layer(nd.concat(pool_data, data, dim=-1))
        #out = self.out_layer(pool_data)
        return pool_data


def parse_aggregator_from_desc(aggregator_desc):
    assert len(aggregator_desc) == 4 or len(aggregator_desc) == 2
    name = aggregator_desc[0]
    args = aggregator_desc[1]
    if len(aggregator_desc) == 4:
        prefix = aggregator_desc[2]
        out_act = aggregator_desc[3]
    else:
        prefix = None
        out_act = None

    if name.lower() == "MuGGA".lower():
        return MuGGA.from_args(prefix, out_act, *args)
    elif name.lower() == "GraphPoolAggregator".lower():
        return GraphPoolAggregator.from_args(prefix, out_act, *args)
    elif name.lower() == "HeterGraphPoolAggregator".lower():
        return HeterGraphPoolAggregator.from_args(prefix, out_act, *args)
    elif name.lower() == "GraphWeightedSumAggregator".lower():
        return GraphWeightedSumAggregator.from_args(prefix, out_act, *args)
    elif name.lower() == "GraphMultiWeightedSumAggregator".lower():
        return GraphMultiWeightedSumAggregator.from_args(prefix, out_act, *args)
    elif name.lower() == "BiGraphPoolAggregator".lower():
        return BiGraphPoolAggregator.from_args(prefix, out_act, *args)
    else:
        raise NotImplementedError("name={name} is not supported!".format(name=name))
