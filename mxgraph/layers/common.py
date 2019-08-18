import math
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import mxnet.gluon as gluon
from mxnet.gluon import nn, HybridBlock, Block


class IdentityActivation(HybridBlock):
    def hybrid_forward(self, F, x):
        return x


class ELU(HybridBlock):
    r"""
    Exponential Linear Unit (ELU)
        "Fast and Accurate Deep Network Learning by Exponential Linear Units", Clevert et al, 2016
        https://arxiv.org/abs/1511.07289
        Published as a conference paper at ICLR 2016
    Parameters
    ----------
    alpha : float
        The alpha parameter as described by Clevert et al, 2016
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return - self._alpha * F.relu(1.0 - F.exp(x)) + F.relu(x)


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or HybridBlock

    Returns
    -------
    ret: HybridBlock
    """
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'identity':
            return IdentityActivation()
        elif act == 'elu':
            return ELU()
        else:
            return nn.Activation(act)
    else:
        return act


class DenseNetBlock(HybridBlock):
    def __init__(self, units, layer_num, act, flatten=False, prefix=None, params=None):
        super(DenseNetBlock, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._layer_num = layer_num
        self._act = get_activation(act)
        print("layer_num", layer_num)
        with self.name_scope():
            self.layers = nn.HybridSequential('dblock_')
            with self.layers.name_scope():
                for _ in range(layer_num):
                    self.layers.add(nn.Dense(units, flatten=flatten))

    def hybrid_forward(self, F, x):
        layer_in_l = [x]
        layer_out = None
        for i in range(self._layer_num):
            if len(layer_in_l) == 1:
                layer_in = layer_in_l[0]
            else:
                layer_in = F.concat(*layer_in_l, dim=-1)
            layer_out = self._act(self.layers[i](layer_in))
            layer_in_l.append(layer_out)
        return layer_out

class HeterDenseNetBlock(Block):
    def __init__(self, units, layer_num, act, num_set, flatten=False, prefix=None, params=None):
        super(HeterDenseNetBlock, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_set = num_set
        self._layer_num = layer_num
        self._act = get_activation(act)
        with self.name_scope():
            self.layers = nn.Sequential('hdblock_')
            with self.layers.name_scope():
                for _ in range(layer_num):
                    self.layers.add(nn.Dense(units*num_set, flatten=flatten))

    def forward(self, x, mask):
        """

        Parameters
        ----------
        F
        x: Shape(batch_size, num_node, input_dim)
        mask: Shape(batch_size, num_node, num_set, 1)

        Returns
        -------

        """
        layer_in_l = [x]
        layer_out = None
        for i in range(self._layer_num):
            if len(layer_in_l) == 1:
                layer_in = layer_in_l[0]
            else:
                layer_in = nd.concat(*layer_in_l, dim=-1)
            ### TODO assume batch_size=1
            x_mW = nd.reshape(self.layers[i](layer_in), shape=(0, 0, self._num_set, self._units))
            layer_out = self._act( nd.sum(nd.broadcast_mul(x_mW, mask), axis=-2) )
            layer_in_l.append(layer_out)
        return layer_out


class L2Normalization(HybridBlock):
    def __init__(self, axis=-1, eps=1E-6, prefix=None, params=None):
        super(L2Normalization, self).__init__(prefix=prefix, params=params)
        self._axis = axis
        self._eps = eps

    def hybrid_forward(self, F, x):
        ret = F.broadcast_div(x, F.sqrt(F.sum(F.square(x), axis=self._axis, keepdims=True)
                                        + self._eps))
        return ret
