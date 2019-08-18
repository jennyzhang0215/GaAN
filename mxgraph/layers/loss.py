from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.loss import Loss, LogisticLoss


class LinkPredLogisticLoss(Loss):
    def __init__(self, neg_weight=1.0, weight=None, **kwargs):
        super(LinkPredLogisticLoss, self).__init__(weight=weight, batch_axis=None, **kwargs)
        self._neg_weight = neg_weight

    def hybrid_forward(self, F, pred, edge_label, edge_weight=None):
        """

        Parameters
        ----------
        F
        pred
        edge_label: 1-> positive edge; -1 -> negative edge
        edge_weight

        Returns
        -------

        """
        binary_label = (edge_label + 1) / 2.0 # 1 -> positive edge; 0 -> negative edge
        if edge_weight is None:
            edge_weight = binary_label * 1.0 + (1 - binary_label) * self._neg_weight
        else:
            edge_weight *= binary_label * 1.0 + (1 - binary_label) / 2. * self._neg_weight
        #print("edge_weight", edge_weight)
        #print("edge_label", edge_label)
        loss = edge_weight * F.log(1.0 + F.exp(-pred * edge_label))
        loss = F.sum(loss)
        return loss


class UnsupWalkLoss(Loss):
    def __init__(self, neg_weight=1.0, weight=None, **kwargs):
        super(UnsupWalkLoss, self).__init__(weight=weight, batch_axis=None, **kwargs)
        self._neg_weight = neg_weight

    def hybrid_forward(self, F, node_emb, pos_emb, neg_emb):
        """

        Parameters
        ----------
        F
        node_emb : (batch_size, dim)
        pos_emb : (batch_size, dim)
        neg_emb : (neg_sample_num, dim)
        edge_weight

        Returns
        -------

        """
        pos_innerproduct = F.sum(F.broadcast_mul(node_emb, pos_emb), axis=1) ## Shape: (batch_size, )
        neg_innerproduct = F.dot(node_emb, neg_emb, transpose_b=True) ## Shape: (batch_size, num_neg)

        pos_loss = F.log(1.0 + F.exp(-1.0 * pos_innerproduct))
        neg_loss = F.log(1.0 + F.exp(neg_innerproduct))
        loss = F.sum(pos_loss) + self._neg_weight * F.sum(neg_loss)

        return loss
