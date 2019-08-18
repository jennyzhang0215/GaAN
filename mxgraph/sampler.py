import numpy as np
import scipy.sparse as ss
import logging
from mxgraph import _graph_sampler

### numpy version of negative sampling TODO out-of-date
class UnsupLossSampler(object):
    def __init__(self, num_neg_sample, **kwargs):
        """
        Parameters
        ----------
        num_neg_sample : Shape scalar
        kwargs
        """
        self.num_neg_sample = num_neg_sample

    def run(self, orig_end_points, orig_indptr, sample_target_indices):
        """
        For training:
            orig_end_points and orig_indptr refer to the training subgraph
            sample_target_indices are the training nodes
        For validation:
            orig_end_points and orig_indptr refer to the training+validation subgraph
            sample_target_indices are the validation nodes
        Parameters
        ----------
        orig_end_points :
            Shape (num_edges，)
        orig_indptr :
            Shape (num_nodes+1, )
        sample_target_indices :
            Shape (#sample_nodes,)

        Returns
        -------

        """
        self.orig_node_size = orig_indptr.size-1

        ### Fetch positive edges
        values = np.ones(shape=orig_end_points.shape, dtype=np.float32)
        pos_adj = ss.csr_matrix((values, orig_end_points, orig_indptr),
                                shape=(self.orig_node_size, self.orig_node_size))

        ### Negative Sampling
        p_unnorm = pos_adj
        p_unnorm.data = values * -1.
        p_unnorm = p_unnorm.toarray() + 1. ## negative edge: 1.. ; positive edge: 0.0 to compute the sampling probability
        np.fill_diagonal(p_unnorm, .0)
        p = p_unnorm / p_unnorm.sum(axis=1, keepdims=True)

        row = np.asarray([np.ones(self.num_neg_sample, dtype=np.int32)*node_id for node_id in sample_target_indices])
        column = np.asarray([np.random.choice(self.orig_node_size,
                                              size=self.num_neg_sample,
                                              replace=False,
                                              p=p[node_id]).tolist()
                                 for node_id in sample_target_indices])
        neg_adj = ss.coo_matrix((np.ones(self.num_neg_sample * sample_target_indices.size) * -1.0,
                                 (row.reshape(-1), column.reshape(-1,))),
                                shape=(self.orig_node_size, self.orig_node_size)).tocsr()
        ### Sum pos_adj and neg_adj
        all_adj = pos_adj + neg_adj
        sampled_adj = all_adj[sample_target_indices]
        end_points, indptr, edge_labels = sampled_adj.indices, sampled_adj.indptr, sampled_adj.data
        return end_points, indptr, edge_labels


class LinkPredEdgeSampler(object):
    def __init__(self, neg_sample_scale, replace=False):
        """

        Parameters
        ----------
        neg_sample_scale: float
            For every node, the number of negative samples will be:

            .. Code::

             expected_neg_num = ceil(pos_num * neg_sample_scale)
             expected_neg_num = min(expected_neg_num, total_num - pos_num)

             If this value is set to be negative, no sampling will be performed and
              all the end_points will be returned.
        replace : bool
            Whether to sample the negative edges with replacement
        """
        self._neg_sample_scale = neg_sample_scale
        self._replace = replace

    def sample_by_indices(self, G, chosen_node_indices):
        """Sample the G with the chosen_node_indices

        Parameters
        ----------
        G : SimpleGraph
        chosen_node_indices : np.ndarray

        Returns
        -------
        end_points : np.ndarray
        indptr : np.ndarray
        edge_label : np.ndarray
        edge_count : np.ndarray
        """
        end_points, indptr, edge_label, edge_count = \
            _graph_sampler.uniform_neg_sampling(G.end_points.astype(np.int32),
                                                G.ind_ptr.astype(np.int32),
                                                chosen_node_indices.astype(np.int32),
                                                float(self._neg_sample_scale),
                                                int(self._replace))
        return end_points, indptr, edge_label, edge_count

    def sample_by_id(self, G, chosen_node_ids):
        """Sample the G with the chosen_node_ids

        Parameters
        ----------
        G : SimpleGraph
        chosen_node_ids : np.ndarray

        Returns
        -------
        end_points : np.ndarray
        indptr : np.ndarray
        edge_label : np.ndarray
        edge_count : np.ndarray
        """
        return self.sample_by_indices(G, G.reverse_map(chosen_node_ids))


class BaseHierarchicalNodeSampler(object):
    def __init__(self, layer_num):
        self._layer_num = layer_num

    @property
    def layer_num(self):
        return self._layer_num

    def layer_sample_and_merge(self, G, sel_indices, depth):
        """

        Parameters
        ----------
        G : SimpleGraph
        sel_indices : np.ndarray
        depth : int

        Returns
        -------
        end_points : np.ndarray
        indptr : np.ndarray
        merged_node_ids : np.ndarray
        indices_in_merged : np.ndarray
        """
        raise NotImplementedError

    def sample_by_indices(self, G, base_node_indices):
        """Note that the order of our samples should be from the lowest layer to the highest layer

        Parameters
        ----------
        G : SimpleGraph
        base_node_indices : np.ndarray

        Returns
        -------
        indices_in_merged_l : list
            The first element will be the indices of the 0th layer in the original graph
        end_points_l : list
        indptr_l : list
        node_ids_l : list
            The original ids of the indices
        """
        base_node_indices = base_node_indices.astype(np.int32)
        indices_in_merged_l = [None for _ in range(self._layer_num + 1)]
        end_points_l = [None for _ in range(self._layer_num)]
        indptr_l = [None for _ in range(self._layer_num)]
        node_ids_l = [None for _ in range(self._layer_num + 1)]
        node_ids_l[self._layer_num] = G.node_ids[base_node_indices]
        for i in range(self._layer_num - 1, -1, -1):
            end_points, indptr, merged_node_ids, indices_in_merged =\
                self.layer_sample_and_merge(G, base_node_indices, self._layer_num - 1 - i)
            base_node_indices = G.reverse_map(merged_node_ids)
            indices_in_merged_l[i + 1] = indices_in_merged
            node_ids_l[i] = merged_node_ids
            end_points_l[i] = end_points
            indptr_l[i] = indptr
        indices_in_merged_l[0] = base_node_indices
        return indices_in_merged_l, end_points_l, indptr_l, node_ids_l

    def sample_by_indices_with_edge_weight(self, G, base_node_indices):
        """Note that the order of our samples should be from the lowest layer to the highest layer

        Parameters
        ----------
        G : SimpleGraph
        base_node_indices : np.ndarray

        Returns
        -------
        indices_in_merged_l : list
            The first element will be the indices of the 0th layer in the original graph
        end_points_l : list
        indptr_l : list
        node_ids_l : list
            The original ids of the indices
        """
        base_node_indices = base_node_indices.astype(np.int32)
        indices_in_merged_l = [None for _ in range(self._layer_num + 1)]
        end_points_l = [None for _ in range(self._layer_num)]
        indptr_l = [None for _ in range(self._layer_num)]
        end_points_edge_weight_l = [None for _ in range(self._layer_num)]
        node_ids_l = [None for _ in range(self._layer_num + 1)]
        node_ids_l[self._layer_num] = G.node_ids[base_node_indices]
        for i in range(self._layer_num - 1, -1, -1):
            end_points, indptr, merged_node_ids, indices_in_merged =\
                self.layer_sample_and_merge(G, base_node_indices, self._layer_num - 1 - i)
            base_node_indices = G.reverse_map(merged_node_ids)
            indices_in_merged_l[i + 1] = indices_in_merged
            node_ids_l[i] = merged_node_ids
            end_points_l[i] = end_points
            indptr_l[i] = indptr
            # print("previous base_node_indices", G.reverse_map(node_ids_l[i+1]).shape, G.reverse_map(node_ids_l[i+1]))
            # print("indices_in_merged", indices_in_merged.shape, indices_in_merged)
            # print("base_node_indices", base_node_indices.shape, base_node_indices)
            sampled_sub_adj = G.adj.submat(row_indices=G.reverse_map(node_ids_l[i+1]), col_indices=G.reverse_map(node_ids_l[i]))
            # print("**********************************")
            # print("sampled_sub_adj.ind_ptr", sampled_sub_adj.ind_ptr.shape, sampled_sub_adj.ind_ptr)
            # print("sampled indptr", indptr.shape, indptr)
            # print("**********************************")
            # print("sampled_sub_adj.end_points", sampled_sub_adj.end_points.shape, sampled_sub_adj.end_points)
            # print("sampled end_points", end_points.shape, end_points)
            # print("**********************************")
            # print("sampled_sub_adj.values", sampled_sub_adj.values.shape, sampled_sub_adj.values)
            end_points_edge_weight_l[i] =sampled_sub_adj.values
            #print("=============================================")

        indices_in_merged_l[0] = base_node_indices
        return indices_in_merged_l, end_points_l, indptr_l, node_ids_l, end_points_edge_weight_l


    # def sample_by_id(self, G, base_node_ids):
    #     return self.sample_by_indices(G, base_node_ids)


class FractionNeighborSampler(BaseHierarchicalNodeSampler):
    def __init__(self, layer_num, neighbor_fraction=None, replace=False):
        super(FractionNeighborSampler, self).__init__(layer_num=layer_num)
        self._neighbor_fraction = neighbor_fraction
        self._replace = replace
        assert len(self._neighbor_fraction) == layer_num
        for ele in self._neighbor_fraction:
            assert ele > 0

    def layer_sample_and_merge(self, G, sel_indices, depth):
        return _graph_sampler.random_sel_neighbor_and_merge(
            G.end_points, G.ind_ptr, G.node_ids, sel_indices.astype(np.int32),
            -1, np.float32(self._neighbor_fraction[depth]), 0, int(self._replace))


class FixedNeighborSampler(BaseHierarchicalNodeSampler):
    def __init__(self, layer_num, neighbor_num=None, replace=False):
        super(FixedNeighborSampler, self).__init__(layer_num=layer_num)
        self._neighbor_num = neighbor_num
        self._replace = replace
        assert len(self._neighbor_num) == self._layer_num
        for ele in self._neighbor_num:
            assert ele > 0

    def layer_sample_and_merge(self, G, sel_indices, depth):
        return _graph_sampler.random_sel_neighbor_and_merge(
            G.end_points, G.ind_ptr, G.node_ids, sel_indices.astype(np.int32),
            self._neighbor_num[depth], -1.0, 0, int(self._replace))



class AllNeighborSampler(BaseHierarchicalNodeSampler):
    def _npy_layer_sample_and_merge(self, G, base_node_indices):
        end_points = []
        indptr = []
        merged_node_ids = []
        indices_in_merged = []
        indptr.append(0)
        for ind in base_node_indices:
            end_points.extend(G.end_points[G.ind_ptr[ind]:G.ind_ptr[ind + 1]].tolist())
            indptr.append(indptr[-1] + G.ind_ptr[ind + 1] - G.ind_ptr[ind])
        counter = 0
        ind_dict = dict()
        for i, val in enumerate(end_points):
            if val not in ind_dict:
                ind_dict[val] = counter
                end_points[i] = counter
                merged_node_ids.append(G.node_ids[val])
                counter += 1
            else:
                end_points[i] = ind_dict[val]
        for val in base_node_indices:
            if val in ind_dict:
                indices_in_merged.append(ind_dict[val])
            else:
                indices_in_merged.append(counter)
                merged_node_ids.append(G.node_ids[val])
                counter += 1
        end_points = np.array(end_points, dtype=np.int32)
        indptr = np.array(indptr, dtype=np.int32)
        merged_node_ids = np.array(merged_node_ids, dtype=np.int32)
        indices_in_merged = np.array(indices_in_merged)
        return end_points, indptr, merged_node_ids, indices_in_merged

    def layer_sample_and_merge(self, G, sel_indices, depth):
        return _graph_sampler.random_sel_neighbor_and_merge(
            G.end_points, G.ind_ptr, G.node_ids, sel_indices.astype(np.int32),
            -1, -1.0, 1, 1)


class NoMergeSampler(FixedNeighborSampler):
    def layer_sample_and_merge(self, G, sel_indices, depth):
        raise NotImplementedError



def parse_hierarchy_sampler_from_desc(desc):
    assert len(desc) == 2
    name = desc[0]
    args = desc[1]
    if name.lower() == "fraction".lower():
        logging.info("FractionNeighborSampler: layer_num=%d, fraction=%s" % (len(args), str(args)))
        return FractionNeighborSampler(layer_num=len(args), neighbor_fraction=args)
    elif name.lower() == "fixed".lower():
        logging.info("FixedNeighborSampler: layer_num=%d, sample_num=%s" % (len(args), str(args)))
        return FixedNeighborSampler(layer_num=len(args), neighbor_num=args)
    elif name.lower() == "all".lower():
        #assert len(args) == 1
        logging.info("AllNeighborSampler: layer_num=%d" % args)
        return AllNeighborSampler(layer_num=int(args))
    else:
        raise NotImplementedError("name={name} is not supported!".format(name=name))
