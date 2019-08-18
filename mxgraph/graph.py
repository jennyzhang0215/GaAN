import numpy as np
import scipy.sparse as ss
import logging
import mxgraph._graph_sampler as _graph_sampler

def set_seed(seed):
    """Set the random seed of the inner sampling handler

    Parameters
    ----------
    seed : int

    Returns
    -------
    ret : bool
    """
    return _graph_sampler.set_seed(seed)

class CSRMat(object):
    """A simple wrapper of the CSR Matrix

    Apart from the traditoinal CSR format, we use two additional arrays: row_ids and col_ids
     to track the original ids of the row/col indices

    We use the C++ API to accelerate the speed if possible
    """
    def __init__(self, end_points, ind_ptr, row_ids, col_ids, values=None, force_contiguous=False):
        self.end_points = end_points
        self.ind_ptr = np.ascontiguousarray(ind_ptr, dtype=np.int32)
        self.values = None if values is None else values.astype(np.float32)
        self.row_ids = row_ids
        self.col_ids = col_ids
        assert self.ind_ptr.size == len(self.row_ids) + 1
        if force_contiguous:
            self.end_points = np.ascontiguousarray(self.end_points, dtype=np.int32)
            self.ind_ptr = np.ascontiguousarray(self.ind_ptr, dtype=np.int32)
            if self.values is not None:
                self.values = np.ascontiguousarray(self.values, dtype=np.float32)
            self.row_ids = np.ascontiguousarray(self.row_ids, dtype=np.int32)
            self.col_ids = np.ascontiguousarray(self.col_ids, dtype=np.int32)
        self._row_id_reverse_mapping = -1 * np.ones(self.row_ids.max() + 1, dtype=np.int32)
        self._col_id_reverse_mapping = -1 * np.ones(self.col_ids.max() + 1, dtype=np.int32)
        self._row_id_reverse_mapping[self.row_ids] = np.arange(self.row_ids.size, dtype=np.int32)
        self._col_id_reverse_mapping[self.col_ids] = np.arange(self.col_ids.size, dtype=np.int32)
        # self._row_id_reverse_mapping = dict()
        # self._col_id_reverse_mapping = dict()
        # for (i, ele) in enumerate(self.row_ids):
        #     self._row_id_reverse_mapping[ele] = i
        # for (i, ele) in enumerate(self.col_ids):
        #     self._col_id_reverse_mapping[ele] = i

    def to_spy(self):
        """Convert to the scipy csr matrix

        Returns
        -------
        ret : ss.csr_matrix
        """
        if self.values is None:
            values = np.ones(shape=self.end_points.shape, dtype=np.float32)
        else:
            values = self.values
        return ss.csr_matrix((values, self.end_points, self.ind_ptr), shape=(self.row_ids.size, self.col_ids.size))

    @staticmethod
    def from_spy(mat):
        """

        Parameters
        ----------
        mat : ss.csr_matrix

        Returns
        -------
        ret : CSRMat
        """
        return CSRMat(end_points=mat.indices,
                      ind_ptr=mat.indptr,
                      row_ids=np.arange(mat.shape[0], dtype=np.int32),
                      col_ids=np.arange(mat.shape[1], dtype=np.int32),
                      values=mat.data,
                      force_contiguous=True)
    @property
    def nnz(self):
        return self.end_points.size

    def reverse_row_map(self, node_ids):
        """Maps node ids back to row indices in the CSRMat

        Parameters
        ----------
        node_ids : np.ndarray or list or tuple or int

        Returns
        -------
        ret : np.ndarray
        """
        # if isinstance(node_ids, (np.ndarray, list, tuple)):
        #     return np.array(list(map(lambda ele: self._row_id_reverse_mapping[ele], node_ids)),
        #                     dtype=np.int32)
        # else:
        return self._row_id_reverse_mapping[node_ids]

    def reverse_col_map(self, node_ids):
        """Maps node ids back to col indices in the CSRMat

        Parameters
        ----------
        node_ids : np.ndarray or list or tuple or int

        Returns
        -------
        ret : np.ndarray
        """
        # if isinstance(node_ids, (np.ndarray, list, tuple)):
        #     return np.array(list(map(lambda ele: self._col_id_reverse_mapping[ele], node_ids)),
        #                     dtype=np.int32)
        # else:
        return self._col_id_reverse_mapping[node_ids]

    def submat(self, row_indices=None, col_indices=None):
        """Get the submatrix of the corresponding row/col indices

        Parameters
        ----------
        row_indices : np.ndarray or None
        col_indices : np.ndarray or None

        Returns
        -------
        ret : CSRMat
        """
        row_indices = None if row_indices is None else row_indices.astype(np.int32)
        col_indices = None if col_indices is None else col_indices.astype(np.int32)
        dst_end_points, dst_values, dst_ind_ptr, dst_row_ids, dst_col_ids\
            = _graph_sampler.csr_submat(self.end_points,
                                        self.values,
                                        self.ind_ptr,
                                        self.row_ids,
                                        self.col_ids,
                                        row_indices,
                                        col_indices)
        return CSRMat(end_points=dst_end_points,
                      ind_ptr=dst_ind_ptr,
                      row_ids=dst_row_ids,
                      col_ids=dst_col_ids,
                      values=dst_values)

    def summary(self):
        print(self.info())

    def info(self):
        info_str = "Summary" + \
                   "\n   Row={}, Col={}, NNZ={}".format(self.row_ids.size,
                                                        self.col_ids.size,
                                                        self.end_points.size)
        return info_str


class SimpleGraph(object):
    """A simple graph container

    We use the CSR format to store the adjacency matrix

    """
    def __init__(self, node_ids, node_types, undirected=True,
                 end_points=None, ind_ptr=None, edge_weight=None, adj=None, edge_features=None):
        """Initialize a SimpleGraph

        Parameters
        ----------
        node_ids : np.ndarray
            Maps the indices to the real node_ids
        node_types : np.ndarray
            Types of the nodes, 1 --> Train, 2 --> Valid, 3 --> Test
        undirected : bool
        end_points : np.ndarray
            Indices of the end-points of the connections
        ind_ptr : np.ndarray
            Pointer to the beginning of end_points for a specific array
        adj : CSRMat
            The CSR matrix that stores the value
        edge_features : None or np.ndarray
            The edge features, should be None or have shape (node_num, edge_feature_dim)
        """
        if adj is not None:
            self.adj = adj
        else:
            assert end_points is not None and ind_ptr is not None
            self.adj = CSRMat(end_points=end_points,
                              ind_ptr=ind_ptr,
                              row_ids=node_ids,
                              col_ids=node_ids,
                              values=edge_weight)
        self.degrees = np.floor(np.array(self.adj.to_spy().sum(axis=1))).astype(np.int32).reshape((-1,))
        self.node_ids = node_ids.astype(np.int32)
        self.node_types = node_types.astype(np.int32)
        self.undirected = undirected
        self.edge_features = edge_features
        self._node_id_reverse_mapping = -1 * np.ones(shape=self.node_ids.max() + 1, dtype=np.int32)
        self._node_id_reverse_mapping[self.node_ids] = np.arange(self.node_ids.size, dtype=np.int32)
        # self._node_id_reverse_mapping = dict()
        # for (i, ele) in enumerate(self.node_ids):
        #     self._node_id_reverse_mapping[ele] = i

    @property
    def end_points(self):
        return self.adj.end_points.astype(np.int32)

    @property
    def ind_ptr(self):
        return self.adj.ind_ptr.astype(np.int32)

    @property
    def node_num(self):
        return self.node_ids.size

    @property
    def edge_num(self):
        return self.adj.end_points.size if not self.undirected else self.adj.end_points.size / 2

    @property
    def avg_degree(self):
        return self.adj.end_points.size / self.node_ids.size

    def to_networkx(self):
        """Convert to a networkx graph

        Returns
        -------
        ret: networkx.Graph
        """
        raise NotImplementedError

    def reverse_map(self, node_ids):
        """Maps node ids back to indices in the graph

        Parameters
        ----------
        node_ids : np.ndarray or list or tuple or int

        Returns
        -------
        ret : np.ndarray
        """
        return self._node_id_reverse_mapping[node_ids]
        # if isinstance(node_ids, (np.ndarray, list, tuple)):
        #     return np.array(list(map(lambda ele: self._node_id_reverse_mapping[ele], node_ids)),
        #                     dtype=np.int32)
        # else:
        #     return self._node_id_reverse_mapping[node_ids]

    def subgraph_by_indices(self, indices):
        """Obtain subgraph by index values, i.e., 0, 1, 2, ...

        Parameters
        ----------
        indices

        Returns
        -------

        """
        subgraph_adj = self.adj.submat(row_indices=indices, col_indices=indices)
        subgraph_node_ids = self.node_ids[indices]
        subgraph_node_types = self.node_types[indices]
        if self.edge_features is not None:
            new_edge_features = self.edge_features[indices, :]
        else:
            new_edge_features = None
        return SimpleGraph(node_ids=subgraph_node_ids,
                           node_types=subgraph_node_types,
                           undirected=self.undirected,
                           adj=subgraph_adj,
                           edge_features=new_edge_features)

    def subgraph_by_node_ids(self, node_ids):
        """Obtain subgraph by the node_ids

        For example, the original graph has node ids: (2, 7, 9, 11) and you decide to take a look at the
        subgraph that contains nodes (2, 9), call G.subgraph_by_node_ids([2, 9])

        Parameters
        ----------
        real_ids

        Returns
        -------

        """
        return self.subgraph_by_indices(self.reverse_map(node_ids))

    def save(self, fname):
        return np.savez_compressed(fname,
                                   node_ids=self.node_ids,
                                   node_types=self.node_types,
                                   end_points=self.adj.end_points,
                                   ind_ptr=self.adj.ind_ptr,
                                   undirected=np.array((self.undirected,), dtype=np.bool))

    @staticmethod
    def load(fname):
        G_data = np.load(fname)
        if 'undirected' in G_data.keys():
            if isinstance(G_data['undirected'], list):
                undirected = G_data['undirected'][0]
            else:
                undirected = G_data['undirected']
        else:
            undirected = True  # default value of undirected is true
        return SimpleGraph(node_ids=G_data['node_ids'],
                           node_types=G_data['node_types'],
                           undirected=undirected,
                           end_points=G_data['end_points'],
                           ind_ptr=G_data['ind_ptr'])

    @property
    def train_indices(self):
        all_indices = np.arange(0, self.node_num, dtype=np.int32)
        return all_indices[self.node_types == 1]

    @property
    def train_node_ids(self):
        return self.node_ids[self.train_indices]

    def fetch_train(self):
        return self.subgraph_by_indices(indices=self.train_indices)

    @property
    def valid_indices(self):
        all_indices = np.arange(0, self.node_num, dtype=np.int32)
        return all_indices[self.node_types == 2]

    @property
    def valid_node_ids(self):
        return self.node_ids[self.valid_indices]

    def fetch_valid(self):
        return self.subgraph_by_indices(indices=self.valid_indices)

    @property
    def test_indices(self):
        all_indices = np.arange(0, self.node_num, dtype=np.int32)
        return all_indices[self.node_types == 3]

    @property
    def test_node_ids(self):
        return self.node_ids[self.test_indices]

    def fetch_test(self):
        return self.subgraph_by_indices(indices=self.test_indices)

    @property
    def train_valid_indices(self):
        all_indices = np.arange(0, self.node_num, dtype=np.int32)
        return all_indices[self.node_types <= 2]

    @property
    def train_valid_node_ids(self):
        return self.node_ids[self.train_valid_indices]

    def fetch_train_valid(self):
        return self.subgraph_by_indices(indices=self.train_valid_indices)

    def summary(self, graph_name="Graph"):
        logging.info(self.info(graph_name))

    def info(self, graph_name="Graph", indent_token="\t"):
        info_str = indent_token + "Summary of {}\n".format(graph_name) + \
                   indent_token + indent_token + "Undirected=%s\n" %str(self.undirected) + \
                   indent_token + indent_token + "Node Number={}, Train={}, Valid={}, Test={}\n".format(self.node_num,
                                                                       (self.node_types == 1).sum(),
                                                                       (self.node_types == 2).sum(),
                                                                       (self.node_types == 3).sum()) + \
                   indent_token + indent_token + "Edge Number={}\n".format(self.edge_num) + \
                   indent_token + indent_token + "Avg Degree={}".format(self.avg_degree)
        if self.edge_features is not None:
            info_str += indent_token + indent_token + "Edge Features Shape={}".format(self.edge_features.shape)
        return info_str

    def random_walk(self,
                    initial_node=-1,
                    walk_length=10000,
                    return_prob=None,
                    max_node_num=-1,
                    max_edge_num=-1):
        """Random Walk

        At every step, we will return to the initial node with return_p.
         Otherwise, we will jump randomly to a conneted node.
        Ref: [KDD06] Sampling from Large Graphs

        Parameters
        ----------
        initial_node : int or None
        walk_length : int
        return_prob : float or None
        max_node_num : int or None
        max_edge_num : int or None

        Returns
        -------
        G_sampled : SimpleGraph
        """
        if initial_node is None:
            initial_node = -1
        if max_node_num is None:
            max_node_num = -1
        if max_edge_num is None:
            max_edge_num = -1
        if return_prob is None:
            return_prob = 0.15
        subgraph_end_points, subgraph_ind_ptr, subgraph_node_ids =\
            _graph_sampler.random_walk(self.end_points,
                                       self.ind_ptr,
                                       self.node_ids,
                                       int(self.undirected),
                                       initial_node,
                                       walk_length,
                                       return_prob,
                                       max_node_num,
                                       max_edge_num)
        indices = self.reverse_map(subgraph_node_ids)
        subgraph_node_types = self.node_types[indices]
        return SimpleGraph(node_ids=subgraph_node_ids,
                           node_types=subgraph_node_types,
                           undirected=self.undirected,
                           end_points=subgraph_end_points,
                           ind_ptr=subgraph_ind_ptr)


class HeterGraph(SimpleGraph):
    def __init__(self, node_ids, node_types, num_set, node_sets, node_indices_in_set, undirected=True,
                 end_points=None, ind_ptr=None, adj=None, edge_weight=None, edge_features=None, edge_types=None):
        super(HeterGraph, self).__init__(node_ids=node_ids, node_types=node_types, undirected=undirected,
                                         end_points=end_points, ind_ptr=ind_ptr, edge_weight=edge_weight, adj=adj,
                                         edge_features=edge_features)
        self._num_set = int(num_set)
        self.node_sets = node_sets ## which node set the node belongs to
        self.node_indices_in_set = node_indices_in_set ## the position where the node in its set
        #self.edge_types = edge_types

    @property
    def num_set(self):
        return self._num_set

    def subgraph_by_indices(self, indices):
        subgraph_adj = self.adj.submat(row_indices=indices, col_indices=indices)
        subgraph_node_ids = self.node_ids[indices]
        subgraph_node_types = self.node_types[indices]
        subgraph_node_sets = self.node_sets[indices]
        subgraph_node_indices_in_set = self.node_indices_in_set[indices]
        if self.edge_features is not None:
            new_edge_features = self.edge_features[indices, :]
        else:
            new_edge_features = None

        return HeterGraph(node_ids=subgraph_node_ids,
                          node_types=subgraph_node_types,
                          num_set=self._num_set,
                          node_sets=subgraph_node_sets,
                          node_indices_in_set=subgraph_node_indices_in_set,
                          adj=subgraph_adj,
                          undirected=self.undirected,
                          edge_features=new_edge_features)

    def subgraph_by_node_ids(self, node_ids):
        return self.subgraph_by_indices(self.reverse_map(node_ids))

    def node_sets_by_indices(self, indices):
        return self.node_sets[indices]
    def node_sets_by_node_ids(self, node_ids):
        return self.node_sets_by_indices(self.reverse_map(node_ids))

    def node_indices_in_set_by_indices(self, indices):
        return self.node_indices_in_set[indices]
    def node_indices_in_set_by_node_ids(self, node_ids):
        return self.node_indices_in_set_by_indices(self.reverse_map(node_ids))

    @staticmethod
    def load(fname):
        print("Loading a Heterogeneous Graph ...")
        G_data = np.load(fname)
        if 'undirected' in G_data.keys():
            if isinstance(G_data['undirected'], list):
                undirected = G_data['undirected'][0]
            else:
                undirected = G_data['undirected']
        else:
            undirected = True  # default value of undirected is true
        if "adj" in G_data and isinstance(G_data["adj"], CSRMat):
            return HeterGraph(node_ids=G_data['node_ids'],
                              node_types=G_data['node_types'],
                              num_set=G_data['num_set'],
                              node_sets=G_data['node_sets'],
                              node_indices_in_set=G_data['node_indices_in_set'],
                              adj=G_data['adj'],
                              undirected=undirected)
        else:
            return HeterGraph(node_ids=G_data['node_ids'],
                              node_types=G_data['node_types'],
                              num_set=G_data['num_set'],
                              node_sets = G_data['node_sets'],
                              node_indices_in_set=G_data['node_indices_in_set'],
                              end_points=G_data['end_points'],
                              ind_ptr=G_data['ind_ptr'],
                              edge_weight=G_data['edge_weight'],
                              undirected=undirected)


    def save(self, fname):
        return np.savez_compressed(fname,
                                   node_ids=self.node_ids,
                                   node_types=self.node_types,
                                   set_num=self._num_set,
                                   node_sets=self.node_sets,
                                   node_indices_in_set=self.node_indices_in_set,
                                   adj=self.adj,
                                   undirected=np.array((self.undirected,), dtype=np.bool))



class BiGraph(SimpleGraph):
    def __init__(self, node_ids, node_types, num_node_set, num_edge_set, node_sets, node_indices_in_set, undirected=True,
                 end_points=None, ind_ptr=None, adj=None, edge_weight=None, edge_features=None):
        super(BiGraph, self).__init__(node_ids=node_ids, node_types=node_types, undirected=undirected,
                                      end_points=end_points, ind_ptr=ind_ptr, edge_weight=edge_weight, adj=adj,
                                      edge_features=edge_features)
        self._num_node_set = int(num_node_set)
        self._num_edge_set = int(num_edge_set)
        self.node_sets = node_sets ## which node set the node belongs to SHAPE np.array(num_node, )
        self.node_indices_in_set = node_indices_in_set ## the position where the node in its set SHAPE np.array(num_node, )
        #self.edge_types = edge_types

    @property
    def num_node_set(self):
        return self._num_node_set
    @property
    def num_edge_set(self):
        return self._num_edge_set

    def subgraph_by_indices(self, indices):
        subgraph_adj = self.adj.submat(row_indices=indices, col_indices=indices)
        subgraph_node_ids = self.node_ids[indices]
        subgraph_node_types = self.node_types[indices]
        subgraph_node_sets = self.node_sets[indices]
        subgraph_node_indices_in_set = self.node_indices_in_set[indices]
        new_edge_features = self.edge_features[indices, :] if self.edge_features is not None else None
        return BiGraph(node_ids=subgraph_node_ids,
                       node_types=subgraph_node_types,
                       num_node_set=self._num_node_set,
                       num_edge_set=self._num_edge_set,
                       node_sets=subgraph_node_sets,
                       node_indices_in_set=subgraph_node_indices_in_set,
                       adj=subgraph_adj,
                       undirected=self.undirected,
                       edge_features=new_edge_features)

    def subgraph_by_node_ids(self, node_ids):
        return self.subgraph_by_indices(self.reverse_map(node_ids))

    def node_sets_by_indices(self, indices):
        return self.node_sets[indices]
    def node_sets_by_node_ids(self, node_ids):
        return self.node_sets_by_indices(self.reverse_map(node_ids))

    def node_indices_in_set_by_indices(self, indices):
        return self.node_indices_in_set[indices]
    def node_indices_in_set_by_node_ids(self, node_ids):
        return self.node_indices_in_set_by_indices(self.reverse_map(node_ids))

    @staticmethod
    def load(fname):
        print("Loading a Bipartite Graph ...")
        G_data = np.load(fname)
        if 'undirected' in G_data.keys():
            if isinstance(G_data['undirected'], list):
                undirected = G_data['undirected'][0]
            else:
                undirected = G_data['undirected']
        else:
            undirected = True  # default value of undirected is true
        if "adj" in G_data and isinstance(G_data["adj"], CSRMat):
            return BiGraph(node_ids=G_data['node_ids'],
                           node_types=G_data['node_types'],
                           num_node_set=G_data['num_node_set'],
                           num_edge_set=G_data['num_edge_set'],
                           node_sets=G_data['node_sets'],
                           node_indices_in_set=G_data['node_indices_in_set'],
                           adj=G_data['adj'],
                           undirected=undirected)
        else:
            return BiGraph(node_ids=G_data['node_ids'],
                           node_types=G_data['node_types'],
                           num_node_set=G_data['num_node_set'][0],
                           num_edge_set=G_data['num_edge_set'][0],
                           node_sets = G_data['node_sets'],
                           node_indices_in_set=G_data['node_indices_in_set'],
                           end_points=G_data['end_points'],
                           ind_ptr=G_data['ind_ptr'],
                           edge_weight=G_data['edge_weight'],
                           undirected=undirected)


    def save(self, fname):
        return np.savez_compressed(fname,
                                   node_ids=self.node_ids,
                                   node_types=self.node_types,
                                   num_node_set=self._num_node_set,
                                   num_edge_set=self._num_edge_set,
                                   node_sets=self.node_sets,
                                   node_indices_in_set=self.node_indices_in_set,
                                   adj=self.adj,
                                   undirected=np.array((self.undirected,), dtype=np.bool))



def check_subgraph(G_sampled, G_all):
    """Check whether G_sampled is a subgraph of G_all

    Parameters
    ----------
    G_sampled : SimpleGraph
    G_all : SimpleGraph

    Returns
    -------
    correct : bool
    """
    correct = True
    for id_index in range(G_sampled.node_ids.size):
        sampled_node_id = G_sampled.node_ids[id_index]
        sampled_node_neighbor_id = G_sampled.node_ids[G_sampled.end_points[G_sampled.ind_ptr[id_index]:
                                                                           G_sampled.ind_ptr[id_index + 1]]]
        # print("sampled_node_id: {}".format(sampled_node_id),"\n",
        #       "\t sampled_neighbor_ids: {}".format(sampled_node_neighbor_id))

        G_all_idx = G_all.reverse_map(sampled_node_id)
        node_neighbor_id = frozenset(G_all.node_ids[G_all.end_points[G_all.ind_ptr[G_all_idx]:
                                                                     G_all.ind_ptr[G_all_idx + 1]]].tolist())
        # print("node_id: {}".format(node_id), "\n",
        #       "\t neighbor_ids: {}".format(node_neighbor_id))

        for end_id in sampled_node_neighbor_id:
            if end_id not in node_neighbor_id:
                print("Wrong edge:", G_sampled.node_ids[id_index], end_id)
                return False
    return correct


if __name__ == '__main__':
    from mxgraph.config import cfg
    import cProfile, pstats
    cfg.DATA_NAME = 'ppi'
    from mxgraph.iterators import cfg_data_loader
    import time
    set_seed(100)
    G_all, features, labels, num_class = cfg_data_loader()

    ########################################################
    ############# Run Graph Sampling Algorithm #############
    ########################################################
    G_train = G_all.fetch_train()
    # pr = cProfile.Profile()
    # pr.enable()
    start = time.time()
    G_sampled_small = G_train.random_walk(initial_node=None,
                                          return_prob=0.0,
                                          walk_length=10000,
                                          max_node_num=2000,
                                          max_edge_num=None)
    end = time.time()
    print('Time spent for random walk sampling, sample %d nodes: %g'%(2000, end - start))
    G_sampled_sub_set = G_train.subgraph_by_node_ids(G_sampled_small.node_ids)
    G_sampled_sub_set.summary("Random Walk Node Subgraph")

    G_sampled_random_node = G_train.subgraph_by_node_ids(np.random.choice(G_train.node_ids, size=2000, replace=False))
    G_sampled_random_node.summary("Random Node Subgraph")
    start = time.time()
    G_sampled = G_train.random_walk(initial_node=None,
                                    return_prob=0.15,
                                    walk_length=2000,
                                    max_node_num=None,
                                    max_edge_num=None)
    end = time.time()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumulative')
    # ps.print_stats(10)
    print('Time spent for random walk:', end - start)
    G_sampled.summary('Random Walk Subgraph')
    print("==================================== Graph Sampling Finished ============================================\n")


    ########################################################
    ### testing the correctness of the samping algorithm ###
    ########################################################
    print("Testing the correctness of the samping algorithm...")
    correct = check_subgraph(G_sampled=G_sampled, G_all=G_train)
    if correct:
        print("Correctness Test Passed, G_sampled is a subgraph of G_train!")
    else:
        raise RuntimeError("Fail Test!")
    correct = check_subgraph(G_sampled=G_sampled, G_all=G_all)
    if correct:
        print("Correctness Test Passed, G_sampled is a subgraph of G_all!")
    else:
        raise RuntimeError("Fail Test!")
    print("==================================== Sampling Correctness Test Finished! ============================================\n")

