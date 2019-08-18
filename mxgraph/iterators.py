import os
import logging
import numpy as np
import scipy.sparse as ss
import json
import pandas as pd
from mxgraph.config import cfg
from mxgraph.graph import SimpleGraph, set_seed, HeterGraph, BiGraph
from mxgraph.sampler import parse_hierarchy_sampler_from_desc
from mxgraph import _graph_sampler
from mxgraph.utils import copy_to_ctx
from mxgraph.sampler import LinkPredEdgeSampler

try:
    import mxnet as mx
    import mxnet.ndarray as nd
except ImportError:
    print("MXNet not installed! The data iterator class will be not available")
    mx = None
    nd = None

def data_loader(base_name, training_num=None, load_walks=False):
    fea_label = np.load(base_name + "_fea_label.npz")
    features = fea_label['feature']
    labels = fea_label['label']
    if training_num is None:
        if cfg.SPLIT_TRAINING:
            assert cfg.TRAIN_SPLIT_NUM in [4, 8, 12, 16, 20]
            G = SimpleGraph.load(base_name + "_G"+str(cfg.TRAIN_SPLIT_NUM)+".npz")
        else:
            G = SimpleGraph.load(base_name + "_G.npz")
    else:
        assert training_num in [4, 8, 12, 16, 20]
        G = SimpleGraph.load(base_name + "_G" + str(training_num) + ".npz")

    walks = None
    if load_walks or cfg.LOAD_WALKS:
        walks_f = np.load(base_name + "_walks.npz")
        walks = walks_f["walks"]

    return G, features, labels, walks
def sampled_edges_loader(base_name, valid_neg_sample_scale, test_neg_sample_scale):
    valid_sampled_data = np.load(base_name + "_valid_pos_neg%d_edge.npz" % valid_neg_sample_scale)
    test_sampled_data = np.load(base_name + "_test_pos_neg%d_edge.npz" % test_neg_sample_scale)
    return valid_sampled_data, test_sampled_data
def get_num_class(dataset_name):
    if dataset_name == "reddit":
        num_class = 41
    elif dataset_name == "ppi":
        num_class = 121
    elif dataset_name == "citeseer":
        num_class = 6
    elif dataset_name == "cora":
        num_class = 7
    elif dataset_name == "pubmed":
        num_class = 3
    elif dataset_name == "movielens":
        num_class = 5
    else:
        raise ValueError("Dataset = %s not supported!" % dataset_name)
    return num_class
def cfg_data_loader(dataset_name=None, training_num=None, load_walks=False):
    if dataset_name is None:
        dataset_name = cfg.DATA_NAME
    G, features, labels, walks = data_loader(os.path.join(cfg.DATASET_PATH, dataset_name, dataset_name),
                                             training_num,
                                             load_walks)
    return G, features, labels, walks, get_num_class(dataset_name)
def cfg_data_loader_unsup(dataset_name=None, training_num=None, load_walks=False,
                          valid_neg_sample_scale=10, test_neg_sample_scale=10):
    if dataset_name is None:
        dataset_name = cfg.DATA_NAME
    G, features, labels, walks = data_loader(os.path.join(cfg.DATASET_PATH, dataset_name, dataset_name),
                                             training_num,
                                             load_walks)
    valid_sampled_data, test_sampled_data = sampled_edges_loader(os.path.join(cfg.DATASET_PATH, dataset_name, dataset_name),
                                                                 valid_neg_sample_scale, test_neg_sample_scale)
    return G, features, labels, walks, get_num_class(dataset_name), valid_sampled_data, test_sampled_data
def _get_hierarchical_sample_feature_label(features_all_nd, labels_all_nd,
                                           G_all, G_sub, node_indices, sampler):
    indices_in_merged_l, end_points_l, indptr_l, node_ids_l =\
        sampler.sample_by_indices(G_sub, node_indices)
    layer0_indices_in_G_all = nd.array(G_all.reverse_map(node_ids_l[0]),
                                       dtype=np.int32,
                                       ctx=features_all_nd.context)
    layer0_features_nd = nd.take(features_all_nd, indices=layer0_indices_in_G_all, axis=0)
    labels_nd = nd.take(labels_all_nd,
                        indices=nd.array(G_all.reverse_map(node_ids_l[-1]),
                                         dtype=np.int32, ctx=features_all_nd.context),
                        axis=0)
    return layer0_features_nd, \
           copy_to_ctx(end_points_l, ctx=features_all_nd.context, dtype=np.int32), \
           copy_to_ctx(indptr_l, ctx=features_all_nd.context, dtype=np.int32), \
           copy_to_ctx(indices_in_merged_l[1:], ctx=features_all_nd.context, dtype=np.int32), \
           labels_nd, \
           node_ids_l


class StaticGraphIterator(object):
    """Iterator for the static graph

    """
    def __init__(self, dataset_name=None, hierarchy_sampler_desc=("all", 2),
                 ctx=None, supervised=True, load_walks=False, neg_sample_size=20,
                 normalize_feature=True, batch_node_num=512,
                 batch_sample_method="uniform",
                 rw_max_step=10000, rw_initial_pos_num=2,
                 rw_sample_unused=True):
        """

        Parameters
        ----------
        dataset_name: str
        hierarchy_sampler_desc: list or tuple
        ctx: mx.context
        supervised: bool
        load_walks: bool
        neg_sample_size: int
        normalize_feature: bool
        batch_node_num: int
        batch_sample_method: str , it can be 'uniform' and 'random_walk'
        rw_max_step: int -- The maximum step of the random walk if batch_sample_method is 'random_walk'
        rw_initial_pos_num: int
        rw_sample_unused: bool
        """
        super(StaticGraphIterator, self).__init__()
        if ctx is None:
            self._ctx = mx.gpu()
        else:
            self._ctx = ctx
        if dataset_name is None:
            self._dataset_name = cfg.DATA_NAME
        else:
            self._dataset_name = dataset_name
        self._sampler = parse_hierarchy_sampler_from_desc(hierarchy_sampler_desc)
        self._supervised = supervised
        self._load_walks = load_walks
        self._neg_sample_size = neg_sample_size
        self._normalize_feature = normalize_feature
        self._batch_node_num = batch_node_num
        if supervised:
            self.G_all, self.features_all, self.labels_all, self.walks, self.num_class =\
                cfg_data_loader(dataset_name=self._dataset_name, load_walks=self._load_walks)
        else:
            self.G_all, self.features_all, self.labels_all, self.walks, self.num_class = \
                cfg_data_loader(dataset_name=self._dataset_name, load_walks=self._load_walks)
            # self.G_all, self.features_all, self.labels_all, self.walks, self.num_class, \
            # self.valid_sampled_data, self.test_sampled_data = \
            #     cfg_data_loader_unsup(dataset_name=self._dataset_name, load_walks=load_walks,
            #                           valid_neg_sample_scale=cfg.STATIC_GRAPH.MODEL.VALID_NEG_SAMPLE_SCALE,
            #                           test_neg_sample_scale=cfg.STATIC_GRAPH.MODEL.TEST_NEG_SAMPLE_SCALE)

            # self.train_neg_sampler = LinkPredEdgeSampler(neg_sample_scale=cfg.STATIC_GRAPH.MODEL.TRAIN_NEG_SAMPLE_SCALE,
            #                                              replace=cfg.STATIC_GRAPH.MODEL.TRAIN_NEG_SAMPLE_REPLACE)
        if normalize_feature:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(self.features_all[self.G_all.train_indices, :])
            self.features_all = scaler.transform(self.features_all)
        self.G_train = self.G_all.fetch_train()
        self.G_train_valid = self.G_all.fetch_train_valid()
        self.train_degrees = self.G_train.degrees
        self.train_degrees_p = self.train_degrees / np.sum(self.train_degrees)

        self._features_all_nd = nd.array(self.features_all, dtype=np.float32, ctx=self._ctx)
        self._labels_all_nd = nd.array(self.labels_all, dtype=np.float32, ctx=self._ctx)
        self._batch_sample_method = batch_sample_method
        if self._batch_sample_method not in ["uniform", "random_walk"]:
            raise NotImplementedError("Batch Sample Method = %s is not supported!"
                                      % self._batch_sample_method)
        self._rw_max_step = rw_max_step
        self._rw_initial_pos_num = rw_initial_pos_num
        self._rw_sample_unused = rw_sample_unused

        self._train_indices = np.random.permutation(self.G_train.train_indices)
        self._train_curr_pos = 0
        self._valid_indices = self.G_train_valid.valid_indices.copy()
        self._valid_curr_pos = 0
        self._test_indices = self.G_all.test_indices.copy()
        self._test_curr_pos = 0
        if not supervised:
            self._walks_indices = np.arange(self.walks.shape[0])
            self._walks_curr_pos = 0
        self._need_train_begin = True

    def set_batch_sample_mode(self, mode, rw_max_step=None,
                              rw_initial_pos_num=None, rw_sample_unused=None):
        self._batch_sample_method = mode
        self._need_train_begin = True
        if mode == "random_walk":
            if rw_max_step is not None:
                self._rw_max_step = rw_max_step
            if rw_initial_pos_num is not None:
                self._rw_initial_pos_num = rw_initial_pos_num
            if rw_sample_unused is not None:
                self._rw_sample_unused = rw_sample_unused

    def begin_epoch(self, mode):
        self._mode = mode
        if self._mode == "train":
            if self._batch_sample_method == "uniform":
                self._train_indices = np.random.permutation(self.G_train.train_indices)
                self._train_curr_pos = 0
            elif self._batch_sample_method == "random_walk":
                self._train_indices = set(self.G_train.train_indices)
                self._train_curr_pos = 0
            self._need_train_begin = False
        elif self._mode == "valid":
            self._valid_curr_pos = 0
        elif self._mode == "test":
            self._test_curr_pos = 0
        else:
            raise NotImplementedError()

    @property
    def epoch_finished(self):
        if self._mode == "train":
            if self._batch_sample_method == "uniform":
                return self._train_curr_pos >= len(self._train_indices)
            else:
                return len(self._train_indices) <= 0
        elif self._mode == "valid":
            return self._valid_curr_pos >= len(self._valid_indices)
        elif self._mode == "test":
            return self._test_curr_pos >= len(self._test_indices)
        else:
            raise NotImplementedError

    def sample(self):
        """

        Parameters
        ----------
        batch_node_num : int

        Returns
        -------
        layer0_features_nd : nd.NDArray
        end_points_l : list of NDArray
        indptr_l : list of NDArray
        indices_in_merged_l : list of NDArray
        labels_nd : nd.NDArray
        node_ids_l : list of np.ndarray
        """
        if self.epoch_finished:
            raise RuntimeError("Epoch has finished, you need to call `.begin_epoch()` again")
        if self._mode == "train":
            if self._need_train_begin:
                raise RuntimeError(
                    "No begin_epoch has been called, you need to call `.begin_epoch()` again")
            if self._batch_sample_method == "uniform":
                node_indices = self._train_indices[self._train_curr_pos:(self._train_curr_pos + self._batch_node_num)]
                self._train_curr_pos += node_indices.size
            elif self._batch_sample_method == "random_walk":
                sample_num = min(self._rw_initial_pos_num, len(self._train_indices))
                initial_inds = np.random.choice(a=list(self._train_indices),
                                                size=sample_num, replace=False)
                node_indices = set()
                for initial_ind in initial_inds:
                    #print('initial_ind:', initial_ind)
                    sampled_indices = _graph_sampler.get_random_walk_nodes(self.G_train.end_points,
                                                                           self.G_train.ind_ptr,
                                                                           initial_ind,
                                                                           self._batch_node_num // len(initial_inds),
                                                                           self._rw_max_step)
                    node_indices.update(sampled_indices)
                if self._rw_sample_unused:
                    # Removed the sampled indices train_indices
                    for ind in node_indices.intersection(self._train_indices):
                        self._train_indices.remove(ind)
                    #print(len(self._train_indices))
                node_indices = np.array(list(node_indices), dtype=np.int32)
                #print("node_indices", node_indices.shape)
            else:
                raise NotImplementedError
            if self.epoch_finished:
                self._need_train_begin = True
            return _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                          labels_all_nd=self._labels_all_nd,
                                                          G_all=self.G_all,
                                                          G_sub=self.G_train,
                                                          node_indices=node_indices,
                                                          sampler=self._sampler)
        elif self._mode == "valid":
            node_indices = self._valid_indices[self._valid_curr_pos:(self._valid_curr_pos + self._batch_node_num)]
            self._valid_curr_pos += node_indices.size
            return _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                          labels_all_nd=self._labels_all_nd,
                                                          G_all=self.G_all,
                                                          G_sub=self.G_train_valid,
                                                          node_indices=node_indices,
                                                          sampler=self._sampler)
        elif self._mode == "test":
            node_indices = self._test_indices[self._test_curr_pos:(self._test_curr_pos + self._batch_node_num)]
            self._test_curr_pos += node_indices.size
            return _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                          labels_all_nd=self._labels_all_nd,
                                                          G_all=self.G_all,
                                                          G_sub=self.G_all,
                                                          node_indices=node_indices,
                                                          sampler=self._sampler)
        else:
            raise NotImplementedError("Mode=%s is not supported" % self._mode)

    def summary(self):
        if self._supervised:
            logging.info("Supervised Training ......")
            self.G_all.summary(graph_name=self._dataset_name)
            logging.info("Feature Shape:" + str(self.features_all.shape))
            logging.info("Labels Shape:" + str(self.labels_all.shape))
            logging.info("Num Class:" + str(self.num_class))
        else:
            logging.info("Unsupervised Training ......")
            self.G_all.summary(graph_name=self._dataset_name)
            logging.info("Feature Shape:" + str(self.features_all.shape))
            assert self._load_walks is True
            logging.info("Walks Shape:" + str(self.walks.shape))

class StaticGraphEdgeIterator(StaticGraphIterator):
    """Iterator for the unsupervised static graph

    """

    def begin_epoch(self, mode):
        self._mode = mode
        if self._mode == "walks":
            self._walks_indices = np.random.permutation(self._walks_indices)
            self._walks_curr_pos = 0
            self._need_train_begin = False
        elif self._mode == "train":
            self._train_curr_pos = 0
        elif self._mode == "valid":
            self._valid_curr_pos = 0
        elif self._mode == "test":
            self._test_curr_pos = 0
        else:
            raise NotImplementedError

    @property
    def epoch_finished(self):
        if self._mode == "walks":
            return self._walks_curr_pos + self._batch_node_num>= len(self._walks_indices)
        elif self._mode == "train":
            return self._train_curr_pos >= len(self._train_indices)
        elif self._mode == "valid":
            return self._valid_curr_pos >= len(self._valid_indices)
        elif self._mode == "test":
            return self._test_curr_pos >= len(self._test_indices)
        else:
            raise NotImplementedError

    # def _neg_sampler(self):
    #     """ Weighted uniform sampler
    #
    #     Returns
    #     -------
    #         np.array: Shape (batch_size*neg_sample_size, )
    #     """
    #
    #     chosen_indices = np.random.choice(self.G_train.train_indices,
    #                                       size=(self._batch_node_num, self._neg_sample_size),
    #                                       p=self.train_degrees_p)
    #     #print("chosen_indices:", chosen_indices.dtype)
    #     return chosen_indices
    def _neg_sampler(self):
        """ Weighted uniform sampler

        Returns
        -------
            np.array: Shape (batch_size*neg_sample_size, )
        """

        chosen_indices = np.random.choice(self.G_train.train_indices,
                                          #size = self._neg_sample_size,
                                          size=(self._batch_node_num, self._neg_sample_size),
                                          p = self.train_degrees_p)
        return chosen_indices

    def sample(self):
        """

        Parameters
        ----------
        batch_node_num : int

        Returns
        -------
        layer0_features_nd : nd.NDArray
        end_points_l : list of NDArray
        indptr_l : list of NDArray
        indices_in_merged_l : list of NDArray
        labels_nd : nd.NDArray
        node_ids_l : list of np.ndarray
        """
        if self.epoch_finished:
            raise RuntimeError("Epoch has finished, you need to call `.begin_epoch()` again")
        if self._mode == "walks":
            if self._need_train_begin:
                raise RuntimeError(
                    "No begin_epoch has been called, you need to call `.begin_epoch()` again")
            walk_indices = self._walks_indices[self._walks_curr_pos:(self._walks_curr_pos + self._batch_node_num)]
            ### to vold the last iter, whose size may be less than batch_node_num
            current_batch_size = walk_indices.size
            assert current_batch_size == self._batch_node_num
            self._walks_curr_pos += current_batch_size

            if self.epoch_finished:
                self._need_train_begin = True
            """
            source_indices = self.walks[walk_indices][:,0] ### Shape: (batch_size, )
            target_indices = self.walks[walk_indices][:,1] ### Shape: (batch_size, )
            #print("source_indices", source_indices.dtype, source_indices.shape)
            #print("target_indices", target_indices.dtype, target_indices.shape)

            #### negtive samplers
            neg_sampled_indices = self._neg_sampler() ## Shape: (neg_sample_size, )
            #print("neg_sampled_indices:", neg_sampled_indices.dtype, neg_sampled_indices.shape)
            # pos_neg_neighbor_indices = np.hstack((target_indices, neg_sampled_indices)) ## Shape: (current_batch_size, 1+neg_sample_size)
            # pos_neg_neighbor_indices = pos_neg_neighbor_indices.reshape(-1) ## Shape: (current_batch_size * (1+neg_sample_size))

            source_info = _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                                 labels_all_nd=self._labels_all_nd,
                                                                 G_all=self.G_all,
                                                                 G_sub=self.G_train,
                                                                 node_indices=source_indices,
                                                                 sampler=self._sampler)
            pos_neighbor_info = _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                                       labels_all_nd=self._labels_all_nd,
                                                                       G_all=self.G_all,
                                                                       G_sub=self.G_train,
                                                                       node_indices=target_indices,
                                                                       sampler=self._sampler)
            neg_neighbor_info = _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                                       labels_all_nd=self._labels_all_nd,
                                                                       G_all=self.G_all,
                                                                       G_sub=self.G_train,
                                                                       node_indices=neg_sampled_indices,
                                                                       sampler=self._sampler)

            # pos_neg_labels = np.ones((current_batch_size, (self._neg_sample_size+1))) * -1.0
            # pos_neg_labels[:,0] = 1.0
            # pos_neg_labels_nd = nd.array(pos_neg_labels.reshape(-1), ctx=source_info[0].context, dtype=np.float32)
            #print("pos_neg_labels_nd", pos_neg_labels_nd.shape, pos_neg_labels_nd)

            return source_info, pos_neighbor_info, neg_neighbor_info
            """

            source_indices = self.walks[walk_indices][:, 0]
            target_indices = self.walks[walk_indices][:, 1].reshape((-1, 1))  # target_indices (batch_size, 1)
            # print("target_indices", target_indices.shape, target_indices)
            #### negtive samplers
            neg_sampled_indices = self._neg_sampler()  ## Shape: (current_batch_size, neg_sample_size)

            pos_neg_neighbor_indices = np.hstack(
                (target_indices, neg_sampled_indices))  ## Shape: (current_batch_size, 1+neg_sample_size)
            pos_neg_neighbor_indices = pos_neg_neighbor_indices.reshape(
                -1)  ## Shape: (current_batch_size * (1+neg_sample_size))

            source_info = _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                                 labels_all_nd=self._labels_all_nd,
                                                                 G_all=self.G_all,
                                                                 G_sub=self.G_train,
                                                                 node_indices=source_indices,
                                                                 sampler=self._sampler)
            pos_neg_neighbor_info = _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                                           labels_all_nd=self._labels_all_nd,
                                                                           G_all=self.G_all,
                                                                           G_sub=self.G_train,
                                                                           node_indices=pos_neg_neighbor_indices,
                                                                           sampler=self._sampler)

            pos_neg_labels = np.ones((current_batch_size, (self._neg_sample_size + 1))) * -1.0
            pos_neg_labels[:, 0] = 1.0
            pos_neg_labels_nd = nd.array(pos_neg_labels.reshape(-1), ctx=source_info[0].context, dtype=np.float32)
            # print("pos_neg_labels_nd", pos_neg_labels_nd.shape, pos_neg_labels_nd)

            return source_info, pos_neg_neighbor_info, pos_neg_labels_nd

        elif self._mode == "train":
            node_indices = self._train_indices[self._train_curr_pos:(self._train_curr_pos + self._batch_node_num)]
            self._train_curr_pos += node_indices.size
            return _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                          labels_all_nd=self._labels_all_nd,
                                                          G_all=self.G_all,
                                                          G_sub=self.G_train,
                                                          node_indices=node_indices,
                                                          sampler=self._sampler)
        elif self._mode == "valid":
            node_indices = self._valid_indices[self._valid_curr_pos:(self._valid_curr_pos + self._batch_node_num)]
            self._valid_curr_pos += node_indices.size
            return _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                          labels_all_nd=self._labels_all_nd,
                                                          G_all=self.G_all,
                                                          G_sub=self.G_train_valid,
                                                          node_indices=node_indices,
                                                          sampler=self._sampler)
        elif self._mode == "test":
            node_indices = self._test_indices[self._test_curr_pos:(self._test_curr_pos + self._batch_node_num)]
            self._test_curr_pos += node_indices.size
            return _get_hierarchical_sample_feature_label(features_all_nd=self._features_all_nd,
                                                          labels_all_nd=self._labels_all_nd,
                                                          G_all=self.G_all,
                                                          G_sub=self.G_all,
                                                          node_indices=node_indices,
                                                          sampler=self._sampler)
        else:
            raise NotImplementedError("Mode=%s is not supported" % self._mode)

class TrafficIterator(object):
    """Iterator for the traffic dataset

    """
    def __init__(self, dataset_name=None, adj_threshold=0.1,
                 ctx=None, add_time_in_day=True, normalize_feature=True,
                 val_ratio=0.1, test_ratio=0.2,
                 in_length=12, out_length=12, train_val_batchsize=64):
        """

        Parameters
        ----------
        dataset_name : str or None
        adj_threshold : float
        ctx : mx.context
        val_ratio : float
        test_ratio : float
        horizon : int
        """
        if ctx is None:
            self._ctx = mx.gpu()
        else:
            self._ctx = ctx
        if dataset_name is None:
            dataset_name = cfg.DATA_NAME
        self._dataset_name = dataset_name
        self._adj_threshold = adj_threshold
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._add_time_in_day = add_time_in_day
        self._normalize_feature = normalize_feature
        self._in_length = in_length
        self._out_length = out_length
        self._train_val_batchsize = train_val_batchsize
        # Load Pandas Data Frame and reorder the Dataframe
        self.pd_data = pd.read_hdf(os.path.join(cfg.DATASET_PATH, dataset_name, 'traffic_data.h5'))
        self.sensor_ids = json.load(open(os.path.join(cfg.DATASET_PATH, dataset_name, 'sensor_ids.json')))
        self.coordinates = np.load(os.path.join(cfg.DATASET_PATH, dataset_name, 'coordinates.npy'))
        self.pd_data = self.pd_data.loc[:, self.sensor_ids]  # Important!
        # Load and transform adj matrix
        self.adj_mx = np.load(os.path.join(cfg.DATASET_PATH, dataset_name, 'adj_mx.npy'))
        self.adj_mx[self.adj_mx < adj_threshold] = 0
        # Loading the training/validation/testing sequences
        n_sample = self.pd_data.shape[0]
        n_val = int(round(n_sample * val_ratio))
        n_test = int(round(n_sample * test_ratio))
        n_train = n_sample - n_val - n_test
        self.pd_train_data, self.pd_val_data, self.pd_test_data = self.pd_data.iloc[:n_train, :],\
                                          self.pd_data.iloc[n_train:(n_train + n_val), :],\
                                          self.pd_data.iloc[(n_train + n_val):, :]
        # Get Mean and Variance
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        self._scaler.fit(self.pd_train_data)

        self.train_data = np.expand_dims(self.pd_train_data.values, axis=-1)
        self.val_data = np.expand_dims(self.pd_val_data.values, axis=-1)
        self.test_data = np.expand_dims(self.pd_test_data.values, axis=-1)
        assert add_time_in_day is True
        assert normalize_feature is True
        if add_time_in_day:
            # Append time within the day to the features
            train_time_ind =\
                (self.pd_train_data.index.values
                 - self.pd_train_data.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
            self.train_data = np.concatenate([self.train_data,
                                              np.broadcast_to(train_time_ind.reshape((-1, 1, 1)), self.train_data.shape)],
                                              axis=2)
            val_time_ind = \
                (self.pd_val_data.index.values
                 - self.pd_val_data.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
            self.val_data = np.concatenate([self.val_data,
                                            np.broadcast_to(val_time_ind.reshape((-1, 1, 1)), self.val_data.shape)],
                                            axis=2)
            test_time_ind = \
                (self.pd_test_data.index.values
                 - self.pd_test_data.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
            self.test_data = np.concatenate([self.test_data,
                                             np.broadcast_to(test_time_ind.reshape((-1, 1, 1)), self.test_data.shape)],
                                             axis=2)  # Shape (total_seq_len, node_num, 2)
        # We just drop the last few instances. This is to make our code consistent with DCRNN
        train_batch_len = self.train_data.shape[0] // self._train_val_batchsize
        val_batch_len = self.val_data.shape[0] // self._train_val_batchsize
        self.train_data = self.train_data[:(self._train_val_batchsize * train_batch_len),
                          :].reshape((self._train_val_batchsize, train_batch_len,
                                      self.train_data.shape[1], self.train_data.shape[2])).transpose((1, 0, 2, 3))  # Shape (total_seq_len, batch_size, node_num, 2)
        self.val_data = self.val_data[:(self._train_val_batchsize * val_batch_len),
                          :].reshape((self._train_val_batchsize, val_batch_len,
                                      self.val_data.shape[1], self.val_data.shape[2])).transpose((1, 0, 2, 3))  # Shape (total_seq_len, batch_size, node_num, 2)
        self._train_begin_inds = np.arange(0, train_batch_len - self._out_length - self._in_length + 1,
                                          dtype=np.int32)
        self._val_begin_inds = np.arange(0, val_batch_len - self._out_length - self._in_length + 1,
                                        dtype=np.int32)
        self._test_begin_inds = np.arange(0, self.test_data.shape[0] -
                                          self._out_length - self._in_length + 1, dtype=np.int32)
        self._train_curr_pos = 0
        self._valid_curr_pos = 0
        self._test_curr_pos = 0

    def begin_epoch(self, mode="train"):
        assert mode in ["train", "valid", "test"]
        self._mode = mode
        if mode == "train":
            self._train_begin_inds = np.random.permutation(self._train_begin_inds)
            self._train_curr_pos = 0
        elif mode == "valid":
            self._valid_curr_pos = 0
        elif mode == "test":
            self._test_curr_pos = 0
        else:
            raise NotImplementedError

    @property
    def epoch_finished(self):
        if self._mode == "train":
            return self._train_curr_pos >= len(self._train_begin_inds)
        elif self._mode == "valid":
            return self._valid_curr_pos >= len(self._val_begin_inds)
        elif self._mode == "test":
            return self._test_curr_pos >= len(self._test_begin_inds)
        else:
            raise NotImplementedError

    @property
    def test_instance_num(self):
        return len(self._test_begin_inds)

    @property
    def out_length(self):
        return self._out_length

    @property
    def node_num(self):
        return self.pd_data.shape[1]

    @property
    def mean(self):
        return self._scaler.mean_

    @property
    def std(self):
        return self._scaler.scale_

    def summary(self):
        logging.info("Data Name: %s" % self._dataset_name)
        logging.info("   In Length=%d, Out Length=%d" %(self._in_length, self._out_length))
        logging.info("   Train=%d, Valid=%d, Test=%d"
                     % (self._train_val_batchsize * len(self._train_begin_inds),
                        self._train_val_batchsize * len(self._val_begin_inds),
                        len(self._test_begin_inds)))

    def get_sym_mat_data_with_edge(self):
        raise NotImplementedError

    def sample(self):
        """Sample data

        Parameters
        ----------

        Returns
        -------
        data_in : NDArray
            Shape (seq_len, batch_size, node_num, 2)
            * Will be transformed
        data_out : NDArray
            Shape (seq_len, batch_size, node_num, 1)
        out_additional_data : NDArray
            Shape (seq_len, batch_size, node_num, 1) or Shape (seq_len, batch_size, node_num, 3)
        """
        if self.epoch_finished:
            raise RuntimeError("Epoch has finished! Need to call begin_epoch() again!")
        if self._mode == 'train':
            begin_ind = self._train_begin_inds[self._train_curr_pos]
            data_in = self.train_data[begin_ind:(begin_ind + self._in_length), :, :, :].copy()
            data_out = self.train_data[(begin_ind + self._in_length):
                                       (begin_ind + self._in_length + self._out_length), :, :, 0:1].copy()
            out_additional_data =\
                self.train_data[(begin_ind + self._in_length):
                                (begin_ind + self._in_length + self._out_length), :, :, 1:2].copy()
            self._train_curr_pos += 1
        elif self._mode == 'valid':
            begin_ind = self._val_begin_inds[self._valid_curr_pos]
            data_in = self.val_data[begin_ind:(begin_ind + self._in_length), :, :, :].copy()
            data_out = self.val_data[(begin_ind + self._in_length):
                                     (begin_ind + self._in_length + self._out_length), :, :, 0:1].copy()
            out_additional_data = \
                self.val_data[(begin_ind + self._in_length):
                              (begin_ind + self._in_length + self._out_length), :, :, 1:2].copy()
            self._valid_curr_pos += 1
        elif self._mode == 'test':
            test_batchsize = min(self._train_val_batchsize,
                                 self.test_instance_num - self._test_curr_pos)
            data_in = np.empty(shape=(self._in_length, test_batchsize, self.node_num, 2),
                               dtype=np.float32)
            data_out = np.empty(shape=(self._out_length, test_batchsize, self.node_num, 1),
                                dtype=np.float32)
            out_additional_data = np.empty(shape=(self._out_length, test_batchsize, self.node_num, 1),
                                       dtype=np.float32)
            for i in range(test_batchsize):
                begin_ind = self._test_begin_inds[self._test_curr_pos]
                data_in[:, i, :, :] = self.test_data[begin_ind:(begin_ind + self._in_length), :, :]
                data_out[:, i, :, :] =\
                    self.test_data[(begin_ind + self._in_length):
                                   (begin_ind + self._in_length + self._out_length), :, 0:1]
                out_additional_data[:, i, :, :] = \
                    self.test_data[(begin_ind + self._in_length):
                                   (begin_ind + self._in_length + self._out_length), :, 1:2]
                self._test_curr_pos += 1
        else:
            raise NotImplementedError
        if self._normalize_feature:
            # Normalize the features
            data_in[:, :, :, 0] = (data_in[:, :, :, 0] - self.mean) / self.std
        if cfg.SPATIOTEMPORAL_GRAPH.USE_COORDINATES:
            normalized_coordinates = (self.coordinates -
                                      self.coordinates.mean(axis=0, keepdims=True))\
                                     / self.coordinates.std(axis=0, keepdims=True)
            data_in_coordinates =\
                np.broadcast_to(np.reshape(normalized_coordinates,
                                           newshape=(1, 1) + normalized_coordinates.shape),
                                shape=data_in.shape[:2] + normalized_coordinates.shape)
            data_out_coordinates =\
                np.broadcast_to(np.reshape(normalized_coordinates,
                                           newshape=(1, 1) + normalized_coordinates.shape),
                                shape=data_out.shape[:2] + normalized_coordinates.shape)
            data_in = np.concatenate([data_in, data_in_coordinates],
                                     axis=-1)
            out_additional_data = np.concatenate([out_additional_data, data_out_coordinates],
                                                 axis=-1)
        data_in = nd.array(data_in, ctx=self._ctx, dtype=np.float32)
        data_out = nd.array(data_out, ctx=self._ctx, dtype=np.float32)
        out_additional_data = nd.array(out_additional_data, ctx=self._ctx, dtype=np.float32)
        return data_in, data_out, out_additional_data


def get_num_set(dataset_name):
    if dataset_name == "movielens":
        num_set = 3
    else:
        raise ValueError("Dataset = %s not supported!" % dataset_name)
    return num_set

def cfg_heter_data_loader(dataset_name, version=None):
    if dataset_name is None:
        dataset_name = cfg.DATA_NAME
    if version is None: ### version can be 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m'
        path_dir = os.path.join(cfg.DATASET_PATH, dataset_name, "processed")
    else:
        path_dir = os.path.join(cfg.DATASET_PATH, dataset_name, version, "processed")

    preds = np.load(os.path.join(path_dir, "preds.npz")) ## user_id, movie_id, genre, rate
    train_preds = preds["train_preds"]
    valid_preds = preds["valid_preds"]
    test_preds = preds["test_preds"]
    G = HeterGraph.load(os.path.join(path_dir, "G.npz"))
    feas = np.load(os.path.join(path_dir, "feas.npz"))
    num_set = get_num_set(dataset_name)
    set_instances = feas["set_instances"]
    feas_l = [feas["set"+str(i)+"_feas"] for i in range(num_set)]
    ### check for None feature sets and then set it to be an identity matrix
    for i in range(num_set):
        if isinstance(feas_l[i], np.ndarray) and feas_l[i].size == 1 and feas_l[i] == None:
            feas_l[i] = np.identity(set_instances[i], dtype=np.int32)
    #print("feas_l[-1] ==None?:", feas_l[-1] == None) ## True
    G.info()
    return G, feas_l, train_preds, valid_preds, test_preds, num_set, get_num_class(dataset_name)


def _heter_get_hierarchical_sample_feature(features_all_set_nd_l, G_sub, node_indices,
                                           sampler, labels, ctx, loss_type, rates, num_set):
    indices_in_merged_l, end_points_l, indptr_l, node_ids_l = sampler.sample_by_indices(G_sub, node_indices)
    # print("indices_in_merged_l", len(indices_in_merged_l), indices_in_merged_l[0].shape,
    #       indices_in_merged_l[1].shape, indices_in_merged_l[2].shape)
    # print("end_points_l", len(end_points_l), end_points_l[0].shape, end_points_l[1].shape)
    # print("indptr_l", len(indptr_l), indptr_l[0].shape, indptr_l[1].shape)
    # print("node_ids_l", len(node_ids_l), len(node_ids_l[0]), len(node_ids_l[1]), len(node_ids_l[2]))
    # print("#Sample nodes", len(node_ids_l[0]))
    sampled_node_order = np.arange(len(node_ids_l[0]), dtype=np.int32)
    node_sets = G_sub.node_sets_by_node_ids(node_ids_l[0])
    #print("node_sets", node_sets.shape, node_sets)
    node_sets_masks = nd.zeros(shape=(len(node_ids_l[0]), num_set, 1), ctx=ctx, dtype=np.float32)
    node_sets_masks[nd.arange(len(node_ids_l[0]), ctx=ctx), nd.array(node_sets, ctx=ctx), 0] = 1.
    node_indices_in_set = G_sub.node_indices_in_set_by_node_ids(node_ids_l[0])
    #all_set_info_np = np.transpose(np.vstack((node_ids_l[0], sampled_set_order, node_sets, node_indices_in_set)))
    layer0_features_nd_l=[] ### one node feature array for one set
    sampled_node_order_nd_l=[]
    for i in range(num_set):
        node_indices_in_set_sub_nd = nd.array(node_indices_in_set[node_sets == i], dtype=np.int32, ctx=ctx)
        sampled_feas_nd = nd.take(features_all_set_nd_l[i], indices=node_indices_in_set_sub_nd, axis=0)
        sampled_node_order_nd = nd.array(sampled_node_order[node_sets==i], dtype=np.int32, ctx=ctx)
        layer0_features_nd_l.append(sampled_feas_nd)
        sampled_node_order_nd_l.append(sampled_node_order_nd)
    sampled_node_order_nd = nd.concat(*sampled_node_order_nd_l, dim=0)
    # print("{}+{}+{}={}?", sampled_node_order_nd_l[0].size,sampled_node_order_nd_l[1].size, sampled_node_order_nd_l[2].size,
    #       sampled_node_order_nd.size)
    if loss_type == "regression":
        labels_nd = nd.array(labels, dtype=np.float32, ctx=ctx)
    elif loss_type == "classification":
        labels_nd = nd.array(labels, dtype=np.float32, ctx=ctx) - 1
    else:
        raise NotImplementedError
    return sampled_node_order_nd,\
           layer0_features_nd_l, \
           copy_to_ctx(end_points_l, ctx=ctx, dtype=np.int32), \
           copy_to_ctx(indptr_l, ctx=ctx, dtype=np.int32), \
           copy_to_ctx(indices_in_merged_l[1:], ctx=ctx, dtype=np.int32), \
           labels_nd, \
           node_ids_l,\
           rates,\
           node_sets_masks


class HeterGraphNodeIterator(object):
    """Iterator for the interaction dataset

    """
    def __init__(self, dataset_name=None, version=None, ctx=None,
                 #supervised=True, neg_sample_size=20, normalize_feature=True,
                 batch_node_num=128, batch_sample_method="uniform", hierarchy_sampler_desc=("all", 2),
                 loss_type = "regression"):
        if ctx is None:
            self._ctx = mx.gpu()
        else:
            self._ctx = ctx
        if dataset_name is None:
            dataset_name = cfg.DATA_NAME ##
        self._dataset_name = dataset_name
        if version is None:
            version = cfg.DATA_VERSION
        self._loss_type = loss_type

        self.G_all, self.feas_l, self.train_preds, self.valid_preds, self.test_preds, self._num_set , self._num_class = \
            cfg_heter_data_loader(dataset_name, version)
        self._num_pred_set = self.train_preds.shape[1] - 1
        #print("_num_pred_set: ", self._num_pred_set)
        if self._loss_type == "regression":
            self._num_class = 1
        if version == "ml-1m" or version == "ml-100k":
            self.rates = nd.array([1., 2., 3., 4., 5.], ctx=self._ctx)
        else:
            raise NotImplementedError

        ### processing the graph
        self.G_train = self.G_all.fetch_train()
        self.G_train_valid = self.G_all.fetch_train_valid()
        self.train_degrees = self.G_train.degrees
        self.train_degrees_p = self.train_degrees / np.sum(self.train_degrees)
        ### processing the graph sampling method
        self._batch_sample_method = batch_sample_method
        self._sampler = parse_hierarchy_sampler_from_desc(hierarchy_sampler_desc)
        if self._batch_sample_method not in ["uniform"]:
            raise NotImplementedError("Batch Sample Method = %s is not supported!"
                                      % self._batch_sample_method)
        ### processing the features
        self.feas_nd_l = []
        for i in range(len(self.feas_l)):
            self.feas_nd_l.append(nd.array(self.feas_l[i], dtype=np.float32, ctx=self._ctx))
        ### processing the batch mode parameters
        self._batch_node_num = batch_node_num
        self._train_indices = np.random.permutation(self.train_preds.shape[0])
        self._train_curr_pos = 0
        if isinstance(self.valid_preds, np.ndarray) and self.valid_preds.size == 1 and self.valid_preds == None:
            print("valid_preds is None")
            self._f_valid = False
        else:
            self._f_valid = True
            self._valid_indices = np.arange(self.valid_preds.shape[0])
            self._valid_curr_pos = 0
        if isinstance(self.test_preds, np.ndarray) and self.test_preds.size == 1 and self.test_preds == None:
            print("test_preds is None")
            self._f_test = False
        else:
            self._f_test = True
            self._test_indices = np.arange(self.test_preds.shape[0])
            self._test_curr_pos = 0

        self._need_train_begin = True

    @property
    def num_class(self):
        return self._num_class
    @property
    def num_set(self):
        return self._num_set

    @property
    def num_pred_set(self):
        return self._num_pred_set

    def set_batch_sample_mode(self, mode):
        self._batch_sample_method = mode
        self._need_train_begin = True

    def begin_epoch(self, mode):
        self._mode = mode
        if self._mode == "train":
            if self._batch_sample_method == "uniform":
                self._train_indices = np.random.permutation(self.train_preds.shape[0])
                self._train_curr_pos = 0
            self._need_train_begin = False
        elif self._mode == "valid" and self._f_valid:
            self._valid_curr_pos = 0
        elif self._mode == "test" and self._f_test:
            self._test_curr_pos = 0
        else:
            raise NotImplementedError()

    @property
    def epoch_finished(self):
        if self._mode == "train":
            if self._batch_sample_method == "uniform":
                return self._train_curr_pos >= len(self._train_indices)
        elif self._mode == "valid" and self._f_valid:
            return self._valid_curr_pos >= len(self._valid_indices)
        elif self._mode == "test" and self._f_test:
            return self._test_curr_pos >= len(self._test_indices)
        else:
            raise NotImplementedError

    def sample(self):
        if self.epoch_finished:
            raise RuntimeError("Epoch has finished, you need to call `.begin_epoch()` again")
        if self._mode == "train":
            if self._need_train_begin:
                raise RuntimeError(
                    "No begin_epoch has been called, you need to call `.begin_epoch()` again")
            if self._batch_sample_method == "uniform":
                chosen_train_preds_indexs = self._train_indices[self._train_curr_pos:
                                                                (self._train_curr_pos + self._batch_node_num)]
                chosen_preds = self.train_preds[chosen_train_preds_indexs]
                chosen_node_indices = np.concatenate([self.G_train.reverse_map(chosen_preds[:, i])
                                                        for i in range(self._num_pred_set)])
                #print("chosen_node_indices", chosen_node_indices.dtype, chosen_node_indices.shape, chosen_node_indices)
                # chosen_node_indices_set_id = np.concatenate([np.ones(chosen_preds[:, i].size, dtype=np.int32)*i
                #                                                for i in range(self._pred_set)])
                #print("chosen_node_indices_set_id", chosen_node_indices_set_id.dtype, chosen_node_indices_set_id.shape, chosen_node_indices_set_id)
                labels = chosen_preds[:, -1]

                #print("labels", labels.dtype, labels.shape, labels)
                self._train_curr_pos += chosen_train_preds_indexs.shape[0]
            else:
                raise NotImplementedError
            if self.epoch_finished:
                self._need_train_begin = True
            return _heter_get_hierarchical_sample_feature(features_all_set_nd_l=self.feas_nd_l,
                                                          G_sub=self.G_train,
                                                          node_indices=chosen_node_indices,
                                                          sampler=self._sampler,
                                                          labels=labels,
                                                          ctx=self._ctx,
                                                          loss_type = self._loss_type,
                                                          rates=self.rates,
                                                          num_set=self._num_set)
        elif self._mode == "test" and self._f_test:
            chosen_test_preds_indexs = self._test_indices[self._test_curr_pos:
                                                          (self._test_curr_pos + self._batch_node_num)]
            chosen_preds = self.test_preds[chosen_test_preds_indexs]
            chosen_node_indices = np.concatenate([self.G_all.reverse_map(chosen_preds[:, i])
                                                  for i in range(self._num_pred_set)])
            labels = chosen_preds[:, -1]
            self._test_curr_pos += chosen_test_preds_indexs.shape[0]
            return _heter_get_hierarchical_sample_feature(features_all_set_nd_l=self.feas_nd_l,
                                                          G_sub=self.G_all,
                                                          node_indices=chosen_node_indices,
                                                          sampler=self._sampler,
                                                          labels=labels,
                                                          ctx=self._ctx,
                                                          loss_type = self._loss_type,
                                                          rates=self.rates,
                                                          num_set=self._num_set)

    def summary(self, indent_token = "\t"):
        logging.info("Supervised Training ......")
        logging.info("Graph ---->")
        self.G_all.summary(graph_name=self._dataset_name)
        #logging.info(indent_token + "Batch sampling method:", str(self._batch_sample_method))
        logging.info("Features ---->")
        logging.info(indent_token + "{} node sets".format(self._num_set))
        for i in range(self._num_set):
            info_str = indent_token + indent_token + "Set" + str(i) + \
                       ": Shape ({}, {})".format(self.feas_l[i].shape[0], self.feas_l[i].shape[1])
            logging.info(info_str)
        logging.info("# Training preds: {}".format(self.train_preds.shape[0]))
        if self._f_valid:
            logging.info("# Valid preds: {}".format(self.valid_preds.shape[0]))
        if self._f_test:
            logging.info("# Testing preds: {}".format(self.test_preds.shape[0]))



def cfg_bi_data_loader(dataset_name, version=None):
    if dataset_name is None:
        dataset_name = cfg.DATA_NAME
    if version is None: ### version can be 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m'
        path_dir = os.path.join(cfg.DATASET_PATH, dataset_name)
    else:
        path_dir = os.path.join(cfg.DATASET_PATH, dataset_name, version)

    path_dir = os.path.join(path_dir, "bipartite")

    preds = np.load(os.path.join(path_dir, "preds.npz")) ## user_id, movie_id, genre, rate
    train_preds = preds["train_preds"]
    valid_preds = preds["valid_preds"]
    test_preds = preds["test_preds"]

    G = BiGraph.load(os.path.join(path_dir, "G.npz"))

    print("G.num_node_set", G.num_node_set, type(G.num_node_set))
    print("G.num_edge_set", G.num_edge_set, type(G.num_edge_set))

    feas = np.load(os.path.join(path_dir, "feas.npz"))
    set_instances = feas["set_instances"]
    feas_l = [feas["set" + str(i) + "_feas"] for i in range(G.num_node_set)]
    ### check for None feature sets and then set it to be an identity matrix
    for i in range(G.num_node_set):
        if isinstance(feas_l[i], np.ndarray) and feas_l[i].size == 1 and feas_l[i] == None:
            ### TODO can change to the label embedding format with the node indices as the embedding input
            feas_l[i] = np.identity(set_instances[i], dtype=np.int32)
    G.info()
    return G, feas_l, train_preds, valid_preds, test_preds


def _bi_get_hierarchical_sample_feature(features_all_set_nd_l, G_sub, node_indices,
                                        sampler, labels, ctx, rates, num_node_set, num_edge_set):
    # print("Input node_indices in subgraphs", node_indices.shape, "\n", node_indices)
    indices_in_merged_l, end_points_l, indptr_l, node_ids_l, end_points_edge_weight_l = \
        sampler.sample_by_indices_with_edge_weight(G_sub, node_indices)
    # print("indices_in_merged_l", len(indices_in_merged_l), indices_in_merged_l[0].shape,
    #       indices_in_merged_l[1].shape, indices_in_merged_l[2].shape)
    # print("end_points_edge_weight_l", len(end_points_edge_weight_l), end_points_edge_weight_l[0].shape, end_points_edge_weight_l[1].shape)
    # print("end_points_l", len(end_points_l), end_points_l[0].shape, end_points_l[1].shape)
    # print("indptr_l", len(indptr_l), indptr_l[0].shape, indptr_l[1].shape)
    # print("node_ids_l", len(node_ids_l), len(node_ids_l[0]), len(node_ids_l[1]), len(node_ids_l[2]))
    # print("#Sample nodes", len(node_ids_l[0]))
    # print("---------------------------------------------------\n\n")
    seg_indices_l=[np.arange(len(end_points)) for end_points in end_points_l]
    sampled_node_order = np.arange(len(node_ids_l[0]), dtype=np.int32)
    node_sets = G_sub.node_sets_by_node_ids(node_ids_l[0])
    #print("node_sets", node_sets.shape, node_sets)

    node_type_mask = nd.zeros(shape=(len(node_ids_l[0]), num_node_set, 1), ctx=ctx, dtype=np.float32)
    node_type_mask[nd.arange(len(node_ids_l[0]), ctx=ctx), nd.array(node_sets, ctx=ctx), 0] = 1.
    #print("node_type_mask", node_type_mask)
    edge_type_mask_l = []
    for end_points_edge_weight in end_points_edge_weight_l:
        nnz = end_points_edge_weight.size
        edge_type_mask = nd.zeros(shape=(nnz, num_edge_set, 1), ctx = ctx, dtype=np.float32)
        edge_type_mask[nd.arange(nnz, ctx=ctx), nd.array(end_points_edge_weight-1, ctx=ctx), 0] = 1.
        edge_type_mask_l.append(edge_type_mask)
        #print("edge_type_mask", edge_type_mask)
    node_indices_in_set = G_sub.node_indices_in_set_by_node_ids(node_ids_l[0])
    #all_set_info_np = np.transpose(np.vstack((node_ids_l[0], sampled_set_order, node_sets, node_indices_in_set)))
    layer0_features_nd_l=[] ### one node feature array for one set
    sampled_node_order_nd_l=[]
    for i in range(num_node_set):
        node_indices_in_set_sub_nd = nd.array(node_indices_in_set[node_sets == i], dtype=np.int32, ctx=ctx)
        sampled_feas_nd = nd.take(features_all_set_nd_l[i], indices=node_indices_in_set_sub_nd, axis=0)
        sampled_node_order_nd = nd.array(sampled_node_order[node_sets==i], dtype=np.int32, ctx=ctx)
        layer0_features_nd_l.append(sampled_feas_nd)
        sampled_node_order_nd_l.append(sampled_node_order_nd)
    sampled_node_order_nd = nd.concat(*sampled_node_order_nd_l, dim=0)
    # print("{}+{}+{}={}?", sampled_node_order_nd_l[0].size,sampled_node_order_nd_l[1].size, sampled_node_order_nd_l[2].size,
    #       sampled_node_order_nd.size)
    labels_nd = nd.array(labels, dtype=np.float32, ctx=ctx)

    return sampled_node_order_nd,\
           layer0_features_nd_l, \
           copy_to_ctx(end_points_l, ctx=ctx, dtype=np.int32), \
           copy_to_ctx(indptr_l, ctx=ctx, dtype=np.int32), \
           copy_to_ctx(indices_in_merged_l[1:], ctx=ctx, dtype=np.int32), \
           labels_nd, \
           node_ids_l,\
           rates, \
           node_type_mask, \
           edge_type_mask_l, \
           copy_to_ctx(seg_indices_l, ctx=ctx, dtype=np.int32)



class BiGraphNodeIterator(object):
    """Iterator for the interaction dataset

    """
    def __init__(self, dataset_name=None, version=None, ctx=None,
                 #supervised=True, neg_sample_size=20, normalize_feature=True,
                 batch_node_num=128, batch_sample_method="uniform", hierarchy_sampler_desc=("all", 2),
                 loss_type = "regression"):
        if ctx is None:
            self._ctx = mx.gpu()
        else:
            self._ctx = ctx
        if dataset_name is None:
            dataset_name = cfg.DATA_NAME
        self._dataset_name = dataset_name
        if version is None:
            version = cfg.DATA_VERSION
        self._loss_type = loss_type

        self.G_all, self.feas_l, self.train_preds, self.valid_preds, self.test_preds = \
            cfg_bi_data_loader(dataset_name, version)

        self._num_node_set = self.G_all.num_node_set
        self._num_edge_set = self.G_all.num_edge_set
        self._num_pred_set = self.train_preds.shape[1] - 1

        if self._loss_type == "regression":
            self._num_class = 1
        if version == "ml-1m" or version == "ml-100k":
            self.rates = nd.array([1., 2., 3., 4., 5.], ctx=self._ctx)
        else:
            raise NotImplementedError

        ### processing the graph
        self.G_train = self.G_all.fetch_train()
        self.G_train_valid = self.G_all.fetch_train_valid()
        ### processing the graph sampling method
        self._batch_sample_method = batch_sample_method
        self._sampler = parse_hierarchy_sampler_from_desc(hierarchy_sampler_desc)
        if self._batch_sample_method not in ["uniform"]:
            raise NotImplementedError("Batch Sample Method = %s is not supported!"
                                      % self._batch_sample_method)
        ### processing the features
        self.feas_nd_l = []
        for i in range(len(self.feas_l)):
            self.feas_nd_l.append(nd.array(self.feas_l[i], dtype=np.float32, ctx=self._ctx))
        ### processing the batch mode parameters
        self._batch_node_num = batch_node_num
        self._train_indices = np.random.permutation(self.train_preds.shape[0])
        self._train_curr_pos = 0
        if isinstance(self.valid_preds, np.ndarray) and self.valid_preds.size == 1 and self.valid_preds == None:
            print("valid_preds is None")
            self._f_valid = False
        else:
            self._f_valid = True
            self._valid_indices = np.arange(self.valid_preds.shape[0])
            self._valid_curr_pos = 0
        if isinstance(self.test_preds, np.ndarray) and self.test_preds.size == 1 and self.test_preds == None:
            print("test_preds is None")
            self._f_test = False
        else:
            self._f_test = True
            self._test_indices = np.arange(self.test_preds.shape[0])
            self._test_curr_pos = 0

        self._need_train_begin = True

    @property
    def num_node_set(self):
        return self._num_node_set
    @property
    def num_edge_set(self):
        return self._num_edge_set
    @property
    def num_pred_set(self):
        return self._num_pred_set

    def set_batch_sample_mode(self, mode):
        self._batch_sample_method = mode
        self._need_train_begin = True

    def begin_epoch(self, mode):
        self._mode = mode
        if self._mode == "train":
            if self._batch_sample_method == "uniform":
                self._train_indices = np.random.permutation(self.train_preds.shape[0])
                self._train_curr_pos = 0
            self._need_train_begin = False
        elif self._mode == "valid" and self._f_valid:
            self._valid_curr_pos = 0
        elif self._mode == "test" and self._f_test:
            self._test_curr_pos = 0
        else:
            raise NotImplementedError()

    @property
    def epoch_finished(self):
        if self._mode == "train":
            if self._batch_sample_method == "uniform":
                return self._train_curr_pos >= len(self._train_indices)
        elif self._mode == "valid" and self._f_valid:
            return self._valid_curr_pos >= len(self._valid_indices)
        elif self._mode == "test" and self._f_test:
            return self._test_curr_pos >= len(self._test_indices)
        else:
            raise NotImplementedError

    def sample(self):
        if self.epoch_finished:
            raise RuntimeError("Epoch has finished, you need to call `.begin_epoch()` again")
        if self._mode == "train":
            if self._need_train_begin:
                raise RuntimeError(
                    "No begin_epoch has been called, you need to call `.begin_epoch()` again")
            if self._batch_sample_method == "uniform":
                chosen_train_preds_indexs = self._train_indices[self._train_curr_pos:
                                                                (self._train_curr_pos + self._batch_node_num)]
                chosen_preds = self.train_preds[chosen_train_preds_indexs]
                chosen_node_indices = np.concatenate([self.G_train.reverse_map(chosen_preds[:, i])
                                                        for i in range(self._num_pred_set)])
                labels = chosen_preds[:, -1]
                #print("labels", labels.dtype, labels.shape, labels)
                self._train_curr_pos += chosen_train_preds_indexs.shape[0]
            else:
                raise NotImplementedError
            if self.epoch_finished:
                self._need_train_begin = True
            return _bi_get_hierarchical_sample_feature(features_all_set_nd_l=self.feas_nd_l,
                                                       G_sub=self.G_train,
                                                       node_indices=chosen_node_indices,
                                                       sampler=self._sampler,
                                                       labels=labels,
                                                       ctx=self._ctx,
                                                       rates=self.rates,
                                                       num_node_set=self._num_node_set,
                                                       num_edge_set=self._num_edge_set)
        elif self._mode == "valid" and self._f_valid:
            chosen_valid_preds_indexs = self._valid_indices[self._valid_curr_pos:
                                                          (self._valid_curr_pos + self._batch_node_num)]
            chosen_preds = self.valid_preds[chosen_valid_preds_indexs]
            chosen_node_indices = np.concatenate([self.G_train_valid.reverse_map(chosen_preds[:, i])
                                                  for i in range(self._num_pred_set)])
            labels = chosen_preds[:, -1]
            self._valid_curr_pos += chosen_valid_preds_indexs.shape[0]
            return _bi_get_hierarchical_sample_feature(features_all_set_nd_l=self.feas_nd_l,
                                                       G_sub=self.G_train_valid,
                                                       node_indices=chosen_node_indices,
                                                       sampler=self._sampler,
                                                       labels=labels,
                                                       ctx=self._ctx,
                                                       rates=self.rates,
                                                       num_node_set=self._num_node_set,
                                                       num_edge_set=self._num_edge_set)
        elif self._mode == "test" and self._f_test:
            chosen_test_preds_indexs = self._test_indices[self._test_curr_pos:
                                                          (self._test_curr_pos + self._batch_node_num)]
            chosen_preds = self.test_preds[chosen_test_preds_indexs]
            chosen_node_indices = np.concatenate([self.G_all.reverse_map(chosen_preds[:, i])
                                                  for i in range(self._num_pred_set)])
            labels = chosen_preds[:, -1]
            self._test_curr_pos += chosen_test_preds_indexs.shape[0]
            return _bi_get_hierarchical_sample_feature(features_all_set_nd_l=self.feas_nd_l,
                                                       G_sub=self.G_all,
                                                       node_indices=chosen_node_indices,
                                                       sampler=self._sampler,
                                                       labels=labels,
                                                       ctx=self._ctx,
                                                       rates=self.rates,
                                                       num_node_set=self._num_node_set,
                                                       num_edge_set=self._num_edge_set)

    def summary(self, indent_token = "\t"):
        logging.info("Supervised Training ......")
        logging.info("Graph ---->")
        self.G_all.summary(graph_name=self._dataset_name)
        #logging.info(indent_token + "Batch sampling method:", str(self._batch_sample_method))
        logging.info("Features ---->")
        logging.info(indent_token + "{} node sets".format(self._num_node_set))
        logging.info(indent_token + "{} edge sets".format(self._num_edge_set))
        for i in range(self._num_node_set):
            info_str = indent_token + indent_token + "Set" + str(i) + \
                       ": Shape ({}, {})".format(self.feas_l[i].shape[0], self.feas_l[i].shape[1])
            logging.info(info_str)
        logging.info("# Training preds: {}".format(self.train_preds.shape[0]))
        if self._f_valid:
            logging.info("# Valid preds: {}".format(self.valid_preds.shape[0]))
        if self._f_test:
            logging.info("# Testing preds: {}".format(self.test_preds.shape[0]))


def test_homo_graph():
    import time
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    for dataset_name in ['ppi', 'reddit']:
        print("Loading", dataset_name, "...")

        start = time.time()
        G, features, labels, num_class = cfg_data_loader(dataset_name)
        end = time.time()
        print("Done! Time Spent:", end - start)
        G.summary("train+val+test")

        start = time.time()
        G.fetch_train_valid().summary("train+val")
        print("Time spent for fetching train+val:", time.time() - start)

        start = time.time()
        G.fetch_train().summary("train")
        print("Time spent for fetching train:", time.time() - start)

        print('   features.shape =', features.shape)
        print('   labels.shape =', labels.shape)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats(20)

    print('Test Reddit Iterator Node Uniform:')
    data_iterator = StaticGraphIterator(dataset_name='reddit',
                                        hierarchy_sampler_desc=['fraction', [0.08, 0.04]],
                                        ctx=mx.gpu(),
                                        batch_sample_method='uniform')

    print('Test train set')
    data_iterator.begin_epoch('train')
    pr = cProfile.Profile()
    pr.enable()
    epoch_id = 0
    node_ids = []
    while not data_iterator.epoch_finished:
        layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l, labels_nd, node_ids_l = \
            data_iterator.sample(batch_node_num=500)
        print("Epoch:", epoch_id, "Train Pos:", data_iterator._train_curr_pos,
              ", End Points:", [ele.size for ele in end_points_l],
              ", Node Num:", [ele.size for ele in indices_in_merged_l])
        epoch_id += 1
        node_ids.extend(node_ids_l[-1].tolist())
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats(20)
    assert len(set(node_ids)) == len(node_ids)
    assert set(node_ids) == set(data_iterator.G_all.train_node_ids)
    print('Test valid set')
    node_ids = []
    epoch_id = 0
    data_iterator.begin_epoch('valid')
    while not data_iterator.epoch_finished:
        layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l, labels_nd, node_ids_l = \
            data_iterator.sample(batch_node_num=500)
        print("Epoch:", epoch_id, "Valid Pos:", data_iterator._valid_curr_pos,
              ", End Points:", [ele.size for ele in end_points_l],
              ", Node Num:", [ele.size for ele in indices_in_merged_l])
        epoch_id += 1
        node_ids.extend(node_ids_l[-1].tolist())
    assert len(set(node_ids)) == len(node_ids)
    assert set(node_ids) == set(data_iterator.G_all.valid_node_ids)
    print('Test test set')
    node_ids = []
    epoch_id = 0
    data_iterator.begin_epoch('test')
    while not data_iterator.epoch_finished:
        layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l, labels_nd, node_ids_l = \
            data_iterator.sample(batch_node_num=500)
        print("Epoch:", epoch_id, "Test Pos:", data_iterator._test_curr_pos,
              ", End Points:", [ele.size for ele in end_points_l],
              ", Node Num:", [ele.size for ele in indices_in_merged_l])
        epoch_id += 1
        node_ids.extend(node_ids_l[-1].tolist())
    assert len(set(node_ids)) == len(node_ids)
    assert set(node_ids) == set(data_iterator.G_all.test_node_ids)

    print('Test Reddit Iterator Node Random Walk:')
    data_iterator.set_batch_sample_mode(mode='random_walk')
    data_iterator.summary()
    data_iterator.begin_epoch('train')
    while not data_iterator.epoch_finished:
        layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l, labels_nd, node_ids_l = \
            data_iterator.sample(batch_node_num=500)
        print("Epoch:", epoch_id, "Train Pos:", data_iterator._train_curr_pos,
              ", End Points:", [ele.size for ele in end_points_l],
              ", Node Num:", [ele.size for ele in indices_in_merged_l])
        epoch_id += 1
        node_ids.extend(node_ids_l[-1].tolist())
        nd.waitall()


def test_heter_graph():
    # G, feas_l, train_preds, valid_preds, test_preds, _num_set, _num_class = cfg_heter_data_loader("movielens", "ml-100k")
    # G.summary("train+val+test")
    iter = HeterGraphNodeIterator(dataset_name="movielens", version="ml-100k", ctx=None)
    iter.begin_epoch("test")
    iter.sample()


if __name__ == '__main__':
    test_heter_graph()
