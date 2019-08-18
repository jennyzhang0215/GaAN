import numpy as np
from mxgraph.iterators import cfg_data_loader
from mxgraph.sampler import FixedNeighborSampler
from mxgraph.graph import set_seed
set_seed(100)

G, features, _, _, _ = cfg_data_loader()
sampler = FixedNeighborSampler(layer_num=2, neighbor_num=[4, 2])
indices_in_merged_l, end_points_l, indptr_l, node_ids_l = sampler.sample_by_indices(G, np.arange(10))
print("node_ids_l", node_ids_l[0].shape, node_ids_l[1].shape, node_ids_l[2].shape)
print(node_ids_l)
print("indptr_l", indptr_l[0].shape, indptr_l[1].shape)
print(indptr_l)
print("end_points_l", end_points_l[0].shape, end_points_l[0].shape)
print(end_points_l)
