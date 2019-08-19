# GaAN

The MXNet implementation of GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs
 in UAI 2018.


We only support python3!

## Installation

Compile the MXNet operators by following the guide in [seg_ops_cuda](seg_ops_cuda).
Install the graph sampler by following the guide in [GraphSampler](GraphSampler).

```bash
python setup.py develop
```

## Download datasets
You can download the datasets via the *download_data.py script. The usage is like
```bash
python download_data.py --dataset ppi
```
The --dataset hyperparameter can be 'cora', 'ppi', and 'reddit'.

## Run experiments
The script is experiments/static_graph/sup_train_sample.py. 

## Citation
```
@inproceedings{zhang18,
  author    = {Jiani Zhang and Xingjian Shi and Junyuan Xie and Hao Ma and Irwin King and Dit{-}Yan Yeung},
  title     = {GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs},
  booktitle = {Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence},
  pages     = {339--349},
  year      = {2018}
}
```
