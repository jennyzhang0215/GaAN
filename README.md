# Deep Learning for Graph in MXNet

Implements the Multi-head Graph Attention code in MXNet.

We only support python3!

## Installation

Compile the MXNet operators by following the guide in [seg_ops_cuda](seg_ops_cuda).
Install the graph sampler by following the guide in [GraphSampler](GraphSampler).

```bash
python setup.py develop
```

## Download atasets
You can download the datasets via the *download_data.py script. The usage is like
```bash
python download_data.py --dataset all
```
The --dataset hyperparameter can be 'cora', 'ppi', and 'reddit'.
