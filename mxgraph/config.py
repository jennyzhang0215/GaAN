import numpy as np
import os
import yaml
import logging
from collections import OrderedDict, namedtuple
from mxgraph.helpers.ordered_easydict import OrderedEasyDict as edict

__C = edict()
cfg = __C  # type: edict()

# Random seed
__C.MX_SEED = 12345
__C.NPY_SEED = 12345

# Project directory, since config.py is supposed to be in $ROOT_DIR/mxgraph
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

__C.DATASET_PATH = os.path.join(__C.ROOT_DIR, 'datasets')

# DATA NAME
# Used by symbols factories who need to adjust for different
# inputs based on dataset used. Should be set by the script.
__C.DATA_NAME = 'movielens'  # choice: ppi; reddit; movielens
__C.DATA_VERSION = "ml-100k"
__C.SPLIT_TRAINING = False
__C.TRAIN_SPLIT_NUM = 20
__C.LOAD_WALKS = False

__C.AGGREGATOR = edict()
__C.AGGREGATOR.ACTIVATION = 'relu'  #'leaky'

__C.AGGREGATOR.GRAPHPOOL = edict()
__C.AGGREGATOR.GRAPHPOOL.ARGS = ["out_units", "mid_units", "mid_layer_num"]
__C.AGGREGATOR.GRAPHPOOL.POOL_TYPE = "avg"  # Can be "max", "avg", "sum" or "mixed" (sum and mixed is deprecated!)

__C.AGGREGATOR.HETERGRAPHPOOL = edict()
__C.AGGREGATOR.HETERGRAPHPOOL.ARGS = ["out_units", "mid_units", "mid_layer_num", "num_set"]
__C.AGGREGATOR.HETERGRAPHPOOL.POOL_TYPE = "avg"  # Can be "max", "avg", "sum" or "mixed" (sum and mixed is deprecated!)

__C.AGGREGATOR.GRAPH_WEIGHTED_SUM = edict()
__C.AGGREGATOR.GRAPH_WEIGHTED_SUM.ARGS = ["out_units", "mid_units", "attend_units"]
__C.AGGREGATOR.GRAPH_WEIGHTED_SUM.ATTEND_W_DROPOUT = 0.0
__C.AGGREGATOR.GRAPH_WEIGHTED_SUM.DIVIDE_SIZE = True
__C.AGGREGATOR.GRAPH_WEIGHTED_SUM.WEIGHT_ACT = 'sigmoid'

__C.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM = edict()
__C.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.ARGS = ["out_units", "mid_units", "attend_units", "K"]
__C.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.ATTEND_W_DROPOUT = 0.0
__C.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.DIVIDE_SIZE = True
__C.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.WEIGHT_ACT = 'sigmoid'

__C.AGGREGATOR.MUGGA = edict()
__C.AGGREGATOR.MUGGA.ARGS = ["out_units", "attend_units", "value_units", "K", "context_units", "context_layer_num"]
__C.AGGREGATOR.MUGGA.USE_EDGE = False
__C.AGGREGATOR.MUGGA.RESCALE_INNERPRODUCT = True
__C.AGGREGATOR.MUGGA.ATTEND_W_DROPOUT = 0.5
__C.AGGREGATOR.MUGGA.CONTEXT = edict()  # TODO(sxjscience) Add flag to enable/disable dense connection in context network
__C.AGGREGATOR.MUGGA.CONTEXT.USE_SUM_POOL = False ## set to be false and do not change it
__C.AGGREGATOR.MUGGA.CONTEXT.USE_MAX_POOL = True
__C.AGGREGATOR.MUGGA.CONTEXT.USE_AVG_POOL = True
__C.AGGREGATOR.MUGGA.CONTEXT.USE_GATE = False
__C.AGGREGATOR.MUGGA.CONTEXT.USE_SHARPNESS = False

__C.AGGREGATOR.BIGRAPHPOOL = edict()
__C.AGGREGATOR.BIGRAPHPOOL.ARGS = ["out_units", "mid_units", "num_node_set", "num_edge_set"]
__C.AGGREGATOR.BIGRAPHPOOL.POOL_TYPE = "avg"  # Can be "max" and "avg"
__C.AGGREGATOR.BIGRAPHPOOL.ACCUM_TYPE = "sum"  # Can be "stack" and "sum"


__C.BI_GRAPH=edict()
__C.BI_GRAPH.MODEL = edict()
#__C.BI_GRAPH.MODEL.LOSS_TYPE = "regression"
__C.BI_GRAPH.MODEL.FEA_EMBED_UNITS = 500
#__C.BI_GRAPH.MODEL.FIRST_EMBED_UNITS = 256
__C.BI_GRAPH.MODEL.AGGREGATOR_ARGS_LIST = [["BiGraphPoolAggregator", [None, 500, None, 5]]]
__C.BI_GRAPH.MODEL.OUT_NODE_EMBED = 75
__C.BI_GRAPH.MODEL.DROPOUT_RATE_LIST = [0.7]
__C.BI_GRAPH.MODEL.DENSE_CONNECT = False
__C.BI_GRAPH.MODEL.L2_NORMALIZATION = False
__C.BI_GRAPH.MODEL.EVERY_LAYER_L2_NORMALIZATION = False
#__C.HETER_GRAPH.MODEL.PRED_HIDDEN_DIM = 64

__C.BI_GRAPH.MODEL.TRAIN = edict()
__C.BI_GRAPH.MODEL.TRAIN.BATCH_SIZE = 128
__C.BI_GRAPH.MODEL.TRAIN.GRAPH_SAMPLER_ARGS = ["all", 1] # ["fixed", [50, 20]]  # Can be all, fraction, fixed, ...
__C.BI_GRAPH.MODEL.TRAIN.VALID_ITER = 500
__C.BI_GRAPH.MODEL.TRAIN.TEST_ITER = 1
__C.BI_GRAPH.MODEL.TRAIN.MAX_ITER = 100000
__C.BI_GRAPH.MODEL.TRAIN.OPTIMIZER = "adam"
__C.BI_GRAPH.MODEL.TRAIN.LR = 1E-2      # initial learning rate
__C.BI_GRAPH.MODEL.TRAIN.MIN_LR = 1E-5  # Minimum learning rate
__C.BI_GRAPH.MODEL.TRAIN.DECAY_PATIENCE = 5  # Patience of the lr decay. If no better train loss occurs for DECAY_PATIENCE epochs, the lr will be multplied by lr_decay
__C.BI_GRAPH.MODEL.TRAIN.EARLY_STOPPING_PATIENCE = 10  # Patience of early stopping
__C.BI_GRAPH.MODEL.TRAIN.LR_DECAY_FACTOR = 0.5
__C.BI_GRAPH.MODEL.TRAIN.GRAD_CLIP = 5.0
__C.BI_GRAPH.MODEL.TRAIN.WD = 0.0


__C.HETER_GRAPH=edict()
__C.HETER_GRAPH.MODEL = edict()
__C.HETER_GRAPH.MODEL.LOSS_TYPE = "regression"
__C.HETER_GRAPH.MODEL.FEA_EMBED_UNITS = 256
#__C.HETER_GRAPH.MODEL.FIRST_EMBED_UNITS = 256
__C.HETER_GRAPH.MODEL.AGGREGATOR_ARGS_LIST = [["HeterGraphPoolAggregator", [128, 512, 1, 3]],
                                              ["HeterGraphPoolAggregator", [128, 512, 1, 3]]]
__C.HETER_GRAPH.MODEL.OUT_NODE_EMBED = 128
__C.HETER_GRAPH.MODEL.DROPOUT_RATE_LIST = [0.5, 0.5]
__C.HETER_GRAPH.MODEL.DENSE_CONNECT = False
__C.HETER_GRAPH.MODEL.L2_NORMALIZATION = True
__C.HETER_GRAPH.MODEL.EVERY_LAYER_L2_NORMALIZATION = True
#__C.HETER_GRAPH.MODEL.PRED_HIDDEN_DIM = 64

__C.HETER_GRAPH.MODEL.TRAIN = edict()
__C.HETER_GRAPH.MODEL.TRAIN.BATCH_SIZE = 128
__C.HETER_GRAPH.MODEL.TRAIN.GRAPH_SAMPLER_ARGS = ["all", 2] # ["fixed", [50, 20]]  # Can be all, fraction, fixed, ...
__C.HETER_GRAPH.MODEL.TRAIN.VALID_ITER = 625
__C.HETER_GRAPH.MODEL.TRAIN.TEST_ITER = 1
__C.HETER_GRAPH.MODEL.TRAIN.MAX_ITER = 100000
__C.HETER_GRAPH.MODEL.TRAIN.OPTIMIZER = "adam"
__C.HETER_GRAPH.MODEL.TRAIN.LR = 1E-3      # initial learning rate
__C.HETER_GRAPH.MODEL.TRAIN.MIN_LR = 1E-5  # Minimum learning rate
__C.HETER_GRAPH.MODEL.TRAIN.DECAY_PATIENCE = 5  # Patience of the lr decay. If no better train loss occurs for DECAY_PATIENCE epochs, the lr will be multplied by lr_decay
__C.HETER_GRAPH.MODEL.TRAIN.EARLY_STOPPING_PATIENCE = 10  # Patience of early stopping
__C.HETER_GRAPH.MODEL.TRAIN.LR_DECAY_FACTOR = 0.5
__C.HETER_GRAPH.MODEL.TRAIN.GRAD_CLIP = 5.0
__C.HETER_GRAPH.MODEL.TRAIN.WD = 0.0


__C.STATIC_GRAPH = edict()
__C.STATIC_GRAPH.MODEL = edict()
__C.STATIC_GRAPH.MODEL.TYP = 'supervised' ## This hyperparameter does not have any meaning but for logging
__C.STATIC_GRAPH.MODEL.FEATURE_NORMALIZE = False
if __C.DATA_NAME == 'ppi':
    __C.STATIC_GRAPH.MODEL.FIRST_EMBED_UNITS = 64
elif __C.DATA_NAME == 'reddit':
    __C.STATIC_GRAPH.MODEL.FIRST_EMBED_UNITS = 256
__C.STATIC_GRAPH.MODEL.AGGREGATOR_ARGS_LIST = [["MuGGA", [128, 16, 16, 8, 16, 3]],
                                               ["MuGGA", [128, 16, 16, 8, 16, 3]]]
__C.STATIC_GRAPH.MODEL.DROPOUT_RATE_LIST = [0.5, 0.5]  # dropout rate (1 - keep probability)'
__C.STATIC_GRAPH.MODEL.DENSE_CONNECT = False
__C.STATIC_GRAPH.MODEL.L2_NORMALIZATION = False
__C.STATIC_GRAPH.MODEL.EVERY_LAYER_L2_NORMALIZATION = False

# The following elements are generally used in unsupervised learning
__C.STATIC_GRAPH.MODEL.EMBED_DIM = 128
__C.STATIC_GRAPH.MODEL.NEG_WEIGHT = 1.0
__C.STATIC_GRAPH.MODEL.TRAIN_NEG_SAMPLE_SCALE = 20
__C.STATIC_GRAPH.MODEL.TRAIN_NEG_SAMPLE_REPLACE = False
__C.STATIC_GRAPH.MODEL.VALID_NEG_SAMPLE_SCALE = 50
__C.STATIC_GRAPH.MODEL.TEST_NEG_SAMPLE_SCALE = 50

__C.STATIC_GRAPH.MODEL.TRAIN = edict()
__C.STATIC_GRAPH.MODEL.TRAIN.BATCH_SIZE = 512
__C.STATIC_GRAPH.MODEL.TRAIN.GRAPH_SAMPLER_ARGS = ["fixed", [25, 10]]  # Can be all, fraction, fixed, ...
__C.STATIC_GRAPH.MODEL.TRAIN.VALID_ITER = 1
__C.STATIC_GRAPH.MODEL.TRAIN.TEST_ITER = 1
__C.STATIC_GRAPH.MODEL.TRAIN.MAX_ITER = 100000
__C.STATIC_GRAPH.MODEL.TRAIN.OPTIMIZER = "adam"
__C.STATIC_GRAPH.MODEL.TRAIN.LR = 1E-3      # initial learning rate
__C.STATIC_GRAPH.MODEL.TRAIN.MIN_LR = 1E-3  # Minimum learning rate
__C.STATIC_GRAPH.MODEL.TRAIN.DECAY_PATIENCE = 15  # Patience of the lr decay. If no better train loss occurs for DECAY_PATIENCE epochs, the lr will be multplied by lr_decay
__C.STATIC_GRAPH.MODEL.TRAIN.EARLY_STOPPING_PATIENCE = 30  # Patience of early stopping
__C.STATIC_GRAPH.MODEL.TRAIN.LR_DECAY_FACTOR = 0.5
__C.STATIC_GRAPH.MODEL.TRAIN.GRAD_CLIP = 1.0
__C.STATIC_GRAPH.MODEL.TRAIN.WD = 0.0

__C.STATIC_GRAPH.MODEL.TEST = edict()
__C.STATIC_GRAPH.MODEL.TEST.BATCH_SIZE = 512
__C.STATIC_GRAPH.MODEL.TEST.SAMPLE_NUM = 5

__C.SPATIOTEMPORAL_GRAPH = edict()
__C.SPATIOTEMPORAL_GRAPH.IN_LENGTH = 12
__C.SPATIOTEMPORAL_GRAPH.OUT_LENGTH = 12
__C.SPATIOTEMPORAL_GRAPH.USE_COORDINATES = True
__C.SPATIOTEMPORAL_GRAPH.MODEL = edict()
__C.SPATIOTEMPORAL_GRAPH.MODEL.RNN_TYPE = "RNN"
__C.SPATIOTEMPORAL_GRAPH.MODEL.AGGREGATOR_ARGS_LIST = [["MuGGA", [64, 8, 16, 4, 32, 1]],
                                                       ["MuGGA", [64, 8, 16, 4, 32, 1]]]
__C.SPATIOTEMPORAL_GRAPH.MODEL.AGGREGATION_TYPE = "all"
__C.SPATIOTEMPORAL_GRAPH.MODEL.ADJ_PREPROCESS = 'undirected'
__C.SPATIOTEMPORAL_GRAPH.MODEL.DROPOUT_RATE = 0.0
__C.SPATIOTEMPORAL_GRAPH.MODEL.DIFFUSSION_STEP = 1
__C.SPATIOTEMPORAL_GRAPH.MODEL.SHARPNESS_LAMBDA = 0.0
__C.SPATIOTEMPORAL_GRAPH.MODEL.DIVERSITY_LAMBDA = 0.0
__C.SPATIOTEMPORAL_GRAPH.MODEL.USE_EDGE = False

__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN = edict()
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.BATCH_SIZE = 64
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.MAX_EPOCH = 100
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.OPTIMIZER = "adam"
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.LR = 1E-3      # initial learning rate
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.MIN_LR = 1E-5  # Minimum learning rate
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.SCHEDULED_SAMPLING = edict()
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.SCHEDULED_SAMPLING.TAU = 3000  # tau / (tau + exp(iter / tau))
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.INITIAL_EPOCHS = 20
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.DECAY_PATIENCE = 10  # Patience of the lr decay. Decay the learning rate every DECAY_PATIENCE epochs
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.EARLY_STOPPING_PATIENCE = 5  # Patience of early stopping
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.LR_DECAY_FACTOR = 0.1
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.GRAD_CLIP = 5.0
__C.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.WD = 0.0



def _merge_two_config(user_cfg, default_cfg):
    """ Merge user's config into default config dictionary, clobbering the
        options in b whenever they are also specified in a.
        Need to ensure the type of two val under same key are the same
        Do recursive merge when encounter hierarchical dictionary
    """
    if type(user_cfg) is not edict:
        return
    for key, val in user_cfg.items():
        # Since user_cfg is a sub-file of default_cfg
        if key not in default_cfg:
            raise KeyError('{} is not a valid config key'.format(key))

        if (type(default_cfg[key]) is not type(val) and
                default_cfg[key] is not None):
            if isinstance(default_cfg[key], np.ndarray):
                val = np.array(val, dtype=default_cfg[key].dtype)
            elif isinstance(default_cfg[key], (int, float)) and isinstance(val, (int, float)):
                pass
            else:
                raise ValueError(
                     'Type mismatch ({} vs. {}) '
                     'for config key: {}'.format(type(default_cfg[key]),
                                                 type(val), key))
        # Recursive merge config
        if type(val) is edict:
            try:
                _merge_two_config(user_cfg[key], default_cfg[key])
            except:
                print('Error under config key: {}'.format(key))
                raise
        else:
            default_cfg[key] = val

def cfg_from_file(file_name, target=__C):
    """ Load a config file and merge it into the default options.
    """
    import yaml
    with open(file_name, 'r') as f:
        print('Loading YAML config file from %s' %f)
        yaml_cfg = edict(yaml.load(f))

    _merge_two_config(yaml_cfg, target)


def ordered_dump(data=__C, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items(), flow_style=False)

    def _ndarray_representer(dumper, data):
        return dumper.represent_list(data.tolist())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    OrderedDumper.add_representer(edict, _dict_representer)
    OrderedDumper.add_representer(np.ndarray, _ndarray_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def save_cfg_file(file_path, source=__C):
    source = source.copy()
    masked_keys = ['DATASET_PATH', 'ROOT_DIR']
    for key in masked_keys:
        if key in source:
            del source[key]
            delattr(source, key)
    with open(file_path, 'w') as f:
        logging.info("Save YAML config file to %s" %file_path)
        ordered_dump(source, f, yaml.SafeDumper, default_flow_style=None)


def save_cfg_dir(dir_path, source=__C):
    cfg_count = 0
    file_path = os.path.join(dir_path, 'cfg%d.yml' %cfg_count)
    while os.path.exists(file_path):
        cfg_count += 1
        file_path = os.path.join(dir_path, 'cfg%d.yml' % cfg_count)
    save_cfg_file(file_path, source)
    return cfg_count

def load_latest_cfg(dir_path, target=__C):
    import re
    cfg_count = None
    source_cfg_path = None
    for fname in os.listdir(dir_path):
        ret = re.search(r'cfg(\d+)\.yml', fname)
        if ret != None:
            if cfg_count is None or (int(re.group(1)) > cfg_count):
                cfg_count = int(re.group(1))
                source_cfg_path = os.path.join(dir_path, ret.group(0))
    cfg_from_file(file_name=source_cfg_path, target=target)


# save_f_name = os.path.join("..", "experiments", "heterogeneous_graph", "baselines", "cfg_template","ml_100k.yml")
# save_cfg_file(save_f_name)
