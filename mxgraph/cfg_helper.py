def generate_file_name_static(local_cfg, model=None):
    """

    Parameters
    ----------
    local_cfg : OrderedEdict
    model : str

    Returns
    -------
    file_name
    """
    file_name = str(local_cfg.MX_SEED) + "_" + local_cfg.DATA_NAME
    if local_cfg.DATA_NAME == 'ppi':
        file_name += "_" + str(local_cfg.TRAIN_SPLIT_NUM)
    if model is None:
        if local_cfg.STATIC_GRAPH.MODEL.AGGREGATOR_ARGS_LIST[0][0].lower() == "mugga":
            model = "mugga"
        elif local_cfg.STATIC_GRAPH.MODEL.AGGREGATOR_ARGS_LIST[0][0].lower() == "GraphPoolAggregator".lower():
            model = "pool"
        elif local_cfg.STATIC_GRAPH.MODEL.AGGREGATOR_ARGS_LIST[0][0].lower() == "GraphWeightedSumAggregator".lower():
            model = "weighted"
        elif local_cfg.STATIC_GRAPH.MODEL.AGGREGATOR_ARGS_LIST[0][0].lower() == "GraphMultiWeightedSumAggregator".lower():
            model = "multi_weighted"
        else:
            raise NotImplementedError()
    if local_cfg.STATIC_GRAPH.MODEL.TYP == "supervised":
        file_name += "_sup"
    elif local_cfg.STATIC_GRAPH.MODEL.TYP == "unsupervised":
        file_name += "_unsup"
        file_name += "_neg%d+%d+%d_%g" % (local_cfg.STATIC_GRAPH.MODEL.TRAIN_NEG_SAMPLE_SCALE,
                                          local_cfg.STATIC_GRAPH.MODEL.VALID_NEG_SAMPLE_SCALE,
                                          local_cfg.STATIC_GRAPH.MODEL.TEST_NEG_SAMPLE_SCALE,
                                          local_cfg.STATIC_GRAPH.MODEL.NEG_WEIGHT)
        file_name += "_emb%d" % local_cfg.STATIC_GRAPH.MODEL.EMBED_DIM
    elif local_cfg.STATIC_GRAPH.MODEL.TYP == "transductive":
        file_name += "_trans"
    file_name += "_d" + str(int(local_cfg.STATIC_GRAPH.MODEL.DENSE_CONNECT))
    file_name += "_" + model
    if model == 'mugga':
        # file_name += "_sp" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_SUM_POOL))
        # file_name += "_mp" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_MAX_POOL))
        # file_name += "_ap" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_AVG_POOL))
        file_name += "_g" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_GATE))
        file_name += "_s" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_SHARPNESS))
    elif model == 'pool':
        file_name += "_" + local_cfg.AGGREGATOR.GRAPHPOOL.POOL_TYPE
    elif model == 'weighted':
        file_name += "_div" + str(int(local_cfg.AGGREGATOR.GRAPH_WEIGHTED_SUM.DIVIDE_SIZE))
        file_name += "_" + local_cfg.AGGREGATOR.GRAPH_WEIGHTED_SUM.WEIGHT_ACT
    elif model == 'multi_weighted':
        file_name += "_div" + str(int(local_cfg.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.DIVIDE_SIZE))
        file_name += "_" + local_cfg.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.WEIGHT_ACT
    else:
        raise NotImplementedError
    for layer_info in local_cfg.STATIC_GRAPH.MODEL.AGGREGATOR_ARGS_LIST:
        for units in layer_info[1]:
            file_name += '_' + str(units)
    file_name += '_' + local_cfg.STATIC_GRAPH.MODEL.TRAIN.GRAPH_SAMPLER_ARGS[0]
    if isinstance(local_cfg.STATIC_GRAPH.MODEL.TRAIN.GRAPH_SAMPLER_ARGS[1], int):
        file_name += '_' + str(local_cfg.STATIC_GRAPH.MODEL.TRAIN.GRAPH_SAMPLER_ARGS[1])
    else:
        for ele in local_cfg.STATIC_GRAPH.MODEL.TRAIN.GRAPH_SAMPLER_ARGS[1]:
            file_name += '_' + str(ele)
    file_name += '_d'
    for dropout in local_cfg.STATIC_GRAPH.MODEL.DROPOUT_RATE_LIST:
        file_name += '%g' % (dropout)
    if local_cfg.STATIC_GRAPH.MODEL.L2_NORMALIZATION:
        file_name += '_norm'
    return file_name


def generate_file_name_spatiotemporal(local_cfg):
    """

    Parameters
    ----------
    local_cfg
    model

    Returns
    -------
    file_name
    """
    file_name = str(local_cfg.MX_SEED) + "_" + local_cfg.DATA_NAME
    file_name += "_" + local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.RNN_TYPE
    file_name += "_" + local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.AGGREGATION_TYPE
    first_layer_args = local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.AGGREGATOR_ARGS_LIST[0]
    if first_layer_args[0].lower() == "mugga":
        model = "mugga"
    elif first_layer_args[0].lower()\
            == "GraphPoolAggregator".lower():
        model = "pool"
    elif first_layer_args[0].lower()\
            == "GraphWeightedSumAggregator".lower():
        model = "weighted"
    elif first_layer_args[0].lower()\
            == "GraphMultiWeightedSumAggregator".lower():
        model = "multi_weighted"
    else:
        raise NotImplementedError()
    file_name += "_" + model
    if model == 'mugga':
        # file_name += "_sp" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_SUM_POOL))
        # file_name += "_mp" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_MAX_POOL))
        # file_name += "_ap" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_AVG_POOL))
        file_name += "_g" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_GATE))
        file_name += "_s" + str(int(local_cfg.AGGREGATOR.MUGGA.CONTEXT.USE_SHARPNESS))
    elif model == 'pool':
        file_name += "_" + local_cfg.AGGREGATOR.GRAPHPOOL.POOL_TYPE
    elif model == 'weighted':
        file_name += "_" + local_cfg.AGGREGATOR.GRAPH_WEIGHTED_SUM.WEIGHT_ACT
    elif model == 'multi_weighted':
        file_name += "_" + local_cfg.AGGREGATOR.GRAPH_MULTI_WEIGHTED_SUM.WEIGHT_ACT
    else:
        raise NotImplementedError
    state_dim = first_layer_args[1][0]
    file_name += "_s" + str(state_dim)
    for i, aggregator_args in enumerate(local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.AGGREGATOR_ARGS_LIST):
        file_name += "_l%d" %i
        for ele in aggregator_args[1]:
            file_name += "_" + str(ele)
    file_name += '_tau%d_%d_%d' %(local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.SCHEDULED_SAMPLING.TAU,
                                  local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.INITIAL_EPOCHS,
                                  local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.TRAIN.DECAY_PATIENCE)
    file_name += '_edge%d' % int(local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.USE_EDGE)
    file_name += '_d%g' % local_cfg.SPATIOTEMPORAL_GRAPH.MODEL.DROPOUT_RATE
    return file_name
