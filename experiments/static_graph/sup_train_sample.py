import logging
import argparse
import mxnet as mx
from mxgraph.graph import *
from mxgraph.iterators import StaticGraphIterator
from mxgraph.layers import *
from mxgraph.utils import *
from mxgraph.helpers.email_sender import send_msg
from mxgraph.config import cfg, cfg_from_file, save_cfg_dir, ordered_dump
from mxgraph.helpers.metric_logger import MetricLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Run the supervised training experiments with sampling.')
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file', default=None, type=str)
    parser.add_argument('--ctx', dest='ctx', default='gpu',
                        help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`', type=str)
    parser.add_argument('--test', dest='test', help="Whether to run in the test mode", action="store_true")
    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.add_argument('--output_inner_result', dest='output_inner_result', action='store_true')
    parser.add_argument('--save_epoch_interval', dest='save_epoch_interval',
                        help="Epoch interval to output the inner result", default=20, type=int)
    parser.add_argument('--load_dir', dest='load_dir', help="The directory to load the pretrained model",
                        default=None, type=str)
    parser.add_argument('--load_iter', dest='load_iter', help="The iteration to load", default=None, type=int)
    parser.add_argument('--save_dir', help='The saving directory', default=None, type=str)
    parser.add_argument('--emails', dest='emails', type=str, default="", help='Email addresses')
    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file, target=cfg)
    ### configure save_fir to save all the info
    if args.save_dir is None:
        if args.cfg_file is None:
            raise ValueError("Must set --cfg if not set --save_dir")
        args.save_dir = os.path.splitext(args.cfg_file)[0]
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    return args


def eval_classification(net, loss_function, data_iterator, num_class, mode):
    """Evaluate the classification accuracy

    Parameters
    ----------
    net : GraphMultiLayerHierarchicalNodes
    loss_function : mx.gluon.loss.Loss
    data_iterator : StaticGraphIterator
    num_class : int
    mode: str
        "valid" or "test"

    Returns
    -------
    avg_loss : float
    f1 : float
    accuracy : float
    """
    assert mode in ['valid', 'test']

    labels_all = None
    node_ids_all = None
    preds_all = None
    total_loss = 0.0
    instance_num = 0
    data_iterator.begin_epoch(mode=mode)

    while not data_iterator.epoch_finished:
        layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l, labels_nd, node_ids_l = \
            data_iterator.sample()
        if net._output_inner_result:
            logits, gate_l, sharpness_l, attend_weights_wo_gate_l = \
                net(layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l)
            np.save(os.path.join(args.save_dir,
                                 'inner_results%d' % args.save_id,
                                 mode+'_gate%d_1.npy' % iter), gate_l[0].asnumpy())
            # print("gate_1", gate_l[0].asnumpy().shape, gate_l[0].asnumpy())
            np.save(os.path.join(args.save_dir,
                                 'inner_results%d' % args.save_id,
                                 mode+'_gate%d_2.npy' % iter), gate_l[1].asnumpy())
            # print("gate_2", gate_l[1].asnumpy().shape, gate_l[1].asnumpy())
            np.save(os.path.join(args.save_dir,
                                 'inner_results%d' % args.save_id,
                                 mode+'_node_id%d_1.npy' % iter), node_ids_l[1])
            # print("node_id_1", node_ids_l[1].shape, node_ids_l[1])
            np.save(os.path.join(args.save_dir,
                                 'inner_results%d' % args.save_id,
                                 mode+'_node_id%d_2.npy' % iter), node_ids_l[2])

        else:
            logits = net(layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l)

        total_loss += nd.sum(loss_function(logits, labels_nd)).asscalar()
        instance_num += labels_nd.shape[0]
        if cfg.DATA_NAME == 'ppi':
            iter_preds = (logits > 0)
            iter_labels = labels_nd
        else:
            iter_preds = nd.argmax(logits, axis=1)
            iter_labels = labels_nd.reshape((-1,))
        # print(list(zip(iter_preds.tolist(), iter_labels.ravel().tolist())))
        if preds_all is None:
            preds_all = iter_preds
            labels_all = iter_labels
            node_ids_all = node_ids_l[-1]
        else:
            preds_all = nd.concatenate([preds_all, iter_preds], axis=0)
            labels_all = nd.concatenate([labels_all, iter_labels], axis=0)
            ### node_ids is numpy array
            node_ids_all = np.concatenate([node_ids_all, node_ids_l[-1]], axis=0)
    avg_loss = total_loss / instance_num

    if cfg.DATA_NAME == 'ppi':
        num_class = 2
    f1 = nd_f1(pred=preds_all, label=labels_all, num_class=num_class, average="micro")
    acc = nd_acc(pred=preds_all, label=labels_all)

    return avg_loss, f1, acc


def build(args):
    ctx = args.ctx[0]
    local_cfg = cfg.STATIC_GRAPH
    ### initialize data_iterator
    data_iterator = StaticGraphIterator(hierarchy_sampler_desc=local_cfg.MODEL.TRAIN.GRAPH_SAMPLER_ARGS,
                                        ctx=ctx,
                                        supervised=True,
                                        batch_node_num=local_cfg.MODEL.TRAIN.BATCH_SIZE,
                                        normalize_feature=local_cfg.MODEL.FEATURE_NORMALIZE,
                                        batch_sample_method="uniform")
    data_iterator.summary()

    ### build net
    net = GraphMultiLayerHierarchicalNodes(out_units=data_iterator.num_class,
                                           aggregator_args_list=local_cfg.MODEL.AGGREGATOR_ARGS_LIST,
                                           dropout_rate_list=local_cfg.MODEL.DROPOUT_RATE_LIST,
                                           dense_connect=local_cfg.MODEL.DENSE_CONNECT,
                                           l2_normalization=local_cfg.MODEL.L2_NORMALIZATION,
                                           first_embed_units=local_cfg.MODEL.FIRST_EMBED_UNITS,
                                           output_inner_result=args.output_inner_result,
                                           prefix='net_')
    net.hybridize()

    ### define loss_function
    if cfg.DATA_NAME == 'ppi':
        loss_function = gluon.loss.LogisticLoss(label_format='binary')
    else:
        loss_function = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
    loss_function.hybridize()

    return net, loss_function, data_iterator


def train(args, net, loss_function, data_iterator):
    """Train the model
    """
    ctx = args.ctx[0]
    local_cfg = cfg.STATIC_GRAPH

    net.initialize(init=mx.init.Xavier(magnitude=3), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(),
                            local_cfg.MODEL.TRAIN.OPTIMIZER,
                            {'learning_rate': local_cfg.MODEL.TRAIN.LR,
                             'wd': local_cfg.MODEL.TRAIN.WD})

    train_loss_logger = MetricLogger(["iter", "loss"], ["%d", "%.4f"],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(["iter", "loss", "f1", "acc", "is_best"],
                                     ["%d", "%.4f", "%.4f", "%.4f", "%d"],
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(["iter", "loss", "f1", "acc"], ["%d", "%.4f", "%.4f", "%.4f"],
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))

    best_valid_f1 = 0
    best_valid_iter_info = []
    best_test_iter_info = []
    no_better_valid = 0
    iter_id = 1
    epoch = 1
    train_moving_avg_loss = 0.0
    data_iterator.begin_epoch('train')
    for iter in range(1, local_cfg.MODEL.TRAIN.MAX_ITER):
        if data_iterator.epoch_finished:
            print("Epoch %d finished! It has %d iterations." % (epoch, iter_id))
            data_iterator.begin_epoch('train')
            iter_id = 1
            epoch += 1
        else:
            iter_id += 1

        layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l, labels_nd, node_ids_l =\
            data_iterator.sample()
        # print("layer0_features_nd", layer0_features_nd.shape, "\n", layer0_features_nd)
        # print("end_points_l", len(end_points_l), end_points_l)
        # print("indptr_l", len(indptr_l), indptr_l)
        # print("indices_in_merged_l", len(indices_in_merged_l), indices_in_merged_l)
        # print("labels_nd", labels_nd.shape, "\n", labels_nd)
        #print("node_id", node_ids_l[0].shape, node_ids_l[1].shape,node_ids_l[2].shape)
        with mx.autograd.record():
            if net._output_inner_result:
                logits, gate_l, sharpness_l, attend_weights_wo_gate_l =\
                    net(layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l)
                # print("gate", len(gate_l), gate_l[0].shape, gate_l[1].shape)
                # print(gate_l[0])
                # print(gate_l[1])
                # if epoch % args.save_epoch_interval == 1 or epoch < args.save_epoch_interval + 1:
                #     #temp_dict = dict([('gate%d' % i, gate.asnumpy())
                #     #                  for i, gate in enumerate(gate_l)] )
                #     np.save(os.path.join(args.save_dir,
                #                           'inner_results%d' % args.save_id,
                #                           'train_gate%d_1.npy' % epoch), gate_l[0].asnumpy())
                #     #print("gate_1", gate_l[0].asnumpy().shape, gate_l[0].asnumpy())
                #     np.save(os.path.join(args.save_dir,
                #                             'inner_results%d' % args.save_id,
                #                             'train_gate%d_2.npy' % epoch), gate_l[1].asnumpy())
                #     #print("gate_2", gate_l[1].asnumpy().shape, gate_l[1].asnumpy())
                #     np.save(os.path.join(args.save_dir,
                #                          'inner_results%d' % args.save_id,
                #                          'train_node_id%d_1.npy' % epoch), node_ids_l[1])
                #     #print("node_id_1", node_ids_l[1].shape, node_ids_l[1])
                #     np.save(os.path.join(args.save_dir,
                #                          'inner_results%d' % args.save_id,
                #                          'train_node_id%d_2.npy' % epoch), node_ids_l[2])
                #print("node_id_2", node_ids_l[2].shape, node_ids_l[2])
                # temp_dict = dict([('gate%d' % i, gate.asnumpy())
                #                   for i, gate in enumerate(gate_l)] +
                #                  [('attend_weights_wo_gate%d' % i, ele)
                #                   for i, ele in enumerate(attend_weights_wo_gate_l)])
                # np.savez(os.path.join(args.save_dir,
                #                       'inner_results%d' % args.save_id,
                #                       'gate_attweight%d.npz' % epoch), **temp_dict)
            else:
                logits = net(layer0_features_nd, end_points_l, indptr_l, indices_in_merged_l)
            loss = loss_function(logits, labels_nd)
            loss = nd.mean(loss)
            loss.backward()
        if iter == 1:
            logging.info("Total Param Number: %d" % gluon_total_param_num(net))
            gluon_log_net_info(net, save_path=os.path.join(args.save_dir, 'net_info%d.txt' % args.save_id))
        ### norm clipping
        if local_cfg.MODEL.TRAIN.GRAD_CLIP <= 0:
            gnorm = get_global_norm([v.grad() for v in net.collect_params().values()])
        else:
            gnorm = gluon.utils.clip_global_norm([v.grad() for v in net.collect_params().values()],
                                                 max_norm=local_cfg.MODEL.TRAIN.GRAD_CLIP)
        trainer.step(batch_size=1)
        iter_train_loss = loss.asscalar()
        train_moving_avg_loss += iter_train_loss
        logging.info('[iter=%d]: loss=%.4f, gnorm=%g' % (iter, iter_train_loss, gnorm))
        train_loss_logger.log(iter=iter, loss=iter_train_loss)

        if iter % local_cfg.MODEL.TRAIN.VALID_ITER == 0:
            valid_loss, valid_f1, valid_accuracy = \
                eval_classification(net=net,
                                    loss_function=loss_function,
                                    data_iterator=data_iterator,
                                    num_class=data_iterator.num_class,
                                    mode="valid")
            logging.info("Iter %d, Epoch %d,: train_moving_loss=%.4f, valid loss=%.4f, f1=%.4f, accuracy=%.4f"
                         % (iter, epoch, train_moving_avg_loss / local_cfg.MODEL.TRAIN.VALID_ITER,
                            valid_loss, valid_f1, valid_accuracy))

            train_moving_avg_loss = 0.0
            if valid_f1 > best_valid_f1:
                logging.info("======================> Best Iter")
                is_best = True
                best_valid_f1 = valid_f1
                best_iter = iter
                best_valid_iter_info = [best_iter, valid_loss, valid_f1, valid_accuracy]
                no_better_valid = 0
                net.save_params(
                    filename=os.path.join(args.save_dir, 'best_valid%d.params' % args.save_id))
                # Calculate the test loss
                test_loss, test_f1, test_accuracy = \
                    eval_classification(net=net,
                                        loss_function=loss_function,
                                        data_iterator=data_iterator,
                                        num_class=data_iterator.num_class,
                                        mode="test")
                test_loss_logger.log(iter=iter, loss=test_loss, f1=test_f1, acc=test_accuracy)
                best_test_iter_info = [best_iter, test_loss, test_f1, test_accuracy]
                logging.info("Iter %d, Epoch %d: test loss=%.4f, f1=%.4f, accuracy=%.4f" %
                             (iter, epoch, test_loss, test_f1, test_accuracy))
            else:
                is_best = False
                no_better_valid += 1
                if no_better_valid > local_cfg.MODEL.TRAIN.EARLY_STOPPING_PATIENCE:
                    # Finish training
                    logging.info("Early stopping threshold reached. Stop training.")
                    valid_loss_logger.log(iter=iter, loss=valid_loss, f1=valid_f1,
                                          acc=valid_accuracy, is_best=is_best)
                    break
                ### add learning rate decay
                elif no_better_valid > local_cfg.MODEL.TRAIN.DECAY_PATIENCE:
                    new_lr = max(trainer.learning_rate * local_cfg.MODEL.TRAIN.LR_DECAY_FACTOR,
                                 local_cfg.MODEL.TRAIN.MIN_LR)
                    if new_lr < trainer.learning_rate:
                        logging.info("Change the LR to %g" % new_lr)
                        trainer.set_learning_rate(new_lr)
                        no_better_valid = 0
            valid_loss_logger.log(iter=iter, loss=valid_loss, f1=valid_f1, acc=valid_accuracy, is_best=is_best)
    ### save best iter info
    logging.info("Best Valid: [Iter, Loss, F1, ACC] = %s" % str(best_valid_iter_info))
    logging.info("Best Test : [Iter, Loss, F1, ACC] = %s" % str(best_test_iter_info))
    valid_loss_logger.log(iter=best_valid_iter_info[0],
                          loss=best_valid_iter_info[1],
                          f1=best_valid_iter_info[2],
                          acc=best_valid_iter_info[3],
                          is_best=True)
    test_loss_logger.log(iter=best_test_iter_info[0],
                         loss=best_test_iter_info[1],
                         f1=best_test_iter_info[2],
                         acc=best_test_iter_info[3])
    if args.emails is not None and len(args.emails) > 0:
        for email_address in args.emails.split(','):
            send_msg(title=os.path.basename(args.save_dir),
                     text="Test: [Iter, Loss, F1, ACC] = %s\n" % str(best_test_iter_info)
                          + "Valid: [Iter, Loss, F1, ACC] = %s\n" % str(best_valid_iter_info)
                          + 'Save Dir: %s\n' % args.save_dir
                          + '\nConfig:\n' + ordered_dump(),
                     dst_address=email_address)
    return

def test(args, net, loss_function, data_iterator, save_id):

    net.load_params(os.path.join(args.save_dir, 'best_valid%d.params' % save_id), ctx=args.ctx[0])
    test_loss, test_f1, test_accuracy = \
        eval_classification(net=net,
                            loss_function=loss_function,
                            data_iterator=data_iterator,
                            num_class=data_iterator.num_class,
                            mode="test")
    logging.info("Test loss=%.4f, f1=%.4f, accuracy=%.4f" % (test_loss, test_f1, test_accuracy))


if __name__ == "__main__":
    args = parse_args()

    local_cfg = cfg.copy()
    del local_cfg.SPATIOTEMPORAL_GRAPH
    del local_cfg['SPATIOTEMPORAL_GRAPH']

    args.save_id = save_cfg_dir(args.save_dir, source=local_cfg)
    if args.output_inner_result:
        if not os.path.isabs(os.path.join(args.save_dir, "inner_results%d" % args.save_id)):
            os.makedirs(os.path.join(args.save_dir, "inner_results%d" % args.save_id))

    logging_config(folder=args.save_dir, name='sup_train_sample%d' % args.save_id, no_console=args.silent)
    logging.info(args)

    np.random.seed(cfg.NPY_SEED)
    mx.random.seed(cfg.MX_SEED)
    from mxgraph.graph import set_seed
    set_seed(cfg.MX_SEED)

    net, loss_function, data_iterator = build(args)

    if args.test:
        test(args, net, loss_function, data_iterator, 0)
    else:
        train(args, net, loss_function, data_iterator)