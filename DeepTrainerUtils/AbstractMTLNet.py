author = 'Simon Kohl, Paul Jaeger, June 2017'

from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
import lasagne
import cPickle
from lasagne.layers import get_output, get_all_params, get_all_layers
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adam
import theano
import theano.tensor as T
import numpy as np
import pickle
from DeepTrainerUtils.utils import binary_dice_per_instance_and_class, get_one_hot_prediction, get_weights
from DeepTrainerUtils.utils import get_one_hot_class_target


class MTLNet():
    def __init__(self, cf, logger):
        self.cf = cf
        self.net = None
        self.loss = None
        self.predict = {}
        self.predict_smax = {}
        self.class_predict_smax = {}
        self.attention_predict = None
        self.predict_one_hot = {}
        self.loss = {}
        self.class_loss = {}
        self.logger = logger
        self.epoch = 0
        self.metrics = {}
        self.metrics['train'] = {'loss': [0.], 'dices': np.zeros(shape=(1, self.cf.num_classes)), 'class_acc': [0.], 'class_loss': [0.]}
        self.metrics['val'] = {'loss': [0.], 'dices': np.zeros(shape=(1, self.cf.num_classes)), 'class_acc': [0.], 'class_loss': [0.]}

    @abstractmethod
    def initialize_net(self):
        pass

    def compile_theano_functions(self, data_type='2D', loss='cross_entropy'):
        assert self.net != None

        ### symbolic theano input
        theano_args = OrderedDict()
        dim = len(self.cf.dim)

        if data_type == '2D':
            assert dim == 2
            theano_args['X'] = T.tensor4()
            theano_args['y'] = T.tensor4()
            theano_args['c'] = T.ivector()
            self.logger.info('Net: Working with 2D data.')
            val_args = deepcopy(theano_args)
            train_args = deepcopy(theano_args)
            train_args['lr'] = T.scalar(name='lr')
            train_args['lw'] = T.scalar(name='lw')


            ### class prediction functions
            class_layer = self.net[self.cf.class_layer]
            class_train_prediction = get_output(class_layer, train_args['X'], deterministic=False)
            class_val_prediction = get_output(class_layer, val_args['X'], deterministic=True)
            attention_val_prediction = get_output(self.net[self.cf.attention_layer], val_args['X'], deterministic=True)

            self.class_predict_smax['train'] = theano.function([train_args['X']], class_train_prediction)
            self.class_predict_smax['val'] = theano.function([val_args['X']], class_val_prediction)
            self.attention_predict = theano.function([val_args['X']], attention_val_prediction)

            # get flattened softmax prediction of shape (pixels, classes), where pixels = b*0*1
            prediction_train_smax_flat = get_output(self.net[self.cf.seg_out_layer_flat], train_args['X'], deterministic=False)
            prediction_val_smax_flat = get_output(self.net[self.cf.seg_out_layer_flat], val_args['X'], deterministic=True)

            # reshape softmax prediction: shapes (pixels,c) -> (b,c,0,1)
            prediction_train_smax = prediction_train_smax_flat.reshape(
                (train_args['X'].shape[0], self.cf.dim[0], self.cf.dim[1], self.cf.num_classes)).transpose((0, 3, 1, 2))
            prediction_val_smax = prediction_val_smax_flat.reshape(
                (val_args['X'].shape[0], self.cf.dim[0], self.cf.dim[1], self.cf.num_classes)).transpose((0, 3, 1, 2))
            self.predict_smax['train'] = theano.function([train_args['X']], prediction_train_smax)
            self.predict_smax['val'] = theano.function([val_args['X']], prediction_val_smax)

            # reshape target vector: shapes (b,c,0,1) -> (b*0*1,c)
            flat_target_train = train_args['y'].transpose((0, 2, 3, 1)).reshape((-1, self.cf.num_classes))
            flat_target_val = val_args['y'].transpose((0, 2, 3, 1)).reshape((-1, self.cf.num_classes))

        elif data_type == '3D':
            assert dim == 3
            theano_args['X'] = T.tensor5()
            theano_args['y'] = T.tensor5()
            theano_args['c'] = T.ivector()
            self.logger.info('Net: Working with 3D data.')
            val_args = deepcopy(theano_args)
            train_args = deepcopy(theano_args)
            train_args['lr'] = T.scalar(name='lr')

            ### prediction functions

            # get flattened softmax prediction of shape (pixels, classes), where pixels = b*0*1*2
            prediction_train_smax_flat = get_output(self.net[self.cf.seg_out_layer_flat], train_args['X'],
                                                    deterministic=False)
            prediction_val_smax_flat = get_output(self.net[self.cf.seg_out_layer_flat], val_args['X'], deterministic=True)

            # reshape softmax prediction: shapes (pixels,c) -> (b,c,0,1,2)
            prediction_train_smax = prediction_train_smax_flat.reshape((train_args['X'].shape[0], self.cf.dim[0],
                                                                        self.cf.dim[1], self.cf.dim[2],
                                                                        self.cf.num_classes)).transpose((0, 4, 1, 2, 3))
            prediction_val_smax = prediction_val_smax_flat.reshape((val_args['X'].shape[0], self.cf.dim[0],
                                                                    self.cf.dim[1], self.cf.dim[2],
                                                                    self.cf.num_classes)).transpose((0, 4, 1, 2, 3))
            self.predict_smax['train'] = theano.function([train_args['X']], prediction_train_smax)
            self.predict_smax['val'] = theano.function([val_args['X']], prediction_val_smax)

            # reshape target vector: shapes (b,c,0,1,2) -> (b*0*1*2,c)
            flat_target_train = train_args['y'].transpose((0, 2, 3, 4, 1)).reshape((-1, self.cf.num_classes))
            flat_target_val = val_args['y'].transpose((0, 2, 3, 4, 1)).reshape((-1, self.cf.num_classes))

        pred_train_one_hot = get_one_hot_prediction(prediction_train_smax, self.cf.num_classes)
        pred_val_one_hot = get_one_hot_prediction(prediction_val_smax, self.cf.num_classes)
        self.predict_one_hot['val'] = theano.function([val_args['X']], pred_val_one_hot)
        self.predict_one_hot['train'] = theano.function([train_args['X']], pred_train_one_hot)

        prediction_val = T.argmax(prediction_val_smax, axis=1)
        prediction_train = T.argmax(prediction_train_smax, axis=1)
        self.predict['val'] = theano.function([val_args['X']], prediction_val)
        self.predict['train'] = theano.function([train_args['X']], prediction_train)

        ### evaluation metrics
        train_dices_hard = binary_dice_per_instance_and_class(pred_train_one_hot, train_args['y'], dim)
        val_dices_hard = binary_dice_per_instance_and_class(pred_val_one_hot, val_args['y'], dim)
        train_dices_soft = binary_dice_per_instance_and_class(prediction_train_smax, train_args['y'], dim)
        val_dices_soft = binary_dice_per_instance_and_class(prediction_val_smax, val_args['y'], dim)

        class_train_acc = T.mean(T.eq(T.argmax(class_train_prediction, axis=1), train_args['c']), dtype=theano.config.floatX)
        class_val_acc = T.mean(T.eq(T.argmax(class_val_prediction, axis=1), val_args['c']), dtype=theano.config.floatX)

        ### loss types
        if loss == 'cross_entropy':
            self.loss['train'] = categorical_crossentropy(prediction_train_smax_flat, flat_target_train).mean()
            self.loss['val'] = categorical_crossentropy(prediction_val_smax_flat, flat_target_val).mean()

        if loss == 'weighted_cross_entropy':
            theano_args['w'] = T.fvector()
            train_args['w'] = T.fvector()
            train_loss = categorical_crossentropy(prediction_train_smax_flat, flat_target_train)
            train_loss *= train_args['w']
            self.loss['train'] = train_loss.mean()

            val_args['w'] = T.fvector()
            val_loss = categorical_crossentropy(prediction_val_smax_flat, flat_target_val)
            val_loss *= val_args['w']
            self.loss['val'] = val_loss.mean()

        if loss == 'dice':
            self.loss['train'] = 1 - train_dices_soft.mean()
            self.loss['val'] = 1 - val_dices_soft.mean()
        self.logger.info('Net: Using {} loss.'.format(loss))

        if self.cf.use_weight_decay:
            training_loss = self.loss['train'] + \
                            self.cf.weight_decay * lasagne.regularization.regularize_network_params(
                                self.net[self.cf.seg_out_layer_flat],
                                lasagne.regularization.l2)
            self.logger.info('Net: Using weight decay of {}.'.format(self.cf.weight_decay))
        else:
            training_loss = self.loss['train']

        class_reg = lasagne.regularization.regularize_network_params(class_layer, lasagne.regularization.l2,
                                                                   {'trainable': True})
        self.class_loss['train'] = lasagne.objectives.categorical_crossentropy(class_train_prediction,
                                                                      train_args['c']).mean()
        self.class_loss['val'] = lasagne.objectives.categorical_crossentropy(class_val_prediction,
                                                                       val_args['c']).mean()


        training_loss += (self.class_loss['train'] + self.cf.class_weight_decay * class_reg) * train_args['lw']


        ### training functions
        params = set(get_all_params(self.net[self.cf.class_layer], trainable=True))
        params = params.union(set(get_all_params(self.net[self.cf.seg_out_layer_flat], trainable=True)))
        params = list(params)
        grads = theano.grad(training_loss, params)
        updates = adam(grads, params, learning_rate=train_args['lr'])

        self.train_fn = theano.function(train_args.values(), [self.loss['train'], train_dices_hard, class_train_acc, self.class_loss['train'], training_loss], updates=updates)
        self.val_fn = theano.function(val_args.values(), [self.loss['val'], val_dices_hard, class_val_acc, self.class_loss['val']])

        self.logger.info('Net: Compiled theano functions.')

    def run_epoch(self, batch_gen):

        ### validation
        val_loss_running_mean = 0.
        val_class_loss_running_mean = 0.
        val_acc_running_mean = 0.
        val_dices_running_batch_mean = np.zeros(shape=(1, self.cf.num_classes))



        for _ in range(self.cf.num_val_batches):
            batch = next(batch_gen['val'])
            args = (batch['data'], batch['seg'], batch['class_target'])
            if self.cf.loss == 'weighted_cross_entropy':
                weights = get_weights(batch['seg'], self.cf.num_classes)
                args = (batch['data'], batch['seg'], batch['class_target'], weights)

            val_loss, val_dices, val_acc, val_class_loss = self.val_fn(*args)
            val_loss_running_mean += val_loss / self.cf.num_val_batches
            val_class_loss_running_mean += val_class_loss / self.cf.num_val_batches
            val_acc_running_mean += val_acc / self.cf.num_val_batches
            val_dices_running_batch_mean[0] += np.mean(val_dices, axis=0) / self.cf.num_val_batches

        self.metrics['val']['loss'].append(val_loss_running_mean)
        self.metrics['val']['class_loss'].append(val_class_loss_running_mean)
        self.metrics['val']['class_acc'].append(val_acc_running_mean)
        self.metrics['val']['dices'] = np.append(self.metrics['val']['dices'], val_dices_running_batch_mean, axis=0)

        ### training
        if type(self.cf.learning_rate) == list:
            lr = self.cf.learning_rate[self.epoch]
        else:
            lr = self.cf.learning_rate

        if type(self.cf.class_loss_weight) == list:
            lw = np.array(self.cf.class_loss_weight[self.epoch]).astype('float32')
        else:
            lw = np.array(self.cf.class_loss_weight).astype('float32')



        train_loss_running_mean = 0.
        train_class_loss_running_mean = 0.
        train_acc_running_mean = 0.
        train_dices_running_batch_mean = np.zeros(shape=(1, self.cf.num_classes))
        for _ in range(self.cf.num_train_batches):
            batch = next(batch_gen['train'])
            args = (batch['data'], batch['seg'], batch['class_target'], lr, lw)
            if self.cf.loss == 'weighted_cross_entropy':
                weights = get_weights(batch['seg'], self.cf.num_classes)
                args = (batch['data'], batch['seg'], batch['class_target'], lr, lw, weights)

            train_loss, train_dices, train_acc, train_class_loss, check_loss = self.train_fn(*args)
            # print grads
            #
            # with open('/home/paul/PhD/my_dlat/pickles/test_grads.pickle', 'wb') as handle:
            #     pickle.dump(grads,handle)
            train_loss_running_mean += train_loss / self.cf.num_train_batches
            train_class_loss_running_mean += train_class_loss / self.cf.num_train_batches
            train_acc_running_mean += train_acc / self.cf.num_train_batches
            train_dices_running_batch_mean += np.mean(train_dices, axis=0) / self.cf.num_train_batches

        self.metrics['train']['loss'].append(train_loss_running_mean)
        self.metrics['train']['class_loss'].append(train_class_loss_running_mean)
        self.metrics['train']['class_acc'].append(train_acc_running_mean)
        self.metrics['train']['dices'] = np.append(self.metrics['train']['dices'], train_dices_running_batch_mean,
                                                   axis=0)

        self.epoch += 1

    def save_weights(self, outfile_pkl=None, spec='', layer='output_layer'):
        assert self.net != None

        if outfile_pkl == None:
            outfile_pkl = self.cf.exp_dir + '/params' + spec + '.pkl'

        params = lasagne.layers.get_all_param_values(self.net[layer])

        with open(outfile_pkl, 'w') as f:
            cPickle.dump(params, f)
        self.logger.info('Net: Saved to {}'.format(outfile_pkl))


    def load_weights(self, outfile_pkl=None, spec=''):

        if outfile_pkl == None:
            outfile_pkl = self.cf.exp_dir + '/params' + spec + '.pkl'

        with open(outfile_pkl, 'r') as f:
            pretrained_weights = cPickle.load(f)

        lasagne.layers.set_all_param_values(self.net[self.cf.class_layer], pretrained_weights)
        self.logger.info('Net: Loaded from {}'.format(outfile_pkl))

    def save_metrics(self, outfile_pkl=None):

        if outfile_pkl == None:
            outfile_pkl = self.cf.exp_dir + '/metrics.pkl'

        with open(outfile_pkl, 'wb') as f:
            cPickle.dump(self.metrics, f)

    def load_metrics(self, outfile_pkl=None):

        if outfile_pkl == None:
            outfile_pkl = self.cf.exp_dir + '/metrics.pkl'

        with open(outfile_pkl, 'rb') as f:
            metrics = cPickle.load(f)

        return metrics

    def plot_architecture(self, f_name=None):

        from DeepTrainerUtils import draw_net
        layers = get_all_layers(self.net[self.cf.class_layer])

        if f_name == None:
            f_name = self.cf.exp_dir + '/architecture.png'

        draw_net.draw_to_file(layers, f_name, output_shape=True, verbose=True)
        self.logger.info('Net: Plotted architecture to {}'.format(f_name))


