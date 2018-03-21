author = 'Simon Kohl, Paul Jaeger, June 2017'

from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
import lasagne
import cPickle
from lasagne.layers import get_output, get_all_params, get_all_layers
from lasagne.objectives import categorical_crossentropy, squared_error
from lasagne.updates import adam
import theano
import theano.tensor as T
from DeepTrainerUtils.utils import get_one_hot_class_target




class DetectionNet():
	
	def __init__(self, cf, logger):
		self.cf = cf
		self.net = None
		self.predict = {}
		self.loss = {}
		self.logger = logger
		self.epoch = 0
		self.metrics = {}
		self.metrics['train'] = {'loss':[0.], 'acc': [0.]}
		self.metrics['val'] = {'loss':[0.], 'acc': [0.]}

	@abstractmethod
	def initialize_net(self):
		pass




	def compile_theano_functions(self, data_type='2D'):
		assert self.net != None

		### symbolic theano input
		theano_args = OrderedDict()
		dim = len(self.cf.dim)

		if data_type == '2D':
			assert dim == 2
			theano_args['X'] = T.tensor4()
			theano_args['y'] = T.dmatrix()
			self.logger.info('Net: Working with 2D data.')

		elif data_type == '3D':
			assert dim == 3
			theano_args['X'] = T.tensor5()
			theano_args['y'] = T.ivector()
			self.logger.info('Net: Working with 3D data.')

		val_args = deepcopy(theano_args)
		train_args = deepcopy(theano_args)
		train_args['lr'] = T.scalar(name='lr')

		### prediction functions

		# get softmax prediction of shape (b, classes)
		prediction_train = get_output(self.net[self.cf.out_layer], train_args['X'], deterministic=False)
		prediction_val = get_output(self.net[self.cf.out_layer], val_args['X'], deterministic=True)

		self.predict['train'] = theano.function([train_args['X']], prediction_train)
		self.predict['val'] = theano.function([val_args['X']], prediction_val)

		### l2 loss
		self.loss['train'] = squared_error(prediction_train, train_args['y']).mean()
		self.loss['val'] = squared_error(prediction_val, val_args['y']).mean()

		if self.cf.use_weight_decay:
			training_loss = self.loss['train'] +\
							self.cf.weight_decay * lasagne.regularization.regularize_network_params(self.net[self.cf.out_layer],
																	   lasagne.regularization.l2)
			self.logger.info('Net: Using weight decay of {}.'.format(self.cf.weight_decay))
		else:
			training_loss = self.loss['train']

		### accuracy
		# train_acc = T.mean(T.eq(T.argmax(prediction_train_smax, axis=1), train_args['y']))
		# val_acc = T.mean(T.eq(T.argmax(prediction_val_smax, axis=1), val_args['y']))

		### training functions
		params = get_all_params(self.net[self.cf.out_layer], trainable=True)
		grads = theano.grad(training_loss, params)
		updates = adam(grads, params, learning_rate=train_args['lr'])

		self.train_fn = theano.function(train_args.values(), [self.loss['train'], prediction_train], updates=updates)
		self.val_fn = theano.function(val_args.values(), [self.loss['val'], prediction_val])

		self.logger.info('Net: Compiled theano functions.')



	def run_epoch(self, batch_gen):

		### validation
		val_loss_running_mean = 0.
		val_acc_running_mean = 0.
		for _ in range(self.cf.num_val_batches):
			batch = next(batch_gen['val'])
			args = (batch['data'], batch['reg_target'])

			val_loss, val_preds = self.val_fn(*args)
			val_loss_running_mean += val_loss/self.cf.num_val_batches
			val_acc_running_mean += 0

		print "VAL TARGET"
		print batch['reg_target']
		print "VAL PREDS"
		print val_preds

		self.metrics['val']['loss'].append(val_loss_running_mean)
		self.metrics['val']['acc'].append(val_acc_running_mean)

		### training
		if type(self.cf.learning_rate) == list:
			lr = self.cf.learning_rate[self.epoch]
		else:
			lr = self.cf.learning_rate

		train_loss_running_mean = 0.
		train_acc_running_mean = 0.
		for _ in range(self.cf.num_train_batches):
			batch = next(batch_gen['train'])
			args = (batch['data'], batch['reg_target'], lr)


			train_loss, train_preds = self.train_fn(*args)
			train_loss_running_mean += train_loss/self.cf.num_train_batches 
			train_acc_running_mean += 0

		print "TRAIN TARGET"
		print batch['reg_target']
		print "TRAIN PREDS"
		print train_preds

		self.metrics['train']['loss'].append(train_loss_running_mean)
		self.metrics['train']['acc'].append(train_acc_running_mean)

		self.epoch += 1

	def save_weights(self, outfile_pkl=None, spec=''):
		assert self.net != None

		if outfile_pkl == None:
			outfile_pkl = self.cf.exp_dir + '/params' + spec + '.pkl'

		params = lasagne.layers.get_all_param_values(self.net[self.cf.out_layer])

		with open(outfile_pkl, 'w') as f:
			cPickle.dump(params, f)
		self.logger.info('Net: Saved to {}'.format(outfile_pkl))

	def load_weights(self, outfile_pkl=None, spec=''):

		if outfile_pkl == None:
			outfile_pkl = self.cf.exp_dir + '/params' + spec + '.pkl'

		with open(outfile_pkl, 'r') as f:
			pretrained_weights = cPickle.load(f)

		lasagne.layers.set_all_param_values(self.net[self.cf.out_layer], pretrained_weights)
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
		layers = get_all_layers(self.net[self.cf.out_layer])

		if f_name == None:
			f_name = self.cf.exp_dir + '/architecture.png'

		draw_net.draw_to_file(layers, f_name, output_shape=True, verbose=True)
		self.logger.info('Net: Plotted architecture to {}'.format(f_name))

	def plot_activations(self):
		raise NotImplementedError


