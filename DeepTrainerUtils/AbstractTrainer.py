from abc import abstractmethod
import os
import logging
import numpy as np
import subprocess
import shutil


class DeepTrainer():

	def __init__(self, cf):
		self.cf = cf
		np.random.seed(cf.seed)
		# self.srng = RandomStreams(seed=cf.seed)
		self.batch_gen = {}

	def prep_exp(self):
		if not os.path.isdir(self.cf.exp_dir):
			os.mkdir(self.cf.exp_dir)
		if not os.path.isdir(self.cf.plot_dir):
			os.mkdir(self.cf.plot_dir)


		self.logger = logging.getLogger('UNet_training')
		log_file = self.cf.exp_dir + '/training.log'
		print("Logging to {}".format(log_file))
		hdlr = logging.FileHandler(log_file)
		self.logger.addHandler(hdlr)
		self.logger.setLevel(logging.DEBUG)
		self.logger.info('Trainer: Created {}.'.format(self.cf.exp_dir))

	def save_config(self):
		if os.path.isdir(self.cf.exp_dir):
			shutil.copy(self.cf.config_path, os.path.join(self.cf.exp_dir, 'config.py'))
			self.logger.info('Trainer: Copied config.')

	@abstractmethod
	def initialize_generators(self):
		pass

	@abstractmethod
	def initialize_training(self):
		pass

	@abstractmethod
	def run_epoch(self):
		pass

