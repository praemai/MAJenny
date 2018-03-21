__author__ = 'Simon Kohl, June 2017'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np



class TrainingPlot_2Panel():

	def __init__(self,
				 num_epochs,
				 file_name,
				 experiment_name,
				 class_dict=None,
				 figsize = (10, 8), ymax=1):

		self.file_name = file_name
		self.exp_name = experiment_name
		self.class_dict = class_dict
		self.f = plt.figure(figsize=figsize)
		gs1 = gridspec.GridSpec(2, 1, height_ratios=[3.5,1], width_ratios=[1])
		self.ax1 = plt.subplot(gs1[0])
		self.ax2 = plt.subplot(gs1[1])

		self.ax1.set_xlabel('epochs')
		self.ax1.set_ylabel('dice')
		self.ax1.set_xlim(0,num_epochs)
		self.ax1.set_ylim(0.0, ymax)
		# self.ax1.set_aspect(num_epochs//2)

		self.ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
		self.ax2.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')


	def update_and_save(self,
						metrics,
						best_metrics,
						type='loss_and_dice'):

		if type == 'loss_and_dice':
			plot_loss_and_dice(self.ax1, self.ax2, metrics, best_metrics, self.exp_name, self.class_dict)

		elif type == 'loss_and_acc':
			plot_loss_and_accuracy(self.ax1, self.ax2, metrics, best_metrics, self.exp_name, self.class_dict)

		else:
			raise NotImplementedError('TrainingPlot: Plot Type {} not implemented!'.format(type))

		plt.savefig(self.file_name)


class TrainingPlot_3Panel():

	def __init__(self,
				 num_epochs,
				 file_name,
				 experiment_name,
				 class_dict=None,
				 figsize = (10, 12)):

		self.file_name = file_name
		self.exp_name = experiment_name
		self.class_dict = class_dict
		self.f = plt.figure(figsize=figsize)
		gs1 = gridspec.GridSpec(3, 1, height_ratios=[3.5,3.5,1], width_ratios=[1])
		self.ax1 = plt.subplot(gs1[0])
		self.ax2 = plt.subplot(gs1[1])
		self.ax3 = plt.subplot(gs1[2])

		self.ax1.set_xlabel('epochs')
		self.ax1.set_ylabel('dice')
		self.ax1.set_xlim(0,num_epochs)
		self.ax1.set_ylim(0.0,1)
		self.ax1.set_aspect(num_epochs//2)

		self.ax2.set_xlabel('epochs')
		self.ax2.set_ylabel('classification acc & loss')
		self.ax2.set_xlim(0,num_epochs)
		self.ax2.set_ylim(0.0,1)
		self.ax2.set_aspect(num_epochs//2)

		self.ax3.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
		self.ax3.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')


	def update_and_save(self,
						metrics,
						best_metrics,
						type='loss_and_dice'):

		if type == 'loss_and_dice':
			plot_loss_dice_acc(self.ax1, self.ax2, self.ax3, metrics, best_metrics, self.exp_name, self.class_dict)
		else:
			raise NotImplementedError('TrainingPlot: Plot Type {} not implemented!'.format(type))

		plt.savefig(self.file_name)



def plot_loss_and_dice(ax, ax2, metrics, best_metrics, experiment_name, class_dict=None):
	"""
	monitor the training process in terms of the loss and dice values of the individual classes
	:param ax:
	:param ax2:
	:param metrics: a dict of the training metrics, use AbstractSegmentationNet's property
	:param best_metrics: a dict of the best metrics, use UNet_2D_Trainer's property
	:param experiment_name:
	:param class_dict: a dict that specifies the mapping between classes and their names
	:return:
	"""

	num_epochs = len(metrics['val']['loss'])
	epochs = range(num_epochs)
	num_classes = len(best_metrics['dices'])

	# prepare colors and linestyle
	num_lines = num_classes + 1
	color=iter(plt.cm.rainbow(np.linspace(0,1,num_lines)))
	colors = []
	for _ in range(num_lines):
		colors.append(next(color))

	colors = colors + colors
	linestyle = ['--']*num_lines + ['-']*num_lines

	# prepare values
	values_to_plot = []
	for l in range(num_classes):
		values_to_plot.append(metrics['train']['dices'][:,l])
	values_to_plot.append(metrics['train']['loss'])
	for l in range(num_classes):
		values_to_plot.append(metrics['val']['dices'][:,l])
	values_to_plot.append(metrics['val']['loss'])

	# prepare legend
	if class_dict != None:
		assert len(class_dict) == num_classes
		raw_labels = [class_dict[i] + ' dice' for i in range(num_classes)]
	else:
		raw_labels = ['class ' + str(i) + ' dice' for i in range(num_classes)]
	raw_labels.append('Loss')

	train_labels = ['Train: ' + l for l in raw_labels]
	val_labels = ['Val: ' + l for l in raw_labels]
	labels = train_labels + val_labels

	if ax.lines:
		for i, elem in enumerate(values_to_plot):
			ax.lines[i].set_xdata(epochs)
			ax.lines[i].set_ydata(elem)
	else:
		for elem, color, linestyle, label in zip(values_to_plot, colors, linestyle, labels):
			ax.plot(epochs, elem, color=color, linestyle=linestyle, label=label)

	handles, labels = ax.get_legend_handles_labels()
	leg = ax.legend(handles, labels, loc=1, fontsize=10)
	leg.get_frame().set_alpha(0.5)
	text = "EXPERIMENT_NAME = '{}'\nBest Val Loss/Ep = {}/{}\n"\
		.format(experiment_name, np.round(best_metrics['loss'][0],3), best_metrics['loss'][1])

	best_metrics_text = ''
	for c in range(num_classes):
		best_metrics_text += 'Best {}-Dice/Ep = {}/{}\n'.format(class_dict[c], np.round(best_metrics['dices'][c][0],3), int(best_metrics['dices'][c][1]))

	text += best_metrics_text
	ax2.clear()
	ax2.text(0.03, 0.1, text, color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0))


def plot_loss_dice_acc(ax, ax2, ax3, metrics, best_metrics, experiment_name, class_dict=None):
	"""
	monitor the training process in terms of the loss and dice values of the individual classes
	:param ax:
	:param ax2:
	:param metrics: a dict of the training metrics, use AbstractSegmentationNet's property
	:param best_metrics: a dict of the best metrics, use UNet_2D_Trainer's property
	:param experiment_name:
	:param class_dict: a dict that specifies the mapping between classes and their names
	:return:
	"""

	num_epochs = len(metrics['val']['loss'])
	epochs = range(num_epochs)

	### AXIS 2
	num_classes = len(best_metrics['dices'])
	# prepare colors and linestyle
	num_lines = num_classes + 1
	color=iter(plt.cm.rainbow(np.linspace(0,1,num_lines)))
	colors = []
	for _ in range(num_lines):
		colors.append(next(color))

	colors = colors + colors
	linestyle = ['--']*num_lines + ['-']*num_lines

	# prepare values
	values_to_plot = []
	for l in range(num_classes):
		values_to_plot.append(metrics['train']['dices'][:,l])
	values_to_plot.append(metrics['train']['loss'])
	for l in range(num_classes):
		values_to_plot.append(metrics['val']['dices'][:,l])
	values_to_plot.append(metrics['val']['loss'])

	# prepare legend
	if class_dict != None:
		assert len(class_dict['dices']) == num_classes
		raw_labels = [class_dict['dices'][i] + ' dice' for i in range(num_classes)]
	else:
		raw_labels = ['class ' + str(i) + ' dice' for i in range(num_classes)]
	raw_labels.append('Loss')

	train_labels = ['Train: ' + l for l in raw_labels]
	val_labels = ['Val: ' + l for l in raw_labels]
	labels = train_labels + val_labels

	if ax.lines:
		for i, elem in enumerate(values_to_plot):
			ax.lines[i].set_xdata(epochs)
			ax.lines[i].set_ydata(elem)
	else:
		for elem, color, linestyle, label in zip(values_to_plot, colors, linestyle, labels):
			ax.plot(epochs, elem, color=color, linestyle=linestyle, label=label)

	handles, labels = ax.get_legend_handles_labels()
	leg = ax.legend(handles, labels, loc=1, fontsize=10)
	leg.get_frame().set_alpha(0.5)

	### AXIS 2
	num_classes = len(class_dict['targets'])
	# prepare colors and linestyle
	num_lines = 2
	color=iter(plt.cm.rainbow(np.linspace(0,1,num_lines)))
	colors = []
	for _ in range(num_lines):
		colors.append(next(color))

	colors = colors + colors
	linestyle = ['--']*num_lines + ['-']*num_lines

	# prepare values
	values_to_plot = []
	values_to_plot.append(metrics['train']['class_acc'])
	values_to_plot.append(metrics['train']['class_loss'])
	values_to_plot.append(metrics['val']['class_acc'])
	values_to_plot.append(metrics['val']['class_loss'])

	# prepare legend
	raw_labels = ['Acc', 'Loss']
	train_labels = ['Train: ' + l for l in raw_labels]
	val_labels = ['Val: ' + l for l in raw_labels]
	labels = train_labels + val_labels

	if ax2.lines:
		for i, elem in enumerate(values_to_plot):
			ax2.lines[i].set_xdata(epochs)
			ax2.lines[i].set_ydata(elem)
	else:
		for elem, color, linestyle, label in zip(values_to_plot, colors, linestyle, labels):
			ax2.plot(epochs, elem, color=color, linestyle=linestyle, label=label)

	handles, labels = ax2.get_legend_handles_labels()
	leg = ax2.legend(handles, labels, loc=1, fontsize=10)
	leg.get_frame().set_alpha(0.5)

	# AXIS 3
	text = "EXPERIMENT_NAME = '{}'\nBest Val Loss/Ep = {}/{}\n"\
		.format(experiment_name, np.round(best_metrics['loss'][0],3), best_metrics['loss'][1])

	best_metrics_text = ''
	for c in range(len(best_metrics['dices'])):
		best_metrics_text += 'Best {}-Dice/Ep = {}/{}\n'.format(class_dict['dices'][c], np.round(best_metrics['dices'][c][0],3), int(best_metrics['dices'][c][1]))

	text += best_metrics_text
	ax3.clear()
	ax3.text(0.03, 0.1, text, color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0))



def plot_loss_and_accuracy(ax, ax2, metrics, best_metrics, experiment_name, class_dict=None):
	"""
	monitor the training process in terms of the loss and dice values of the individual classes
	:param ax:
	:param ax2:
	:param metrics: a dict of the training metrics, use AbstractClassificationNet's property
	:param best_metrics: a dict of the best metrics, use DenseNet_Trainer's property
	:param experiment_name:
	:param class_dict: a dict that specifies the mapping between classes and their names
	:return:
	"""

	num_epochs = len(metrics['val']['loss'])
	epochs = range(num_epochs)
	ax.set_ylabel('accuracy')

	# prepare colors and linestyle
	num_lines = 2
	color=iter(plt.cm.rainbow(np.linspace(0,1,num_lines)))
	colors = []
	for _ in range(num_lines):
		colors.append(next(color))

	colors = colors + colors
	linestyle = ['--']*num_lines + ['-']*num_lines

	# prepare values
	values_to_plot = []
	values_to_plot.append(metrics['train']['acc'])
	values_to_plot.append(metrics['train']['loss'])
	values_to_plot.append(metrics['val']['acc'])
	values_to_plot.append(metrics['val']['loss'])

	# prepare legend
	raw_labels = ['Acc', 'Loss']
	train_labels = ['Train: ' + l for l in raw_labels]
	val_labels = ['Val: ' + l for l in raw_labels]
	labels = train_labels + val_labels

	if ax.lines:
		for i, elem in enumerate(values_to_plot):
			ax.lines[i].set_xdata(epochs)
			ax.lines[i].set_ydata(elem)
	else:
		for elem, color, linestyle, label in zip(values_to_plot, colors, linestyle, labels):
			ax.plot(epochs, elem, color=color, linestyle=linestyle, label=label)

	handles, labels = ax.get_legend_handles_labels()
	leg = ax.legend(handles, labels, loc=1, fontsize=10)
	leg.get_frame().set_alpha(0.5)
	text = "EXPERIMENT_NAME = '{}'\nBest Val Loss/Ep = {}/{}\n"\
		.format(experiment_name, np.round(best_metrics['loss'][0],3), best_metrics['loss'][1])

	best_metrics_text = 'Best Acc/Ep = {}/{}\n'.format(np.round(best_metrics['acc'][0],3), int(best_metrics['acc'][1]))
	text += best_metrics_text
	ax2.clear()
	ax2.text(0.03, 0.1, text, color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0))




def plot_batch_prediction(data, batch, prediction, num_classes, outfile, gt_is_one_hot=True, class_prediction=None, n_select=None):
		"""
		plot the predictions of a batch
		:param input_batch: shape bc01/bc012
		:param ground_truth: shape bc01/bc012
		:param prediction: shape b01/b012
		:return:
		"""
		input_batch = data[:n_select]
		ground_truth = batch['seg'][:n_select]
		if  ground_truth.shape[1] ==1:
			import pytorch_utils
			ground_truth = pytorch_utils.get_one_hot_encoding(ground_truth)
		pids = batch['patient_ids'][:n_select]
		targets = batch['class_target'][:n_select]
		prediction = prediction[:n_select]
        
		print("CHECK", prediction.shape, ground_truth.shape)
        

		if class_prediction is not None:
			class_prediction = class_prediction[:n_select,1]

		if gt_is_one_hot:
			ground_truth = np.argmax(ground_truth, axis=1)

		ground_truth = ground_truth[:, np.newaxis]
		prediction = prediction[:, np.newaxis]

		try:
			# all dimensions except for the 'channel-dimension' are required to match
			for i in [0,2,3]:
				assert input_batch.shape[i] == ground_truth.shape[i] == prediction.shape[i]
		except:
			raise Warning('Shapes of arrays to plot not in agreement! Shapes {} vs. {} vs {}'.format(input_batch.shape, ground_truth.shape, prediction.shape))

		show_arrays = np.concatenate([input_batch, ground_truth, prediction], axis=1)

		approx_figshape = (2.5 * show_arrays.shape[0], 2.5 * show_arrays.shape[1])
		fig = plt.figure(figsize=approx_figshape)
		gs = gridspec.GridSpec(show_arrays.shape[1], show_arrays.shape[0])
		gs.update(wspace=0.1, hspace=0.1)

		for b in range(show_arrays.shape[0]):
			for m in range(show_arrays.shape[1]):

				ax = plt.subplot(gs[m,b])
				ax.axis('off')
				arr = show_arrays[b,m]

				if m < input_batch.shape[1]:
					cmap = 'gray'
					vmin = None
					vmax = None
				else:
					cmap = None
					vmin = 0
					vmax = num_classes - 1

				if m==0:
					if class_prediction is not None:
						plt.title('{}|{}|{:0.2f}'.format(pids[b], targets[b], class_prediction[b]))
					else:
						plt.title('{}|{}'.format(pids[b], targets[b]))
				plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)


		plt.savefig(outfile)
		plt.close(fig)

def plot_batch_detection(data, batch, pred_bbs, num_classes, outfile, gt_is_one_hot=True,
						  class_prediction=None, n_select=None):
	"""
	plot the predictions of a batch
	:param input_batch: shape bc01/bc012
	:param ground_truth: shape bc01/bc012
	:param prediction: shape b01/b012
	:return:
	"""
	input_batch = data[:n_select]
	ground_truth = batch['seg'][:n_select]
	pids = batch['patient_ids'][:n_select]
	pred_bbs = pred_bbs[:n_select]
	reg_ground_truth = batch['reg_target'][:n_select]
	class_ground_truth = batch['class_target'][:n_select]

	if class_prediction is not None:
		class_prediction = class_prediction[:n_select, 1]

	if gt_is_one_hot:
		ground_truth = np.argmax(ground_truth, axis=1)

	ground_truth = ground_truth[:, np.newaxis]


	try:
		# all dimensions except for the 'channel-dimension' are required to match
		for i in [0, 2, 3]:
			assert input_batch.shape[i] == ground_truth.shape[i]
	except:
		raise Warning(
			'Shapes of arrays to plot not in agreement! Shapes {} vs. {} '.format(input_batch.shape, ground_truth.shape))

	show_arrays = np.concatenate([input_batch, ground_truth], axis=1)

	approx_figshape = (2.5 * show_arrays.shape[0], 2.5 * show_arrays.shape[1])
	fig = plt.figure(figsize=approx_figshape)
	gs = gridspec.GridSpec(show_arrays.shape[1], show_arrays.shape[0])
	gs.update(wspace=0.1, hspace=0.1)

	for b in range(show_arrays.shape[0]):
		for m in range(show_arrays.shape[1]):

			ax = plt.subplot(gs[m, b])
			ax.axis('off')
			arr = show_arrays[b, m]

			if m < input_batch.shape[1]:
				cmap = 'gray'
				vmin = None
				vmax = None
			else:
				cmap = None
				vmin = 0
				vmax = num_classes - 1

			plt.plot(pred_bbs[b][0], pred_bbs[b][1], linewidth=2, color='r', marker='+')  # up
			plt.plot(reg_ground_truth[b][0], reg_ground_truth[b][1], linewidth=2, color='g', marker='+')  # up

			if m == 0:
				if class_prediction is not None:
					plt.title('{}|{}'.format(pids[b], class_ground_truth[b]))
				else:
					plt.title('{}|{}'.format(pids[b], class_ground_truth[b]))



			plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

	plt.savefig(outfile)
	plt.close(fig)


def plot_softmax_hist(pred, batch, outfile):


	f, axarr = plt.subplots(1, pred.shape[0])
	f.set_figheight(5)
	f.set_figwidth(15)
	for b in range(pred.shape[0]):

		hist_values = pred[b, 1, :, :].flatten()
		if any(np.isnan(hist_values)):
			print("nan!!")
			print("softmax also nan?", any(np.isnan(pred)))
		axarr[b].hist(hist_values, normed=True, bins=20)
		axarr[b].set_title(batch['class_target'][b])
		axarr[b].axis('off')

	plt.savefig(outfile)
	plt.close(f)


