import theano.tensor as T
import numpy as np

def binary_dice_per_instance_and_class(y_pred, y_true, dim, first_spatial_axis=2):
	"""
	valid for 2D and 3D, expects binary class labels in channel c
	y_pred is softmax output of shape (b,c,0,1) or (b,c,0,1,2)
	y_true's shape is equivalent
	"""

	spatial_axes = tuple(range(first_spatial_axis,first_spatial_axis+dim,1))
	# sum over spatial dimensions
	intersect = T.sum(y_pred * y_true, axis=spatial_axes)
	denominator = T.sum(y_pred, axis=spatial_axes) + T.sum(y_true, axis=spatial_axes)
	dice_scores = T.constant(2) * intersect / (denominator + T.constant(1e-6))

	# dices_scores has shape (batch_size, num_channels/num_classes)
	return dice_scores

def binary_dice_per_batch_and_class(y_pred, y_true, dim, first_spatial_axis=1):
	"""
	valid for 2D and 3D, expects binary class labels in channel c
	y_pred is softmax output of shape (b,c,0,1) or (b,c,0,1,2)
	y_true's shape is equivalent
	"""

	y_pred = y_pred.dimshuffle((1, 0, 2, 3))
	y_true = y_true.dimshuffle((1, 0, 2, 3))

	spatial_axes = tuple(range(first_spatial_axis,first_spatial_axis+dim+1,1))
	# sum over spatial dimensions
	intersect = T.sum(y_pred * y_true, axis=spatial_axes)
	denominator = T.sum(y_pred, axis=spatial_axes) + T.sum(y_true, axis=spatial_axes)
	dice_scores = T.constant(2) * intersect / (denominator + T.constant(1e-6))
	dice_scores = dice_scores.reshape((1, dice_scores.shape[0]))
	dice_scores_stacked = T.extra_ops.repeat(dice_scores, repeats=y_pred.shape[1], axis=0)

	# dices_scores has shape (batch_size, num_channels/num_classes)
	return dice_scores_stacked

def get_one_hot_prediction(pred, num_classes):
	"""
	transform a softmax prediction to a one-hot prediction of the same shape
	"""
	pred_max = T.argmax(pred, axis=1)
	pred_one_hot = T.zeros(pred.shape)
	for cl in range(num_classes):
		cl_val = T.constant(cl)
		pred_one_hot = T.set_subtensor(pred_one_hot[:,cl], T.eq(pred_max, cl_val))

	return pred_one_hot

def get_one_hot_class_target(class_target, num_classes):
	"""
	transform a class target vector to a one-hot encoded vector of shape (b, cl) for the cross entropy loss.
	"""
	pred_one_hot = T.zeros((class_target.shape[0], num_classes))
	for cl in range(num_classes):
		cl_val = T.constant(cl)
		pred_one_hot = T.set_subtensor(pred_one_hot[:, cl], T.eq(class_target, cl_val))

	return pred_one_hot

def get_weights(seg, num_classes):
	"""
	get class weight values for a vector of pixels with shape (b*0*1) or (b*0*1*2).
	return weight vector of shape b, x*y
	"""
	class_counts = np.sum(np.sum(seg, axis=3), axis=2)
	# weighted_class = seg.shape[2]**2/np.clip(class_counts, 1e-8,1e8)
	weighted_class = 1 - (class_counts / float(seg.shape[2] ** 2))
	weighted_class_flat = np.repeat(weighted_class, seg.shape[2]**2, axis=0) #static vector with general class weights
	flat_target =seg.transpose((0, 2, 3, 1)).reshape((-1, num_classes))
	weights = (flat_target*weighted_class_flat).sum(axis=1).astype('float32')*100 # select weight for pixels using the gt class. blow up for larger loss.
	if any(np.isnan(weights)):
		print "nan!!!! now i know where...."
	return weights
