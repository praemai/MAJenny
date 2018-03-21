
import numpy as np



def get_one_hot_encoding(y):
	"""
	transform a softmax prediction to a one-hot prediction of the same shape
	"""
	n_classes = int(np.max(y)) + 1
	y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3])).astype('int32')
	for cl in range(n_classes):
		y_ohe[:, cl][y[:,0] == cl] = 1
	return y_ohe



def get_weights(seg):
	"""
	get class weight values for a vector of pixels with shape (b*0*1) or (b*0*1*2).
	return weight vector of shape b, x*y
	"""
	seg = get_one_hot_encoding(seg)
	class_counts = np.sum(np.sum(seg, axis=3), axis=2)
	# weighted_class = seg.shape[2]**2/np.clip(class_counts, 1e-8,1e8)
	weighted_class = 1 - (class_counts / float(seg.shape[2] ** 2))
	return weighted_class


def get_dice_per_batch_and_class(pred, y):

	pred = get_one_hot_encoding(pred)
	y = get_one_hot_encoding(y)
	axes = tuple(range(2, len(pred.shape)))
	intersect = np.sum(pred*y, axis=axes)
	denominator = np.sum(pred, axis=axes)+np.sum(y, axis=axes)
	dice = 2.0*intersect / denominator

	return dice


