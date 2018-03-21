author = 'Simon Kohl, Paul Jaeger, June 2017'

from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np
from DeepTrainerUtils.pytorch_utils import get_weights, get_dice_per_batch_and_class
from torch.autograd import Variable



class SegmentationNet():
    def __init__(self, cf, logger):
        self.cf = cf
        self.net = None
        self.loss = None
        self.predict = {}
        self.predict_smax = {}
        self.predict_one_hot = {}
        self.loss = {}
        self.logger = logger
        self.epoch = 0
        self.metrics = {}
        self.metrics['train'] = {'loss': [0.], 'dices': np.zeros(shape=(1, self.cf.num_classes))}
        self.metrics['val'] = {'loss': [0.], 'dices': np.zeros(shape=(1, self.cf.num_classes))}
        self.smax = nn.LogSoftmax()
        self.criterion= nn.NLLLoss2d()


    @abstractmethod
    def initialize_net(self):
        pass



    def build_loss(self, x, seg):


        var_x = self.np_to_variable(x)
        var_seg = self.np_to_variable(seg[:, 0], dtype=torch.LongTensor)
        output = self.net(var_x)
        predict_smax = self.smax(output)

        # flat_smax = self.predict_smax.permute(0, 2, 3, 1).contiguous().view((-1, self.cf.num_classes))
        # flat_target = seg.permute(0, 2, 3, 1).contiguous().view((-1, self.cf.num_classes))

        weights = np.mean(get_weights(seg), axis=0)
        criterion = nn.NLLLoss2d(weight=self.np_to_variable(weights))
        self.loss = criterion(predict_smax, var_seg)
        metric_loss = self.loss.cpu().data.numpy()
        dices = get_dice_per_batch_and_class(np.argmax(output.cpu().data.numpy(), axis=1)[:, None], seg)
        return metric_loss, dices

    def get_prediction(self, x):

        self.net.eval()
        var_x = self.np_to_variable(x)
        output = self.net(var_x)
        prediction = np.argmax(output.cpu().data.numpy(), axis=1)
        return prediction

    def run_epoch(self, batch_gen):

        self.net.eval()
        ### validation
        val_loss_running_mean = 0.
        val_dices_running_batch_mean = np.zeros(shape=(1, self.cf.num_classes))
        for _ in range(self.cf.num_val_batches):
            batch = next(batch_gen['val'])

            val_loss, val_dices = self.build_loss(batch['data'], batch['seg'])
            val_loss_running_mean += val_loss / self.cf.num_val_batches
            val_dices_running_batch_mean[0] += np.mean(val_dices, axis=0) / self.cf.num_val_batches

        self.metrics['val']['loss'].append(val_loss_running_mean)
        self.metrics['val']['dices'] = np.append(self.metrics['val']['dices'], val_dices_running_batch_mean, axis=0)

        ### training
        self.net.train()

        train_loss_running_mean = 0.
        train_dices_running_batch_mean = np.zeros(shape=(1, self.cf.num_classes))
        for _ in range(self.cf.num_train_batches):

            self.optimizer.zero_grad()
            batch = next(batch_gen['train'])

            train_loss, train_dices = self.build_loss(batch['data'], batch['seg'])
            train_loss_running_mean += train_loss / self.cf.num_train_batches
            train_dices_running_batch_mean += np.mean(train_dices, axis=0) / self.cf.num_train_batches

            self.loss.backward()
            self.optimizer.step()

        self.metrics['train']['loss'].append(train_loss_running_mean)
        self.metrics['train']['dices'] = np.append(self.metrics['train']['dices'], train_dices_running_batch_mean,
                                                   axis=0)

        self.epoch += 1

    def np_to_variable(self, x, is_cuda=True, dtype=torch.FloatTensor):
        v = Variable(torch.from_numpy(x).type(dtype))
        if is_cuda:
            v = v.cuda()
        return v

    def save_net(self, fname):
        # import h5py
        # h5f = h5py.File(fname, mode='w')
        # for k, v in net.state_dict().items():
        #     h5f.create_dataset(k, data=v.cpu().numpy())
        torch.save(self.net.state_dict(), fname)
