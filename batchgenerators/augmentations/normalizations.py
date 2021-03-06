# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

def range_normalization(data, rnge=(0, 1), per_channel=True):
    for b in range(data.shape[0]):
        if per_channel:
            for c in range(data.shape[1]):
                mn = data[b, c].min()
                mx = data[b, c].max()
                data[b, c] -= mn
                data[b, c] /= (mx - mn)
                data[b, c] *= (rnge[1] - rnge[0])
                data[b, c] += rnge[0]
        else:
            mn = data[b].min()
            mx = data[b].max()
            data[b] -= mn
            data[b] /= (mx - mn)
            data[b] *= (rnge[1] - rnge[0])
            data[b] += rnge[0]
    return data


def zero_mean_unit_variance_normalization(data, per_channel=True, epsilon=1e-7):
    for b in range(data.shape[0]):
        if per_channel:
            for c in range(data.shape[1]):
                mean = data[b, c].mean()
                std = data[b, c].std() + epsilon
                data[b, c] = (data[b, c] - mean) / std
        else:
            mean = data[b].mean()
            std = data[b].std() + epsilon
            data[b] = (data[b] - mean) / std
    return data


def mean_std_normalization(data, mean, std, per_channel=True):
    if isinstance(data, np.ndarray):
        data_shape = tuple(list(data.shape))
    elif isinstance(data, (list, tuple)):
        assert len(data) > 0 and isinstance(data[0], np.ndarray)
        data_shape = (len(data), list(data[0].shape))
    else:
        raise TypeError("Data has to be either a numpy array or a list")

    if per_channel and isinstance(mean, float) and isinstance(std, float):
        mean = [mean] * data_shape[1]
        std = [std] * data_shape[1]
    elif per_channel and isinstance(mean, (tuple, list, np.ndarray)):
        assert len(mean) == data_shape[1]
    elif per_channel and isinstance(std, (tuple, list, np.ndarray)):
        assert len(std) == data_shape[1]


    for b in range(data_shape[0]):
        if per_channel:
            for c in range(data_shape[1]):
                data[b][c] = (data[b][c] - mean[c]) / std[c]
        else:
            data[b] = (data[b] - mean) / std
    return data


def cut_off_outliers(data, percentile_lower=0.2, percentile_upper=99.8, per_channel=False):
    for b in range(len(data)):
        if not per_channel:
            cut_off_lower = np.percentile(data[b], percentile_lower)
            cut_off_upper = np.percentile(data[b], percentile_upper)
            data[b][data[b] < cut_off_lower] = cut_off_lower
            data[b][data[b] > cut_off_upper] = cut_off_upper
        else:
            for c in range(data.shape[1]):
                cut_off_lower = np.percentile(data[b, c], percentile_lower)
                cut_off_upper = np.percentile(data[b, c], percentile_upper)
                data[b, c][data[b, c] < cut_off_lower] = cut_off_lower
                data[b, c][data[b, c] > cut_off_upper] = cut_off_upper
    return data