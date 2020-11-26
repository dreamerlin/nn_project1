import numpy as np

from .activation import (affine_leaky_relu, affine_leaky_relu_backward,
                         affine_relu, affine_relu_backward)
from .inference import affine_backward, affine_forward
from .initialization import normal_init, unit_init, zero_init
from .loss import mse_loss
from .normalization import batchnorm, batchnorm_backward


class MultiLayerClassifier:
    """Multi layer classifier.

    Args:
        dims (tuple): Tuple of dimension for each layer. Default: (10, ).
        leaky_ratio (float): Ratio for leaky relu. Default: 0.0.
        mean (float): Mean value for initialization. Default: 0.0.
        std (float): Std value for initialization. Default: 0.05.
        use_batchnorm (bool): Whether to apply batchnorm. Default: False.
        momentum (float): Momentum value for batchnorm. Default: 0.9.
        eps (float): Epsilon value fpr batchnorm. Default: 1e-5.
        mode (str): Mode of model. Default: 'train'.
    """

    def __init__(self,
                 dims=(10, ),
                 leaky_ratio=0.0,
                 mean=0.0,
                 std=0.05,
                 use_batchnorm=False,
                 momentum=0.9,
                 eps=1e-5,
                 mode='train'):
        self.hidden_num = len(dims)
        self.params = dict()
        self.params['w1'] = normal_init((mean, std), (2, dims[0]))
        self.params['b1'] = zero_init((dims[0], ))
        for i in range(1, self.hidden_num):
            self.params[f'w{i+1}'] = normal_init((mean, std), (dims[i - 1], dims[i]))
            self.params[f'b{i+1}'] = zero_init((dims[i], ))
        self.params[f'w{self.hidden_num+1}'] = normal_init((mean, std), (dims[-1], 1))
        self.params[f'b{self.hidden_num+1}'] = zero_init((1, ))

        self.leaky_ratio = leaky_ratio
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            for i in range(self.hidden_num):
                self.params[f'gamma{i+1}'] = unit_init((dims[i], ))
                self.params[f'beta{i+1}'] = zero_init((dims[i], ))

        bn_params = dict()
        bn_params['mode'] = mode
        bn_params['momentum'] = momentum
        bn_params['eps'] = eps
        self.bn_params = bn_params

    def loss(self, x, y=None):
        """Calculate loss and grads

        Args:
            x (np.ndarray): x values.
            y (np.ndarray): y values.

        Returns:
            tuple[float, dict]: loss and gradients.
        """
        grads = dict()
        cache_list = list()
        out = x
        for i in range(1, self.hidden_num + 1):
            if np.abs(self.leaky_ratio) < 1e-6:
                out, cache = affine_relu(out, self.params[f'w{i}'],
                                         self.params[f'b{i}'])
            else:
                out, cache = affine_leaky_relu(out, self.params[f'w{i}'],
                                               self.params[f'b{i}'],
                                               self.leaky_ratio)
            cache_list.append(cache)
            if self.use_batchnorm:
                out, bn_cache, self.bn_params = batchnorm(out, self.params[f'gamma{i}'],
                                          self.params[f'beta{i}'],
                                          **self.bn_params)
                cache_list.append(bn_cache)
        out, cache = affine_forward(out, self.params[f'w{self.hidden_num+1}'],
                                    self.params[f'b{self.hidden_num+1}'])
        cache_list.append(cache)

        if y is None:
            return out

        loss, dout = mse_loss(out, y)
        cache = cache_list.pop()
        dout, grads[f'w{self.hidden_num+1}'], grads[
            f'b{self.hidden_num+1}'] = affine_backward(dout, cache)
        for i in range(self.hidden_num, 0, -1):
            if self.use_batchnorm:
                bn_cache = cache_list.pop()
                dout, grads[f'gamma{i}'], grads[
                    f'beta{i}'] = batchnorm_backward(dout, **bn_cache)
            cache = cache_list.pop()
            if np.abs(self.leaky_ratio) < 1e-6:
                dout, grads[f'w{i}'], grads[f'b{i}'] = affine_relu_backward(
                    dout, cache)
            else:
                dout, grads[f'w{i}'], grads[
                    f'b{i}'] = affine_leaky_relu_backward(dout, cache)
        return loss, grads
