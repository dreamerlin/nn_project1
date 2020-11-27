import numpy as np

from .activation import leaky_relu, leaky_relu_backward, relu, relu_backward
from .inference import affine_backward, affine_forward
from .initialization import normal_init, unit_init, zero_init
from .loss import mse_loss
from .normalization import batchnorm, batchnorm_backward


class MultiLayerPerceptron:
    """Multi layer classifier.

    Args:
        dims (tuple): Tuple of dimension for each layer. Default: (10, ).
        leaky_ratio (float): Ratio for leaky relu. Default: 0.0.
        mean (float): Mean value for initialization. Default: 0.0.
        std (float): Std value for initialization. Default: 0.05.
        use_batchnorm (bool): Whether to apply batchnorm. Default: False.
    """

    def __init__(self,
                 dims=(10, ),
                 leaky_ratio=0.0,
                 mean=0.0,
                 std=0.05,
                 use_batchnorm=False):
        self.hidden_num = len(dims)
        self.params = dict()
        all_layer_dims = list(dims)
        all_layer_dims.insert(0, 2)
        all_layer_dims.append(1)
        for i in range(1, len(all_layer_dims)):
            self.params[f'w{i}'] = normal_init(
                (mean, std), (all_layer_dims[i - 1], all_layer_dims[i]))
            self.params[f'b{i}'] = zero_init((all_layer_dims[i], ))

        self.leaky_ratio = leaky_ratio
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            for i in range(self.hidden_num):
                self.params[f'gamma{i+1}'] = unit_init((dims[i], ))
                self.params[f'beta{i+1}'] = zero_init((dims[i], ))

    def loss(self, x, y=None):
        """Calculate loss and grads.

        Args:
            x (np.ndarray): x values.
            y (np.ndarray): y values.

        Returns:
            tuple[float, dict]: loss and gradients.
        """

        cache_list = list()
        out = x
        for i in range(1, self.hidden_num + 1):
            out, cache = affine_forward(out, self.params[f'w{i}'],
                                        self.params[f'b{i}'])
            cache_list.append(cache)
            if self.use_batchnorm:
                out, bn_cache = batchnorm(out, self.params[f'gamma{i}'],
                                          self.params[f'beta{i}'])
                cache_list.append(bn_cache)
            out, cache = relu(out) if np.abs(
                self.leaky_ratio) < 1e-5 else leaky_relu(
                    out, self.leaky_ratio)
            cache_list.append(cache)
        out, cache = affine_forward(out, self.params[f'w{self.hidden_num+1}'],
                                    self.params[f'b{self.hidden_num+1}'])
        cache_list.append(cache)

        if y is None:
            return out

        grads = dict()
        loss, dout = mse_loss(out, y)
        cache = cache_list.pop()
        dout, dw, db = affine_backward(dout, cache)
        grads[f'w{self.hidden_num + 1}'] = dw
        grads[f'b{self.hidden_num + 1}'] = db
        for i in range(self.hidden_num, 0, -1):
            cache = cache_list.pop()
            dout = relu_backward(dout, cache) if np.abs(
                self.leaky_ratio) < 1e-5 else leaky_relu_backward(dout, cache)
            if self.use_batchnorm:
                bn_cache = cache_list.pop()
                dout, dgamma, dbeta = batchnorm_backward(dout, **bn_cache)
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta
            cache = cache_list.pop()
            dout, dw, db = affine_backward(dout, cache)
            grads[f'w{i}'] = dw
            grads[f'b{i}'] = db
        return loss, grads
