import numpy as np


class Runner:
    """Runner object to conduct training and inference.

    Args:
        model (object): model to be trained.
        data (dict): data dict containing ``x`` and ``y``.
        kwargs (dict): keywords dict for training.
    """

    def __init__(self, model, data, **kwargs):
        self.x = data['x']
        self.y = data['y']
        self.n = self.x.shape[0]
        self.model = model
        self.lr = kwargs.get('lr', 1e-3)
        self.batch_size = kwargs.get('batch_size', 20)
        self.num_epochs = kwargs.get('num_epochs', 20)
        self.verbose = kwargs.get('verbose', False)
        self.log_interval = kwargs.get('log_interval', 100)
        self._init_params()

    def _init_params(self):
        """initialization."""
        self.min_loss = float('inf')
        self.loss_list = list()
        self.best_params = dict()
        self.cur_iter = 0

    def _step(self):
        """inference step."""
        self.cur_iter += 1
        idxes = np.random.choice(self.n, self.batch_size)
        sample_x, sample_y = self.x[idxes], self.y[idxes]
        loss, grads = self.model.loss(sample_x, sample_y)
        self.loss_list.append(loss)
        # 更新网络参数
        for weight in grads:
            self.model.params[weight] -= self.lr * grads[weight]
        # 保存最优网络参数
        if self.loss_list[-1] < self.min_loss:
            self.min_loss = self.loss_list[-1]
            for weight in self.model.params:
                self.best_params[weight] = self.model.params[weight].copy()

    def train(self):
        """training network."""
        iters_per_epoch = (self.n + self.batch_size - 1) // self.batch_size
        num_iterations = iters_per_epoch * self.num_epochs
        for i in range(num_iterations):
            self._step()
            if self.verbose and i % self.log_interval == 0:
                print(f'Training loss: {self.loss_list[-1]} at iteration {i}')
        if len(self.best_params) != 0:
            self.model.params = self.best_params
