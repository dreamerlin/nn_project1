import argparse
import numpy as np

from naive_nn.nn import Runner, MultiLayerClassifier
from naive_nn.utils import check_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Project demo for Neural Networks course')
    parser.add_argument('num', help='Number of generated data')
    parser.add_argument('batch_size', help='Batch size', default=64)
    parser.add_argument('lr', help='learning rate', default=1e-3)
    parser.add_argument('momentum', help='momentum of the bn module', default=0.9)
    parser.add_argument('eps', help='epsilon of the bn module', default=1e-5)
    parser.add_argument('dims', help='dimension of each layer', type=tuple, nargs='+', default=(10, 10))
    parser.add_argument('total_epochs', help='Number of epochs in training', default=1000)
    parser.add_argument('log_interval', help='Interval of log printing', default=20000)
    parser.add_argument('--verbose', help='Whether to print the log', action='store_true')
    parser.add_argument('--bn', help='Whether to apply bn', action='store_true')
    args = parser.parse_args()
    return args


def generate_data(n):
    x = (np.random.rand(n, 2) - 0.5) * 10
    y = np.sin(x[:, 0]) - np.cos(x[:, 1])
    data = dict(x=x, y=y)
    return data


def main():
    args = parse_args()
    data = generate_data(args.num)
    model = MultiLayerClassifier(dims=args.dims, leaky_ratio=0.0, use_batchnorm=args.bn, momentum=args.momentum, eps=args.eps)
    solver = Runner(model, data, lr=args.lr, batch_size=args.batch_size, num_epochs=args.total_epochs, verbose=args.verbose, log_interval=args.log_interval)
    solver.train()
    model.bn_params['mode'] = 'test'
    loss = check_accuracy(model)
    print(f'lowest mse loss: {loss}')


if __name__ == '__main__':
    main()
