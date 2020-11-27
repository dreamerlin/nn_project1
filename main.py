import argparse

import numpy as np
from naive_nn.nn import MultiLayerPerceptron, Trainer
from naive_nn.utils import check_accuracy, draw2d, draw3d, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Project demo for Neural Networks course')
    parser.add_argument('num', type=int, help='Number of generated data')
    parser.add_argument('--batch_size', help='Batch size', default=64)
    parser.add_argument('--lr', help='learning rate', default=1e-3)
    parser.add_argument(
        '--dims',
        help='dimension of each layer',
        type=tuple,
        nargs='+',
        default=(10, 10))
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument(
        '--total_epochs',
        type=int,
        help='Number of epochs in training',
        default=1000)
    parser.add_argument(
        '--log_interval',
        type=int,
        help='Interval of log printing',
        default=20000)
    parser.add_argument(
        '--verbose', help='Whether to print the log', action='store_true')
    parser.add_argument(
        '--bn', help='Whether to apply bn', action='store_true')
    parser.add_argument(
        '--draw', help='Whether to draw the result', action='store_true')
    parser.add_argument(
        '--num_grid', type=int, help='Number of grid', default=1000)
    args = parser.parse_args()
    return args


def generate_data(n):
    """Generate input data.

    Args:
        n (int): Number of data to be generated.

    Returns:
        np.ndarray: Generated data
    """
    x = (np.random.rand(n, 2) - 0.5) * 10
    y = np.sin(x[:, 0]) - np.cos(x[:, 1])
    data = dict(x=x, y=y)
    return data


def main():
    args = parse_args()
    if args.seed is not None:
        set_random_seed(args.seed)
    data = generate_data(args.num)
    model = MultiLayerPerceptron(
        dims=args.dims,
        leaky_ratio=0.0,
        use_batchnorm=args.bn)
    solver = Trainer(
        model,
        data,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.total_epochs,
        verbose=args.verbose,
        log_interval=args.log_interval)
    solver.train()
    loss, grid_dict = check_accuracy(model, args.num_grid)
    print(f'lowest mse loss: {loss}')

    if args.draw:
        draw2d(solver.loss_list)
        draw3d(**grid_dict)


if __name__ == '__main__':
    main()
