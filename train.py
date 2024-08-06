import os

os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import time

import torch
import torch.multiprocessing as _mp
from torch.optim import Adam

from model import ActorCritic
from process import local_train


def get_args():
    parser = argparse.ArgumentParser("A3C Mario - train")

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=0.9, help='bias-variance tradeoff of GAE: commonly known as lambda')
    parser.add_argument('--sigma', type=float, default=0.01, help='controls entropy regularization of A3C: commonly known as beta')
    
    parser.add_argument('--num_local_steps', type=int, default=20, help='A3C parameter')
    parser.add_argument('--num_global_steps', type=int, default=1e6, help='A3C parameter')  # 1M
    parser.add_argument('--num_processes', type=int, default=4, help='the number of processes')

    parser.add_argument('--save_interval', type=int, default=50, help='the number of horizons between savings')
    parser.add_argument('--save_path', type=str, default='trained_models', help='path to save global nets')

    args = parser.parse_args()
    return args

def train(arg):
    print(time.ctime(time.time()))

    torch.manual_seed(arg.seed)

    if not os.path.isdir(arg.save_path):
        os.makedirs(arg.save_path)
    mp = _mp.get_context("spawn")

    global_model = ActorCritic(1, 12)
    global_model.share_memory()

    optimizer = Adam(params = global_model.parameters(), lr = arg.lr)
    processes = []

    for index in range(arg.num_processes):
        process = mp.Process(target = local_train, args=(index, arg, global_model, optimizer))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    print(time.ctime(time.time()))

    return

if __name__=="__main__":
    args = get_args()
    train(args)