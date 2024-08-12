import os

os.environ['OMP_NUM_THREADS'] = '1'

import argparse

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from mario import create_mario
from model import ActorCritic


def get_args():
    parser = argparse.ArgumentParser("A3C Mario - test")

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--saved_path', type=str, default='trained_models', help='path saved global nets')
    
    args = parser.parse_args()
    return args

def test(arg):
    color_env = False
    c_in = 3 if color_env else 1

    torch.manual_seed(arg.seed)
    env = create_mario(color_env=color_env, save_video=True)
    num_actions = env.action_space.n
    model = ActorCritic(c_in, num_actions)
    model.load_state_dict(torch.load(f"{arg.saved_path}/a3c_mario", map_location = lambda storage, loc : storage))
    model.eval()

    state, *_ = env.reset()
    state = torch.from_numpy(state)
    termin, trunc = False, False
    lstm_h, lstm_c = 0,0

    while True:
        logits, _, lstm_h, lstm_c = model(state, lstm_h, lstm_c)
        policy = F.softmax(logits)
        action = torch.argmax(policy).item()    # greedy selection
        next_state, _, termin, trunc, _ = env.step(action)
        state = torch.from_numpy(next_state)

        if termin or trunc:
            print("Game over")
            break
    
    env.save_video(fps=30)
    env.close()

if __name__=="__main__":
    args = get_args()
    test(args)