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
    num_stack = 4
    torch.manual_seed(arg.seed)

    env = create_mario(save_video=True, num_skip=1, num_stack=num_stack)
    num_actions = env.action_space.n
    model = ActorCritic(num_stack, num_actions)

    ## To test in real-time,
    # model.load_state_dict(torch.load(f"{arg.saved_path}/a3c_mario"))

    ## To test a specific model,
    model.load_state_dict(torch.load(f"{arg.saved_path}/a3c_mario_20240901_180627/90000")["global_model_state_dict"])

    state, *_ = env.reset()
    state = torch.from_numpy(state)
    termin, trunc = False, False
    lstm_h, lstm_c = 0,0
    agent_step = 0

    while True:
        logits, _, lstm_h, lstm_c = model(state, lstm_h, lstm_c)
        policy = F.softmax(logits, dim=2)

        ## 1. random selection
        # action = env.action_space.sample()

        ## 2. deterministic selection
        action = torch.argmax(policy).item()

        ## 3. stochastic selection
        # action = Categorical(policy).sample().item()

        next_state, _, termin, trunc, info = env.step(action)
        agent_step += 1
        state = torch.from_numpy(next_state)

        if agent_step % 50 == 0:
            print(f"step {agent_step}, position {info['x_pos']}")

        if termin or trunc:
            print(f"Game over at {agent_step}th step, x = {info['x_pos']}")
            break
    
    env.save_video(fps=30)
    env.close()

if __name__=="__main__":
    args = get_args()
    test(args)