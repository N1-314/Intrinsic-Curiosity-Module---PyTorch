import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from mario_env import create_mario_env
from model import ActorCritic, ICM

import os
import sys
import time
import getpass
import argparse
from datetime import datetime
from itertools import count
from setproctitle import setproctitle

os.environ['OMP_NUM_THREADS'] = '1'

class SharedAdam(optim.Adam):
    def __init__(self, *args, **kwargs):
        super(SharedAdam, self).__init__(*args, **kwargs)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
        
def torch_obs(obs, device='cpu'):
    return torch.from_numpy(obs).contiguous().to(torch.get_default_dtype()).div(255).to(device)

def evaluation(global_model, global_icm, training_step, best, tb_log, args):
    torch.manual_seed(123)
    # writer = SummaryWriter(f'runs/{args.log_path}')
    setproctitle(args.process_name + f'+eval_{training_step}')
    walltime = time.time()

    env = create_mario_env(actions=args.actions)
    env.record = (training_step % 100_000 == 0)
    model = ActorCritic(env, args.action_lstm, args.actions)
    model.eval()
    model.load_state_dict(global_model.state_dict())
    if global_icm:
        icm = ICM(env)
        icm.load_state_dict(global_icm.state_dict())

    hx = torch.zeros((1, 256), dtype=torch.float32)
    cx = torch.zeros((1, 256), dtype=torch.float32)
    action = 0
    total_reward = 0

    obs, info = env.reset()
    for local_step in count():
        action, policy, hx, cx = model.get_action(torch_obs(obs), (hx, cx), action)
        env.set_record_info(policy.detach().cpu().numpy().flatten().copy(), local_step)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f'eval_step: {training_step}, total_reward: {total_reward}, time_step: {local_step}, x_pos: {info["x_pos"]}')

    # writer.add_scalar('eval/return', total_reward, training_step)
    # writer.add_scalar('eval/time_step', local_step, training_step)
    # writer.add_scalar('eval/x_pos', info['x_pos'], training_step)
    # writer.add_histogram('eval/x_pos_hist', info['x_pos'], training_step)
    # # if training_step % 500 == 0:
    if env.record:
        frames = np.stack(env.record_frames, axis=0)
        frames = frames.transpose(0, 3, 1, 2)
        frames = np.expand_dims(frames, axis=0)
    #     writer.add_video('video/eval', frames, training_step, fps=30)
    # writer.flush()
    data = {
        'training_step': training_step, 
        'total_reward': total_reward,
        'local_step': local_step,
        'x_pos': info['x_pos'],
        'frames': frames if env.record else None,
        'time': walltime,
    }
    tb_log.put(data)

    best_reward, best_x_pos = best
    if total_reward >= best_reward:
        best[0] = total_reward
        torch.save(model.state_dict(), f'{args.saved_path}/best_model_{int(best_reward)}.pt')
        if global_icm:
            torch.save(icm.state_dict(), f'{args.saved_path}/best_icm_{int(best_reward)}.pt')
        print(f'saved best model with reward: {best_reward}')
    if info['x_pos'] >= best_x_pos:
        best[1] = info['x_pos']
        torch.save(model.state_dict(), f"{args.saved_path}/best_model_x_pos_{int(info['x_pos'])}.pt")
        if global_icm:
            torch.save(icm.state_dict(), f"{args.saved_path}/best_icm_x_pos_{int(info['x_pos'])}.pt")
        print(f"saved best model with x_pos: {info['x_pos']}")

    return best
    # return total_reward, info['x_pos']

def worker(rank, global_model, global_icm, optimizer, args):
    torch.manual_seed(123 + rank)
    setproctitle(args.process_name + f'+worker_{rank}')
    pid = os.getpid()
    print(f'rank: {rank}, pid: {pid}')

    writer = SummaryWriter(f'runs/{args.log_path}')
    writer_video = SummaryWriter(f'runs/{args.log_path}')
    
    env = create_mario_env(actions=args.actions)
    env.set_record(rank == 0)
    num_actions = env.action_space.n
    
    if args.local_device:
        device = args.local_device        
    else:
        device = f'cuda:{rank%2+2}' # for 2, 3

    model = ActorCritic(env, args.action_lstm, args.actions).to(device)
    model.train()
    icm = None
    if args.use_icm:
        icm = ICM(env).to(device)
        icm.train()
    
    inv_criteria = nn.CrossEntropyLoss()
    fwd_criteria = nn.MSELoss()

    terminated = truncated = True
    total_step = 0
    curr_step = 0
    curr_episode = 0
    prev_score = 0
    action = 0
    is_underground = False
    underground_img = None
    prev_x_pos = 40

    for local_update_num in count():
        model.load_state_dict(global_model.state_dict())
        icm.load_state_dict(global_icm.state_dict()) if icm else None

        if terminated or truncated:
            hx = torch.zeros((1, 256), dtype=torch.float32).to(device)
            cx = torch.zeros((1, 256), dtype=torch.float32).to(device)

            writer.add_scalar(f"info/episode_worker_{rank}", curr_episode, curr_episode)
            if rank == 0 and curr_episode > 0:
                writer.add_scalar('train/return', episode_return, curr_episode)
                writer.add_scalar('train/intrinsic_return', episode_intrinsic_return, curr_episode)
                writer.add_scalar('train/extrinsic_return', episode_extrinsic_return, curr_episode)
                writer.add_scalar('train/x_pos', info['x_pos'], curr_episode)
                writer.add_scalar('train/score', info['score'], curr_episode)
                # writer.add_scalar('train/prev_score', prev_score, curr_episode)
                if underground_img is not None:
                    writer_video.add_image('underground', underground_img, curr_episode, dataformats='HWC')
                if (curr_episode-1) % 500 == 0:
                    frames = np.stack(env.record_frames, axis=0)
                    frames = frames.transpose(0, 3, 1, 2)
                    frames = np.expand_dims(frames, axis=0)
                    writer_video.add_video('video/train', frames, curr_episode, fps=30)
                writer.flush()

            obs, info = env.reset()
            obs = torch_obs(obs, device)
            episode_return = 0
            episode_intrinsic_return = 0
            episode_extrinsic_return = 0
            curr_step = 0
            prev_score = 0
            is_underground = False
            underground_img = None
            prev_x_pos = 40
            curr_episode += 1
        else:
            hx = hx.detach()
            cx = cx.detach()

        # ROLLOUT
        values, log_probs, rewards, entropies = [], [], [], []
        inv_losses, fwd_losses = [], []
        for i in range(args.num_local_steps):
            total_step += 1
            curr_step += 1
            action, value, log_prob, entropy, policy, hx, cx = model(obs, (hx, cx), action)
            
            env.set_record_info(policy.detach().cpu().numpy().flatten().copy(), 0) if rank == 0 else None # call before env.step
            next_obs, reward, terminated, truncated, info = env.step(action)
            if args.no_reward:
                reward = 0
            if args.use_sparse_reward:
                reward = 0
                if info['score'] > prev_score:
                    reward = info['score'] - prev_score
                    prev_score = info['score']
                if info['flag_get']:
                    reward += info['time'] * 50

            # check if mario is underground
            x_pos = info['x_pos']
            if prev_x_pos >= 800 and x_pos <= 30: #(898~942) to (24)
                is_underground = True
                if rank == 0:
                    underground_img = np.hstack(env.record_frames[-4*8-1::6]) # action repeat 6
                    underground_img= np.vstack([underground_img[:,:1536//2], underground_img[:,1536//2:]])
            if is_underground and x_pos >= 2000: # (194) to (2616)
                is_underground = False
            prev_x_pos = x_pos
                
            next_obs = torch_obs(next_obs, device)

            action_onehot = torch.zeros((1, num_actions)).to(device)
            action_onehot[0, action] = 1

            if icm:
                if not args.use_detach:
                    pred_logits, pred_phi, phi = icm(obs, next_obs, action_onehot) # inverse, forward, next_state
                else:
                    pred_logits, pred_phi, phi = icm.forward_detach(obs, next_obs, action_onehot) # inverse, forward, next_state

            inv_loss = inv_criteria(pred_logits, torch.tensor([action], dtype=torch.long).to(device)) if icm else 0
            if not args.use_detach:
                fwd_loss = fwd_criteria(pred_phi, phi) / 2 if icm else 0 # 
            else:
                fwd_loss = fwd_criteria(pred_phi, phi.detach()) / 2 if icm else 0 # if forward model이 icm.conv를 학습하지 않는다면 phi.detach()와 icm.forward_detach() 사용
                
            intrinsic_reward = args.eta * fwd_loss.detach() if icm else 0
            episode_intrinsic_return += intrinsic_reward
            episode_extrinsic_return += reward
            reward += intrinsic_reward
            episode_return += reward

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            inv_losses.append(inv_loss)
            fwd_losses.append(fwd_loss)

            obs = next_obs

            # from tensorboard.backend.event_processing import event_accumulator 으로 로그 읽고 후처리 용
            writer.add_scalar(f'db/{rank}/x_pos', info['x_pos'], total_step)
            writer.add_scalar(f'db/{rank}/y_pos', info['y_pos'], total_step)
            writer.add_scalar(f'db/{rank}/fwd_loss', fwd_loss, total_step)
            writer.add_scalar(f'db/{rank}/inv_loss', inv_loss, total_step)
            writer.add_scalar(f'db/{rank}/action', action, total_step)
            writer.add_scalar(f'db/{rank}/underground', int(is_underground), total_step)
            if terminated or truncated:
                break

        # UPDATE
        gae = torch.zeros(1, 1, dtype=torch.float, device=device)
        R = torch.zeros(1, 1, dtype=torch.float, device=device)
        if not (terminated or truncated):
            with torch.no_grad():
                _, R, _, _, _, _, _ = model(obs, (hx, cx))

        next_value = R
        actor_loss, critic_loss, entropy_loss, curiosity_loss = 0, 0, 0, 0

        for value, log_prob, reward, entropy, inv, fwd in reversed(list(zip(values, log_probs, rewards, entropies, inv_losses, fwd_losses))):
            δ = (reward + args.gamma*next_value.detach()) - value.detach()
            gae = δ + args.gamma*args.tau *gae
            next_value = value
            actor_loss = actor_loss + log_prob * gae
            R = args.gamma * R + reward
            critic_loss = critic_loss + 0.5*(R - value)**2
            entropy_loss = entropy_loss + entropy
            curiosity_loss = curiosity_loss + (1-args.beta)*inv + args.beta*fwd if icm else 0

        total_loss = args.lambda_*(-actor_loss + critic_loss - args.sigma*entropy_loss) + curiosity_loss

        optimizer.zero_grad()
        model.zero_grad()
        icm.zero_grad() if icm else None
        total_loss.backward()
        for local_param, global_param in zip(model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param.grad = local_param.grad.to(args.global_device)
        if icm:
            for local_param, global_param in zip(icm.parameters(), global_icm.parameters()):
                if global_param.grad is not None:
                    break
                global_param.grad = local_param.grad.to(args.global_device)
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), 40)
        torch.nn.utils.clip_grad_norm_(global_icm.parameters(), 40) if icm else None
        optimizer.step()
        
        # LOGGING
        global_update_num = int(optimizer.state_dict()['state'][0]['step'].item())
        writer.add_scalar(f'loss/actor', actor_loss.item(), global_update_num)
        writer.add_scalar(f'loss/critic', critic_loss.item(), global_update_num)
        writer.add_scalar(f'loss/entropy', entropy_loss.item(), global_update_num)

        if icm:
            writer.add_scalar(f'loss/inv', torch.stack(inv_losses).sum().item(), global_update_num)
            writer.add_scalar(f'loss/fwd', torch.stack(fwd_losses).sum().item(), global_update_num)
            writer.add_scalar(f'loss/curiosity', curiosity_loss.item(), global_update_num)

        writer.add_scalar(f'loss/total', total_loss.item(), global_update_num)

        LOGGING_GRAD = False
        if LOGGING_GRAD:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.grad, global_update_num)
            if icm:
                for name, param in icm.named_parameters():
                    writer.add_histogram(name, param.grad, global_update_num)
        # if rank == 0:
        #     print(f'training_step: {local_update_num}, x_pos: {info["x_pos"]}, curiosity_loss: {curiosity_loss.item():.3f}', end='\r')
        
        # local update
        # writer.add_scalar(f'loss/actor/{rank}', actor_loss.item(), local_update_num)
        # writer.add_scalar(f'loss/critic/{rank}', critic_loss.item(), local_update_num)
        # writer.add_scalar(f'loss/entropy/{rank}', entropy_loss.item(), local_update_num)

        # writer.add_scalar(f'loss/inv/{rank}', torch.stack(inv_losses).sum().item(), local_update_num)
        # writer.add_scalar(f'loss/fwd/{rank}', torch.stack(fwd_losses).sum().item(), local_update_num)
        # writer.add_scalar(f'loss/curiosity/{rank}', curiosity_loss.item(), local_update_num)

        # writer.add_scalar(f'loss/total/{rank}', total_loss.item(), local_update_num)

        if (global_update_num-1) % 100_000 == 0:
            writer_video.add_image('frame', np.hstack([*obs.cpu().numpy()]), global_update_num, dataformats='HW')
            

        # print(f'rank: {rank}, step: {update_num}, local_step: {i}')
        # EVERY EPISODE

class DummyEnv:
    def __init__(self, args=None):
        if args:
            if args.actions == 'RIGHT_ONLY':
                n = 5
            else:
                n = 12
        self.observation_space = np.empty((4, 42, 42))
        self.action_space = type('ActionSpace', (object,), {'n':n})() # 임의의 클래스를 인스턴스화

def tb_logging(tb_log, args):
    setproctitle(args.process_name + '+eval_tb')
    print(f'tb_logging pid: {os.getpid()}')
    writer = SummaryWriter(f'runs/{args.log_path}')
    writer_video = SummaryWriter(f'runs/{args.log_path}')
    waiting_data = []
    log_step = 0

    while True:
        while not tb_log.empty():
            waiting_data.append(tb_log.get())

        pop_idx = None
        for i, data in enumerate(waiting_data):
            if data['training_step'] == log_step:
                pop_idx = i
                break
        
        if pop_idx is not None:
            data = waiting_data.pop(pop_idx)
            step = data['training_step']
            walltime = data['time']

            writer.add_scalar('eval/return', data['total_reward'], step, walltime)
            writer.add_scalar('eval/time_step', data['local_step'], step, walltime)
            writer.add_scalar('eval/x_pos', data['x_pos'], step, walltime)
            writer.add_histogram('eval/x_pos_hist', data['x_pos'], step, walltime=walltime)
            if data['frames'] is not None:
                writer_video.add_video('video/eval', data['frames'], step, fps=30, walltime=walltime)
            writer.flush()
            log_step += 100
        time.sleep(1)

def loop_evaluation(global_model, global_icm, optimizer, args):
    training_step = 0
    # best_reward = sys.float_info.min
    # best_x_pos = 0
    # reward, x_pos
    best = torch.tensor(([sys.float_info.min, 0])).share_memory_()
    tb_log = mp.Queue()

    mp.Process(target=tb_logging, args=(tb_log, args), daemon=True).start()
    while True:
        if training_step <= optimizer.state_dict()['state'][0]['step']:
            mp.Process(target=evaluation, args=(global_model, global_icm, training_step, best, tb_log, args), daemon=True).start()
            training_step += 100
        time.sleep(0.5)

def get_args():
    parser = argparse.ArgumentParser(
    """Implementation of model described in the paper: Curiosity-driven Exploration by Self-supervised Prediction for Street Fighter""")
    parser.add_argument('--lr', type=float, default=1e-4) # 1e-4
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--sigma', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--lambda_', type=float, default=0.1, help='a3c loss coefficient')
    parser.add_argument('--eta', type=float, default=0.2, help='intrinsic coefficient')
    parser.add_argument('--beta', type=float, default=0.2, help='curiosity coefficient')

    parser.add_argument("--num_local_steps", type=int, default=50)
    # parser.add_argument("--num_global_steps", type=int, default=1e8)
    parser.add_argument("--num_processes", type=int, default=5)
    
    # parser.add_argument("--max_actions", type=int, default=500, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default=datetime.now().strftime("%b%d_%H-%M-%S") + "_train_mario")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    # parser.add_argument("--use_gpu", type=bool, default=True)
    # parser.add_argument("--use_shared_optim", type=bool, default=True)
    parser.add_argument("--use_icm", type=bool, default=False)
    parser.add_argument("--no_reward", type=bool, default=False)
    parser.add_argument("--use_sparse_reward", type=bool, default=True)
    parser.add_argument("--process_name", type=str, default=f'{getpass.getuser()}/Mario')
    parser.add_argument("--global_device", type=str, default="cpu")
    parser.add_argument("--local_device", type=str, default="cuda:0")
    parser.add_argument("--my_text", type=str, default="only_gpu, shared_optim")

    parser.add_argument("--use_detach", type=bool, default=False)
    parser.add_argument("--action_lstm", type=bool, default=False)
    parser.add_argument("--actions", type=str, default="")
    args = parser.parse_args()
    return args

def train(args):
    torch.manual_seed(123)

    global_model = ActorCritic(DummyEnv(args), args.action_lstm, args.actions).to(args.global_device)
    if args.use_icm:
        global_icm = ICM(DummyEnv(args)).to(args.global_device)
        optimizer = SharedAdam(list(global_model.parameters())+list(global_icm.parameters()), lr=args.lr)
    else:
        global_icm = None
        optimizer = SharedAdam(global_model.parameters(), lr=args.lr)
    
    # training
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=worker, args=(rank, global_model, global_icm, optimizer, args), daemon=True)
        p.start()
        processes.append(p)

    # evaluation
    # eval = mp.Process(target=loop_evaluation, args=(global_model, optimizer, args), daemon=True)
    # eval.start()
    loop_evaluation(global_model, global_icm, optimizer, args)
    
    for process in processes:
        process.join()
    eval.join()

    print('All processes joined')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = get_args()
    print(args)
    input('Check args and Press enter to continue')

    os.makedirs(args.saved_path, exist_ok=True)
    setproctitle(args.process_name + '+main')
    args.main_pid = os.getpid()
    writer = SummaryWriter(f'runs/{args.log_path}')
    writer.add_text('args', str(vars(args)))
    writer.close()

    train(args)