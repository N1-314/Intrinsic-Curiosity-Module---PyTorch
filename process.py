import os

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from mario import create_mario
from model import ActorCritic


def local_train(index:int, timestamp:str, arg, global_model, optimizer, concat:dict=None):
    '''
    
    Parameters:
        index (int) : idx of each process
        arg : parsed arguments
        global_model : global network for A3C

    '''

    ### Setting
    seed = arg.seed + index
    if not os.path.isdir(f"{arg.save_path}/a3c_mario_{timestamp}"):
        os.makedirs(f"{arg.save_path}/a3c_mario_{timestamp}")

    torch.manual_seed(seed)
    writer = SummaryWriter(arg.log_path + timestamp)

    num_stack = 4
    env = create_mario(save_video=False, num_stack=num_stack)
    num_actions = env.action_space.n
    local_model = ActorCritic(num_stack, num_actions)
    local_model.train()

    state, *_ = env.reset()
    state = torch.from_numpy(state)
    cur_step, cur_experience = 0,0
    cnt_episode = 0
    termin, trunc = False, False
    lstm_h, lstm_c = 0,0

    if concat is not None:
        cur_experience = concat['cur_experience']

    ### Train loop
    while True:
        cur_experience += 1
        local_model.load_state_dict(global_model.state_dict())
        
        collect_data = {'value':[], 'log_policy':[], 'reward':[], 'entropy':[]}
        
        ### Collect data

        for _ in range(arg.num_local_steps):
            cur_step += 1
            logits, value, lstm_h, lstm_c = local_model(state, lstm_h, lstm_c)  # logits.shape == torch.Size([1, 12])
            lstm_h, lstm_c = lstm_h.detach(), lstm_c.detach()
            policy = F.softmax(logits, dim=2)
            log_policy = F.log_softmax(logits, dim=2)
            entropy = -(policy * log_policy).sum(2, keepdim=True)

            action = Categorical(policy).sample().item()
            next_state, reward, termin, trunc, info = env.step(action)

            state = torch.from_numpy(next_state)

            collect_data['value'].append(value)
            collect_data['log_policy'].append(log_policy[0, 0, action])
            collect_data['reward'].append(reward)
            collect_data['entropy'].append(entropy)

            if cur_step > arg.num_global_steps:
                trunc = True
            if termin or trunc:
                break

        ### Calculate loss

        if termin or trunc: # previous for-loop ends since the episode is finished.
            ret = torch.zeros((1, 1), dtype=torch.float)
            next_state, *_ = env.reset()
            cur_step = 0
            cnt_episode += 1
            # termin, trunc = False, False
            lstm_h, lstm_c = 0,0
        
        else:    # previous for-loop ends since `num_local_steps` step is processed.
            _, ret, _, _ = local_model(state, lstm_h, lstm_c)

        gae = 0
        actor_loss, critic_loss, entropy_loss = 0, 0, 0
        next_value = ret
        
        for value, log_policy, reward, entropy in list(zip(collect_data['value'], collect_data['log_policy'], collect_data['reward'], collect_data['entropy']))[::-1]:
            
            with torch.no_grad():
                gae = gae * arg.gamma * arg.tau
                gae = gae + reward + arg.gamma * next_value - value
            
            actor_loss = actor_loss + log_policy * gae

            ret = ret * arg.gamma + reward
            critic_loss = critic_loss + F.mse_loss(ret, value)

            entropy_loss = entropy_loss + entropy

            next_value = value
        
        ### Backprop
        loss = -actor_loss + critic_loss - arg.sigma * entropy_loss

        writer.add_scalar(f"Process_{index}/Loss", loss, cur_experience)
        writer.add_scalar(f"Process_{index}/actor", actor_loss, cur_experience)
        writer.add_scalar(f"Process_{index}/critic", critic_loss, cur_experience)
        writer.add_scalar(f"Process_{index}/entropy", entropy_loss, cur_experience)
        writer.add_scalar(f"Process_{index}/cnt_episode", cnt_episode, cur_experience)

        optimizer.zero_grad()
        local_model.zero_grad()
        loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break

            # global_param._grad = local_param.grad
            global_param.grad = local_param.grad
        
        optimizer.step()

        if cur_experience % arg.save_interval == 0:
            if index==0:
                # 대표로 하나만 출력
                print(f"Process {index}. Experience {cur_experience}. Loss {loss[0].item():.2f} (Actor: {actor_loss[0].item():.5f}, critic: {critic_loss:.5f}, Entropy: {entropy_loss.item():.5f})")
                
                # save for real-time test
                torch.save(global_model.state_dict(), f"{arg.save_path}/a3c_mario")

                # Checkpoint
                torch.save({
                    "global_model_state_dict" : global_model.state_dict(),
                    "optimizer_state_dict" : optimizer.state_dict(),
                    }, f=f"{arg.save_path}/a3c_mario_{timestamp}/{cur_experience}")

        if cur_experience == arg.num_global_steps // arg.num_local_steps:
            print(f"Process #{index} has been ended.")
            return
