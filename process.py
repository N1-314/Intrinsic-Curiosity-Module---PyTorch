import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from mario import create_mario
from model import ActorCritic


def local_train(index, arg, global_model, optimizer):
    '''
    
    Parameters:
        index (int) : idx of each process
        arg : parsed arguments
        global_model : global network for A3C

    '''

    ### Setting
    seed = arg.seed + index
    color_env = False
    c_in = 3 if color_env else 1

    torch.manual_seed(seed)
    env = create_mario(color_env=color_env, save_video=False)
    num_actions = env.action_space.n
    local_model = ActorCritic(c_in, num_actions)
    local_model.train()

    state, *_ = env.reset()
    state = torch.from_numpy(state) ## <<< !!!
    cur_step, cur_experience = 0,0
    termin, trunc = False, False
    lstm_h, lstm_c = 0,0

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
            policy = F.softmax(logits)          # torch.Size([1, 12])
            log_policy = F.log_softmax(logits)  # torch.Size([1, 12])
            entropy = -(policy * log_policy).sum()

            action = Categorical(policy).sample().item()
            next_state, reward, termin, trunc, info = env.step(action)

            state = torch.from_numpy(next_state)

            collect_data['value'].append(value)
            collect_data['log_policy'].append(log_policy[0, action])
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
            # termin, trunc = False, False
            lstm_h, lstm_c = 0,0
        
        else:    # previous for-loop ends since `num_local_steps` step is processed.
            _, ret, _, _ = local_model(state, lstm_h, lstm_c)

        gae = 0
        actor_loss, critic_loss, entropy_loss = 0, 0, 0
        next_value = ret
        
        for value, log_policy, reward, entropy in list(zip(collect_data['value'], collect_data['log_policy'], collect_data['reward'], collect_data['entropy']))[::-1]:
            
            gae = gae * arg.gamma * arg.tau
            gae = gae + reward + arg.gamma * next_value - value.detach()
            actor_loss = actor_loss + log_policy * gae

            ret = ret * arg.gamma + reward
            critic_loss = critic_loss + F.mse_loss(ret, value)

            entropy_loss = entropy_loss + entropy

            next_value = value
        
        ### Backprop
        loss = -actor_loss + critic_loss - arg.sigma * entropy_loss
        optimizer.zero_grad()
        loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            # if global_param.grad is not None:
                # break
            # 이 if문을 켜면 업데이트가 처음 한번만 됨.. 왜지???

            # global_param._grad = local_param.grad
            global_param.grad = local_param.grad
        
        optimizer.step()

        if cur_experience % arg.save_interval == 0:
            if index==0:    # 대표로 하나만 출력
                print(f"Process {index}. Experience {cur_experience}. Loss {loss[0].item():.2f} (Actor: {actor_loss[0].item():.5f}, critic: {critic_loss:.5f}, Entropy: {entropy_loss.item():.5f})")
            torch.save(global_model.state_dict(), f"{arg.save_path}/a3c_mario")
            
        if cur_experience == arg.num_global_steps // arg.num_local_steps:
            print(f"이 프로세스는 종료되었습니다 #{index}")
            return
