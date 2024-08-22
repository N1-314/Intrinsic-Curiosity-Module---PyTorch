import torch
from mario_env import create_mario_env
from model import ActorCritic
from itertools import count

# torch.manual_seed(123)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

action_repeat = False
env = create_mario_env(test = not action_repeat)

model = ActorCritic(env)
# path = 'best_model.pt'
path = f'trained_models/best_model_3161.pt'
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

obs, info = env.reset()
hx = torch.zeros((1, 256), dtype=torch.float32).to(device)
cx = torch.zeros((1, 256), dtype=torch.float32).to(device)

# for local_step in range(1000):
for local_step in count():
    obs = torch.from_numpy(obs).contiguous().to(torch.get_default_dtype()).div(255).to(device)
    action, policy, hx, cx = model.get_action(obs, (hx, cx))
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'local_step: {local_step}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, x_pos: {info["x_pos"]}')
    if terminated or truncated:
        break

print(f'info: {info}')
env.save_video(f"{path.split('/')[-1].split('.')[0]}.mp4", fps=30)

# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# log_name = 'Aug18_03-05-33' + '_train_mario'
# writer = SummaryWriter(f'runs/{log_name}')
# frames = np.stack(env.record_frames, axis=0)
# frames = frames.transpose(0, 3, 1, 2)
# frames = np.expand_dims(frames, axis=0)
# writer.add_video('video/test', frames, training_step, fps=30)