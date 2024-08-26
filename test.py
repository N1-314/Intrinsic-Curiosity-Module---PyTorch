import torch
from mario_env import create_mario_env
from model import ActorCritic
from itertools import count

import cv2
import numpy as np
from PIL import Image
from captum.attr import LayerGradCam, LayerAttribution, visualization as viz

XAI = False

# torch.manual_seed(123)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

action_repeat = True
env = create_mario_env(test = not action_repeat)

model = ActorCritic(env)
# path = 'best_model.pt'
path = f'./best_model_3161.pt'
model.load_state_dict(torch.load(path, map_location=device))
model.eval()
if XAI:
    grad_cam = LayerGradCam(model.f, model.univers_head[6])
    masked_images = []

obs, info = env.reset()
hx = torch.zeros((1, 256), dtype=torch.float32).to(device)
cx = torch.zeros((1, 256), dtype=torch.float32).to(device)

# for local_step in range(1000):
for local_step in count():
    obs = torch.from_numpy(obs).contiguous().to(torch.get_default_dtype()).div(255).to(device)
    action, policy, hx, cx = model.get_action(obs, (hx, cx))
    obs, reward, terminated, truncated, info = env.step(action)
    if XAI:
        attr = attr = grad_cam.attribute(torch.from_numpy(obs).contiguous().to(torch.get_default_dtype()).div(255).to(device).unsqueeze(0), target=action)
        # upsamp_attr = LayerAttribution.interpolate(attr, (240, 256))
        upsamp_attr = LayerAttribution.interpolate(attr, (240, 256), 'bilinear')
        upsamp_attr = upsamp_attr[0].detach().cpu().permute(1,2,0).numpy()
        
        if False:
            positive_attr = ( upsamp_attr > 0 ) * upsamp_attr
            percentile = 98
            sorted_vals = np.sort(positive_attr.flatten())
            cum_sums = np.cumsum(sorted_vals)
            threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
            threshold = sorted_vals[threshold_id]

            if threshold < 1e-5:
                threshold = 1e-5
            norm_attr = np.clip(positive_attr / threshold, -1, 1)
            masked_image = np.clip((env.record_frames[-1] * norm_attr).astype(np.uint8), 0, 255)
        else:
            upsamp_attr = np.clip(upsamp_attr, 0, 1)
            norm_attr = upsamp_attr / np.max(upsamp_attr)
            norm_attr = cv2.applyColorMap(np.uint8(norm_attr*255), cv2.COLORMAP_JET)
            masked_image = cv2.addWeighted(env.record_frames[-1], 0.6, norm_attr, 0.4, 0)
        
        masked_images.append(Image.fromarray(masked_image))

    print(f'local_step: {local_step}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, x_pos: {info["x_pos"]}')
    if terminated or truncated:
        break

print(f'info: {info}')
env.save_video(f"{path.split('/')[-1].split('.')[0]}.mp4", fps=30)
if XAI:
    masked_images[0].save(f"{path.split('/')[-1].split('.')[0]}_xai.gif", format='GIF', save_all=True, append_images=masked_images[1:], duration=50, loop=0)


# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# log_name = 'Aug18_03-05-33' + '_train_mario'
# writer = SummaryWriter(f'runs/{log_name}')
# frames = np.stack(env.record_frames, axis=0)
# frames = frames.transpose(0, 3, 1, 2)
# frames = np.expand_dims(frames, axis=0)
# writer.add_video('video/test', frames, training_step, fps=30)