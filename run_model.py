import random
from glob import glob
import numpy as np
from PIL import Image
from hashlib import md5
import os
import pickle

import dnnlib
from scripts.legacy import load_network_pkl
import torch

import torchvision.transforms.functional as TVF
from training.networks import SynthesisLayer
from training.networks import PatchWiseSynthesisLayer

# ** For compiling bias_act and upfirdn2d_plugins
os.environ['CC'] = 'gcc-10'
os.environ['CXX'] = 'g++-10'

def input_hasher(data):
  return md5(pickle.dumps(data)).hexdigest()[:7]

device = 'cuda'
G = None

def buildModel():
  global G

  torch.set_grad_enabled(False)

  np.float = float # ***** Monkey patch to fix loading of old snapshot.

  network_pkl = 'file:///home/chris/projects/ml/alis/lhq1024-snapshot.pkl' # ***** Replace with new full model save and simpler loader

  with dnnlib.util.open_url(network_pkl) as f:
    G = load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.eval()
    G.progressive_growing_update(100000)  #** Thought this was disabled?

  for res in G.synthesis.block_resolutions:
    block = getattr(G.synthesis, f'b{res}')
    if hasattr(block, 'conv0'):
        block.conv0.use_noise = False
    block.conv1.use_noise = False


def mainSession(buffer_size, img_callback, shuffle_flag, input_images, seed_hash, mix_strength):

  buildModel()

  while True:
    if len(input_images) == 0:

      #** Generate some input Z and map them to W = f(Z)
      num_frames = 5
      num_frames_per_w = G.synthesis_cfg.patchwise.w_coord_dist // 2
      num_ws = num_frames // num_frames_per_w + 1
      w_range = 2 * num_frames_per_w * G.synthesis_cfg.patchwise.grid_size
      zs = torch.randn(num_ws, G.z_dim).to(device)  # [3, 512]
      mode_idx = 0  #** What is this about?
      modes_idx = (torch.ones(1, device=zs.device).repeat(num_ws).float() * mode_idx).long()
      ws = G.mapping(zs, c=None, modes_idx=modes_idx)  # [num_ws, 19, 512]
      seed_hash.set('t_' + input_hasher(ws))
    else:
      #** Project input_images to Wl, Wc, Wr
      # ***** How?
      pass

    imgs = []
    shift = 0
    curr_w_idx = 1
    curr_ws = ws[curr_w_idx].unsqueeze(0)
    curr_ws_context = torch.stack([ws[curr_w_idx - 1].unsqueeze(0), ws[curr_w_idx + 1].unsqueeze(0)], dim=1)
    while not shuffle_flag.get() : #***** TEMP - just trying to get 5 patches
 
      if shift % w_range == 0:
        new_z =  torch.randn(2, G.z_dim).to(device) # ** Seed stability could be implemented as new_ws =  curr_ws + ss * rand?
        new_ws = G.mapping(new_z, c=None, modes_idx=torch.zeros(1).long().to(device))
        ws = torch.cat((ws, new_ws))
        # ** Should drop oldest ws
        curr_w_idx += 2
        curr_ws = ws[curr_w_idx].unsqueeze(0)
        curr_ws_context = torch.stack([ws[curr_w_idx - 1].unsqueeze(0), ws[curr_w_idx + 1].unsqueeze(0)], dim=1)

      curr_left_borders_idx = torch.zeros(1, device=zs.device).long() + (shift % w_range)  #** What does this do?

      img = G.synthesis(curr_ws, ws_context=curr_ws_context, left_borders_idx=curr_left_borders_idx, noise='const')

      imgs.append(img[0].cpu().clamp(-1, 1) * 0.5 + 0.5)

      if len(imgs) > buffer_size.get():
        imgs.pop(0)

      img_callback(TVF.to_pil_image(torch.cat(imgs, dim=2)))
      shift += 2
    shuffle_flag.set(False)

def mainSessionMock(buffer_size, img_callback, shuffle_flag, input_images, seed_hash, mix_strength):
    eval_width = 128
    eval_height = 128
    file_list = glob('mock_images/endless*.jpg')
    while True:
        random.shuffle(file_list)
        mockImage = Image.open(file_list[0])
        seed_hash.set('m_' + input_hasher(mockImage))

        prediction_count = 0
        first = True
        while not shuffle_flag.get():
            if first:
                # The first frame is double-wide
                img_callback(mockImage.crop((0, 0, eval_width * 2, eval_height)))
                prediction_count = 2
                first = False
            else:
                prediction_count += 1
                start_column = 128 * (prediction_count - buffer_size.get()) if prediction_count >= buffer_size.get() else 0
                img_callback(mockImage.crop((start_column, 0, prediction_count * eval_width, eval_height)))
        shuffle_flag.set(False)