import random
from glob import glob
import numpy as np
from PIL import Image
import os

import dnnlib
from scripts.legacy import load_network_pkl
import torch

import torchvision.transforms.functional as TVF
from training.networks import SynthesisLayer
from training.networks import PatchWiseSynthesisLayer

# ** For compiling bias_act and upfirdn2d_plugins
os.environ['CC'] = 'gcc-10'
os.environ['CXX'] = 'g++-10'

device = 'cuda'
G = None
ws_mean = None
truncation_factor = 0.7


def build_model():
  global G, ws_mean

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

  z_mean = torch.randn(1000, G.z_dim).to(device)
  ws_mean = G.mapping(z_mean, c=None, modes_idx=torch.zeros(1000).long().to(device)).mean(dim=0, keepdim=True)


def generate_neighbor_ws(num, ws, dist=0.8):
  global ws_mean

  ws_list = []

  if num == 25:  # Spatial layout for grid
    ws_matrix = [[None] * 5 for _ in range(5)]
    ws_matrix[2][2] = ws
    east_west = torch.randn(ws.shape).to(device)
    ws_matrix[2][1] = ws + dist/2 * east_west
    ws_matrix[2][0] = ws + dist * east_west
    ws_matrix[2][3] = ws + dist/2 * -east_west
    ws_matrix[2][4] = ws + dist * -east_west
    north_south = torch.randn(ws.shape).to(device)
    ws_matrix[0][2] = ws + dist/2 * north_south
    ws_matrix[1][2] = ws + dist * north_south
    ws_matrix[3][2] = ws + dist/2 * -north_south
    ws_matrix[4][2] = ws + dist * -north_south
    north_east = (east_west + north_south) * 0.5
    ws_matrix[1][3] = ws + dist/2 * north_east
    ws_matrix[0][4] = ws + dist * north_east
    ws_matrix[3][1] = ws + dist/2 * -north_east
    ws_matrix[4][0] = ws + dist * -north_east
    south_east = (east_west - north_south) * 0.5
    ws_matrix[3][3] = ws + dist/2 * south_east
    ws_matrix[4][4] = ws + dist * south_east
    ws_matrix[1][1] = ws + dist/2 * -south_east
    ws_matrix[0][0] = ws + dist * -south_east

    ws_matrix[0][1] = (ws_matrix[0][0] + ws_matrix[0][2]) * 0.5
    ws_matrix[0][3] = (ws_matrix[0][2] + ws_matrix[0][4]) * 0.5

    ws_matrix[1][0] = (ws_matrix[0][0] + ws_matrix[2][0]) * 0.5
    ws_matrix[1][4] = (ws_matrix[0][4] + ws_matrix[2][4]) * 0.5

    ws_matrix[3][0] = (ws_matrix[2][0] + ws_matrix[4][0]) * 0.5
    ws_matrix[3][4] = (ws_matrix[2][4] + ws_matrix[4][4]) * 0.5

    ws_matrix[4][1] = (ws_matrix[4][0] + ws_matrix[4][2]) * 0.5
    ws_matrix[4][3] = (ws_matrix[4][2] + ws_matrix[4][4]) * 0.5

    for i in range(5):
       for j in range(5):
          ws_list.append(ws_matrix[i][j])
  else:
    for _ in range(num):
      ws_list.append(ws + dist * torch.randn(ws.shape).to(device))

  out_ws = torch.stack(ws_list)
  out_ws = out_ws * truncation_factor + (1 - truncation_factor) * ws_mean

  if num == 25:
    out_ws[12] = ws

  return out_ws

def get_batch_from_ws(all_ws, img_callback=None, batch_size=None):

    if batch_size is None:
      batch_size = all_ws.shape[0]

    imgs = []
    dists = []
    for ws in torch.split(all_ws, batch_size):
      for w in ws:
        dists.append(torch.nn.PairwiseDistance()(w[0], ws_mean[0][0]).item())

      ws_context = torch.stack([ws, ws,], dim=1)

      preds = G.synthesis(ws, ws_context=ws_context, left_borders_idx=torch.zeros(batch_size, device=device).long() + 3, noise='const')
      for img in preds:
        imgs.append(TVF.to_pil_image(img.cpu().clamp(-1, 1) * 0.5 + 0.5))

    retval = zip(imgs, all_ws, dists)

    if img_callback is not None:
      return img_callback(retval)
    return retval

def get_rand_batch(batch_size = 4, img_callback=None):
    z = torch.randn(batch_size, G.z_dim).to(device)
    ws = G.mapping(z, c=None, modes_idx=torch.zeros(batch_size).long().to(device))
    ws = ws * truncation_factor + (1 - truncation_factor) * ws_mean
    return get_batch_from_ws(ws, img_callback)

build_model()

def main_session(buffer_size, img_callback, shuffle_flag, input_images, trunc_factor):

  while True:
    if len(input_images) == 0:

      #** Generate some input Z and map them to W = f(Z)
      num_frames = 5
      num_frames_per_w = G.synthesis_cfg.patchwise.w_coord_dist // 2
      num_ws = num_frames // num_frames_per_w + 1
      w_range = 2 * num_frames_per_w * G.synthesis_cfg.patchwise.grid_size
      zs = torch.randn(num_ws, G.z_dim).to(device)  # [3, 512]
      ws = G.mapping(zs, c=None, modes_idx=torch.zeros(1).long().to(device))  # [num_ws, 19, 512]

      # Truncating
      truncation_factor = 1 - (trunc_factor.get() / 100)
      ws = ws * truncation_factor + (1 - truncation_factor) * ws_mean

    else:
      #** Project input_images to Wl, Wc, Wr
      # ***** How?
      pass

    imgs = []
    shift = 0
    curr_w_idx = 1
    curr_ws = ws[curr_w_idx].unsqueeze(0)
    curr_ws_context = torch.stack([ws[curr_w_idx - 1].unsqueeze(0), ws[curr_w_idx + 1].unsqueeze(0)], dim=1)
    while not shuffle_flag.get():
      truncation_factor = 1 - (trunc_factor.get() / 100)

      if shift % w_range == 0 and shift > 0:
        new_z =  torch.randn(2, G.z_dim).to(device)
        new_ws = G.mapping(new_z, c=None, modes_idx=torch.zeros(1).long().to(device))
        new_ws = new_ws * truncation_factor + (1 - truncation_factor) * ws_mean
        ws = torch.cat((ws[1:], new_ws))
        curr_w_idx += 1
        curr_ws = ws[curr_w_idx].unsqueeze(0)
        curr_ws_context = torch.stack([ws[curr_w_idx - 1].unsqueeze(0), ws[curr_w_idx + 1].unsqueeze(0)], dim=1)

      curr_left_borders_idx = torch.zeros(1, device=zs.device).long() + (shift % w_range)  # [0-6] offset in the 3-ws grid, 3 is center image only

      img = G.synthesis(curr_ws, ws_context=curr_ws_context, left_borders_idx=curr_left_borders_idx, noise='const')

      imgs.append(img[0].cpu().clamp(-1, 1) * 0.5 + 0.5)

      if len(imgs) > buffer_size.get():
        imgs.pop(0)

      img_callback(TVF.to_pil_image(torch.cat(imgs, dim=2)))
      shift += 2
    shuffle_flag.set(False)

def main_session_mock(buffer_size, img_callback, shuffle_flag, input_images, trunc_factor):
    eval_width = 1024
    eval_height = 1024
    file_list = glob('mock_images/endless*.jpg')
    while True:
        random.shuffle(file_list)
        mockImage = Image.open(file_list[0])

        prediction_count = 0
        first = True
        while not shuffle_flag.get():
            if first:
                # The first frame is double-wide
                img_callback(mockImage.crop((0, 0, eval_width * 2, eval_height)))
                prediction_count = 1
                first = False
            else:
                prediction_count += 1
                start_column = 1024 * (prediction_count - buffer_size.get()) if prediction_count >= buffer_size.get() else 0
                img_callback(mockImage.crop((start_column, 0, prediction_count * eval_width, eval_height)))
        shuffle_flag.set(False)