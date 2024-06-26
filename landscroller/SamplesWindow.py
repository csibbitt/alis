from hashlib import md5
import pickle
from PIL import Image, ImageTk
import threading
import tkinter as tk

from run_model import get_rand_batch, get_batch_from_ws, generate_neighbor_ws

def input_hasher(data = None, pdata = None):
  if pdata is not None:
    return(md5(pdata)).hexdigest()[:7]
  elif data is not None:
    return md5(pickle.dumps(data.cpu().numpy())).hexdigest()[:7]


class SamplesWindow(tk.Toplevel):
  def __init__(self, container, ws=None):
    tk.Toplevel.__init__(self, container)

    self.app = container
    self.ws = ws

    if ws is None:
      self.title("landscroller - Random Samples")
    else:
      self.title(f"landscroller - Variations on {input_hasher(ws)}")

    self.grid_size_i = 5
    self.grid_size_j = 5
    self.grid_total = self.grid_size_i * self.grid_size_j
    self.populate_grid_i = 0
    self.populate_grid_j = 0
    self.sample_res = 128

    self.batch_size = 5

    self.top_frame = tk.Frame(self)
    self.top_frame.grid(column=0, row=0, sticky=('n', 's', 'e', 'w'))
    button = tk.Button(self.top_frame, text="Resample All", command=self.populate_samples)
    button.grid(row=0, column=0)

    main_frame = tk.Frame(self)
    main_frame.grid(column=0, row=1, sticky=('n', 's', 'e', 'w'))

    self.sample_labels = []

    for i in range(self.grid_size_i):
      self.sample_labels.append([])
      for j in range(self.grid_size_j):
        sample_frame = tk.Frame(main_frame, highlightbackground="black", highlightthickness=1)
        sample_frame.grid(row=i, column=j)

        pimg = ImageTk.PhotoImage(Image.open("startup.jpg").resize((self.sample_res,self.sample_res)))
        self.sample_labels[i].append(tk.Label(sample_frame, image=pimg))
        label = self.sample_labels[i][j]
        label.pimg = pimg
        label.grid(row=0, column=0, rowspan=3)
        label.status = tk.StringVar(value="...waiting...")

        button = tk.Button(sample_frame, text="+", command=lambda i=i, j=j: self.bookmark(i, j))
        button.grid(row=0, column=1)
        button = tk.Button(sample_frame, text="r", command=lambda i=i, j=j: self.resample(i, j))
        button.grid(row=1, column=1)
        button = tk.Button(sample_frame, text="v", command=lambda i=i, j=j: self.variations(i, j))
        button.grid(row=2, column=1)

        status_label = tk.Label(sample_frame, textvariable=label.status)
        status_label.grid()

    self.populate_samples()


  def populate_samples(self):
    self.populate_grid_i = 0
    self.populate_grid_j = 0
    for i in range(self.grid_size_i):
      for j in range(self.grid_size_j):
        self.sample_labels[i][j].status.set("...waiting...")
    if self.ws is None:
      self.target = get_rand_batch
      self.target_args = (self.batch_size, self.populate_callback)
    else:
      neighbor_ws = generate_neighbor_ws(self.grid_total, self.ws)
      self.target = get_batch_from_ws
      self.target_args = (neighbor_ws, self.populate_callback, self.batch_size)

    threading.Thread(name='populate_eval', target=self.target, args=self.target_args).start()

  def bookmark(self, i, j):
    import torch
    label = self.sample_labels[i][j]
    ws = label.ws.cpu()
    pickled_ws = pickle.dumps(ws)
    hash = input_hasher(pdata=pickled_ws)
    torch.save(label.ws, f'outputs/{hash}')
    self.app.control_window.add_bookmark(label.pimg, hash, ws)

  def variations(self, i, j):
    label = self.sample_labels[i][j]
    SamplesWindow(self.app, label.ws)

  def resample(self, i, j):
    if self.ws is not None:
      neighbor_ws = generate_neighbor_ws(1, self.ws)
      img, ws, dist = next(get_batch_from_ws(neighbor_ws))
    else:
      img, ws, dist = next(get_rand_batch(1))
    self.update_sample(i, j, img, ws, dist)

  def update_sample(self, i, j, img, ws, dist):
    label = self.sample_labels[i][j]
    label.pimg = ImageTk.PhotoImage(img.resize((self.sample_res, self.sample_res)))
    label.ws = ws
    label.status.set(f"{input_hasher(ws)} d({dist:.2f})")
    label.config(image=label.pimg)

  def populate_callback(self, img_tuples):
    for img, ws, dist in img_tuples:
      self.update_sample(self.populate_grid_i, self.populate_grid_j, img, ws, dist)
      self.populate_grid_j += 1
      if self.populate_grid_j >= self.grid_size_j:
          self.populate_grid_j = 0
          self.populate_grid_i += 1
      if self.populate_grid_i >= self.grid_size_i:
        return
    threading.Thread(name='populate_eval', target=self.target, args=self.target_args).start()
