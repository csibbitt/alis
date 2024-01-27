from hashlib import md5
import pickle
from PIL import Image, ImageTk
import threading
import tkinter as tk

from run_model import get_rand_batch, get_batch_from_ws

def input_hasher(data):
  return md5(pickle.dumps(data)).hexdigest()[:7]

class SamplesWindow(tk.Toplevel):
  def __init__(self, container, ws=None):
    tk.Toplevel.__init__(self, container)

    self.app = container

    self.title("landscroller - Samples")

    self.grid_size_i = 5
    self.grid_size_j = 5
    self.populate_grid_i = 0
    self.populate_grid_j = 0
    self.sample_res = 128

    self.batch_size = 5

    self.main_frame = tk.Frame(self)
    self.main_frame.grid(column=0, row=0, sticky=('n', 's', 'e', 'w'))

    self.sample_labels = []

    for i in range(self.grid_size_i):
      self.sample_labels.append([])
      for j in range(self.grid_size_j):
        sample_frame = tk.Frame(self.main_frame, highlightbackground="black", highlightthickness=1)
        sample_frame.grid(row=i, column=j)

        pimg = ImageTk.PhotoImage(Image.open("startup.jpg").resize((self.sample_res,self.sample_res)))
        self.sample_labels[i].append(tk.Label(sample_frame, image=pimg))
        label = self.sample_labels[i][j]
        label.pimg = pimg
        label.grid(row=0, column=0, rowspan=3)
        label.status = tk.StringVar(value="...waiting...")

        button = tk.Button(sample_frame, text="+")
        button.grid(row=0, column=1)
        button = tk.Button(sample_frame, text="r", command=lambda i=i, j=j: self.resample(i,j))
        button.grid(row=1, column=1)
        button = tk.Button(sample_frame, text="v")
        button.grid(row=2, column=1)

        status_label = tk.Label(sample_frame, textvariable=label.status)
        status_label.grid()

    if ws is None:
      threading.Thread(name='populate_eval', target=get_rand_batch, args=(self.batch_size, self.populate_callback)).start()
    else:
      threading.Thread(name='populate_eval', target=get_batch_from_ws, args=(self.batch_size, self.populate_callback, ws)).start()


  def resample(self, i, j):
    img, ws = next(get_rand_batch(1))
    self.update_sample(i, j, img, ws)

  def update_sample(self, i, j, img, ws):
    label = self.sample_labels[i][j]
    label.pimg = ImageTk.PhotoImage(img.resize((self.sample_res, self.sample_res)))
    label.ws = ws
    label.status.set(input_hasher(ws))
    label.config(image=label.pimg)

  def populate_callback(self, img_tuples):
    for img, ws in img_tuples:
      self.update_sample(self.populate_grid_i, self.populate_grid_j, img, ws)
      self.populate_grid_j += 1
      if self.populate_grid_j >= self.grid_size_j:
          self.populate_grid_j = 0
          self.populate_grid_i += 1
      if self.populate_grid_i >= self.grid_size_i:
        return
    threading.Thread(name='populate_eval', target=get_rand_batch, args=(self.batch_size, self.populate_callback)).start()