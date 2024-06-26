#!/usr/bin/env python
import argparse
import threading
import run_model as Eval
from landscroller import App, SamplesWindow, ControlWindow, DisplayWindow

import os
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infinite landscape scroller')
    parser.add_argument('-mock', action='store_true', default=False)

    args = parser.parse_args()

    app = App()
    DisplayWindow(app)
    ControlWindow(app)

    target =  Eval.main_session if not args.mock else Eval.main_session_mock
    threading.Thread(name='eval', target=target, args=(app.buffer_size, app.display_window.eval_callback, app.shuffle_flag, app.trunc_factor, app.mix_ws),  daemon=True).start()

    app.mainloop()