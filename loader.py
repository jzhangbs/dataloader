import multiprocessing as mp
import multiprocessing.managers as managers
import random
import numpy as np
import atexit
import time
import argparse
import os
import signal

from process import gen_sample_list, split_sample_list, read


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_batch(self, batch_size, phase):
    outputs = []
    for i in range(batch_size):
        outputs.append(self.output_queue[phase].get())
    output_np = []
    for image_list in zip(*outputs):
        output_np.append(np.stack(image_list, axis=0))
    return output_np


class Loader:
    def __init__(self, data_root, dataset_name, val_ratio, preproc_args, sys_args, server=False):
        self.subproc = mp.Queue()
        atexit.register(self.kill_children)

        self.num_worker = sys_args.num_worker
        self.queue_capacity = sys_args.queue_capacity
        if server: self.port = sys_args.port
        if server: self.auth = sys_args.auth

        self.data_root = data_root
        self.dataset_name = dataset_name
        self.val_ratio = val_ratio
        self.preproc_args = preproc_args

        filename_list = gen_sample_list(self.data_root, self.dataset_name)
        training_list, validation_list = split_sample_list(filename_list, self.val_ratio, seed=0)
        self.training_sample_size = len(training_list)
        self.validation_sample_size = len(validation_list)

        self.filename_queue = {}
        self.output_queue = {}
        self.filename_queue['train'] = mp.Queue(self.queue_capacity)
        self.output_queue['train'] = mp.Queue(self.queue_capacity)
        self.train_process = mp.Process(target=self.main, name='train_loader', kwargs={
            'filename_list': training_list,
            'phase': 'train'
        })
        self.filename_queue['val'] = mp.Queue(self.queue_capacity)
        self.output_queue['val'] = mp.Queue(self.queue_capacity)
        self.val_process = mp.Process(target=self.main, name='val_loader', kwargs={
            'filename_list': validation_list,
            'phase': 'val'
        })

        if server:
            managers.SyncManager.register('global_namespace', callable=lambda: {
                'training_sample_size': self.training_sample_size,
                'validation_sample_size': self.validation_sample_size
            })
            managers.SyncManager.register('train_output_queue', callable=lambda: self.output_queue['train'])
            managers.SyncManager.register('val_output_queue', callable=lambda: self.output_queue['val'])
            self.manager = managers.SyncManager(('', self.port), self.auth)
            self.manager.start()

    def kill_children(self):
        while not self.subproc.empty():
            pid = self.subproc.get_nowait()
            os.kill(pid, signal.SIGTERM)

    def start(self):
        self.train_process.start()
        self.subproc.put(self.train_process.pid)
        self.val_process.start()
        self.subproc.put(self.val_process.pid)
        return self

    def stop(self):
        pass

    def main(self, filename_list, phase):
        self.phase = phase

        process = []
        for i in range(self.num_worker):
            p = mp.Process(target=self.worker, name=phase+'_worker-%d' % i, kwargs={})
            process.append(p)
            p.start()
            self.subproc.put(p.pid)

        while True:
            if phase == 'train':
                random.shuffle(filename_list)
            for filename in filename_list:
                self.filename_queue[phase].put(filename)

    def worker(self):
        while True:
            filename = self.filename_queue[self.phase].get()
            images = read(filename, dotdict(
                dataset_name=self.dataset_name,
                phase=self.phase,
                preproc_args=self.preproc_args
            ))
            self.output_queue[self.phase].put(images)

    def get_batch(self, batch_size, phase):
        return get_batch(self, batch_size, phase)


class LoaderClient:

    def __init__(self, addr, port, auth):
        self.addr = addr
        self.port = port
        self.auth = auth

        managers.SyncManager.register('train_output_queue')
        managers.SyncManager.register('val_output_queue')
        managers.SyncManager.register('global_namespace')
        self.manager = managers.SyncManager((self.addr, self.port), self.auth)
        self.manager.connect()
        self.training_sample_size = self.manager.global_namespace().get('training_sample_size')
        self.validation_sample_size = self.manager.global_namespace().get('validation_sample_size')
        self.output_queue = {
            'train': self.manager.train_output_queue(),
            'val': self.manager.val_output_queue()
        }

    def get_batch(self, batch_size, phase):
        return get_batch(self, batch_size, phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--dataset_name', type=str, default='monkaa,flyingthings3d,driving')
    parser.add_argument('--val_ratio', type=float, default=0.1)

    parser.add_argument('--crop_width', type=int, default=512)
    parser.add_argument('--crop_height', type=int, default=256)

    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--queue_capacity', type=int, default=10)
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--auth', type=str, default='auth')
    args = parser.parse_args()

    Loader(args.data_root, args.dataset_name.split(','), args.val_ratio,
           dotdict(crop_width=args.crop_width, crop_height=args.crop_height),
           dotdict(num_worker=args.num_worker, queue_capacity=args.queue_capacity, port=args.port, auth=args.auth),
           server=True).start()
    while True:
        time.sleep(5)
