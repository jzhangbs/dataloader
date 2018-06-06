import multiprocessing as mp
import sys, os
import traceback as tb


class Loader:
    def __init__(self):
        self.num_worker = 8  # TODO
        self.queue_capacity = 50

        self.filename_queue = mp.Queue(self.queue_capacity)
        self.output_queue = mp.Queue(self.queue_capacity)

        self.main_process = mp.Process(target=self.main, name='loader')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        tb.print_exception(exc_type, exc_val, exc_tb)

    def start(self):
        self.main_process.start()

    def stop(self):
        self.main_process.terminate()

    def main(self):
        process = []
        for i in range(self.num_worker):
            p = mp.Process(target=self.worker, name='worker-%d' % i, kwargs={}, daemon=True)  # TODO
            process.append(p)
            p.start()

        filename_list = []  # TODO

        while True:
            for filename in filename_list:
                self.filename_queue.put(filename)

    def worker(self):
        while True:
            filename = self.filename_queue.get()

            output = None  # TODO

            self.output_queue.put(output)

    def get_batch(self, batch_size, aug=True):
        for i in range(batch_size):
            output = self.output_queue.get()
