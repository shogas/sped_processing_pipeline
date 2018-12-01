from datetime import datetime
import os
import threading
import time

import winstats

def current_timestamp():
    return '{0:%Y}{0:%m}{0:%d}_{0:%H}_{0:%M}_{0:%S}_{0:%f}'.format(datetime.now())


def python_process_instance_number(pid):
    instance_number = 0
    while True:
        try:
            python_pid = winstats.get_perf_data(r'\Process(python{})\ID Process'.format(
                '' if instance_number == 0 else '#{}'.format(instance_number)),
                fmts='long')[0]
            if pid == python_pid:
                return '' if instance_number == 0 else '#{}'.format(instance_number)
            instance_number += 1
        except OSError:
            print('Did not find PID after {} attempts'.format(instance_number))
            exit(0)


class LogThread(threading.Thread):
    def __init__(self, log_filename):
        super().__init__()
        self.log_filename = log_filename
        self.stopper = threading.Event()
        self.instance_number = python_process_instance_number(os.getpid())


    def run(self):
        with open(self.log_filename, 'w') as log_file:
            while not self.stopper.is_set():
                private_bytes = winstats.get_perf_data(
                    r'\Process(python{})\Private Bytes'.format(self.instance_number),
                   fmts='long')[0]
                log_file.write('{}\t{}\n'.format(current_timestamp(), private_bytes))
                time.sleep(0.1)


    def stop(self):
        self.stopper.set()


def time_log_call(result_file, func, *args):
    time_start = time.process_time()
    func()
    time_end = time.process_time()

    time_elapsed = time_end - time_start
    result_file.write(('{}\t'*(1 + len(args) + 1)).format(
        current_timestamp(),
        *args,
        time_elapsed))

    return time_elapsed
