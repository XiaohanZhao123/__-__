import yaml
from data_struture import *
from schedulers import ProcessScheduler, StorageScheduler
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from typing import List


def code_from_str(code_str: str):
    contents = code_str.strip(' ').split(' ')
    type = contents[0]
    if type == 'read':
        return Code(type=type, block_num=int(contents[1]), shift=int(contents[2]))
    if type == 'write':
        return Code(type=type, block_num=int(contents[1]), shift=int(contents[2]), input=int(contents[3]))
    if type == 'output':
        return Code(type=type,)
    if type == 'input':
        return Code(type=type, input=code_from_str(' '.join(contents[1:])))


def process_from_config(config: dict):
    process_list = []
    for process_dict in config.values():
        codes = [code_from_str(x) for x in process_dict['codes']]
        process_list.append(PCB(idx=process_dict['idx'],
                                codes=codes,
                                name=process_dict['name'],
                                state=process_dict['state'],
                                priority=process_dict['priority']))
    return process_list


class BackEnd(QThread):
    clock_finished: pyqtSignal(List[str])

    def __init__(self, config_path, message_queue):
        super(BackEnd, self).__init__()
        self.message_queue = message_queue
        with open(config_path, encoding='utf-8') as f:
            config = yaml.load(f)
            process_list = process_from_config(config['process'])
            self.storage_scheduler = StorageScheduler(**config['storage_scheduler'])
            self.process_scheduler = ProcessScheduler(process_list, self.storage_scheduler, **config['process_scheduler'])

    def run(self) -> None:
        while True:
            message = self.message_queue.get()
            if message is None:
                break
            # else:
            self.process_scheduler.run_one_clock()
            self.clock_finished.emit(self.process_scheduler.event)



