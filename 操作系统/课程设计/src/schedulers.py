from typing import Union, Dict, List
from data_struture import *
import numpy as np


class MissingPageError(Exception):
    def __init__(self, page_id):
        self.page_id = page_id


class ProcessScheduler:
    process_list: List[PCB]
    ready_list: List[PCB]
    block_list: List[PCB]
    storage_list: List[PCB]
    working_process: Union[None, PCB]
    max_ready_size: int
    storage_scheduler: 'StorageScheduler'
    event: List[str]

    def __init__(self, processes: List[PCB], storage_scheduler: 'StorageScheduler', max_ready_size):
        self.storage_scheduler = storage_scheduler
        self.max_ready_size = max_ready_size
        processes.sort(key=lambda x: x.priority, reverse=True)
        self.ready_list = processes[:max_ready_size - 1]
        self.storage_list = processes[max_ready_size - 1:]
        self.working_process = None
        self.event = []
        self.block_list = []
        for process in self.ready_list:
            process.assign_storage(self.storage_scheduler.assign_storage(len(process.page_table)))

    def _preprocess(self):
        if self.working_process is None and len(self.ready_list):
            self.working_process = self.ready_list[0]
            self.working_process.state = 'running'
            del self.ready_list[0]

        if len(self.ready_list):
            if self.working_process.priority < self.ready_list[0].priority:
                self.working_process.state = 'ready'
                self.ready_list.append(self.working_process)
                self.working_process = self.ready_list[0]
                del self.ready_list[0]
                self.working_process.state = 'running'

    def run_one_clock(self, ):
        if self.working_process is None and len(self.block_list) == 0 and len(self.ready_list) == 0:
            self.event.append('所有进程已经运行完成!')
            return
        self._preprocess()
        if self.working_process:
            with self.working_process as (codes, memory, page_frame):
                self.storage_scheduler.memory.items = memory
                self.storage_scheduler.page_frame = page_frame
                code = codes[0]
                try:
                    self.run_code(code)
                except MissingPageError as e:
                    page_id = e.page_id
                    page = self.working_process.page_table[page_id]
                    self.storage_scheduler.missing_page(page)
                    self.run_code(code)
                self.working_process.cpu_time += 1
        self.update()

    def update(self):
        """内函数，用于在时钟周期结束后更新状态"""
        if self.working_process:
            self.working_process.clock()
            if self.working_process.cpu_time == self.working_process.cpu_need:
                self.event.append(f'进程{self.working_process.name}已完成！')
                storage_indices = [x.storage_num for x in self.working_process.page_table]
                self.storage_scheduler.return_storage(storage_indices)
                self.working_process = None
            elif self.working_process.state == 'block':
                self.block_list.append(self.working_process)
                self.working_process = None
        for process in self.ready_list:
            process.clock()

        for idx in range(len(self.block_list) - 1, -1, -1):
            process = self.block_list[idx]
            process.block_time += 1
            if process.block_time == process.block_need:
                self.ready_list.append(process)
                process.state = 'ready'
                del self.block_list[idx]

        if len(self.ready_list) < self.max_ready_size and len(self.storage_list):
            self.storage_list[0].assign_storage(
                self.storage_scheduler.assign_storage(len(self.storage_list[0].page_table)))
            self.ready_list.append(self.storage_list[0])
            del self.storage_list[0]

        self.ready_list.sort(key=lambda x: x.priority, reverse=True)

    def run_code(self, code: Code):
        if code.type in ['input', 'read', 'write']:
            if code.type == 'input':
                block_num = code.input.block_num
                shift = code.input.shift
            else:
                block_num = code.block_num
                shift = code.shift
            if not self.working_process.page_table[block_num].is_in_memory:
                raise MissingPageError(block_num)
            if code.type == 'read':
                item = self.storage_scheduler.read(block_num, shift)
                self.event.append(f'指令：读第{block_num}块页面，偏移量{shift}；结果{item}')
            if code.type == 'write':
                input = code.input
                self.storage_scheduler.write(block_num, shift, input)
                self.event.append(f'指令：写第{block_num}块页面，偏移量{shift}， 值{input}')
            if code.type == 'input':
                self.working_process.state = 'block'
                self.working_process.block_time = 0
                self.working_process.block_need = np.random.randint(2, 4)
                self.event.append(f'进程：{self.working_process.name} 遇到外部输入，阻塞时间{self.working_process.block_need}')
            if code.type == 'output':
                self.working_process.state = 'block'
                self.working_process.block_time = 0
                self.working_process.block_need = np.random.randint(2, 4)
                self.event.append(f'进程：{self.working_process.name} 遇到外部输入，阻塞时间{self.working_process.block_need}')
                self.event.append(f'指令：打印进程状态:\n {self.working_process.state}, {self.working_process.name}, \n'
                                  f'{self.working_process.priority}')


class StorageScheduler:
    memory: Memory
    storage: Storage
    page_frame: List[Page]

    def __init__(self, memory_size, storage_size):
        self.memory = Memory(memory_size)
        self.storage = Storage(storage_size)

    def assign_storage(self, size):
        indices = []
        for idx, avaliable in self.storage.is_avalible:
            if avaliable == 1:
                self.storage.is_avalible[idx] = 0
                indices.append(idx)

        return indices

    def return_storage(self, indices):
        for idx in indices:
            self.storage.is_avalible[idx] = 1

    def read(self, block_num, shift):
        for idx, page in enumerate(self.page_frame):
            if block_num == page.idx:
                break
        del self.page_frame[idx]
        self.page_frame.append(page)  # 将page置于栈顶
        return self.memory.items[page.memory_num][shift]

    def write(self, block_num, shift, input):
        for idx, page in enumerate(self.page_frame):
            if block_num == page.idx:
                break
        del self.page_frame[idx]
        self.page_frame.append(page)  # 将page置于栈顶
        self.memory.items[page.memory_num][shift] = input

    def missing_page(self, page: Page):
        # 缺页调度算法
        if len(self.page_frame) < self.memory.max_memory:
            self.page_frame.append(page)
            self.memory.items.append(self.storage.items[page.storage_num])  # 将外存调入内存
            page.memory_num = len(self.memory.items) - 1
            page.is_in_memory = True
        else:
            memory_num = self.page_frame[0].memory_num
            storage_num = self.page_frame[0].storage_num
            self.storage.items[storage_num] = self.memory.items[memory_num]  # 将内存写回外存
            self.page_frame[0].is_in_memory = False
            del self.page_frame[0]
            self.page_frame.append(page)
            self.memory.items[memory_num] = self.storage.items[page.storage_num]
            page.memory_num = memory_num
            page.is_in_memory = True  # 将空出来的内存给新置换的页面
