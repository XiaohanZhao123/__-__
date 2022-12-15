from typing import Union, List, Tuple, Dict
from dataclasses import dataclass

__all__ = ['PCB', 'Page', 'Memory', 'Storage', 'Code']


class PCB:
    idx: int  # 内部进程编号
    name: str  # 外部名称
    page_frame: List['Page']  # 上下文： LRU堆栈状态
    state: str  # 进程状态：运行、就绪、阻塞
    priority: int # 动态优先级
    page_table: Union[List['Page'], None]  # 页表
    codes: List['Code']  # 指令集
    pc: int  # 上下文：pc 指针
    memory: List[List[int]]  # 上下文：内存状态
    cpu_time: int
    cpu_need: int
    block_time: int
    block_need: int

    def __init__(self, idx, name, state, codes: List['Code'], priority, cpu_need=None):
        assert state in ['ready', 'running', 'block'], '初始化进程状态非法'
        self.state = state
        self.idx = idx
        self.name = name
        page_table_size = max(*[x.return_block_num() for x in codes]) + 1
        self.page_table = [Page(x) for x in range(page_table_size)]
        self.memory = []
        self.page_frame = []
        self.codes = codes
        self.pc = -1
        self.block_need = -1
        self.block_time = 0
        self.priority = priority
        if not cpu_need:
            self.cpu_need = len(codes)
        self.cpu_time = 0

    def clock(self):
        """内函数，修改动态优先级"""
        if self.state == 'ready':
            self.priority += 1
        elif self.state == 'running':
            self.priority -= 3

    def assign_storage(self, block_numbers: List[int]):
        for page, storage_number in zip(self.page_table, block_numbers):
            page.storage_num = storage_number


    def __enter__(self):
        self.pc += 1
        return self.codes[self.pc], self.memory, self.page_frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


@dataclass
class Code:
    type: str  # 指令类型
    block_num: Union[int, None] = None  # 指令逻辑页号
    shift: Union[int, None] = None  # 页内偏移
    input: Union['Code', int, None] = None  # 指令输入值

    def __init__(self, type, block_num=None, shift=None, input=None):
        if type == 'output':
            assert block_num is None and shift is None and input is None, '输入指令output 格式不合法'
            self.type = type
        elif type == 'read':
            assert input is None, '输入read 指令格式不合法'
            self.type = type
            self.block_num = block_num
            self.shift = shift
        elif type == 'write':
            assert input is not None, '输入指令需要指定输入的二进制位'
            self.type = type
            self.block_num = block_num
            self.shift = shift
            self.input = input
        elif type == 'input':
            assert block_num is None and self.shift is None, 'input 指令位外部输入指令，不需要指定页号和偏移'
            assert isinstance(input, Code), 'input 指令的input 值必须是指令'
            self.type = type
            self.input = input

    def return_block_num(self):
        if self.type in ['read', 'write']:
            return self.block_num
        elif self.type == 'input':
            return self.input.block_num
        else:
            return -1

class Page:
    idx: int  # 页号
    memory_num: Union[int, None]  # 内存地址
    storage_num: Union[int, None]  # 外存地址
    time: int  # LRU算法计数器
    is_in_memory: bool  # 是否在内存中
    is_modified: bool  # 是否被修改，若替换算法中被修改，则需写回内存

    def __init__(self, idx, ):
        self.idx = idx
        self.memory_num = None
        self.storage_num = None
        self.time = 0
        self.is_in_memory = False
        self.is_modified = False

    def __repr__(self):
        return f'idx: {self.idx}, memory_num: {self.memory_num}, storage_num: {self.storage_num}'


class Blocks:
    items: List[List[int]]  # items[x][y] x为块号， y为块内偏移

    def __init__(self, items):
        self.items = items


class Memory(Blocks):
    max_memory: int

    def __init__(self, max_memory: int):
        super(Memory, self).__init__([])
        self.max_memory = max_memory


class Storage(Blocks):
    max_storage: int
    is_avalible: List[int]

    def __init__(self, max_storage):
        super(Storage, self).__init__([[0 for _ in range(16)] for __ in range(max_storage)])
        self.max_storage = max_storage
        self.is_avalible = [1 for _ in range(max_storage)]

