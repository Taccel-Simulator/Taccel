import numpy as np
import colorful
import os
from icecream import ic
from enum import Enum
from functools import wraps

class LoggingLevel(Enum):
    DEBUG = (0,)
    INFO = (1,)
    IMPORTANT_INFO = (2,)
    WARN = (3,)
    ERROR = (4,)
    NONE = (999,)
_current_logging_level = LoggingLevel.INFO

def set_logging_level(level: LoggingLevel):
    global _current_logging_level
    _current_logging_level = level

def _logging_level():
    return int(os.environ['WTF_IPC_LOG_LEVEL']) if 'WTF_IPC_LOG_LEVEL' in os.environ else _current_logging_level.value[0]

def debug(*args, **kwargs):
    if _logging_level() <= LoggingLevel.DEBUG.value[0]:
        print(colorful.cyan('[DEBUG]'), *args, **kwargs)

def info(*args, **kwargs):
    if _logging_level() <= LoggingLevel.INFO.value[0]:
        np.set_printoptions(suppress=True)
        print(colorful.green('[INFO]'), *args, **kwargs)

def important_info(*args, **kwargs):
    if _logging_level() <= LoggingLevel.IMPORTANT_INFO.value[0]:
        np.set_printoptions(suppress=True)
        print(colorful.bold_green('[INFO]'), *args, **kwargs)

def warn(text: str):
    if _logging_level() <= LoggingLevel.WARN.value[0]:
        print(colorful.bold_coral(f'[WARNING]\t{text}'))

def error(text: str):
    if _logging_level() <= LoggingLevel.ERROR.value[0]:
        print(colorful.bold_red(f'[ERROR]\t{text}'))

def separate(char: str='-'):
    print(colorful.bold_black(char * 80))
ic.configureOutput(outputFunction=debug)