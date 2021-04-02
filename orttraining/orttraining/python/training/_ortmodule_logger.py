import os
import sys
from contextlib import contextmanager
from enum import IntEnum


class LogLevel(IntEnum):
    VERBOSE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4


@contextmanager 
def suppress_os_stream_output(suppress_stdout=True,suppress_stderr=True, log_level=LogLevel.WARNING):
    """Supress output from being printed to stdout and stderr if log_level is WARNING or higher."""

    # stdout and stderr is written to devnull instead
    stdout = sys.stdout
    stderr = sys.stderr

    with open(os.devnull, 'w') as fo:
        try:
            if suppress_stdout and log_level >= LogLevel.WARNING:
                sys.stdout = fo
            if suppress_stderr and log_level >= LogLevel.WARNING:
                sys.stderr = fo
            yield
        finally:
            if suppress_stdout:
                sys.stdout = stdout
            if suppress_stderr:
                sys.stderr = stderr
