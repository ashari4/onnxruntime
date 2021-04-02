import os
import sys
from contextlib import contextmanager


@contextmanager 
def suppress_output(suppress_stdout=True,suppress_stderr=True, f=os.devnull):
    """Supress output from being printed to stdout and stderr."""

    # stdout and stderr is written to f instead
    stdout = sys.stdout
    stderr = sys.stderr

    with open(f, 'w') as fo:
        try:
            if suppress_stdout:
                sys.stdout = fo
            if suppress_stderr:
                sys.stderr = fo
            yield
        finally:
            if suppress_stdout:
                sys.stdout = stdout
            if suppress_stderr:
                sys.stderr = stderr
