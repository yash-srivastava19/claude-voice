import sys
import time


def log(*parts):
    """One line in the daemon log, stamped to the millisecond (bench/turn_report.py reads them)."""
    now = time.time()
    print(time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}", *parts,
          file=sys.stderr, flush=True)
