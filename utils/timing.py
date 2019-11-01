# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import collections
import time

import torch

from utils.moving_averages import ExponentialMovingAverage

__registry = collections.OrderedDict()
__tic_name = None
__tic_time = None


def tic(name):
    global __tic_name, __tic_time
    if __tic_name is None:
        __tic_name = name
        __tic_time = time.perf_counter()
    else:
        __tic_time = toc()
        __tic_name = name


def toc(alpha=0.7, sync=False):
    global __tic_name, __tic_time
    if __tic_name is not None:
        if sync:
            torch.cuda.synchronize()
        current_time = time.perf_counter()
        diff_time = current_time - __tic_time
        avg = __registry[__tic_name] if __tic_name in __registry.keys() else ExponentialMovingAverage(alpha=alpha)
        avg.add_value(diff_time)
        __registry[__tic_name] = avg
        __tic_name = None
        __tic_time = None
        return current_time
    return None


def gputoc(alpha=0.7):
    return toc(alpha=alpha, sync=True)


def _format_timing(secs):
    ms = round(secs * 1000)
    return '%ims' % ms


def stats():
    # stop last tic, if measuring..
    toc()

    # accumulate results
    values = [
        (k, _format_timing(v.mean()))
        for k, v in __registry.items()
    ]
    return collections.OrderedDict(values)
