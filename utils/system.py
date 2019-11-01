# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import fnmatch
import functools
import hashlib
import itertools
import os
import random
import socket
import subprocess
import sys
from datetime import datetime

import constants


# Taken from: https://gist.github.com/durden/0b93cfe4027761e17e69c48f9d5c4118
def get_size_of(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size_of(v, seen) for v in obj.values()])
        size += sum([get_size_of(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size_of(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size_of(i, seen) for i in obj])
    return size


def deterministic_indices(k, n, seed):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return sorted(indices[0:k])


def generate_sha224(name):
    return hashlib.sha224(name).hexdigest()


def cd_dotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), ".."))


def cd_dotdotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), "../.."))


def cd_dotdotdotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), "../../.."))


def datestr():
    now = datetime.now(constants.LOGGING_TIMEZONE)
    return '{}{:02}{:02}_{:02}{:02}'.format(
        now.year, now.month, now.day, now.hour, now.minute)


# Taken from: https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# THIS IS HACKY FOR NOW
def hostname():
    try:
        name = socket.gethostname()
        n = name.find('.')
        if n > 0:
            name = name[:n]
        return name
    except:
        return "N/A"


def git_hash():
    try:
        git_cmd = ["git", "rev-parse", "HEAD"]
        current_hash = subprocess.check_output(git_cmd).rstrip().decode('UTF-8')
        return current_hash
    except:
        return "N/A"


def screen_identifier():
    try:
        screen_cmd = ["bash", "-c", "echo $STY | cut -d\".\" -f 1-2;"]
        identifier = subprocess.check_output(screen_cmd).rstrip().decode('UTF-8')
        if identifier:
            return "screen:{}".format(identifier)
    except Exception:
        pass
    return ""


def fileparts(x):
    if os.path.isdir(x):
        slash_pos = x.rfind('/') + 1
        path = x[:slash_pos]
        subpath = x[slash_pos:]
        return path, subpath, None
    else:
        slash_pos = x.rfind('/') + 1
        path = x[:slash_pos]
        subpath = x[slash_pos:]
        basename, ext = os.path.splitext(subpath)
        return path, basename, ext


# Taken from: https://stackoverflow.com/questions/229186/os-walk-without-digging-into-directories-below
def walk_level(some_dir, level=0):
    """ allows to restrict hierarchy levels when walking """
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def get_filenames(directory, match='*.*', not_match=(), level=None):
    walker_function = os.walk
    if level is not None:
        walker_function = functools.partial(walk_level, level=level)
    if match is not None:
        if isinstance(match, str):
            match = [match]
    if not_match is not None:
        if isinstance(not_match, str):
            not_match = [not_match]

    result = []
    for dirpath, _, filenames in walker_function(directory):
        filtered_matches = list(itertools.chain.from_iterable(
            [fnmatch.filter(filenames, x) for x in match]))
        filtered_nomatch = list(itertools.chain.from_iterable(
            [fnmatch.filter(filenames, x) for x in not_match]))
        matched = list(set(filtered_matches) - set(filtered_nomatch))
        result += [os.path.join(dirpath, x) for x in matched]
    return result


def get_subdirs(directory, match='*', not_match=(), level=None):
    walker_function = os.walk
    if level is not None:
        walker_function = functools.partial(walk_level, level=level)
    if match is not None:
        if isinstance(match, str):
            match = [match]
    if not_match is not None:
        if isinstance(not_match, str):
            not_match = [not_match]
    result = []
    for dirpath, subdirs, filenames in walker_function(directory):
        filtered_matches = list(itertools.chain.from_iterable(
            [fnmatch.filter(subdirs, x) for x in match]))
        filtered_nomatch = list(itertools.chain.from_iterable(
            [fnmatch.filter(subdirs, x) for x in not_match]))
        matched = list(set(filtered_matches) - set(filtered_nomatch))
        result += [os.path.join(dirpath, x) for x in matched]
    return result
