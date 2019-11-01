# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import ast
import fnmatch
import itertools
import re
import unicodedata


def from_unicode(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')


def as_unicode(msg):
    return ''.join([i if ord(i) < 128 else ' ' for i in msg])


def search_and_replace(string, regex, replace):
    while True:
        match = re.search(regex, string)
        if match:
            string = string.replace(match.group(0), replace)
        else:
            break
    return string


def replace_index(string, index=0, replace=''):
    return '%s%s%s' % (string[:index], replace, string[index + 1:])


def filter_list_of_strings(lst, include='*', exclude=()):
    include = [include] if type(include) is str else include
    exclude = [exclude] if type(exclude) is str else exclude
    filtered_matches = list(itertools.chain.from_iterable([fnmatch.filter(lst, x) for x in include]))
    filtered_nomatch = list(itertools.chain.from_iterable([fnmatch.filter(lst, x) for x in exclude]))
    matched = list(set(filtered_matches) - set(filtered_nomatch))
    return matched


def as_bool_or_none(v):
    vstrip = v.strip()
    if vstrip.lower() == "none":
        return None
    if vstrip.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif vstrip.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def as_string_or_none(v):
    if v.strip().lower() == "none":
        return None
    return v


def as_int_or_none(v):
    if v.strip().lower() == "none":
        return None
    return int(v)


def as_dict_or_none(v):
    if v.strip().lower() == "none":
        return None
    return ast.literal_eval(v)


def as_list_or_none(v, astype):
    vstrip = v.strip()
    # check if supposed to be none
    if vstrip.lower() == "none":
        return None
    # check for brackets
    if vstrip[0] != '[' and vstrip[-1] != ']':
        v = vstrip
    else:
        v = vstrip[1:-1]
    # check for empty list
    if len(v.strip()) == 0:
        return []
    # cast elements as list of astype
    return [astype(x.strip()) for x in v.split(',')]


def as_intlist_or_none(v):
    return as_list_or_none(v, int)


def as_stringlist_or_none(v):
    return as_list_or_none(v, str)


def as_booleanlist_or_none(v):
    return as_list_or_none(v, as_bool_or_none)


def as_floatlist_or_none(v):
    return as_list_or_none(v, float)


def dict_as_string(value):
    sorted_keys = sorted(value.keys())

    def _x2str(x):
        if isinstance(x, str):
            return '\'%s\'' % str(x)
        else:
            return str(x)

    if len(sorted_keys) == 0:
        return '{}'
    else:
        dict_string = '{\'%s\': %s' % (sorted_keys[0], _x2str(value[sorted_keys[0]]))
        for i in range(1, len(sorted_keys)):
            dict_string += ", \'%s\': %s" % (sorted_keys[i], _x2str(value[sorted_keys[i]]))
    dict_string += "}"
    return dict_string
