# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import logging
import os
import re
import sys

import tqdm

import constants
from utils import strings
from utils import summary
from utils import telegram


def get_default_logging_format(colorize=False, brackets=False):
    color = constants.COLOR_TIMESTAMP if colorize else ''
    reset = constants.COLOR_RESET if colorize else ''
    if brackets:
        result = "{}[%(asctime)s]{} %(message)s".format(color, reset)
    else:
        result = "{}%(asctime)s{} %(message)s".format(color, reset)
    return result


def log_module_info(module):
    lines = module.__str__().split("\n")
    for line in lines:
        logging.info(line)


class LogbookFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._re = re.compile(r"\033\[[0-9]+m")

    def remove_colors_from_msg(self, msg):
        msg = re.sub(self._re, "", msg)
        return msg

    def format(self, record=None):
        record.msg = self.remove_colors_from_msg(record.msg)
        record.msg = strings.as_unicode(record.msg)
        return super().format(record)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record=None):
        indent = sys.modules[__name__].global_indent
        record.msg = " " * indent + record.msg
        return super().format(record)


class SkipLogbookFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.LOGBOOK


# -----------------------------------------------------------------
# Subclass tqdm to achieve two things:
#   1) Output the progress bar into the logbook.
#   2) Remove the comma before {postfix} because it's annoying.
# -----------------------------------------------------------------
class TqdmToLogger(tqdm.tqdm):
    def __init__(self, iterable=None, desc=None, total=None, leave=True,
                 file=None, ncols=None, mininterval=0.1,
                 maxinterval=10.0, miniters=None, ascii=None, disable=False,
                 unit='it', unit_scale=False, dynamic_ncols=False,
                 smoothing=0.3, bar_format=None, initial=0, position=None,
                 postfix=None,
                 logging_on_close=True,
                 logging_on_update=False,
                 track_eta=False):

        super().__init__(
            iterable=iterable, desc=desc, total=total, leave=leave,
            file=file, ncols=ncols, mininterval=mininterval,
            maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
            unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
            smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
            postfix=postfix)

        self.logging_on_close = logging_on_close
        self.logging_on_update = logging_on_update
        self.closed = False
        self.track_eta = track_eta
        self.eta = '?'

    def eta_str(self):
        return self.eta

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False,
                     unit='it', unit_scale=False, rate=None, bar_format=None,
                     postfix=None, unit_divisor=1000, **extra_kwargs):

        meter = tqdm.tqdm.format_meter(
            n=n, total=total, elapsed=elapsed, ncols=ncols, prefix=prefix, ascii=ascii,
            unit=unit, unit_scale=unit_scale, rate=rate, bar_format=bar_format,
            postfix=postfix, unit_divisor=unit_divisor, **extra_kwargs)

        # get rid of that stupid comma before the postfix
        if postfix is not None:
            postfix_with_comma = ", %s" % postfix
            meter = meter.replace(postfix_with_comma, postfix)

        return meter

    @staticmethod
    def format_eta(t):
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        d, h = divmod(h, 24)
        if d:  # if we have more than 24hours, only display days and rounded hours
            h += round(m / 60)
            return '{0:d}d:{1:02}h'.format(d, h)
        elif h:
            # if we have more than an hour, just display hours and minutes
            return '{0:d}h:{1:02d}m'.format(h, m)
        else:
            # otherwise, minutes and seconds
            return '{0:02d}m:{1:02d}s'.format(m, s)

    def update(self, n=1):
        if self.logging_on_update:
            msg = self.__repr__()
            logging.logbook(msg)

        res = super().update(n=n)

        # remember eta if tracking is enabled
        if self.track_eta:
            if self.total is not None:
                rate = 1 / self.avg_time if self.avg_time else None
                if rate is None:
                    elapsed = self._time() - self.start_t
                    rate = self.n / elapsed
                self.eta = TqdmToLogger.format_eta((self.total - self.n) / rate)

        return res

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        super().set_postfix(ordered_dict, refresh, **kwargs)

    def close(self):
        if self.logging_on_close and not self.closed:
            msg = self.__repr__()
            logging.logbook(msg)
            self.closed = True
        return super().close()


def tqdm_with_logging(iterable=None, desc=None, total=None, leave=True,
                      ncols=None, mininterval=0.1,
                      maxinterval=10.0, miniters=None, ascii=None, disable=False,
                      unit="it", unit_scale=False, dynamic_ncols=False,
                      smoothing=0.3, bar_format=None, initial=0, position=None,
                      postfix=None,
                      logging_on_close=True,
                      logging_on_update=False,
                      track_eta=False):
    return TqdmToLogger(
        iterable=iterable, desc=desc, total=total, leave=leave,
        ncols=ncols, mininterval=mininterval,
        maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
        unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
        smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
        postfix=postfix,
        logging_on_close=logging_on_close,
        logging_on_update=logging_on_update,
        track_eta=track_eta)


# ----------------------------------------------------------------------------------------
# Comprehensively adds a new logging level to the `logging` module and the
# currently configured logging class.
# e.g. addLoggingLevel('TRACE', logging.DEBUG - 5)
# ----------------------------------------------------------------------------------------
def add_logging_level(level_name, level_num, method_name=None):
    if not method_name:
        method_name = level_name.lower()
    if hasattr(logging, level_name):
        raise AttributeError('{} already defined in logging module'.format(level_name))
    if hasattr(logging, method_name):
        raise AttributeError('{} already defined in logging module'.format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError('{} already defined in logger class'.format(method_name))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)


def configure_telegram(filename):
    telegram_bot = telegram.Bot(filename, flush_secs=constants.TELEGRAM_FLUSH_SECS)
    setattr(logging, "_telegram_bot", telegram_bot)
    setattr(logging, "telegram", telegram_bot.sendmessage)
    setattr(logging, "telegram_flush", telegram_bot.flush)


def configure_tensorboard_summaries(save):
    logdir = os.path.join(save, 'tb')
    writer = summary.SummaryWriter(logdir, flush_secs=constants.TENSORBOARD_FLUSH_SECS)
    setattr(summary, "_summary_writer", writer)
    with LoggingBlock("Tensorboard", emph=True):
        logging.value('  flush_secs:', constants.TENSORBOARD_FLUSH_SECS)
        logging.value('      logdir: ', logdir)


def configure_logging(filename):
    # set global indent level
    sys.modules[__name__].global_indent = 0

    # add custom tqdm logger
    add_logging_level("LOGBOOK", 1000)

    # create logger
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)

    # create console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = get_default_logging_format(colorize=True, brackets=False)
    datefmt = constants.LOGGING_TIMESTAMP_FORMAT
    formatter = ConsoleFormatter(fmt=fmt, datefmt=datefmt)
    console.setFormatter(formatter)

    # Skip logging.tqdm requests for console outputs
    skip_logbook_filter = SkipLogbookFilter()
    console.addFilter(skip_logbook_filter)

    # add console to root_logger
    root_logger.addHandler(console)

    # Show warnings in logger
    logging.captureWarnings(True)

    def _log_key_value_pair(key, value=None):
        if value is None:
            logging.info("{}{}".format(constants.COLOR_KEY_VALUE, str(key)))
        else:
            logging.info("{}{}{}".format(key, constants.COLOR_KEY_VALUE, str(value)))

    def _log_dict(indict):
        for key, value in sorted(indict.items()):
            logging.info("{}: {}{}".format(key, constants.COLOR_KEY_VALUE, str(value)))

    # this is for logging key value pairs or dictionaries
    setattr(logging, "value", _log_key_value_pair)
    setattr(logging, "values", _log_dict)

    # this is for logging blocks
    setattr(logging, "block", LoggingBlock)

    # add logbook
    if filename is not None:
        # ensure dir
        d = os.path.dirname(filename)
        if not os.path.exists(d):
            os.makedirs(d)
        with logging.block("Creating Logbook", emph=True):
            logging.info(filename)

        # --------------------------------------------------------------------------
        # Configure handler that removes color codes from logbook
        # --------------------------------------------------------------------------
        logbook = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")
        logbook.setLevel(logging.INFO)
        fmt = get_default_logging_format(colorize=False, brackets=True)
        logbook_formatter = LogbookFormatter(fmt=fmt, datefmt=datefmt)
        logbook.setFormatter(logbook_formatter)
        root_logger.addHandler(logbook)

        # --------------------------------------------------------------------------
        # Not necessary
        # --------------------------------------------------------------------------
        # logbook_tqdm = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")
        # logbook_tqdm.setLevel(logging.TQDM)
        # fmt = get_default_logging_format(colorize=False, brackets=True)
        # remove_colors_formatter = LogbookFormatter(fmt=fmt, datefmt=datefmt)
        # logbook_tqdm.setFormatter(remove_colors_formatter)
        # root_logger.addHandler(logbook_tqdm)


class LoggingBlock:
    def __init__(self, title=None, emph=False):
        self._emph = emph
        if title is not None:
            if emph:
                logging.info("%s==>%s %s%s%s" % (constants.COLOR_BLOCK_ARROW,
                                                 constants.COLOR_RESET,
                                                 constants.COLOR_BLOCK_EMPH,
                                                 title,
                                                 constants.COLOR_RESET))
            else:
                logging.info(title)

    def __enter__(self):
        sys.modules[__name__].global_indent += constants.LOGGING_GLOBAL_INDENT
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.modules[__name__].global_indent -= constants.LOGGING_GLOBAL_INDENT
