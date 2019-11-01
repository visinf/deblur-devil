# Copyright (c) 2014, Anton Backer <olegov@gmail.com>
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


# Styles:
#   autumn
#   borland
#   bw
#   colorful
#   default
#   emacs
#   friendly
#   fruity
#   manni
#   monokai
#   mu
#   native
#   pastie
#   perldoc
#   tango
#   trac
#   vim
#   vs

def add_hook(always=False, style='native', debug=False):
    isatty = getattr(sys.stderr, 'isatty', lambda: False)
    if always or isatty():
        try:
            import pygments  # flake8:noqa
            colorizer = Colorizer(style, debug)
            sys.excepthook = colorizer.colorize_traceback
        except ImportError:
            if debug:
                sys.stderr.write("Failed to add coloring hook; pygments not available\n")


class Colorizer(object):

    def __init__(self, style, debug=False):
        self.style = style
        self.debug = debug

    def colorize_traceback(self, type, value, tb):
        import traceback
        import pygments.lexers
        tb_text = "".join(traceback.format_exception(type, value, tb))
        lexer = pygments.lexers.get_lexer_by_name("pytb", stripall=True)

        import pygments.util

        from pygments.style import Style
        from pygments.token import Keyword, Name, Comment, String, Error, \
            Number, Operator, Generic, Token, Whitespace, Literal
        from pygments.formatters import Terminal256Formatter

        class MyStyle(Style):
            """
            Pygments version of the "native" vim theme.
            """

            background_color = '#202020'
            highlight_color = '#404040'

            styles = {
                Token: '#d0d0d0',
                Whitespace: '#666666',

                Comment: 'italic #999999',
                Comment.Preproc: 'noitalic bold #cd2828',
                Comment.Special: 'noitalic bold #e50808 bg:#520000',

                Keyword: 'bold #cda869',
                # Keyword.Constant:   '#cf6a4c',
                # Keyword.Namespace:         "#f92672", # class: 'kn'
                Keyword.Pseudo: 'nobold',
                Operator.Word: 'bold #cda869',

                # Punctuation:        '#f8f8f2',

                String: '#8f9d6a',
                String.Other: '#8f9d6a',

                Number: '#cf6a4c',

                Name: '#f8f8f2',
                Name.Builtin: '#24909d',
                Name.Variable: '#40ffff',
                Name.Constant: '#40ffff',
                Name.Class: 'underline #447fcf',
                Name.Function: '#447fcf',
                Name.Namespace: 'underline #447fcf',
                Name.Exception: '#bbbbbb',
                Name.Tag: 'bold #6ab825',
                Name.Attribute: '#bbbbbb',
                Name.Decorator: '#ffa500',
                # Name.Entity:               "bold #999999",

                Literal: '#ae81ff',
                Literal.Date: '#e6db74',

                Generic.Heading: 'bold #ffffff',
                Generic.Subheading: 'underline #ffffff',
                Generic.Deleted: '#cf6a4c',
                Generic.Inserted: '#589819',
                Generic.Error: '#cf6a4c',
                Generic.Emph: 'italic',
                Generic.Strong: 'bold',
                Generic.Prompt: '#aaaaaa',
                Generic.Output: '#cccccc',
                Generic.Traceback: '#cf6a4c',

                Error: 'bg:#e3d2d2 #a61717'
            }

        formatter = Terminal256Formatter(style=MyStyle)

        tb_colored = pygments.highlight(tb_text, lexer, formatter)
        self.stream.write(tb_colored)

    @property
    def formatter(self):
        colors = _get_term_color_support()
        if self.debug:
            sys.stderr.write("Detected support for %s colors\n" % colors)

        if colors == 256:
            fmt_options = {'style': self.style}
        elif self.style in ('light', 'dark'):
            fmt_options = {'bg': self.style}
        else:
            fmt_options = {'bg': 'dark'}

        from pygments.formatters import get_formatter_by_name
        import pygments.util

        fmt_alias = 'terminal256' if colors == 256 else 'terminal'
        try:
            return get_formatter_by_name(fmt_alias, **fmt_options)
        except pygments.util.ClassNotFound as ex:
            if self.debug:
                sys.stderr.write(str(ex) + "\n")
            return get_formatter_by_name(fmt_alias)

    @property
    def stream(self):
        try:
            import colorama
            return colorama.AnsiToWin32(sys.stderr)
        except ImportError:
            return sys.stderr


def _get_term_color_support():
    try:
        import curses
    except ImportError:
        # Probably Windows, which doesn't have great curses support
        return 16
    curses.setupterm()
    return curses.tigetnum('colors')
