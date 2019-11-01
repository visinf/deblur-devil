# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import argparse
import re as _re
from gettext import gettext as _

import constants

# This code extends bits of the HelpFormatter presented in
# https://github.com/python/cpython/blob/3.7/Lib/argparse.py

# The purpose of the changes is to better format long lists of choices
# of the default argparse usage message.


class LongChoicesFormatter(argparse.HelpFormatter):

    # multiple options per line
    def _format_long_choices(self, expanded):
        indent = ' ' * self._current_indent
        if self._current_indent + len(expanded) > self._max_help_position:
            expanded = expanded[1:-1]  # take the part between curly brackets { .. }
            expanded = expanded.split(',')  # split arguments
            expanded = sorted(expanded)  # sort arguments
            expanded = ["{}{}{}".format(
                constants.COLOR_HELP_ARG_OPTION, opt, constants.COLOR_RESET) for opt in expanded]
            lines = []
            line = []
            line_len = self._current_indent - 1
            text_width = self._width - self._current_indent
            for part in expanded:
                if line_len + 1 + len(part) > text_width and line:
                    lines.append(indent + ' '.join(line))
                    line = []
                    line_len = len(indent) - 1
                line.append(part)
                line_len += len(part) + 1
            if line:
                lines.append(indent + ' '.join(line))
            result = '$'.join(lines)
        else:
            result = expanded

        return result

    # new line for every option
    # def _format_long_choices(self, expanded):
    #     if self._current_indent + len(expanded) > self._max_help_position:
    #         expanded = expanded[1:-1]  # take the part between curly brackets { .. }
    #         expanded = expanded.split(',')  # split arguments
    #         expanded = sorted(expanded)  # sort arguments
    #         expanded = ["{}{}{}".format(constants.COLOR_KEY_VALUE, opt, constants.COLOR_RESET) for opt in expanded]
    #         result = '$'.join(expanded)
    #     else:
    #         result = expanded
    #
    #     return result

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            expanded = '{%s}' % ','.join([str(choice) for choice in action.choices])
            result = self._format_long_choices(expanded)
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result,) * tuple_size

        return format

    def _format_actions_usage(self, actions, groups):
        # find group indices and identify actions in groups
        group_actions = set()
        inserts = {}
        for group in groups:
            try:
                start = actions.index(group._group_actions[0])
            except ValueError:
                continue
            else:
                end = start + len(group._group_actions)
                if actions[start:end] == group._group_actions:
                    for action in group._group_actions:
                        group_actions.add(action)
                    if not group.required:
                        if start in inserts:
                            inserts[start] += ' ['
                        else:
                            inserts[start] = '['
                        if end in inserts:
                            inserts[end] += ']'
                        else:
                            inserts[end] = ']'
                    else:
                        if start in inserts:
                            inserts[start] += ' ('
                        else:
                            inserts[start] = '('
                        if end in inserts:
                            inserts[end] += ')'
                        else:
                            inserts[end] = ')'
                    for i in range(start + 1, end):
                        inserts[i] = '|'

        # collect all actions format strings
        parts = []
        for i, action in enumerate(actions):

            # suppressed arguments are marked with None
            # remove | separators for suppressed arguments
            if action.help is argparse.SUPPRESS:
                parts.append(None)
                if inserts.get(i) == '|':
                    inserts.pop(i)
                elif inserts.get(i + 1) == '|':
                    inserts.pop(i + 1)

            # produce all arg strings
            elif not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                part = self._format_args(action, default)

                # if it's in a group, strip the outer []
                if action in group_actions:
                    if part[0] == '[' and part[-1] == ']':
                        part = part[1:-1]

                # add the action string to the list
                parts.append(part)

            # produce the first way to invoke the option in brackets
            else:
                option_string = action.option_strings[0]

                # if the Optional doesn't take a value, format is:
                #    -s or --long
                if action.nargs == 0:
                    part = ''
                    # part = action.format_usage()

                # if the Optional takes a value, format is:
                #    -s ARGS or --long ARGS
                else:
                    default = self._get_default_metavar_for_optional(action)
                    args_string = self._format_args(action, default)
                    part = '{}{}{} {}{}{}'.format(
                        constants.COLOR_HELP_ARG, option_string, constants.COLOR_RESET,
                        constants.COLOR_HELP_ARG_VALUE, args_string, constants.COLOR_RESET)

                # make it look optional if it's not required or in a group
                if not action.required and action not in group_actions:
                    part = '[%s]' % part

                # add the action string to the list
                parts.append(part)

        # insert things at the necessary indices
        for i in sorted(inserts, reverse=True):
            parts[i:i] = [inserts[i]]

        # join all the action items with spaces
        text = ' '.join([item for item in parts if item is not None])

        # clean up separators for mutually exclusive groups
        open = r'[\[(]'
        close = r'[\])]'
        text = _re.sub(r'(%s) ' % open, r'\1', text)
        text = _re.sub(r' (%s)' % close, r'\1', text)
        text = _re.sub(r'%s *%s' % (open, close), r'', text)
        text = _re.sub(r'\(([^|]*)\)', r'\1', text)
        text = text.strip()

        # return the text
        return text

    # def _fill_text(self, text, width, indent):
    #   return ''.join(indent + line for line in text.splitlines(keepends=True))

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = _('usage: ')

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)

        # if optionals and positionals are available, calculate usage
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)

            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            # build full usage string
            format = self._format_actions_usage
            action_usage = format(optionals + positionals, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:

                # break usage into wrappable parts
                part_regexp = (
                    r'\(.*?\)+(?=\s|$)|'
                    r'\[.*?\]+(?=\s|$)|'
                    r'\S+'
                )
                opt_usage = format(optionals, groups)
                pos_usage = format(positionals, groups)
                opt_parts = _re.findall(part_regexp, opt_usage)
                pos_parts = _re.findall(part_regexp, pos_usage)
                assert ' '.join(opt_parts) == opt_usage
                assert ' '.join(pos_parts) == pos_usage

                opt_parts = [x[1:-1] for x in opt_parts]  # strip brackets

                # helper for wrapping lines
                def get_lines(parts, indent, prefix=None):
                    lines = []
                    line = []
                    new_parts = []
                    for part in parts:
                        # find indent of option
                        opt_indent = part.find(' ') - 12

                        if '$' in part:
                            split_parts = part.split('$')
                            for i, p in enumerate(split_parts):
                                if i == 0:
                                    new_parts.append(p)
                                else:
                                    new_parts.append(' ' * opt_indent + p)
                        else:
                            new_parts.append(part)

                    parts = new_parts
                    for part in parts:
                        #
                        # Changed to ensure newline for every argument!
                        #
                        lines.append(indent + ' '.join(line))
                        line = []
                        line_len = len(indent) - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + ' '.join(line))
                    if prefix is not None:
                        lines[0] = lines[0][len(indent):]
                    return lines

                indent = ' ' * len(prefix)
                parts = opt_parts + pos_parts
                lines = get_lines(parts, indent)
                if len(lines) > 1:
                    lines = []
                    lines.extend(get_lines(opt_parts, indent))
                    lines.extend(get_lines(pos_parts, indent))
                lines = [prog] + lines

                # join lines into usage
                usage = '\n'.join(lines)

        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)
