class ProgressBar:

    def __init__(self):

        self.progress_maxes = {}
        self.progress = {}

        self.progress_char = '#'
        self.current_progress_char = '#'
        self.empty_char = '-'
        self.progress_char_length = 30

        self._prefix_expression = None
        self._postfix_expression = None
        self._prefix_kwargs = {}
        self._postfix_kwargs = {}

        self._last_printed_line_length = 0

    def set_length(self, length):
        """
        Set progress bar length (number of characters).
        :param length:
        """
        if length < 0:
            raise AttributeError('length should be positive')
        self.progress_char_length = length

    def set_empty_character(self, char):
        """
        Set displayed character (or string) for empty progress part of the bar.
        :param char: str
        """
        self.empty_char = char

    def set_progress_character(self, char):
        """
        Set displayed character (or string) for finished progress part of the bar.
        :param char: str
        """
        self.progress_char = char

    def set_current_progress_char(self, char):
        """
        Set character for (or string) current progress identification.
        :param char: str
        :return:
        """
        self.current_progress_char = char

    def __str__(self):
        """
        Renders progress bar as a string.
        :return: str
        """
        # Render prefix
        prefix = ''
        if self._prefix_expression and len(self._prefix_kwargs):
            prefix = self._prefix_expression.format(**self._prefix_kwargs)
        elif self._prefix_expression:
            prefix = self._prefix_expression

        # Render postfix
        postfix = ''
        if self._postfix_expression and len(self._postfix_kwargs):
            postfix = self._postfix_expression.format(**self._postfix_kwargs)
        elif self._postfix_expression:
            postfix = self._postfix_expression

        # Render bar
        bar = ''
        current_progress_length = self.progress_char_length
        nested_progress_positions = []
        # Re-calculate actual progress in screen characters
        for level, max_progress in self.progress_maxes.items():

            if level not in self.progress:
                value = 0
            else:
                value = self.progress[level]

            nested_progress_positions.append((value / max_progress) * current_progress_length)
            current_progress_length = int(current_progress_length / max_progress)

        # Round and floor progress to fit into the character limit
        for i in range(len(nested_progress_positions) - 1):
            nested_progress_positions[i] = int(nested_progress_positions[i])
        nested_progress_positions[-1] = round(nested_progress_positions[-1])

        # Actual bar render
        total_progress_chars = sum(nested_progress_positions)
        if total_progress_chars == self.progress_char_length:
            bar = self.progress_char * total_progress_chars
        else:
            bar = self.progress_char * max(0, total_progress_chars - 1) + \
                  self.current_progress_char + \
                  self.empty_char * (self.progress_char_length - total_progress_chars)

        return prefix + bar + postfix  # concatenate the bar

    def print(self, *progress):
        """
        Prints the bar to stdout.
        :param progress: indicates progress if specified, equal to calling set_progress without level
            (with single progress value or set of values for multiple levels) before print.
            Does not change current progress if not specified. Default: None
        :return:
        """
        self.set_progress(*progress)

        bar = self.__str__()
        print('\r' + ' ' * self._last_printed_line_length + '\r' + bar, sep = '', end = '', flush = True)
        self._last_printed_line_length = len(bar)

    def set_max(self, *max_progress, **max_progress_dictionary):
        """
        Sets max progress values for progress bar rendering. Values can be int or float.
        :param max_progress: one or more arguments for max progress values
        :param max_progress_dictionary: use if you want named progress levels
        :return: list of keywords for current progress levels
        """
        if not len(max_progress) and not len(max_progress_dictionary):
            return
        if len(max_progress) and len(max_progress_dictionary):
            raise AttributeError('max progress should be either all positional or all keyword arguments')

        self.progress_maxes = {}

        if len(max_progress):
            for i, max_val in enumerate(max_progress):
                self.progress_maxes[str(i)] = max_val
        else:
            self.progress_maxes = max_progress_dictionary

        return self.progress_maxes.keys()

    def add_max(self, level, value, insert_after = None):
        """
        Adds new max progress level.
        :param level:
        :param value:
        :param insert_after: str or None - if not specified, insert new level at the end of the list.
        :return:
        """
        pass

    def change_max(self, level, value):
        """
        Change max progress for particular progress level.
        :param level:
        :param value:
        :return:
        """
        if value < 0:
            raise AttributeError('max value should be greater than zero')
        if level in self.progress_maxes:
            self.progress_maxes[level] = value

    def remove_progress_level(self, level):
        """
        Removes progress level data, max and value.
        :param level: str - keyword for the progress level to remove.
        """
        if level in self.progress_maxes:
            self.progress_maxes.pop(level, None)
        if level in self.progress:
            self.progress.pop(level, None)

    def set_progress(self, *progress, level = None,
                     fraction = False, percent = False, print=True):
        """
        Sets progress for a single level or for or existing levels as an absolute value, if progress consists of
        multiple values.
        :param progress:
        :param level:
        :param fraction:
        :param percent:
        """
        if not len(progress):
            return
        if not level and len(progress) > 1:
            raise AttributeError('multiple progress values with specified level are not compatible')
        if not level and len(progress) != len(self.progress_maxes):
            raise AttributeError(f'progress values count ({len(progress)}) should be equal'
                                 f' to the number of progress levels ({len(self.progress_maxes)})')
        if fraction and percent:
            raise AttributeError('both fraction and percent could not be True simultaneously')

        if not level:

            self.progress = {}
            for value, (level, max_progress) in zip(progress, self.progress_maxes.items()):
                if fraction:
                    value = min(value, 1.)
                    self.progress[level] = value * max_progress
                elif percent:
                    value = min(value, 100.)
                    self.progress[level] = (value * max_progress) / 100.
                else:
                    value = min(value, max_progress)
                    self.progress[level] = value

        else:

            if level not in self.progress_maxes:
                return
            if type(level) is not str:
                level = str(level)

            value = progress[0]
            max_progress = self.progress_maxes[level]

            if fraction:
                value = min(value, 1.)
                self.progress[level] = value * max_progress
            elif percent:
                value = min(value, 100.)
                self.progress[level] = (value * max_progress) / 100.
            else:
                value = min(value, max_progress)
                self.progress[level] = value

        if print:
            self.print()

    def set_progress_kwargs(self, fraction = False, percent = False, **progress):
        """
        Sets progress by dictionary of level keywords with progress values.
        :param progress:
        :param fraction:
        :param percent:
        """
        if not len(progress):
            return

        for level, value in progress.items():

            if level not in self.progress_maxes:
                return
            max_progress = self.progress_maxes[level]

            if fraction:
                value = min(value, 1.)
                self.progress[level] = value * max_progress
            elif percent:
                value = min(value, 100.)
                self.progress[level] = (value * max_progress) / 100.
            else:
                value = min(value, max_progress)
                self.progress[level] = value

    def set_prefix_expression(self, expression, clear_args = True):
        """
        Setter for the prefix expression.
        This expression will be used when printing the progress bar. Prefix keyword arguments will be used
        if specified with the expression.format(prefix_keyword_args) like call.
        :param expression: expression string in Pythons format specification mini-language (or just plain string
            if no formatting is needed).
        :param clear_args:
        """
        if expression and type(expression) is not str:
            raise TypeError('expression should be either string or None or False')
        if clear_args:
            self._prefix_kwargs = {}
        self._prefix_expression = expression

    def set_postfix_expression(self, expression, clear_args = True):
        """
        Setter for the postfix expression.
        This expression will be used when printing the progress bar. Postfix keyword arguments will be used
        if specified with the expression.format(postfix_keyword_args) like call.
        :param expression: expression string in Pythons format specification mini-language (or just plain string
            if no formatting is needed).
        :param clear_args:
        """
        if expression and type(expression) is not str:
            raise TypeError('expression should be either string or None or False')
        if clear_args:
            self._prefix_kwargs = {}
        self._postfix_expression = expression

    def set_prefix(self, expression):
        """
        Sets prefix string. Note: if you want to use expression formating with dynamic parameters, use
        set_prefix_expression and set_prefix_kwargs instead.
        :param expression:
        """
        self.set_prefix_expression(self, expression, clear_args = True)

    def set_postfix(self, expression):
        """
        Sets postfix string. Note: if you want to use expression formating with dynamic parameters, use
        set_postfix_expression and set_postfix_kwargs instead.
        :param expression:
        """
        self.set_postfix_expression(self, expression, clear_args = True)

    def set_prefix_kwargs(self, **kwargs):
        """
        Set prefix keyword arguments
        :param kwargs:
        """
        self._prefix_kwargs = kwargs

    def set_postfix_kwargs(self, **kwargs):
        """
        Set postfix keyword arguments
        :param kwargs:
        """
        self._postfix_kwargs = kwargs

    def set_prefix_arg(self, name, value):
        """
        Set one prefix keyword argument by its keyword
        :param name: str - keyword
        :param value:
        """
        self._prefix_kwargs[name] = value

    def set_postfix_arg(self, name, value):
        """
        Set one postfix keyword argument by its keyword
        :param name: str - keyword
        :param value:
        """
        self._postfix_kwargs[name] = value

    def pop_prefix_arg(self, name):
        """
        Pop prefix argument by its keyword (name)
        :param name: str - keyword
        :return: argument value or None
        """
        return self._prefix_kwargs.pop(name, None)

    def pop_postfix_arg(self, name):
        """
        Pop postfix argument by its keyword (name)
        :param name: str - keyword
        :return: argument value or None
        """
        return self._postfix_kwargs.pop(name, None)
