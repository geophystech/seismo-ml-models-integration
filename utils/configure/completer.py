import readline


def init():
    readline.parse_and_bind('tab: complete')


def get_completions(text, options):
    """
    Returns a list of options, which could be used to complete provided text.
    """
    completions = []
    l = len(text)
    for x in options:
        if len(x) < l:
            continue
        if x.startswith(text):
            completions.append(x)
    return completions


def create_completer(options):
    """
    Creates a completer functions.
    :param options:  list of strings of auto-completion options.
    :return: function
    """
    options = list(set(sorted(options)))
    def _completer(text, state):
        completions = get_completions(text, options)
        if state >= len(completions):
            return None
        return completions[state]
    return _completer


def set_completer(options):
    """
    Sets a completer function, based on list of completion options.
    :param options:  list of strings of auto-completion options.
    """
    if type(options) is not list:
        raise TypeError('create_completer failed: "options" must be a list of strings!')
    completer = create_completer(options)
    readline.set_completer(completer)
