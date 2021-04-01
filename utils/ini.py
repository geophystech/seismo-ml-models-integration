def remove_chars(line, chars=' \t', quotes='\'\"', comments=None):
    """
    Removes all specified characters but leaves quotes intact. Removes comments if comment character is specified.
    """
    new_line = ''
    quote_stack = ''
    remove_comments = (type(comments) is list) or (type(comments) is str)

    for c in line:

        if remove_comments and len(quote_stack) == 0 and c in comments:
            break

        if len(quote_stack) == 0 and c in chars:
            continue

        if c in quotes:
            if len(quote_stack) == 0 or c != quote_stack[-1]:
                quote_stack += c
            elif len(quote_stack) != 0:
                quote_stack = quote_stack[:-1]

            continue

        new_line += c

    return new_line


def parse_value(key, value):
    """
    Parses .ini value and returns tuple of value and type.
    """
    # Parse dictionary
    if value[0] == '{':

        value = value[1:-1]
        v_split = value.split(',')

        d = {}
        for x in v_split:
            x_split = x.split(':')
            d[x_split[0]] = x_split[1]

        return key, d, 'dictionary'

    # Parse single variable or list
    typ = 'var'
    if key[-2:] == '[]':
        typ = 'list'
        key = key[:-2]

    split = value.split(',')

    if len(split) == 1:
        return key, value, typ
    else:
        return key, split, 'list'


def parse_line(line):
    """
    Parses single .ini file line. Returns tuple of key, value and type of param.
    """
    # Trim line
    line = line.strip(' \t\n')

    # Check if empty
    if len(line) == 0:
        return None, None, None

    # Check if comment
    if line[0] in '#;':
        return None, None, 'comment'

    # Check if section: and ignore it
    if line[0] == '[':
        return None, None, 'section'

    # Remove all whitespaces unless in quotes and remove inline comments
    line = remove_chars(line, comments=';#')

    # Get key
    split = line.split('=')
    if len(split) < 2:
        return None, None, None

    key = split[0]
    val = line[len(key) + 1:]

    # Check value type
    key, val, typ = parse_value(key, val)

    return key, val, typ


def parse_ini(filename, params = None, param_names = None, param_set = None):
    """
    Parses .ini file.
    """
    var_dictionary = params
    if params is None:
        var_dictionary = {}

    with open(filename, 'r') as f:

        for line in f:

            key, value, typ = parse_line(line)

            if not typ or typ == 'comment' or typ == 'section':
                continue

            if key in param_set:
                continue

            if typ is not None and (param_names is None or key in param_names):

                if typ == 'var':
                    var_dictionary[key] = value

                if typ == 'list':

                    if type(value) is not list:
                        value = [value]

                    if key in var_dictionary:
                        var_dictionary[key] += value
                    else:
                        var_dictionary[key] = value

                if typ == 'dictionary':
                    var_dictionary[key] = value

    return var_dictionary


def convert_params(params, types):
    """
    Converts params dictionary elements to specified types.
    """
    for k in types:

        if k not in params:
            continue

        var = params[k]

        if type(var) is list:
            var = [types[k](x) for x in var]

        elif type(var) is dict:

            new_var = {}
            for key in var:

                new_var[key] = types[k](var[key])

            var = new_var

        else:
            var = types[k](var)

        params[k] = var
