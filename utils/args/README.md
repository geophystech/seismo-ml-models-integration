# utils.args
This module's main method is `.args.archive_scan` which generates 
utils.params.Params  object with all required variables for `archive_scan.py` script.

## How to add a variable
You can add a variable on five different steps: 
- Command line arguments
- Config file
- Environmental argument
- Default values
- Applied functions (although this one is not recommended for variables addition, only for existing variables values initialization)

Note, this steps are executed in the order they are listed above.


### 1. Command line arguments
Go to `.command_line.py` script and find a function, with a name, similar to the main method's name 
(e.g. `archive_scan`) with postfix `_args` (e.g. `<name>_args`, `archive_scan_args`).
Add `parser.add_argument` call for new argument.

In `<name>_dictionary` add a dictionary entry `<variable name>: <name in argument parser>`.

You can setup default value (during command line stage) in the `<name>_defaults`.

### 2. Config file
Just add new variable to any config file and it will be read automatically. In '.config.py'  you can also add 
default values logic in method `<name>_defaults`.


### 3. Environmental argument
In `.env.py` in method `<name>` you can add logic to process your environmental variables.

### 4. Default values
Back in `.args.py` script you can find a function `defaults`. Add your argument and its value
there. It will be used, if argument was not set before this stage.

### 5. Applied functions
Module: `.applied_functions`. Applied functions are function, that applied to variables as a last stage of initialization process.
Applied functions could be used to convert types, set default values, process complex values (e.g. dates),
and even to create new variables.

In `<name>` method you can modify a dictionary of applied functions. 
Dictionary entry structure:
`<variable_name>: [<function_1>, <function_2>, ...]` - function are applied in the order they are listed.

All applied functions are located above in the same module.

You can write your own functions. Each function takes three variables: `(value, params, key)`:
- `value` - value of processed variable
- `params` - `utils.params.Params` object with all the variables (supports numpy style indexing for
variables access).
- `key` - current variable index.
Function returns value which is then applied to current variable.
You can also alter existing variable or add new by manipulating `params`.