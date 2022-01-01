# twos.py for defining functions in a python 2 manner

# _print function to print
def _print(*messages):

    # merge messages
    message = ','.join([str(entry) for entry in messages])

    # print the message
    print message

    return None

# define error functions in terms of python2 errors
FileExistsError = OSError
FileNotFoundError = IOError
PermissionError = OSError