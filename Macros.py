import sys


def fprint(type, *args, **kwargs):
    if type == "E":
        print(*args, file=sys.stderr, **kwargs)
    elif type == "M":
        print("\033[3;34m", *args, "\033[0m", **kwargs)
    elif type == "C":
        print("\033[1;32m", *args, "\033[0m", **kwargs)
