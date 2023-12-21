import numpy as np


def count_opcodes(program: list, opcodes: list) -> int:
    """
    Count the number of opcodes in the program
    :param program: The program
    :param opcodes: The opcodes
    :return: The number of opcodes
    """
    count = 0
    for elem in opcodes:
        for code in opcodes:
            if elem == code:
                count += 1
                break
    return count
