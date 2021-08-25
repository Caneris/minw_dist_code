import numpy as np

def pick_element(set, num):
    N = len(set)
    i = (num%N)
    num = np.floor(num/N)
    return int(i), num

def ind(set, num):
    N = len(set)
    i = (num%N)
    return int(i)