import numpy as np


def softmax(L):
    """
    Takes in a list of numbers and returns the numbers from 0 to 1, where 1 
    represents high percentage and 0 represents low percentage.
    Together this numbers sum up to 1.
    """
    myList = []
    expL = np.exp(L)
    total = sum(expL)
    for x in expL:
        myList.append((1.0 * x) / total)
    return myList
    
