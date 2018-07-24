import numpy as np


def cross_entropy(Y, P):
  """
  The function that takes as input two lists Y, P:
  Y: a list of 0s and 1: 0 means the scenrio is not happening and 1 means it does
  P: a list of values we will need to log to calculate the cross-entropy
  The function returns float corresponding to the cross-entropy of the values.
  """
  total = 0.0
  for y, p in zip(Y, P):
      if y == 0:
          total += np.log(1 - p)
      elif y == 1:
          total += np.log(p)
  # I use the negative symbol because log of decimal returns negative number
  return -total
