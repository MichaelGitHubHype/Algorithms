# By: Michael Sheinman, with the help of Udacity Nanodegree
# Created in: July 21, 2018
# A single weight update using gradient descent 

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

def gradient_descent():
  learnrate = 0.5
  x = np.array([1, 2, 3, 4])
  y = np.array(0.5)

  # Initial weights
  w = np.array([0.5, -0.5, 0.3, 0.1])

  ### Calculating one gradient descent step for each weight
  
  # Calculate the node's linear combination of inputs and weights
  h = np.dot(x, w)

  # Calculating output of neural network
  nn_output = sigmoid(h)

  # Calculating error of neural network
  error = y - nn_output  # true value minus network predicion

  # Calculating the error term
  error_term = error * sigmoid_prime(h)

  # Calculating change in weights
  del_w = error_term * learnrate * x
  return nn_output, error, del_w
  
if __name__ == 'main':
  nn_output, error, del_w = gradient_descent()
  print('Neural Network output:')
  print(nn_output)
  print('Amount of Error:')
  print(error)
  print('Change in Weights:')
  print(del_w)
