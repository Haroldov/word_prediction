#!/usr/bin/env python3
"""
Module for simple RNN functions. Taken from Coursera (deeplearning spec)
"""

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    txt = ' '.join(ix_to_char[ix] for ix in sample_ix)
    #txt = txt[0].upper() + txt[1:]  # capitalize first character
    print ('%s' % (txt, ), end='')

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values

    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state
    p_t = softmax(np.dot(Wya, a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
    return a_next, p_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def rnn_forward(X, Y, a0, parameters, vocab_size):
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size,1))
        if (X[t] != None):
            x[t][X[t]] = 1
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        loss -= np.log(y_hat[t][Y[t],0])
    cache = (y_hat, a, x)
    return loss, cache

def rnn_backward(X, Y, parameters, cache):
    gradients = {}
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    return gradients, a

def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients

def sample(parameters, char_to_ix, seed, a_prev = [], x = []):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    if len(x) == 0:
        x = np.zeros((vocab_size, 1))
    if (len(a_prev) == 0):
        a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']
    np.random.seed(seed)
    while (idx != newline_character and counter != 40):
        a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        idx = np.random.choice(vocab_size, p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        seed += 1
        counter +=1

    if (counter == 40):
        indices.append(char_to_ix['\n'])
    return indices

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """

    loss, cache = rnn_forward(X, Y, a_prev, parameters, parameters["Wax"].shape[1])
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)
    return loss, gradients, a[len(X)-1]

def model(data, ix_to_char, char_to_ix, dino_names, vocab_size, num_iterations = 25000, n_a = 20):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number o iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)
    examples = data
    np.random.seed(0)
    np.random.shuffle(examples)
    a_prev = np.zeros((n_a, 1))
    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
        loss = smooth(loss, curr_loss)
        if j % 2000 == 0:
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            seed = 0
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                seed += 1
            print('\n')
    return parameters

def infer(string, parameters, wrd2ix, ix2wrd):
    X = [None] + [wrd2ix[word] for word in string]
    Y = X[1:] + [wrd2ix["\n"]]
    n_a = 20
    seed = 0
    a_prev = np.zeros((n_a, 1))
    loss, cache = rnn_forward(X, Y, a_prev, parameters, parameters["Wax"].shape[1])
    sampled_indices = sample(parameters, wrd2ix, seed, cache[1][len(X) - 1], cache[0][len(X) - 1])
    print_sample(sampled_indices, ix2wrd)
