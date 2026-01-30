import numpy as np 

def likelihood(model, x, t, data):
    pred = model.forward(x, t)
    chi2 = np.sum((data - pred)**2)
    return np.exp(-0.5 * chi2)
def log_likelihood(model, t, x, data):
    pred = model.forward(t, x)
    return -0.5 * np.sum((data - pred)**2)
