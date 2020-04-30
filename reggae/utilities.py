import numpy as np
from tensorflow import linalg

def discretise(t, num_disc = 10):
    gcd = np.gcd.reduce(t)
    t_augmented = np.arange(0, t[-1]+gcd, gcd)
    N = t_augmented.shape[0]
    
    def calc_N_p(N_p, num_disc):
        '''A helper recursive function to ensure t is a subset of τ'''
        if num_disc <= 0:
            return N_p
        return N_p -1 + calc_N_p(N_p, num_disc-1)
    N_p = calc_N_p(N, num_disc)  # Number of time discretisations
    τ = np.linspace(0, t_augmented[-1], N_p, dtype='float64')    # Discretised observation times
    i = int(t[0]/gcd)
    τ = τ[i*num_disc+i:]
    common_indices = np.searchsorted(τ, t)
    return τ, common_indices

def get_rbf_dist(times, N):
    t_1 = np.reshape(np.tile(times, N), [N, N]).T
    t_2 = np.reshape(np.tile(times, N), [N, N])
    return t_1-t_2

def exp(x):
    '''Safe exp'''
    with np.errstate(under='ignore', over='ignore'):
        return np.exp(x)
    
def mult(a, b):
    '''Safe multiplication'''
    with np.errstate(under='ignore', over='ignore', invalid='ignore'):
        c = a*b
        return np.where(np.isnan(c), 0, c)

def jitter_cholesky(A):
    try:
        jitter1 = linalg.diag(1e-7 * np.ones(A.shape[0]))
        return linalg.cholesky(A + jitter1)
    except:
        jitter2 = diag(1e-5 * np.ones(A.shape[0]))
        return linalg.cholesky(A + jitter2)

class ArrayList:
    def __init__(self, shape):
        self.capacity = 100
        self.shape = shape
        self.data = np.zeros((self.capacity, *shape))
        self.size = 0

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity, *self.shape))
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    def get(self):
        data = self.data[:self.size]
        return data
