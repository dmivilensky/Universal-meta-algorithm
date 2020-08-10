import numpy as np
import time
np.random.seed(7)

def fgm(f, func, grad_func, x0, T=1000, L_const=0, return_f=True, time_scale=False, verbose=False):
    """
    Fast Gradient method
    
    param: func             objective functional
    param: grad_func        gradient of the objective functional
    param: x0               starting point
    param: T                maximum number of iterations
    param: L_const          Lipschitz constant
    param: return_f         if true, then procedure returns the history of functional values
    param: time_scale       if true, then procedure returns the times of all iterations
    param: verbose          if true, then procedure logs debug information
    
    :return: x                     result value
    :return: funcs (if return_f)   history of functional values    
    :return: times (if return_f)   times for all iterations
    """
    
    x = x0.copy()
    v = x0.copy()
    
    A = 0
    S = L_const / 2
    
    funcs = []
    times = []
    
    grad_h_count = [0]
    grad_g_count = [0]
    
    for k in range(T):
        a = np.roots([S, -1, -A]).max()
        A = A + a
        alpha = a / A
        
        y = (1 - alpha) * x + alpha * v
        x = y - (1 / L_const) * grad_func(y)
        v = v - a * grad_func(x)
        
        grad_h_count.append(grad_h_count[-1] + 2.2)
        grad_g_count.append(grad_g_count[-1] + 2.2)
        
        if time_scale:
            times.append(time.time())
        
        funcs.append(func(x))
        
        if verbose:
            print(f(x))
        
    if return_f:
        return x, np.array(funcs), grad_h_count, grad_g_count, times
    else:
        return x