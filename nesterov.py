import numpy as np
import time
np.random.seed(7)

def nesterov(n, func, f, grad_func_stoch, x0, A_matrix, Li, T=100, L_add=0, L_mult=1, restart=None, beta=1/2, return_f=True, inner=False, time_scale=False, verbose=False, print_grads=False, grad_func=None):
    """
    Nesterov's Fast Coordinate Descent method
    paper: Nesterov and Stich
           "Efficiency of the Accelerated Coordinate Descent Method on Structured Optimization Problems"
           (presentation http://www.mathnet.ru:8080/PresentFiles/11909/7_nesterov.pdf , page 6)
    
    param: func             objective functional
    param: grad_func_stoch  stochastic gradient of the objective functional
    param: x0               starting point
    param: T                maximum number of iterations
    param: L_add            summand for all directional Lipschitz constants
    param: restart          count of iterations between two consequent restarts
    param: beta             parameter of randomizer normalization constants
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
    S = ((Li*L_mult+L_add)**beta).sum()
    
    funcs = []
    times = []
    
    grad_h_count = [0]
    grad_g_count = [0]
    
    time_start = time.time()
    
    eye = np.eye(n)
    
    for k in range(T):
        i = int(np.random.choice(np.linspace(0, n-1, n), 
                                 p=(Li*L_mult+L_add)**beta/S))
        a = np.roots([S**2, -1, -A]).max()
        A = A + a
        alpha = a / A
        
        y = (1 - alpha) * x + alpha * v
        
        stoch_grad = grad_func_stoch(y, i)
        
        if not inner:
            grad_h_count.append(grad_h_count[-1] + 1)
            grad_g_count.append(grad_g_count[-1] + 1)
        else:
            grad_h_count.append(grad_h_count[-1])
            grad_g_count.append(grad_g_count[-1] + 1)
        
        gamma = - (1 / (Li[i]*L_mult+L_add)) * stoch_grad 
        zeta = - (a * S) / ((Li[i]*L_mult+L_add)**beta) * stoch_grad
        
        x = y + gamma * eye[i]
        
        v = v + zeta * eye[i]
        
        if restart is not None and k % restart == 0:
            v = x.copy()
            
        if time_scale:
            times.append(time.time())
        
        funcs.append(f(x))
        if verbose:
            print(f(x))
            
        if print_grads:
            print(np.linalg.norm(grad_func(x)))
    
    if return_f:
        return x, np.array(funcs), np.array(grad_h_count), np.array(grad_g_count), times
    else:
        return x