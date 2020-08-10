import numpy as np
import time
np.random.seed(7)

# \psi_{x, L_h}(y) = h(x) + \langle \nabla h(x), y - x \rangle + L_h \|y - x\|_2^2 + g(y)

def psi(grad_h, g, x, y, L_h, h_val=None, h_grad=None):
    if h_grad is None:
        h_grad = grad_h(x)
        
    if h_val is None:
        h_val = h(x)
    
    return h_val + h_grad.dot(y - x) + L_h * np.linalg.norm(y - x)**2 + g(y)


def grad_psi_stoch(grad_h_stoch, grad_g_stoch, x, y, L_h, i, h_grad=None):
    if h_grad is None:
        h_grad_stoch = grad_h_stoch(x, i)
    else:
        h_grad_stoch = h_grad[i]
    
    return h_grad_stoch + 2 * L_h * (y[i] - x[i]) + grad_g_stoch(y, i)


def taylor_acc_prox_method(n, x0, inner_method, grad_h, h, g, grad_h_stoch, grad_g_stoch, grad_f, f, Li, L, Lh, A_matrix, T_inner, T, time_scale, verbose=False):
    """
    Taylor Descent
    
    param: x0         starting point
    param: T          maximum number of iterations
    param: L, Lh      Lipschitz constants
    param: time_scale if true, then procedure returns the times of all iterations
    param: verbose    if true, then procedure logs debug information
    
    :return: zeta         result value
    :return: history_comp history of functional values of the M_inn method (Nesterov FCD)
    :return: times        times for all iterations
    """
    
    x = x0.copy()
    y = x0.copy()

    history = []
    times = []
    
    grad_h_count = [0]
    grad_g_count = [0]
    
    lambda_k = 1 / (2*L)
    Ak = 0
    for it in range(T):
        ak = (lambda_k + np.sqrt(lambda_k**2 + 4*Ak*lambda_k)) / 2
        Ak_new = Ak + ak
        
        x_wave = (Ak/Ak_new) * y + (ak/Ak_new) * x
        
        h_val = h(x_wave)
        
        y, history_comp, grad_h_count_comp, grad_g_count_comp, times_comp = inner_method(
            n, lambda y_var: psi(grad_h, g, x_wave.copy(), y_var, Lh, h_val=h_val),
            f,
            lambda y_var, i: grad_psi_stoch(grad_h_stoch, grad_g_stoch, x_wave.copy(), y_var, Lh, i),
            x_wave.copy(),
            A_matrix,
            T=T_inner,
            Li=Li, 
            L_add=3*Lh,
            restart=None,
            inner=True,
            return_f=True,
            time_scale=time_scale
        )
        
        grad_h_count += (grad_h_count[-1] + 1 + grad_h_count_comp).tolist()[1:]
        grad_g_count += (grad_g_count[-1] + grad_g_count_comp).tolist()[1:]
        
        x = x - ak*grad_f(y)
        
        grad_h_count[-1] = grad_h_count[-1] + 2.2
        grad_g_count[-1] = grad_g_count[-1] + 2.2
        
        Ak = Ak_new
        
        times += times_comp
        history += history_comp.tolist()
        
        if verbose:
            print("y", history_comp[-1])
            print("--", ak)
        
    return y, np.array(history), grad_h_count, grad_g_count, times