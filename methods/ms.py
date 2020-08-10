import numpy as np
import time
np.random.seed(7)

# \varphi_{\zeta^k, \zeta^0}(\zeta) = \langle \nabla h(\zeta^k), \zeta - \zeta^k \rangle + g(\zeta) + \frac{L}{2} \|\zeta - \zeta^0\|_2^2 + \frac{L_h}{2} \|\zeta - \zeta^k\|_2^2

def phi(grad_h, g, zeta, zeta_k, zeta_0, L_h, L, h_grad=None):
    """
    Proximal phi function
    
    param: zeta   variable
    param: zeta_k variable value at the certain iteration of composite method
    param: zeta_0 initial variable value for the composite method
    param: L_h, L Lipschitz constants
    
    :return: result 
    """
    if h_grad is None:
        h_grad = grad_h(zeta_k)
    
    return np.dot(h_grad, zeta - zeta_k) + g(zeta) + \
           0.5 * L_h * np.linalg.norm(zeta - zeta_k)**2 + \
           0.5 * L * np.linalg.norm(zeta - zeta_0)**2


def grad_phi_stoch(grad_h_stoch, grad_g_stoch, zeta, zeta_k, zeta_0, L_h, L, i, h_grad=None):
    """
    Gradient of the proximal phi function
    
    param: zeta   variable
    param: zeta_k variable value at the certain iteration of composite method
    param: zeta_0 initial variable value for the composite method
    param: L_h, L Lipschitz constants
    param: i      component
    
    :return: result 
    """
    if h_grad is None:
        h_grad_stoch = grad_h_stoch(zeta_k, i)
    else:
        h_grad_stoch = h_grad[i]
    
    return h_grad_stoch + grad_g_stoch(zeta, i) + \
           L_h * (zeta[i] - zeta_k[i]) + \
           L * (zeta[i] - zeta_0[i])


def grad_composite(n, method, grad_h, g, grad_h_stoch, grad_g_stoch, f, grad_f, zeta0, Li, L, Lh, A_matrix, time_scale):
    """
    Composite Gradient method for optimizing proximal phi function
    
    param: zeta0      starting point
    param: T          maximum number of iterations
    param: L, Lh      Lipschitz constants
    param: time_scale if true, then procedure returns the times of all iterations
    
    :return: zeta         result value
    :return: history_comp history of functional values of the M_inn method (Nesterov FCD)
    :return: times        times for all iterations
    """
    
    zeta = zeta0.copy()
    history_comp = []
    
    times = []
    
    grad_h_count = []
    grad_g_count = []
    
    for i in range(1):
        zeta, history_inner, grad_h_count_inner, grad_g_count_inner, times_inner = method(
            n, lambda zeta_var: phi(grad_h, g, zeta_var, zeta, zeta0, Lh, L),#, h_grad=h_grad),
            f,
            lambda zeta_var, i: grad_phi_stoch(grad_h_stoch, grad_g_stoch, zeta_var, zeta, zeta0, Lh, L, i),#, h_grad=h_grad),
            zeta.copy(),
            A_matrix,
            T=50,
            Li=Li, 
            L_add=2*Lh,
            restart=None,
            inner=True,
            return_f=True,
            time_scale=time_scale
        )
        
        grad_h_count = (1 + grad_h_count_inner).tolist()[1:]
        grad_g_count = (grad_g_count_inner).tolist()[1:]
            
        times += times_inner
        history_comp += history_inner.tolist()
        
        if np.linalg.norm(grad_f(zeta) + L*(zeta - zeta0)) <= L / 2 * np.linalg.norm(zeta - zeta0):
            break
    
    return zeta, history_comp, np.array(grad_h_count), np.array(grad_g_count), times


def ms_acc_prox_method(n, x0, inner_method, grad_h, g, grad_h_stoch, grad_g_stoch, f, grad_f, Li, L, Lh, A_matrix, T, time_scale, verbose=False):
    """
    Monteiro–Svaiter algorithm
    
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
    z = x0.copy()

    history = []
    times = []
    
    grad_h_count = [0]
    grad_g_count = [0]
    
    Ak = 0
    for it in range(T):
        ak = (1/L + np.sqrt(1/L**2 + 4 * Ak / L)) / 2
        Ak_new = Ak + ak
        
        x = (Ak / Ak_new) * y + (ak / Ak_new) * z
        y, history_comp, grad_h_count_comp, grad_g_count_comp, times_comp = grad_composite(
            n, inner_method, grad_h, g, grad_h_stoch, grad_g_stoch, f, grad_f, x, Li, Lh, L, A_matrix, time_scale=time_scale
        )
        
        if len(grad_h_count) == 1:
            grad_h_count = (grad_h_count[-1] + grad_h_count_comp).tolist()
            grad_g_count = (grad_g_count[-1] + grad_g_count_comp).tolist()
        else:
            grad_h_count += (grad_h_count[-1] + grad_h_count_comp).tolist()
            grad_g_count += (grad_g_count[-1] + grad_g_count_comp).tolist()
        
        z = z - ak*0.04*grad_f(y)
        
        grad_h_count[-1] = grad_h_count[-1] + 2
        grad_g_count[-1] = grad_g_count[-1] + 2
        
        Ak = Ak_new
        
        times += times_comp
        history += history_comp
        
        if verbose:
            print("y", history_comp[-1])
            print("--", ak)
        
    return y, np.array(history), grad_h_count, grad_g_count, times