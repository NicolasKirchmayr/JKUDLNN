"""
Utility functions for checking neural network implementations.
"""

import numpy as np


def numerical_gradient(module, *x, eps=1e-6):
    """
    Compute numerical gradient using two-sided numerical gradient approximation.

    Parameters
    ----------
    module : Module
        Numpy deep learning module with parameters.
    x0, x1, ..., xn : ndarray
        Input data to compute gradients for.
    eps : float, optional
        The finite difference to use in the computation.
    """
    gradients = []
    for par in module.parameters():
        # establish flat views
        old_par_flat = par.copy().flat
        par_flat = par.flat
        for i in range(par.size):
            # larger weight
            par_flat[i] = old_par_flat[i] + eps
            pred, _ = module.compute_outputs(*x)
            f_up = np.sum(pred)

            # smaller weight
            par_flat[i] = old_par_flat[i] - eps
            pred, _ = module.compute_outputs(*x)
            f_low = np.sum(pred)

            # restore weight
            par_flat[i] = old_par_flat[i]
            grad = (f_up - f_low) / (2 * eps)
            gradients.append(grad)

    return np.stack(gradients)


def gradient_check(module, *xs, eps=1e-6, debug=False):
    """
    Compare analytical gradients with numerical gradient approximation.

    Parameters
    ----------
    module : Module
        Numpy deep learning module with parameters.
    x0, x1, ..., xn : ndarray
        Input data to check gradients on.
    eps : float, optional
        The finite difference to use for numerical gradient computation.
    debug : bool, optional
        Flag to print gradients for debugging.
    """
    pred = module.forward(*xs)
    module.zero_grad()

    # hack to get numerical input gradients
    input_module = type('DummyModule', (object,), {
        'parameters': lambda: xs,
        'compute_outputs': lambda _: (module(*xs), None)
    })
    numeric_dx = numerical_gradient(input_module, (None, ), eps=eps)
    analytic_grads = module.backward(np.ones_like(pred))
    analytic_dx = np.r_[tuple(dx.flat for dx in analytic_grads)]
    dx_diff = np.linalg.norm(numeric_dx - analytic_dx.flat) / len(analytic_dx)
    dx_check = np.isclose(0., dx_diff, atol=eps)

    if debug and not dx_check:
        print("dx diff:    ", dx_diff)
        print("dx numeric: ", numeric_dx)
        print("dx analytic:", analytic_dx)

    try:
        numeric_dws = numerical_gradient(module, *xs, eps=eps)
        dws_check = True
        for name, w in module.named_parameters():
            numeric_dw, numeric_dws = np.split(numeric_dws, [w.size])
            analytic_dw = w.grad.flatten()
            dw_diff = np.linalg.norm(numeric_dw - analytic_dw)
            dw_diff /= w.size * np.amax(numeric_dw) + eps
            dw_check = np.isclose(0., dw_diff, atol=eps)
            dws_check = dws_check and dw_check

            if debug and not dw_check:
                print("{}.grad diff:    ".format(name), dw_diff)
                print("{}.grad numeric: ".format(name), numeric_dw)
                print("{}.grad analytic:".format(name), analytic_dw)
    except (ValueError, AttributeError):
        dws_check = True
        if debug:
            print("No parameter gradients to check!")

    return dx_check and dws_check
