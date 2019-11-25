import operator
import numpy as np

class Parameter(np.ndarray):
    """ Numpy array with gradient information. """

    def __new__(cls, value=None, shape=None, dtype=None):
        if value is None:
            value = np.empty(shape, dtype)
        obj = np.asarray(value, dtype=dtype).view(cls)
        obj._grad = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        # discard gradients
        self._grad = None

    @property
    def grad(self):
        """ Accumulated gradient for this parameter. """
        return self._grad

    @grad.setter
    def grad(self, value):
        """
        Accumulates the gradient for this parameter.

        Notes
        -----
        Since gradients are accumulated, the setter for the gradients does not
        simply overwrite the existing gradients, but accumulates them on
        assignment. This might lead to unexpected results when doing something
        like:
        >>> par.grad = par.grad + 2  # actual result will be `par.grad * 2 + 2`
        Use in-place operations for this kind of situations!
        """
        if value is None:
            del self.grad
        elif np.all(value == 0):
            self.zero_grad()
        elif self._grad is None:
            msg = "Gradients have not yet been initialised!"
            raise ValueError(msg)
        elif self._grad is not value:
            self._grad += value

    @grad.deleter
    def grad(self):
        """ Remove the gradients from the parameter. """
        self._grad = None

    def zero_grad(self):
        """ Set the gradients of this parameter to all zeros. """
        self._grad = np.zeros_like(self)



class Module:
    """Base class for all NNumpy modules."""

    def __init__(self):
        self.predicting = False
        self._parameters = {}

        self._forward_cache = []
        self._shape_cache = []

    def __call__(self, *inputs):
        if self.predicting:
            # keep cache clean
            return self.compute_outputs(*inputs)[0]

        return self.forward(*inputs)

    # # # attribute stuff # # #

    def __dir__(self):
        yield from super().__dir__()
        yield from self._parameters.keys()

    def __getattr__(self, name):
        try:
            return self._parameters[name]
        except KeyError:
            msg = "'{}' object has no attribute '{}'"
            raise AttributeError(msg.format(type(self).__name__, name))

    def __setattr__(self, name, value):
        try:
            # necessary to allow initialisation of _parameters in __init__
            _parameters = self.__dict__.get('_parameters', {})
            par = _parameters[name]
        except KeyError:
            if isinstance(value, Parameter):
                return self.register_parameter(name, value)

            return super().__setattr__(name, value)

        if par is not value:
            par[...] = value
        del par.grad

    def __delattr__(self, name):
        try:
            return self.deregister_parameter(name)
        except KeyError:
            return super().__delattr__(name)

    # # # non-magic stuff # # #

    def register_parameter(self, name, value):
        """
        Register a parameter with gradients to this module.

        Parameters
        ----------
        name : str
            Name of the parameter.
        value : ndarray or Parameter
            Values of the parameters to be registered.
        """
        name = str(name)
        param = Parameter(value)
        self._parameters[name] = param
        return param

    def deregister_parameter(self, name):
        """
        Deregister a parameter with its gradients from this module.

        Parameters
        ----------
        name : str
            Name of the parameter.

        Returns
        -------
        arr : ndarray
            Values of the unregistered parameter.
        """
        par = self._parameters.pop(str(name))
        return np.asarray(par)

    def named_parameters(self):
        """
        Iterator over module parameter (name, value) pairs.

        Yields
        -------
        name : str
            Name of the parameter
        param : Parameter
            Module parameter values with gradients.
        """
        yield from self._parameters.items()

    def parameters(self):
        """
        Iterator over module parameter values.

        Yields
        ------
        param : Parameter
            Module parameter values with gradients.
        """
        for _, par in self.named_parameters():
            yield par

    def reset_parameters(self, **kwargs):
        """ Initialise parameters. """
        for name, _ in self.named_parameters():
            self.__setattr__(name, 0)

    def zero_grad(self):
        """ (Re)set parameter gradients to zero. """
        for param in self.parameters():
            param.zero_grad()

    def train(self):
        """ Put the module in training mode. """
        self.predicting = False
        return self

    def eval(self):
        """ Put the module in evaluation mode. """
        self.predicting = True
        return self

    def forward(self, *inputs):
        """
        Forward pass through this module.

        Parameters
        ----------
        input0, input1, ..., inputn : ndarray
            One or more inputs.

        Returns
        -------
        out : ndarray or tuple of ndarrays
            One or more module outputs.

        See Also
        --------
        Module.compute_outputs : forward implementation without magic.
        Module.backward : automagic backward pass.

        Notes
        -----
        This is a wrapper around `Module.compute_outputs` that caches
        the values that are necessary for gradient computation automagically.
        """
        self._shape_cache.append(tuple(np.shape(x) for x in inputs))
        out, cache = self.compute_outputs(*inputs)
        self._forward_cache.append(cache)
        return out

    def backward(self, *grads):
        """
        Backward pass through this module.

        Parameters
        ----------
        grad0, grad1, ..., gradn : ndarray
            Gradients from one or more subsequent modules.

        Returns
        -------
        dx : ndarray or tuple of ndarrays
            Gradients w.r.t. to the input(s).

        See Also
        --------
        Module.compute_gradients : backward implementation without magic.
        Module.forward : automagic forward pass.

        Notes
        -----
        This is a wrapper around `Module.compute_gradients` that uses
        the values cached when calling `Module.forward` automagically.
        It also allows adds the ability to accumulate gradients
        when the output of this module is used in multiple subsequent modules.
        Updates the parameter gradients.
        """
        if self.predicting:
            raise ValueError("module not in training mode")

        shapes = self._shape_cache.pop()
        dx_accs = tuple(np.zeros(shape) for shape in shapes)

        cache = self._forward_cache.pop()
        try:
            for grad in grads:
                dxs = self.compute_grads(grad, cache)
                if len(shapes) == 1:
                    dxs = [dxs]

                for dx_acc, dx in zip(dx_accs, dxs):
                    dx_acc += dx
        except TypeError as e:
            # restore caches if zero_grad was not called (annoying in ipython)
            self._shape_cache.append(shapes)
            self._forward_cache.append(cache)
            raise TypeError(e.args[0] + "\nDid you forget to call zero_grad()?")

        return dx_accs[0] if len(dx_accs) == 1 else dx_accs

    def compute_outputs(self, *inputs):
        """
        Compute outputs for this module.

        Parameters
        ----------
        input0, input1, ..., inputn : ndarray
            One or more inputs.

        Returns
        -------
        out : ndarray or tuple of ndarrays
            One or more module outputs.
        cache : ndarray or tuple of ndarrays
            One or more values that are necessary for the gradient computation.

        See Also
        --------
        Module.forward : wrapper around this method for auto-magic caching.
        Module.compute_gradients : needs the cache data.
        """
        raise NotImplementedError()

    def compute_grads(self, grads, cache):
        """
        Compute gradients for this module.

        Parameters
        ----------
        grads : ndarray
            Gradients from subsequent module.
        cache : ndarray or tuple of ndarrays
            Cached values from the forward pass.

        Returns
        -------
        grad0, grad1, ..., gradn : ndarray
            Gradients w.r.t. to the input(s).

        See Also
        --------
        Module.backward : wrapper around this method for auto-magic caching.
        Module.compute_outputs : provides the cache data.

        Notes
        -----
        Updates the parameter gradients.
        """
        raise NotImplementedError()


class LossFunction(Module):
    """ Base class for NNumpy loss functions. """

    def __init__(self, reduction='mean', target_grads=False):
        """
        Set up the loss function.

        Parameters
        ----------
        reduction : {'none', 'sum', 'mean'}, optional
            Specification of how to reduce the results on the sample dimension.
        target_grads : bool, optional
            Flag to enable gradients w.r.t. to the target values.
        """
        super().__init__()
        self.reduction = reduction
        self.disable_target_grads = not target_grads
        self.reduction = get_reduction(reduction, axis=0)

    def __call__(self, predictions, targets):
        if self.predicting:
            out, _ = self.compute_outputs(predictions, targets)
            return self.reduction.compute_outputs(out)[0]

        return self.forward(predictions, targets)

    def forward(self, predictions, targets):
        out = super().forward(predictions, targets)
        return self.reduction.forward(out)

    def backward(self, *grads):
        grad = self.reduction.backward(*grads)
        return super().backward(grad)

    def compute_outputs(self, predictions, targets):
        raise NotImplementedError

    def compute_grads(self, grads, cache):
        raise NotImplementedError

class LogitCrossEntropy(LossFunction):
    """
    NNumpy implementation of the cross entropy loss function
    computed from the logits, i.e. before applying the softmax nonlinearity.
    """

    def compute_outputs(self, logits, targets):
        """
        Parameters
        ----------
        logits : (N, K) ndarray
        targets : (N, K) ndarray
        
        Returns
        -------
        cross_entropy : (N, 1) or (1, ) ndarray
        cache : ndarray or iterable of ndarrays
        """
        exp_s = np.exp(logits - np.max(logits))
        stable_softmax = exp_s / np.sum(exp_s)
        cross_entropy = np.sum(-targets.T*np.log( np.maximum(stable_softmax, np.ones(stable_softmax.shape)*1e-15).T), axis=0)
        cache = {'targets': targets, 'stable_softmax': stable_softmax}
        return cross_entropy, cache

    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, 1) or (1, ) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        dlogits : (N, K) ndarray
        dtargets : (N, K) ndarray
        """
        dlogits = cache['stable_softmax'] - cache['targets']
        dtargets = np.log(np.maximum(cache['stable_softmax'], np.ones(cache['stable_softmax'].shape)*1e-15))
        return dlogits, dtargets

class Tanh(Module):
    """ NNumpy implementation of the hyperbolic tangent function. """
        
    def compute_outputs(self, s):
        """
        Parameters
        ----------
        s : (N, K) ndarray
        
        Returns
        -------
        a : (N, K) ndarray
        cache : ndarray or iterable of ndarrays
        """
        cache = s
        return np.tanh(s), cache
        raise NotImplementedError("TODO: implement Tanh.compute_outputs method!")

    
    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, K) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        ds : (N, K) ndarrays
        """
        return grads*(1-np.tanh(cache)**2)
        raise NotImplementedError("TODO: implement Tanh.compute_grads method!")
        
class Identity(Module):
    """ NNumpy implementation of the identity function. """
        
    def compute_outputs(self, s):
        return s, None
    
    def compute_grads(self, grads, cache):
        return grads


class Sum(Module):
    """
    NNumpy implementation of sum reduction.
    """

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def compute_outputs(self, x):
        return np.sum(x, axis=self.axis), x.shape

    def compute_grads(self, grads, shape):
        grads = np.expand_dims(grads, self.axis)
        return np.broadcast_to(grads, shape)


class Mean(Module):
    """
    NNumpy implementation of mean reduction.
    """

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def compute_outputs(self, x):
        return np.mean(x, axis=self.axis), x.shape

    def compute_grads(self, grads, shape):
        grads = np.expand_dims(grads, self.axis)
        return np.broadcast_to(grads, shape) / shape[self.axis]

def get_reduction(name, axis=0, **kwargs):
    name = str(name).lower()
    if not name or name == 'none':
        return Identity()
    elif name == 'sum':
        return Sum(axis)
    elif name == 'mean' or name == 'avg' or name == 'average':
        return Mean(axis)
    else:
        raise ValueError("unknown aggregation: {}".format(name))
        
class MSE(LossFunction):
    """
    NNumpy implementation of the mean squared error function
    computed from the logits.
    """

    def compute_outputs(self, logits, targets):
        """
        Parameters
        ----------
        logits : (N, K) ndarray
        targets : (N, K) ndarray
        
        Returns
        -------
        mse : (N, 1) or (1, ) ndarray
        cache : ndarray or iterable of ndarrays
        """
        mse = (logits-targets)**2
        cache = {'targets': targets, 'logits': logits}
        return mse, cache

    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, 1) or (1, ) ndarray
        cache : ndarray or iterable of ndarrays

        Returns
        -------
        dlogits : (N, K) ndarray
        dtargets : (N, K) ndarray
        """
        dlogits = float(2/cache['logits'].shape[0])*(cache['logits'] - cache['targets'])
        dtargets = float(2/cache['logits'].shape[0])*(cache['logits'] - cache['targets'])
        return dlogits, dtargets