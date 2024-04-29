"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias == True:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose())
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if X.shape[-1] != self.in_features:
            raise ValueError("Input tensor's last dimension size does not match the linear layer's input features")
        origin_shape = list(X.shape)
        batch_size = int(np.prod(origin_shape)) // self.in_features
        X = X.reshape((batch_size, self.in_features))

        res = X.matmul(self.weight)
        if self.bias != None:
            res += self.bias.broadcast_to(res.shape)
        
        origin_shape[-1] = self.out_features
        res = res.reshape(tuple(origin_shape))
        return res
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_oh = init.one_hot(logits.shape[1], y)
        return (ops.summation(ops.logsumexp(logits, (1,)) / logits.shape[0]) - 
            ops.summation(y_oh * logits / logits.shape[0]))
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones((dim), requires_grad=True, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros((dim), requires_grad=True, dtype=dtype, device=device))
        self.running_mean = init.zeros((dim), dtype=dtype, device=device)
        self.running_var = init.ones((dim), dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = (x.sum((0,)) / x.shape[0]) #(dim)
            var = (((x - mean.broadcast_to(x.shape))**2).sum((0,)) / x.shape[0]) #(dim)
            self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + var * self.momentum
            x_norm = (x - mean.broadcast_to(x.shape)) / ((
                var.broadcast_to(x.shape) + self.eps)**0.5)
        else:
            x_norm = (x - self.running_mean.broadcast_to(x.shape)) / ((
                self.running_var.broadcast_to(x.shape) + self.eps)**0.5)
        return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones((dim), requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros((dim), requires_grad=True, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        origin_shape = x.shape
        batch_size = int(np.prod(x.shape)) // self.dim
        x = x.reshape((batch_size, self.dim))
        # x: (N, dim) mean: (N, dim) var: (N, dim)
        mean = (x.sum((1,)) / self.dim).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = (((x - mean)**2).sum((1,)) / self.dim).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        deno = (var + self.eps)**0.5
        out = self.weight.reshape((1, self.dim)).broadcast_to(x.shape) * (x - mean
                ) / deno + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return out.reshape(origin_shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5, device=None, dtype="float32"):
        super().__init__()
        self.p = p
        self.device = device
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=self.device, dtype=self.dtype)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION