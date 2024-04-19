from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def __init__(self, dim: Optional[int] = None):
        if dim == None:
            self.dim = 1
        else:
            self.dim = dim

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.dim, keepdims=True)
        ez = array_api.exp(Z - maxz)
        sum_exp = array_api.sum(ez, axis=self.dim, keepdims=True)
        log_sum_exp = array_api.log(sum_exp)
        self.softmax = ez / sum_exp
        self.output = Z - maxz - log_sum_exp
        return self.output
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Z: (N, dim) 
        # ∂(log-softmax(Z)_i) / ∂Z_j = softmax(Z)_i - (i == j)
        grad_input = out_grad - (out_grad * self.softmax).sum(
            axes=(self.dim,)).broadcast_to(out_grad.shape) * self.softmax
        return grad_input
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        maxz_r = array_api.max(Z, axis=self.axes)
        x = array_api.sum(array_api.exp(Z - maxz), axis=self.axes)
        return array_api.log(x) + maxz_r
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        maxz = Z.realize_cached_data().max(axis=self.axes, keepdims=True)
        sumexp = summation(exp(Z - maxz), self.axes)
        grad_Z = out_grad / sumexp
        # Adjust shapes: Due to axis in sum
        expand_shape = list(Z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_Z = grad_Z.reshape(expand_shape).broadcast_to(Z.shape)
        return grad_Z * exp(Z - maxz)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

