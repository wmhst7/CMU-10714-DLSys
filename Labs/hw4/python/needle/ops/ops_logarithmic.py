from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = Z.max(axis=self.dim, keepdims=True)
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
        max_z_original = Z.max(axis=self.axes, keepdims=True) 
        max_z_reduce = Z.max(axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_original.broadcast_to(Z.shape)), axis=self.axes)) + max_z_reduce 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = Tensor(z.realize_cached_data().max(axis=self.axes, keepdims=True), device=z.device)
        exp_z = exp(z - max_z.broadcast_to(z.shape))
        sum_exp_z = summation(exp_z, axes=self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION

    # def compute(self, Z):
    #     ### BEGIN YOUR SOLUTION
    #     maxz = Z.max(axis=self.axes, keepdims=True)
    #     maxz_r = Z.max(axis=self.axes)
    #     x = array_api.sum(array_api.exp(Z - maxz), axis=self.axes)
    #     return array_api.log(x) + maxz_r
    #     ### END YOUR SOLUTION

    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     Z = node.inputs[0]
    #     maxz = Z.realize_cached_data().max(axis=self.axes, keepdims=True)
    #     sumexp = summation(exp(Z - maxz), self.axes)
    #     grad_Z = out_grad / sumexp
    #     # Adjust shapes: Due to axis in sum
    #     expand_shape = list(Z.shape)
    #     axes = range(len(expand_shape)) if self.axes is None else self.axes
    #     for axis in axes:
    #         expand_shape[axis] = 1
    #     grad_Z = grad_Z.reshape(expand_shape).broadcast_to(Z.shape)
    #     return grad_Z * exp(Z - maxz)
    #     ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

