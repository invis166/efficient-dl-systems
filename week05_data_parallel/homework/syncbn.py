import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float, world_size: int):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        batch_size = input.shape[0]
        global_batch_size = batch_size * world_size

        # getting global sum of elements and global sum of squared elements
        input_squared = input * input
        input_with_squared = torch.concat([input, input_squared], dim=-1)
        dist.all_reduce(input_with_squared, dist.ReduceOp.SUM)
        input_reduced, input_squared_reduced = torch.chunk(input_with_squared, chunks=2, dim=-1)

        # Var[X] = E[X^2] - E[X]^2
        global_mean = input_reduced.sum(axis=0) / global_batch_size
        global_std = torch.sqrt(input_squared_reduced.sum(axis=0) / global_batch_size - global_mean**2 + eps)

        # updating running statistics
        running_mean_new = (1 - momentum) * global_mean + momentum * running_mean
        running_std_new = (1 - momentum) * global_std + momentum * running_std

        # saving statistics for the backward pass
        ctx.save_for_backward(global_mean, global_std, input, torch.tensor(global_batch_size))

        return (input - global_mean) / (global_std), running_mean_new, running_std_new

    @staticmethod
    def backward(ctx, grad_output, running_mean, running_std):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        global_mean, global_std, input, global_batch_size = ctx.saved_tensors

        grad_reduced = torch.clone(grad_output)
        grad_prod_inp_minus_mu_reduced = grad_reduced * (input - global_mean)
        combined = torch.cat([grad_reduced, grad_prod_inp_minus_mu_reduced], dim=-1)
        dist.all_reduce(combined, dist.ReduceOp.SUM)
        grad_reduced, grad_prod_inp_minus_mu_reduced = torch.chunk(combined, 2, dim=-1)

        d_act_d_inp = 1 / global_std

        d_loss_d_mu = -1 / global_std * grad_reduced.sum(axis=0)
        d_mu_d_inp = 1 / global_batch_size

        d_loss_d_sigma = -1 / global_std**2 * grad_prod_inp_minus_mu_reduced.sum(axis=0)
        d_sigma_d_inp = 1 / global_std * (input - global_mean) / global_batch_size

        return grad_output * d_act_d_inp + d_loss_d_mu * d_mu_d_inp + d_loss_d_sigma * d_sigma_d_inp, None, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        # self.running_mean = torch.zeros((num_features,))
        # self.running_std = torch.ones((num_features,))

        self._world_size = dist.get_world_size()
        self._bn = sync_batch_norm.apply

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        normalized, running_mean_new, running_std_new = self._bn(
            input, self.running_mean, self.running_var, self.eps, self.momentum, self._world_size
        )
        self.running_mean.copy_(running_mean_new.detach())
        self.running_var.copy_(running_std_new.detach())

        return normalized
