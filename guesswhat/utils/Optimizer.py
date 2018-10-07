import torch.optim as optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


class Optimizer():

    def __init__(self, algorithm, parameters, *args, **kwargs):

        if not issubclass(algorithm, optim.Optimizer):
            raise ValueError(
                "Expected algorithm to be subclass of torch.optim.Optimizer")
        self.parameters = parameters
        self.optimizer = algorithm(parameters, *args, **kwargs)

    def optimize(self, loss, clip_norm_args=None, clip_val_args=None):
        """Short summary.

        Parameters
        ----------
        loss : torch.Tensor
        clip_norm_args : list, tuple
            If provided the norm of the gradients will be clipped.
            First value represents the max. grad. value, second the norm
            (optional).
        clip_val_args : int
            If provided the gradients will be clipped in range
            (-clip_val_args, clip_val_args)
        Note
        ----------
        clip_norm_args and clip_val_args are mutually exclusive.
        """
        
        if clip_norm_args is not None and clip_val_args is not None:
            raise ValueError(
                "'clip_norm_args' and 'clip_val_args' are mutually exclusive.")

        self.optimizer.zero_grad()
        loss.backward()
        if clip_norm_args is not None:
            clip_grad_norm_(self.parameters, *clip_norm_args)
        if clip_val_args is not None:
            clip_grad_value_(self.parameters, clip_val_args)
        self.optimizer.step()
