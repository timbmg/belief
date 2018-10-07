import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, sizes, activation, final_activation=None, bias=True):

        super().__init__()

        str2activation = {'relu': nn.ReLU, 'tanh': nn.Tanh,
                          'sigmoid': nn.Sigmoid, 'log_softmax': nn.LogSoftmax,
                          'softmax': nn.Softmax}

        if isinstance(activation, str):
            activation = str2activation[activation]

        if final_activation is not None and isinstance(final_activation, str):
            final_activation = str2activation[final_activation]

        if isinstance(bias, bool):
            bias = [bias] * (len(sizes) - 1)
        elif isinstance(bias, list):
            assert len(bias) == len(sizes) - 1, \
                "Expected bias to be list of length {}, bot got {}."\
                .format(len(sizes)-1, len(bias))

        args = list()
        for li, (in_sz, out_sz) in enumerate(zip(sizes[:-1], sizes[1:])):
            args.append(nn.Linear(in_sz, out_sz, bias[li]))
            if li < len(sizes) - 2:
                args.append(activation())
            else:
                if final_activation is not None:
                    args.append(final_activation())

        self.mlp = nn.Sequential(*args)

    def forward(self, x):
        return self.mlp(x)
