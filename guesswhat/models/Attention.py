import torch
import torch.nn as nn


def mask_attn(scores, mask):
    scores.masked_fill_(1-mask, float(-1e30))
    return scores


class MLBAttention(nn.Module):

    def __init__(self, hidden_size, context_size, input_size, glimpses=1):

        super().__init__()

        self.ih = nn.Linear(input_size, hidden_size, bias=False)
        self.ch = nn.Linear(context_size, hidden_size, bias=False)
        self.score_linear = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()

        # TODO: multiple glimpses
        if glimpses > 1:
            raise NotImplementedError()

    def forward(self, context, inputs, mask=None):

        if inputs.dim() > 3:
            size = inputs.size()
            inputs = inputs.view(size[0], -1, size[-1])

        i = self.tanh(self.ih(inputs))
        c = self.tanh(self.ch(context))
        scores = self.score_linear(i*c.unsqueeze(1))

        if mask is not None:
            scores = mask_attn(scores, mask)

        attn_weights = nn.functional.softmax(scores, dim=1)

        return torch.bmm(attn_weights.transpose(1, 2), inputs)


class DotAttention(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, context, inputs, mask=None):

        scores = torch.bmm(context, inputs.transpose(1, 2)).squeeze(1)

        if mask is not None:
            scores = mask_attn(scores, mask)

        attn_weights = nn.functional.softmax(scores, dim=1)

        return torch.bmm(attn_weights.unsqueeze(1), inputs).squeeze(1)
