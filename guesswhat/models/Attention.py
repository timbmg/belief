import torch
import torch.nn as nn


def mask_attn(scores, mask):
    scores.masked_fill_(1-mask, float(-1e30))
    return scores


class MLBAttention(nn.Module):

    def __init__(self, hidden_size, context_size, annotation_size, glimpses=1):

        super().__init__()

        self.ah = nn.Linear(annotation_size, hidden_size, bias=False)
        self.ch = nn.Linear(context_size, hidden_size, bias=False)
        self.score_linear = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()

        # TODO: multiple glimpses
        if glimpses > 1:
            raise NotImplementedError()

    def forward(self, annotations, context, mask=None):
        """
        annotations:  [B, A, F] (A beeing the dimension to do the attention over)
        context: [B, F]
        """

        a = self.tanh(self.ah(annotations))
        c = self.tanh(self.ch(context)).unsqueeze(1)

        scores = self.score_linear(a*c)

        if mask is not None:
            scores = mask_attn(scores, mask)

        attn_weights = nn.functional.softmax(scores, dim=1)

        return torch.bmm(attn_weights.transpose(1, 2), annotations).squeeze(1)


class DotAttention(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, context, inputs, mask=None):

        scores = torch.bmm(context, inputs.transpose(1, 2)).squeeze(1)

        if mask is not None:
            scores = mask_attn(scores, mask)

        attn_weights = nn.functional.softmax(scores, dim=1)

        return torch.bmm(attn_weights.unsqueeze(1), inputs).squeeze(1)
