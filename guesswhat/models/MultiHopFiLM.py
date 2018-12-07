import torch
import torch.nn as nn

from .Attention import MLBAttention, DotAttention


class ResNetBlock(nn.Module):

    def __init__(self, dims, context_size, flim_gen_hidden):

        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dims+2, dims, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(dims, affine=False)

        self.film_gen = nn.Sequential(
            nn.Linear(context_size, flim_gen_hidden),
            nn.ReLU(),
            nn.Linear(flim_gen_hidden, dims*2))

    def forward(self, x, spat, segm, context):

        gamma, beta = torch.chunk(self.film_gen(context), chunks=2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(2)
        beta = beta.unsqueeze(2).unsqueeze(2)

        x = torch.cat((x, spat, segm), dim=1)  # concat along channels
        x = self.conv1(x)
        x = self.relu(x)
        residual = x  # where is the residual taken?

        x = self.conv2(x)
        x = self.bn2(x)
        x = gamma * x + beta
        x = self.relu(x)
        x = x + residual

        return x


class MultiHopFiLM(nn.Module):

    def __init__(self, dims, context_size, flim_gen_hidden):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1024, dims, (3, 3), padding=1)  # QUESTION: downscaling happens here?
        self.bn1 = nn.BatchNorm2d(dims)

        self.blocks = [ResNetBlock(dims, context_size, flim_gen_hidden),
                       ResNetBlock(dims, context_size, flim_gen_hidden),
                       ResNetBlock(dims, context_size, flim_gen_hidden),
                       ResNetBlock(dims, context_size, flim_gen_hidden)]
        self.lns = [nn.LayerNorm(context_size), nn.LayerNorm(context_size),
                    nn.LayerNorm(context_size), nn.LayerNorm(context_size)]  # QUESTION: affine?
        self.dot_attn = DotAttention()

        self.conv_out = nn.Conv2d(dims, 512, kernel_size=(1, 1))
        self.bn_out = nn.BatchNorm2d(512)  # QUESTION: affine?
        self.mlb_attn = MLBAttention(hidden_size=256, context_size=context_size, input_size=512)

    def forward(self, x, encoder_states, mask, spat, segm):

        # QUESTION: spat and segm here too?
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        context = encoder_states[:, -1]

        for block, layer_norm in zip(self.blocks, self.lns):
            # TODO: incorperate attention masking
            context = self.dot_attn(context.unsqueeze(1), encoder_states, mask)
            context = layer_norm(context)
            x = block(x, spat, segm, context)

        # QUESTION: spat and segm here too?
        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.relu(x)
        x = self.mlb_attn(
            encoder_states[:, -1], x.permute(0, 2, 3, 1)).squeeze()

        return x
