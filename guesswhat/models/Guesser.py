import torch
import torch.nn as nn
from .Encoder import Encoder


class Guesser(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_categories,
                 category_dim, hidden_size, mlp_hidden):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = Encoder(embedding_dim, hidden_size)

        self.cat = nn.Embedding(num_categories, category_dim)
        self.mlp = nn.Sequential(
            nn.Linear(category_dim+8, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_size)
        )

    def forward(self, dialogue, dialogue_lengths,
                object_categories, object_bboxes):

        word_emb = self.emb(dialogue)
        _, dialogue_emb = self.rnn(word_emb, dialogue_lengths)

        cat_emb = self.cat(object_categories)
        obj_emb = self.mlp(torch.cat([cat_emb, object_bboxes], dim=-1))

        return torch.bmm(obj_emb, dialogue_emb[0].permute(1, 2, 0)).squeeze(-1)
