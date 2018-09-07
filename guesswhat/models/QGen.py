import torch
import torch.nn as nn
from .Encoder import Encoder


class QGen(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_visual_features,
                 visual_embedding_dim, hidden_size):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.visual_emb = nn.Linear(num_visual_features, visual_embedding_dim)
        self.rnn = Encoder(embedding_dim+visual_embedding_dim, hidden_size)
        self.hidden2vocab = nn.Linear(hidden_size, num_embeddings)

    def forward(self, dialogue, dialogue_lengths, visual_features):

        word_emb = self.emb(dialogue)

        visual_emb = self.visual_emb(visual_features)
        visual_emb = visual_emb.unsqueeze(1).repeat(1, word_emb.size(1), 1)

        input_emb = torch.cat([word_emb, visual_emb], dim=-1)

        outputs, _ = self.rnn(input_emb, dialogue_lengths)

        return self.hidden2vocab(outputs).view(-1, self.emb.num_embeddings)

    def save(self, file='bin/qgen.pt'):

        params = dict()
        params['num_embeddings'] = self.emb.num_embeddings
        params['embedding_dim'] = self.emb.embedding_dim
        params['num_visual_features'] = self.visual_emb.in_features
        params['visual_embedding_dim'] = self.visual_emb.out_features
        params['hidden_size'] = self.rnn.rnn.hidden_size
        params['state_dict'] = self.state_dict()

        torch.save(params, file)

    @classmethod
    def load(cls, file='bin/qgen.pt'):
        params = torch.load(file)

        qgen = cls(params['num_embeddings'],
                   params['embedding_dim'],
                   params['num_visual_features'],
                   params['visual_embedding_dim'],
                   params['hidden_size'])

        qgen.load_state_dict(params['state_dict'])

        return qgen
