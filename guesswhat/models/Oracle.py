import torch
import torch.nn as nn
from .Encoder import Encoder


class Oracle(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_categories,
                 category_dim, hidden_size, mlp_hidden, global_film=None,
                 crop_film=None):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.cat = nn.Embedding(num_categories, category_dim)
        self.encoder = Encoder(embedding_dim, hidden_size)
        self.global_film = global_film
        self.crop_film = crop_film

        mlp_in_features = hidden_size+category_dim+8
        if global_film is not None:
            mlp_in_features += 512
        if crop_film is not None:
            mlp_in_features += 512

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_features, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 3)
        )

    def forward(self, question, question_lengths, object_categories,
                object_bboxes, crop=None, question_mask=None,
                global_features=None, crop_features=None):

        word_emb = self.emb(question)
        encoder_states, last_hidden = self.encoder(
            word_emb, question_lengths)

        cat_emb = self.cat(object_categories)
        mlp_input = torch.cat([last_hidden[0].squeeze(0),
                               cat_emb, object_bboxes], dim=-1)

        spat = torch.zeros(word_emb.size(0), 1, 14, 14)
        segm = torch.zeros(word_emb.size(0), 1, 14, 14)
        if self.global_film is not None:
            global_emb = self.global_film(
                global_features, encoder_states, question_mask, spat, segm)
            mlp_input = torch.cat([mlp_input, global_emb], dim=-1)
        if self.crop_film is not None:
            crop_emb = self.crop_film(
                crop_features, encoder_states, question_mask, spat, segm)
            mlp_input = torch.cat([mlp_input, crop_emb], dim=-1)

        logits = self.mlp(mlp_input)

        return logits

    def save(self, file='bin/oracle.pt'):

        params = dict()
        params['num_embeddings'] = self.emb.num_embeddings
        params['embedding_dim'] = self.emb.embedding_dim
        params['num_categories'] = self.cat.num_embeddings
        params['category_dim'] = self.cat.embedding_dim
        params['hidden_size'] = self.encoder.rnn.hidden_size
        params['mlp_hidden'] = self.mlp[0].out_features
        params['state_dict'] = self.state_dict()
        for k, v in params['state_dict'].items():
            params['state_dict'][k] = v.cpu()

        torch.save(params, file)

    @classmethod
    def load(cls, device, file='bin/oracle.pt'):
        params = torch.load(file)

        oracle = cls(params['num_embeddings'], params['embedding_dim'],
                     params['num_categories'], params['category_dim'],
                     params['hidden_size'], params['mlp_hidden'])

        oracle.load_state_dict(params['state_dict'])
        oracle = oracle.to(device)

        return oracle
