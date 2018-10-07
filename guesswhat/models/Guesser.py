import torch
import torch.nn as nn
from .Encoder import Encoder


class Guesser(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_categories,
                 category_dim, hidden_size, mlp_hidden, setting):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder = Encoder(embedding_dim, hidden_size)

        self.cat = nn.Embedding(num_categories, category_dim)

        spatial_dim = 8
        visual_dim = 1024
        if setting == 'baseline':
            input_size = category_dim + spatial_dim
        elif setting == 'category-only':
            input_size = category_dim
        elif setting == 'mrcnn':
            input_size = category_dim + spatial_dim + visual_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_size)
        )

        self.setting = setting

    def forward(self, dialogue, dialogue_lengths, object_categories=None,
                object_bboxes=None, num_objects=None, visual_features=None):

        word_emb = self.emb(dialogue)
        _, dialogue_emb = self.encoder(word_emb, dialogue_lengths)

        if self.setting in 'baseline':
            cat_emb = self.cat(object_categories)
            obj_emb = self.mlp(torch.cat([cat_emb, object_bboxes], dim=-1))
        elif self.setting == 'category-only':
            # take all categories
            cat_emb = self.cat.weight\
                .unsqueeze(0).repeat(dialogue.size(0), 1, 1)
            obj_emb = self.mlp(cat_emb)
        if self.setting == 'mrcnn':
            cat_emb = self.cat(object_categories)
            obj_emb = self.mlp(
                torch.cat([cat_emb, object_bboxes, visual_features], dim=-1))

        logits = torch.bmm(obj_emb,
                           dialogue_emb[0].permute(1, 2, 0)).squeeze(-1)

        if self.setting not in ['category-only']:
            # mask padding objects
            idx = num_objects.new_tensor(range(logits.size(1))).unsqueeze(0)\
                .repeat(logits.size(0), 1)
            mask = num_objects.unsqueeze(1).repeat(1, logits.size(1)) <= idx
            logits.masked_fill(mask, float('-1e30'))

        return logits

    def save(self, file='bin/guesser_belief.pt'):

        params = dict()
        params['num_embeddings'] = self.emb.num_embeddings
        params['embedding_dim'] = self.emb.embedding_dim
        params['num_categories'] = self.cat.num_embeddings
        params['category_dim'] = self.cat.embedding_dim
        params['hidden_size'] = self.encoder.rnn.hidden_size
        params['setting'] = self.setting
        params['mlp_hidden'] = self.mlp[0].out_features
        params['state_dict'] = self.state_dict()
        for k, v in params['state_dict'].items():
            params['state_dict'][k] = v.cpu()

        torch.save(params, file)

    @classmethod
    def load(cls, device, file='bin/guesser.pt'):
        params = torch.load(file)

        # legacy
        if 'setting' not in params:
            params['setting'] = 'baseline'
        else:
            if params['setting'] == 'object-only':
                params['setting'] = 'category-only'

        guesser = cls(params['num_embeddings'], params['embedding_dim'],
                      params['num_categories'], params['category_dim'],
                      params['hidden_size'], params['mlp_hidden'],
                      params['setting'])

        if 'state_dict' in params:
            guesser.load_state_dict(params['state_dict'])
        else:
            print("No state_dict for Guesser found.")
        guesser = guesser.to(device)

        return guesser
