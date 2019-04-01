import torch
import torch.nn as nn
from .Encoder import Encoder
from .Attention import MLBAttention
from collections import defaultdict


class QGen(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_visual_features,
                 visual_embedding_dim, hidden_size, num_additional_features=0,
                 visual_representation='vgg', query_tokens=None):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

        if visual_embedding_dim > 0:
            self.visual = True
            self.num_visual_features = num_visual_features
            self.visual_emb = nn.Linear(num_visual_features,
                                        visual_embedding_dim)
        else:
            self.visual = False

        self.add = num_additional_features > 0

        self.encoder = Encoder(embedding_dim + visual_embedding_dim +
                               num_additional_features, hidden_size)
        self.hidden2vocab = nn.Linear(hidden_size, num_embeddings)

        self.visual_representation = visual_representation
        if self.visual_representation == 'resnet-mlb':
            self.attn = MLBAttention(hidden_size=512,
                                     context_size=hidden_size,
                                     annotation_size=self.num_visual_features)
            self.query_tokens = query_tokens

    @property
    def hidden_size(self):
        return self.encoder.rnn.hidden_size

    def forward(self, dialogue, dialogue_lengths, visual_features=None,
                additional_features=None, hx=None, flatten_output=True,
                total_length=None):

        # prepare input embedding
        input_emb = self.emb(dialogue)
        if self.visual_representation == 'vgg':
            if self.visual:
                visual_emb = self.visual_emb(visual_features)
                if visual_emb.dim() == 2:
                    visual_emb = visual_emb.unsqueeze(1)
                    visual_emb = visual_emb.repeat(1, input_emb.size(1), 1)
                input_emb = torch.cat((input_emb, visual_emb), dim=-1)
            if self.add:
                input_emb = torch.cat((input_emb, additional_features), dim=-1)

            # pass through encoder
            outputs, self.last_hidden = \
                self.encoder(input_emb, dialogue_lengths, hx, total_length)

        elif self.visual_representation == 'resnet-mlb':

            batch_size = input_emb.size(0)
            visual_features = visual_features.permute(
                0, 2, 3, 1).view(-1, 14*14, self.num_visual_features)

            h, c = (input_emb.new_zeros(1, batch_size, self.hidden_size),
                    input_emb.new_zeros(1, batch_size, self.hidden_size))
            attn_vis = self.visual_emb(
                self.attn(
                    visual_features,
                    h.view(-1, self.hidden_size)))

            outputs = list()
            for i in range(input_emb.size(1)):

                # check if previous input was an answer
                m = input_emb.new_zeros(batch_size).byte()
                for t in self.query_tokens:
                    m += dialogue[:, i-1] == t
                if m.sum() > 0:
                    # update visual input
                    attn_vis[m] = self.visual_emb(
                        self.attn(
                            visual_features[m],
                            h.view(-1, self.hidden_size)[m]))

                _, (h, c) = self.encoder(
                    torch.cat(
                        (input_emb[:, i], attn_vis), dim=-1).unsqueeze(1),
                    dialogue.new_ones(batch_size), (h, c))

                outputs.append(h.transpose(0, 1))

            outputs = torch.cat(outputs, dim=1)
            m = (dialogue == 0).unsqueeze(2)
            outputs.masked_fill_(m, 0)

        out = self.hidden2vocab(outputs)

        if flatten_output:
            return out.view(-1, self.emb.num_embeddings)
        else:
            return out

    def inference(self, input, hidden, end_of_question_token,
                  visual_features=None, resnet_features=None,
                  additional_features=None, max_tokens=100, strategy='greedy',
                  return_keys=['generations', 'log_probs', 'hidden_states',
                               'mask']):

        input.squeeze_(1)
        batch_size = input.size(0)
        batch_idx = input.new_tensor(list(range(batch_size))).long()
        running_idx = batch_idx.clone()

        if hidden is None:
            h = input.new_zeros((1, batch_size, self.hidden_size))\
                .float()
            c = input.new_zeros((1, batch_size, self.hidden_size))\
                .float()
        else:
            h, c = hidden

        lengths = input.new_zeros((batch_size)).long()
        return_dict = defaultdict(list)

        if self.visual_representation == 'resnet-mlb':
            visual_features = resnet_features.permute(
                0, 2, 3, 1).view(-1, 14*14, self.num_visual_features)
            visual_emb = self.visual_emb(
                self.attn(
                    visual_features,
                    h.view(-1, self.hidden_size)))

        first_token = True
        while True and torch.max(lengths[running_idx]).item() < max_tokens:

            # QGen forward pass
            input_emb = self.emb(input[running_idx])\
                .view(len(running_idx), 1, -1)
            if self.visual:
                if self.visual_representation == 'vgg':
                    visual_emb = self.visual_emb(visual_features[running_idx])
                    if visual_emb.dim() == 2:
                        visual_emb = visual_emb.unsqueeze(1)
                        visual_emb = visual_emb.repeat(1, input_emb.size(1), 1)
                    input_emb = torch.cat((input_emb, visual_emb), dim=-1)
                elif self.visual_representation == 'resnet-mlb':
                    # check if previous input was an answer
                    m = torch.zeros_like(running_idx).byte()
                    for t in self.query_tokens:
                        m += input[running_idx] == t
                    if m.sum() > 0:
                        # update visual input
                        mask_idx = torch.nonzero(running_idx*m.long()).view(-1)
                        visual_emb[mask_idx] = self.visual_emb(
                            self.attn(
                                visual_features[mask_idx],
                                h.view(-1, self.hidden_size)[mask_idx]))
                    input_emb = torch.cat(
                        (input_emb, visual_emb[running_idx].unsqueeze(1)),
                        dim=-1)

            if self.add:
                input_emb = torch.cat(
                    (input_emb, additional_features[running_idx]), dim=-1)

            outputs, (h[:, running_idx], c[:, running_idx]) = \
                self.encoder(input_emb,
                             input.new_ones(len(running_idx)),
                             (h[:, running_idx], c[:, running_idx]))
            logits = self.hidden2vocab(outputs).squeeze(1)  # B x V

            # get generated token
            if first_token:
                logits[:, 3] = -1e9  # dont sample end of question token
                first_token = False
            if strategy == 'greedy':
                input[running_idx] = logits.topk(1)[1].squeeze(1)
                logp = torch.nn.functional.log_softmax(logits, dim=-1)
                logp = torch.gather(logp, 1, input[running_idx].unsqueeze(1))
                logp.unsqueeze_(1)
            elif strategy == 'sampling':
                d = torch.distributions.Categorical(logits=logits)
                input[running_idx] = d.sample()
                logp = d.log_prob(input[running_idx])
            else:
                raise ValueError("Expected strategy to be 'greedy' or " +
                                 "'sampling' but got {}.".format(strategy))

            # update running idx
            m = (input != end_of_question_token)
            if m.sum() > 0:
                running_idx = batch_idx.masked_select(m)

                # save generation (excluding eoq tokens)
                if 'generations' in return_keys:
                    generated_idx = input.new_zeros((batch_size))
                    generated_idx.masked_scatter_(m, input[running_idx])
                    return_dict['generations'].append(generated_idx)

                # save log probabilites
                if 'log_probs' in return_keys:
                    padded_log_probs = input.new_zeros((batch_size)).float()
                    padded_log_probs.masked_scatter_(m, logp)
                    return_dict['log_probs'].append(padded_log_probs)

                # save hidden states
                if 'hidden_states' in return_keys:
                    hidden_states = h.new_zeros((batch_size, self.hidden_size))
                    hidden_states.masked_scatter_(
                        m.unsqueeze(1).repeat(1, self.hidden_size),
                        h[:, running_idx].squeeze(0))
                    return_dict['hidden_states'].append(hidden_states)

                # save mask
                if 'mask' in return_keys:
                    return_dict['mask'].append(m)

                # update lengths
                lengths[running_idx] = lengths[running_idx] + 1

            else:
                break

        for key in return_dict:
            return_dict[key] = torch.stack(return_dict[key], dim=1)

        return lengths, h, c, return_dict

    def save(self, file='bin/qgen.pt', **kwargs):

        params = dict()
        for k, v in kwargs.items():
            params[k] = v

        params['num_embeddings'] = self.emb.num_embeddings
        params['embedding_dim'] = self.emb.embedding_dim
        params['num_visual_features'] = self.visual_emb.in_features\
            if self.visual else 0
        params['visual_embedding_dim'] = self.visual_emb.out_features\
            if self.visual else 0
        params['hidden_size'] = self.hidden_size
        params['num_additional_features'] = self.encoder.rnn.input_size \
            - params['visual_embedding_dim'] \
            - params['embedding_dim']
        params['state_dict'] = self.state_dict()
        for k, v in params['state_dict'].items():
            params['state_dict'][k] = v.cpu()

        torch.save(params, file)

    @classmethod
    def load(cls, device, file='bin/qgen.pt', legacy_attn=False):
        params = torch.load(file)

        # legacy fix
        if 'visual_representation' not in params:
            params['visual_representation'] = \
                'vgg' if params['num_visual_features'] == 1000 else 'resnet-mlb'
            if legacy_attn:
                params['visual_representation'] = 'vgg'
        params['query_tokens'] = params.get('query_tokens', [4, 5, 6])

        for k, v in params.items():
            if k!='state_dict':
                print(k, v)

        qgen = cls(params['num_embeddings'],
                   params['embedding_dim'],
                   params['num_visual_features'],
                   params['visual_embedding_dim'],
                   params['hidden_size'],
                   num_additional_features=params['num_additional_features'],
                   visual_representation=params['visual_representation'],
                   query_tokens=params['query_tokens'])

        if 'state_dict' in params:
            qgen.load_state_dict(params['state_dict'])
        else:
            print("No state_dict for QGen found.")
        qgen = qgen.to(device)

        return qgen
