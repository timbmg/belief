import torch
import torch.nn as nn
from .Encoder import Encoder
from collections import defaultdict


class QGen(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_visual_features,
                 visual_embedding_dim, hidden_size, num_additional_features=0):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

        if visual_embedding_dim > 0:
            self.visual = True
            self.visual_emb = nn.Linear(num_visual_features,
                                        visual_embedding_dim)
        else:
            self.visual = False

        self.add = num_additional_features > 0

        self.encoder = Encoder(embedding_dim + visual_embedding_dim +
                               num_additional_features, hidden_size)
        self.hidden2vocab = nn.Linear(hidden_size, num_embeddings)

    @property
    def hidden_size(self):
        return self.encoder.rnn.hidden_size

    def forward(self, dialogue, dialogue_lengths, visual_features,
                additional_features=None, hx=None, flatten_output=True):

        # prepare input embedding
        input_emb = self.emb(dialogue)
        if self.visual:
            visual_emb = self.visual_emb(visual_features).unsqueeze(1)
            visual_emb = visual_emb.repeat(1, input_emb.size(1), 1)
            input_emb = torch.cat((input_emb, visual_emb), dim=-1)
        if self.add:
            input_emb = torch.cat((input_emb, additional_features), dim=-1)

        # pass through encoder
        outputs, self.last_hidden = \
            self.encoder(input_emb, dialogue_lengths, hx)

        if flatten_output:
            return self.hidden2vocab(outputs)\
                .view(-1, self.emb.num_embeddings)
        else:
            return self.hidden2vocab(outputs)

    def inference(self, input, hidden, visual_features, end_of_question_token,
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

        visual_features = visual_features.unsqueeze(1)

        lengths = input.new_zeros((batch_size)).long()
        return_dict = defaultdict(list)

        first_token = True
        while True and torch.max(lengths[running_idx]).item() < max_tokens:

            # QGen forward pass
            input_emb = self.emb(input[running_idx])\
                .view(len(running_idx), 1, -1)
            if self.visual:
                visual_emb = self.visual_emb(visual_features[running_idx])
                input_emb = torch.cat((input_emb, visual_emb), dim=-1)
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

        for key in return_keys:
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
    def load(cls, device, file='bin/qgen.pt'):
        params = torch.load(file)

        qgen = cls(params['num_embeddings'],
                   params['embedding_dim'],
                   params['num_visual_features'],
                   params['visual_embedding_dim'],
                   params['hidden_size'],
                   num_additional_features=params['num_additional_features'])

        if 'state_dict' in params:
            qgen.load_state_dict(params['state_dict'])
        else:
            print("No state_dict for QGen found.")
        qgen = qgen.to(device)

        return qgen
