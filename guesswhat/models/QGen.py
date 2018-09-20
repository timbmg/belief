import torch
import torch.nn as nn
from .Encoder import Encoder


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
                  additional_features=None, strategy='greedy'):

        input.squeeze_(1)
        batch_size = input.size(0)
        batch_idx = input.new_tensor(list(range(batch_size))).long()
        running_idx = batch_idx.clone()

        if hidden is None:
            h = input.new_zeros((1, batch_size, self.encoder.rnn.hidden_size))\
                .float()
            c = input.new_zeros((1, batch_size, self.encoder.rnn.hidden_size))\
                .float()
        else:
            h, c = hidden

        visual_features = visual_features.unsqueeze(1)

        lengths = input.new_zeros((batch_size)).long()
        generations = list()

        while True and torch.max(lengths[running_idx]).item() < 100:

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
            if strategy == 'greedy':
                input[running_idx] = logits.topk(1)[1].squeeze(1)
            elif strategy == 'sampling':
                probs = nn.functional.softmax(logits, dim=-1)
                input[running_idx] = probs.multinomial(num_samples=1)
            else:
                raise ValueError("Expected strategy to be 'greedy' or " +
                                 "'sampling' but got {}.".format(strategy))

            # update running idx
            m = (input != end_of_question_token)
            if m.sum() > 0:
                running_idx = batch_idx.masked_select(m)

                # save generation (excluding eoq tokens)
                generated_idx = input.new_zeros((batch_size))
                generated_idx.masked_scatter_(m, input[running_idx])
                generations.append(generated_idx)

                # update lengths
                lengths[running_idx] = lengths[running_idx] + 1

            else:
                break

        return torch.stack(generations, dim=1), lengths, h, c

    def save(self, file='bin/qgen.pt'):

        params = dict()
        params['num_embeddings'] = self.emb.num_embeddings
        params['embedding_dim'] = self.emb.embedding_dim
        params['num_visual_features'] = self.visual_emb.in_features\
            if self.visual else 0
        params['visual_embedding_dim'] = self.visual_emb.out_features\
            if self.visual else 0
        params['hidden_size'] = self.encoder.rnn.hidden_size
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
