import torch
import torch.nn as nn
from .Encoder import Encoder


class QGen(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_visual_features,
                 visual_embedding_dim, hidden_size):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.visual_emb = nn.Linear(num_visual_features, visual_embedding_dim)
        self.encoder = Encoder(embedding_dim+visual_embedding_dim, hidden_size)
        self.hidden2vocab = nn.Linear(hidden_size, num_embeddings)

    def forward(self, dialogue, dialogue_lengths, visual_features):

        word_emb = self.emb(dialogue)

        visual_emb = self.visual_emb(visual_features)
        visual_emb = visual_emb.unsqueeze(1).repeat(1, word_emb.size(1), 1)

        input_emb = torch.cat([word_emb, visual_emb], dim=-1)

        outputs, _ = self.encoder(input_emb, dialogue_lengths)

        return self.hidden2vocab(outputs).view(-1, self.emb.num_embeddings)

    def inference(self, dialogue, hidden, visual_features,
                  end_of_question_token, strategy='greedy'):

        batch_size = input.size(0)
        batch_idx = input.new_tensor(list(range(batch_size))).long()
        running_idx = batch_idx.clone()
        m = input.new_ones((batch_size)).byte()

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
            emb = self.emb(input[running_idx]).view(len(running_idx), 1, -1)
            visual_emb = self.visual_emb(visual_features[running_idx])
            input_emb = torch.cat((emb, visual_emb), dim=-1)
            outputs, (h[:, running_idx], c[:, running_idx]) = \
                self.encoder(input_emb, (h[:, running_idx], c[:, running_idx]))
            logits = self.h2v(outputs).squeeze(1)

            # get generated token
            if strategy == 'greedy':
                _, input[running_idx] = logits.topk(1)
            elif strategy == 'sampling':
                probs = nn.functional.softmax(logits, dim=-1)
                input[running_idx] = probs.multinomial(num_samples=1)
            else:
                raise ValueError("Expected strategy to be 'greedy' or " +
                                 "'sampling' but got {}.".format(strategy))

            # update running idx
            m = (input != end_of_question_token).squeeze(1)
            if m.sum() > 0:
                running_idx = batch_idx.masked_select(m)

                # save generation (excluding eoq tokens)
                generated_idx = input.new_zeros((batch_size))
                generated_idx.masked_scatter_(m, input[running_idx].squeeze(1))
                generations.append(generated_idx)

                # update lengths
                lengths[running_idx] = lengths[running_idx] + 1

            else:
                break

        return torch.stack(generations, dim=1), lengths, (h, c)

    def save(self, file='bin/qgen.pt'):

        params = dict()
        params['num_embeddings'] = self.emb.num_embeddings
        params['embedding_dim'] = self.emb.embedding_dim
        params['num_visual_features'] = self.visual_emb.in_features
        params['visual_embedding_dim'] = self.visual_emb.out_features
        params['hidden_size'] = self.encoder.rnn.hidden_size
        params['state_dict'] = self.state_dict()
        for k, v in params['state_dict'].items():
            params['state_dict'][k] = v.cpu()

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
