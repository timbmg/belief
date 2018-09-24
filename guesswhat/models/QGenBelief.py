import os
import torch
import torch.nn as nn
from models import QGen, Guesser


class QGenBelief(nn.Module):

    def __init__(self, qgen, guesser, category_embedding_dim,
                 use_guesser_cat_emb, object_bow_baseline, train_guesser):

        super().__init__()

        self.qgen = qgen
        self.guesser = guesser
        self.use_guesser_cat_emb = use_guesser_cat_emb
        if self.use_guesser_cat_emb:
            self.relu = nn.ReLU()
            self.linear = nn.Linear(self.guesser.encoder.rnn.hidden_size,
                                    category_embedding_dim)
        else:
            self.category_emb = nn.Embedding(self.guesser.cat.num_embeddings,
                                             category_embedding_dim,
                                             padding_idx=0)

        self.tanh = nn.Tanh()
        self.category_embedding_dim = category_embedding_dim
        self.object_bow_baseline = object_bow_baseline
        self.train_guesser = train_guesser

    def get_object_probs(self, dialogue, dialogue_lengths, object_categories,
                         object_bboxes, num_objects):
        object_probs = nn.functional.softmax(
            self.guesser(
                dialogue=dialogue,
                dialogue_lengths=dialogue_lengths,
                object_categories=object_categories,
                object_bboxes=object_bboxes,
                num_objects=num_objects),
            dim=-1)

        return object_probs

    def get_object_emb(self, object_categories, object_bboxes):
        if self.use_guesser_cat_emb:
            category_emb = self.guesser.cat(object_categories)
            object_emb = torch.cat((category_emb, object_bboxes), dim=-1)
            guesser_emb = self.guesser.mlp(object_emb)
            if not self.train_guesser:
                guesser_emb.detach_()
            object_emb = self.linear(self.relu(guesser_emb))
        else:
            object_emb = self.category_emb(object_categories)

        return object_emb

    def get_belief_state(self, object_categories, object_bboxes,
                         object_probs):
        object_emb = self.get_object_emb(object_categories, object_bboxes)
        belief_state = self.tanh(torch.bmm(object_probs, object_emb))

        return belief_state

    def forward(self, dialogue, dialogue_lengths, visual_features,
                cumulative_dialogue, cumulative_lengths, num_questions,
                object_categories, object_bboxes, num_objects):

        if self.object_bow_baseline:
            additional_features = self.get_object_emb(object_categories,
                                                      object_bboxes)
            additional_features = torch.sum(additional_features, dim=1)\
                / num_objects.unsqueeze(1)\
                .repeat(1, self.category_embedding_dim).float()
            additional_features = additional_features.unsqueeze(1)\
                .repeat(1, dialogue.size(1), 1)
        else:
            batch_size = dialogue.size(0)
            max_que = torch.max(num_questions)  # most q's in a dialogue
            max_dln = torch.max(dialogue_lengths)  # most tokens in a dialogue
            pad_que = len(cumulative_lengths.view(-1))  # total num of q's
            max_obj = object_categories.size(1)  # most obj's in the batch

            if self.train_guesser:
                torch.enable_grad()
            else:
                torch.no_grad()

            # mask to remove q's which only contain pad tokens
            # note cumulative_lengths is zero padded
            rem_pad = cumulative_lengths.view(-1) > 0

            # merge batch and num_questions dimension
            # then filter out any questions which were added for padding
            cumulative_dialogue = cumulative_dialogue\
                .view(-1, cumulative_dialogue.size(2))[rem_pad]

            # repeat over padding dimension, then filter as above
            object_categories_repeated = object_categories.unsqueeze(1)\
                .repeat(1, max_que, 1).view(-1, max_obj)[rem_pad]
            object_bboxes_repeated = object_bboxes.unsqueeze(1)\
                .repeat(1, max_que, 1, 1).view(-1, max_obj, 8)[rem_pad]
            num_objects_repeated = num_objects.unsqueeze(1)\
                .repeat(1, max_que).view(-1)[rem_pad]

            # get object probabilities after each q/a in the dialogue
            object_probs = object_bboxes.new_zeros((pad_que, max_obj))
            object_probs[rem_pad] = \
                self.get_object_probs(cumulative_dialogue,
                                      cumulative_lengths.view(-1)[rem_pad],
                                      object_categories_repeated,
                                      object_bboxes_repeated,
                                      num_objects_repeated)
            object_probs = object_probs.view(-1, max_que, max_obj)
            if not self.train_guesser:
                object_probs.detach_()
                torch.enable_grad()

            belief_state = self.get_belief_state(object_categories,
                                                 object_bboxes, object_probs)

            # repeat belief_state over question tokens
            additional_features = visual_features.new_zeros(
                batch_size, max_dln, self.category_embedding_dim)
            for qi in range(max_que-1):
                running = qi < (num_questions-1)
                from_token = cumulative_lengths.new_zeros(batch_size) \
                    if qi == 0 else (cumulative_lengths[:, qi]-1)
                to_token = cumulative_lengths[:, qi+1]-1
                for bi in range(batch_size):
                    if running[bi].item() == 0:
                        continue
                    fr = from_token[bi].item()
                    to = to_token[bi].item()
                    additional_features[bi, fr:to] = belief_state[bi, qi]\
                        .unsqueeze(0).repeat(to - fr, 1)\
                        .view(-1, self.category_embedding_dim)

        logits = self.qgen(
            dialogue=dialogue,
            dialogue_lengths=dialogue_lengths,
            visual_features=visual_features,
            additional_features=additional_features)
        return logits

    def inference(self, input, dialogue, dialogue_lengths, hidden,
                  visual_features, end_of_question_token, object_categories,
                  object_bboxes, num_objects, strategy='greedy'):

        object_probs = \
            self.get_object_probs(dialogue, dialogue_lengths,
                                  object_categories, object_bboxes,
                                  num_objects)
        object_probs.unsqueeze_(1)

        belief_state = self.get_belief_state(object_categories,
                                             object_bboxes, object_probs)

        generations, lengths, h, c = self.qgen.inference(
            input=input,
            hidden=hidden,
            visual_features=visual_features,
            end_of_question_token=end_of_question_token,
            additional_features=belief_state,
            strategy=strategy)

        return generations, lengths, h, c

    def save(self, file='bin/qgen_belief.pt'):

        params = dict()

        # save parameters
        params['state_dict'] = self.state_dict()
        for k, v in params['state_dict'].items():
            params['state_dict'][k] = v.cpu()

        # save belief hyperparameters
        belief_params = dict()
        belief_params['category_embedding_dim'] = self.category_embedding_dim
        belief_params['use_guesser_cat_emb'] = self.use_guesser_cat_emb
        belief_params['object_bow_baseline'] = self.object_bow_baseline
        belief_params['train_guesser'] = self.train_guesser
        params['belief'] = belief_params

        # save qgen hyperparameters
        qgen_tmp_file = file[:-3] + '_qgen.pt'
        self.qgen.save(qgen_tmp_file)
        params['qgen'] = torch.load(qgen_tmp_file)
        os.remove(qgen_tmp_file)
        params['qgen'].pop('state_dict', None)

        # save guesser hyperparameters
        guesser_tmp_file = file[:-3] + '_guesser.pt'
        self.guesser.save(guesser_tmp_file)
        params['guesser'] = torch.load(guesser_tmp_file)
        os.remove(guesser_tmp_file)
        params['guesser'].pop('state_dict', None)

        torch.save(params, file)

    @classmethod
    def load(cls, device, file='bin/qgen_belief.pt'):

        params = torch.load(file)

        torch.save(params['qgen'], 'qgen_tmp.pt')
        qgen = QGen.load(device, file='qgen_tmp.pt')
        os.remove('qgen_tmp.pt')

        torch.save(params['guesser'], 'guesser_tmp.pt')
        guesser = Guesser.load(device, file='guesser_tmp.pt')
        os.remove('guesser_tmp.pt')

        # legacy
        if 'use_guesser_cat_emb' not in params['belief']:
            params['belief']['use_guesser_cat_emb'] = False
        if 'object_bow_baseline' not in params['belief']:
            params['belief']['object_bow_baseline'] = False
        if 'train_guesser' not in params['belief']:
            params['belief']['train_guesser'] = False

        qgen_belief = cls(qgen, guesser,
                          params['belief']['category_embedding_dim'],
                          params['belief']['use_guesser_cat_emb'],
                          params['belief']['object_bow_baseline'],
                          params['belief']['train_guesser'])
        qgen_belief.load_state_dict(params['state_dict'])
        qgen_belief = qgen_belief.to(device)
        print("Guesser and QGen loaded from QGenBelief.")

        return qgen_belief
