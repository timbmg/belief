import os
import torch
import datetime
import torch.nn as nn
from torchvision.models import resnet50
from models import QGen, Guesser
from .Attention import MLBAttention


class QGenBelief(nn.Module):

    def __init__(self, qgen, guesser, category_embedding_dim,
                 object_emebdding_setting, object_probs_setting,
                 train_guesser_setting, visual_representation='vgg',
                 visual_query=None):

        super().__init__()

        self.qgen = qgen
        self.guesser = guesser
        if object_emebdding_setting == 'from-guesser':
            self.relu = nn.ReLU()
            self.linear = nn.Linear(self.guesser.encoder.rnn.hidden_size,
                                    category_embedding_dim)
        elif object_emebdding_setting == 'learn-emb':
            self.category_emb = nn.Embedding(self.guesser.cat.num_embeddings,
                                             category_embedding_dim,
                                             padding_idx=0)
        elif object_emebdding_setting == 'from-mrcnn':
            self.relu = nn.ReLU()
            self.linear = nn.Linear(1024, category_embedding_dim)

        self.visual_representation = visual_representation
        if self.visual_representation == 'resnet-mlb':
            if visual_query == 'belief':
                context_size = category_embedding_dim
            elif visual_query == 'hidden':
                context_size = qgen.hidden_size
            self.attn = MLBAttention(hidden_size=512,
                                     context_size=context_size,
                                     input_size=1024)
            self.attn_linear = nn.Linear(1024, 512)

            self.resnet = resnet50(pretrained=True)
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-3])
            self.resnet.eval()
            for p in self.resnet.parameters():
                p.requires_grad = False

        self.tanh = nn.Tanh()
        self.category_embedding_dim = category_embedding_dim

        self.object_emebdding_setting = object_emebdding_setting
        self.object_probs_setting = object_probs_setting
        self.train_guesser_setting = train_guesser_setting

    @property
    def hidden_size(self):
        return self.qgen.encoder.rnn.hidden_size

    def get_object_probs(self, dialogue, dialogue_lengths, object_categories,
                         object_bboxes, num_objects,
                         guesser_visual_features=None):
        object_probs = nn.functional.softmax(
            self.guesser(
                dialogue=dialogue,
                dialogue_lengths=dialogue_lengths,
                object_categories=object_categories,
                object_bboxes=object_bboxes,
                num_objects=num_objects,
                visual_features=guesser_visual_features),
            dim=-1)

        return object_probs

    def get_object_emb(self, object_categories, object_bboxes,
                       guesser_visual_features=None):
        if self.object_emebdding_setting == 'from-guesser':
            emb = self.guesser.cat(object_categories)
            if self.guesser.setting == 'baseline':
                emb = torch.cat((emb, object_bboxes), dim=-1)
            guesser_emb = self.guesser.mlp(emb)
            if not self.train_guesser_setting:
                guesser_emb.detach_()
            object_emb = self.linear(self.relu(guesser_emb))
        elif self.object_emebdding_setting == 'from-mrcnn':
            return self.relu(self.linear(guesser_visual_features))
        else:
            object_emb = self.category_emb(object_categories)

        return object_emb

    def get_belief_state(self, object_categories, object_bboxes,
                         object_probs, guesser_visual_features=None):
        object_emb = self.get_object_emb(object_categories, object_bboxes,
                                         guesser_visual_features)
        belief_state = self.tanh(torch.bmm(object_probs, object_emb))

        return belief_state

    def forward(self, dialogue, dialogue_lengths, cumulative_dialogue,
                cumulative_lengths, num_questions, question_lengths,
                object_categories, object_bboxes, num_objects,
                visual_features=None, guesser_visual_features=None,
                resnet_features=None):

        if self.object_probs_setting == 'uniform':
            additional_features = self.get_object_emb(object_categories,
                                                      object_bboxes)
            additional_features = torch.sum(additional_features, dim=1)\
                / num_objects.unsqueeze(1)\
                .repeat(1, self.category_embedding_dim).float()
            additional_features = additional_features.unsqueeze(1)\
                .repeat(1, dialogue.size(1), 1)
        elif self.object_probs_setting in ['guesser-probs',
                                           'guesser-all-categories']:
            batch_size = dialogue.size(0)
            max_que = torch.max(num_questions)  # most q's in a dialogue
            max_dln = torch.max(dialogue_lengths)  # most tokens in a dialogue
            pad_que = len(cumulative_lengths.view(-1))  # total num of q's
            max_que_len = torch.max(question_lengths)

            if self.train_guesser_setting:
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
            if self.object_probs_setting == 'guesser-probs':
                max_obj = object_categories.size(1)  # most obj's in the batch
                # repeat over padding dimension, then filter as above
                object_categories_repeated = object_categories.unsqueeze(1)\
                    .repeat(1, max_que, 1).view(-1, max_obj)[rem_pad]
                object_bboxes_repeated = object_bboxes.unsqueeze(1)\
                    .repeat(1, max_que, 1, 1).view(-1, max_obj, 8)[rem_pad]
                num_objects_repeated = num_objects.unsqueeze(1)\
                    .repeat(1, max_que).view(-1)[rem_pad]
                if guesser_visual_features is not None:
                    guesser_visual_features_repeated = guesser_visual_features\
                        .unsqueeze(1).repeat(1, max_que, 1, 1)\
                        .view(-1, max_obj, 1024)[rem_pad]
                else:
                    guesser_visual_features_repeated = None
            elif self.object_probs_setting == 'guesser-all-categories':
                max_obj = self.guesser.cat.num_embeddings
                object_categories_repeated = None
                object_bboxes_repeated = None
                num_objects_repeated = None
                object_categories = object_categories.new_tensor(
                    range(max_obj)).unsqueeze(0).repeat(batch_size, 1)

            # get object probabilities after each q/a in the dialogue
            object_probs = object_bboxes.new_zeros((pad_que, max_obj))
            object_probs[rem_pad] = \
                self.get_object_probs(cumulative_dialogue,
                                      cumulative_lengths.view(-1)[rem_pad],
                                      object_categories_repeated,
                                      object_bboxes_repeated,
                                      num_objects_repeated,
                                      guesser_visual_features_repeated)
            if not self.train_guesser_setting:
                object_probs.detach_()
                torch.enable_grad()
            object_probs = object_probs.view(-1, max_que, max_obj)

            belief_state = self.get_belief_state(object_categories,
                                                 object_bboxes, object_probs,
                                                 guesser_visual_features)

            if self.visual_representation == 'resnet-mlb':
                with torch.no_grad():
                    resnet_features = self.resnet(resnet_features)
                resnet_features = resnet_features.detach()
                attn_vis = self.attn(
                    resnet_features.repeat(1, 1, 1, belief_state.size(1)).view(-1, 14*14, 1024),
                    belief_state.view(-1, 512))
                attn_vis = attn_vis.view(belief_state.size(0), belief_state.size(1), -1) # b x q x 1024

            # repeat belief_state over question tokens
            additional_features = dialogue.new_zeros(
                batch_size, max_dln, self.category_embedding_dim).float()
            attn_vis_features = dialogue.new_zeros(
                batch_size, max_dln, 1024).float()
            chunked_dialogue = dialogue.new_zeros(
                batch_size, max_que-1, max_que_len+1)  # +1 for answer
            chunked_lengths = dialogue.new_zeros(
                batch_size, max_que-1)
            for qi in range(max_que - 1):

                running = qi < (num_questions - 1)
                from_token = cumulative_lengths.new_zeros(batch_size) \
                    if qi == 0 else (cumulative_lengths[:, qi] - 1)
                to_token = cumulative_lengths[:, qi + 1] - 1
                for bi in range(batch_size):
                    if running[bi].item() == 0:
                        continue
                    fr = from_token[bi].item()
                    to = to_token[bi].item()
                    additional_features[bi, fr:to] = belief_state[bi, qi]\
                        .unsqueeze(0).repeat(to - fr, 1)\
                        .view(-1, self.category_embedding_dim)
                    if self.visual_representation == 'resnet-mlb':
                        attn_vis_features[bi, fr:to] = attn_vis[bi, qi]\
                            .unsqueeze(0).repeat(to - fr, 1)\
                            .view(-1, 1024)

                    if False:
                        last_question = num_questions[bi]-1 == qi+1
                        offset1, offset2, offset3 = 0, 1, 1
                        if qi == 0:
                            offset1 += 1  # +1 from <sos>
                            offset2 -= 1
                        if last_question:
                            offset1 -= 1
                            offset3 -= 1

                        chunked_dialogue[bi, qi, :to - fr + offset1] = \
                            dialogue[bi, fr + offset2:to + offset3]

                        chunked_lengths[bi, qi] = (to + offset3) - (fr + offset2)

        if False:
            # this part is still not working correctly...
            batch_ids = torch.arange(0, batch_size).long()
            logits = dialogue.new_zeros(
                batch_size, max_que-1,
                max_que_len+1, self.qgen.emb.num_embeddings).float()
            hx = dialogue.new_zeros(
                1, batch_size, self.qgen.hidden_size).float()
            hx = (hx, hx)
            for qi in range(max_que-1):
                run = batch_ids[qi < (num_questions - 1)]
                additional_features = belief_state[run, qi]\
                    .unsqueeze(1).repeat(1, max_que_len+1, 1)

                if self.visual_representation == 'resnet-mlb':
                    context = hx[0][:, run].squeeze(0)
                    visual_rep = self.attn(
                        inputs=resnet_features.permute(0, 2, 3, 1)[run],
                        context=context)
                    visual_rep = self.tanh(self.attn_linear(visual_rep))
                    visual_rep = visual_rep.unsqueeze(1).repeat(
                        1, max_que_len+1, 1)
                    additional_features = torch.cat(
                        (additional_features, visual_rep), dim=-1)

                vf = visual_features[run] if visual_features is not None \
                    else None
                logits[run, qi] = self.qgen(
                    dialogue=chunked_dialogue[run, qi],
                    dialogue_lengths=chunked_lengths[run, qi],
                    visual_features=vf,
                    additional_features=additional_features,
                    hx=(hx[0][:, run], hx[1][:, run]),
                    flatten_output=False,
                    total_length=max_que_len+1)
                hx[0][:, run] = self.qgen.last_hidden[0]
                hx[1][:, run] = self.qgen.last_hidden[1]

            m = chunked_dialogue.unsqueeze(-1) > 0

            logits = logits.masked_select(m).view(
                -1, self.qgen.emb.num_embeddings)
        else:
            if self.visual_representation == 'resnet-mlb':
                visual_features = attn_vis_features

            logits = self.qgen(
                dialogue=dialogue,
                dialogue_lengths=dialogue_lengths,
                visual_features=visual_features,
                additional_features=additional_features)

        return logits

    def inference(self, input, dialogue, dialogue_lengths, hidden,
                  visual_features, end_of_question_token, object_categories,
                  object_bboxes, num_objects, guesser_visual_features=None,
                  max_tokens=100, strategy='greedy',
                  return_keys=['generations', 'log_probs', 'hidden_states',
                               'mask', 'object_probs']):

        object_probs = \
            self.get_object_probs(dialogue, dialogue_lengths,
                                  object_categories, object_bboxes,
                                  num_objects, guesser_visual_features)
        object_probs = object_probs.unsqueeze(1)
        if not self.train_guesser_setting:
            object_probs = object_probs.detach()

        belief_state = self.get_belief_state(object_categories,
                                             object_bboxes, object_probs,
                                             guesser_visual_features)

        lengths, h, c, return_dict = self.qgen.inference(
            input=input,
            hidden=hidden,
            visual_features=visual_features,
            end_of_question_token=end_of_question_token,
            additional_features=belief_state,
            max_tokens=max_tokens,
            strategy=strategy,
            return_keys=return_keys)

        if 'object_probs' in return_keys:
            return_dict['object_probs'] = object_probs

        return lengths, h, c, return_dict

    def save(self, file='bin/qgen_belief.pt', **kwargs):

        params = dict()
        for k, v in kwargs.items():
            params[k] = v

        # save parameters
        params['state_dict'] = self.state_dict()
        for k, v in params['state_dict'].items():
            params['state_dict'][k] = v.cpu()

        # save belief hyperparameters
        belief_params = dict()
        belief_params['category_embedding_dim'] = self.category_embedding_dim
        belief_params['object_emebdding_setting'] = \
            self.object_emebdding_setting
        belief_params['object_probs_setting'] = self.object_probs_setting
        belief_params['train_guesser_setting'] = self.train_guesser_setting
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

        ts = datetime.datetime.now().timestamp()
        torch.save(params['qgen'], 'qgen_tmp_{}.pt'.format(ts))
        qgen = QGen.load(device, file='qgen_tmp_{}.pt'.format(ts))
        os.remove('qgen_tmp_{}.pt'.format(ts))

        torch.save(params['guesser'], 'guesser_tmp_{}.pt'.format(ts))
        guesser = Guesser.load(device, file='guesser_tmp_{}.pt'.format(ts))
        os.remove('guesser_tmp_{}.pt'.format(ts))

        # legacy
        if 'object_emebdding_setting' not in params['belief']:
            if params['belief'].get('use_guesser_cat_emb', False):
                params['belief']['object_emebdding_setting'] = 'from-guesser'
            else:
                params['belief']['object_emebdding_setting'] = 'learn-emb'
        if 'object_probs_setting' not in params['belief']:
            params['belief']['object_probs_setting'] = \
                params['belief'].get('object_bow_baseline', 'guesser-probs')
        if 'train_guesser_setting' not in params['belief']:
            params['belief']['train_guesser_setting'] = \
                params['belief'].get('train_guesser', False)

        qgen_belief = cls(qgen, guesser,
                          params['belief']['category_embedding_dim'],
                          params['belief']['object_emebdding_setting'],
                          params['belief']['object_probs_setting'],
                          params['belief']['train_guesser_setting'])

        qgen_belief.load_state_dict(params['state_dict'])
        qgen_belief = qgen_belief.to(device)
        print("Guesser and QGen loaded from QGenBelief.")

        return qgen_belief
