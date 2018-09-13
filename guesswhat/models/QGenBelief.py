import torch
import torch.nn as nn


class QGenBelief(nn.Module):

    def __init__(self, qgen, guesser):

        super().__init__()

        self.qgen = qgen
        self.guesser = guesser
        self.softmax = nn.functional.softmax

    def forward(self, source_questions, question_lengths, visual_features,
                unrolled_dialogue, cumulative_lengths, num_questions,
                object_categories, object_bboxes):

        torch.no_grad()
        batch_size = source_questions.size(0)
        max_que = torch.max(num_questions)  # most q's in a dialogue
        max_qln = torch.max(question_lengths)  # most tokens in a dialogue
        pad_que = len(cumulative_lengths.view(-1))  # total num of q's
        max_obj = object_categories.size(1)  # most obj's in the batch

        # mask to remove q's which only contain pad tokens
        # note cumulative_lengths is zero padded
        rem_pad = cumulative_lengths.view(-1) > 0

        # merge batch and num_questions dimension
        # then filter out any questions which were added for padding
        unrolled_dialogue = unrolled_dialogue\
            .view(-1, unrolled_dialogue.size(2))[rem_pad]
        cumulative_lengths = cumulative_lengths.view(-1)[rem_pad]

        # repeat over padding dimension, then filter as above
        object_categories_repeated = object_categories.unsqueeze(1)\
            .repeat(1, max_que, 1).view(-1, max_obj)[rem_pad]
        object_bboxes_repeated = object_bboxes.unsqueeze(1)\
            .repeat(1, max_que, 1, 1).view(-1, max_obj, 8)[rem_pad]

        # get object probabilities after each q/a in the dialogue
        object_beliefs = object_bboxes.new_zeros((pad_que, max_obj))
        object_beliefs[rem_pad] = self.softmax(
            self.guesser(
                dialogue=unrolled_dialogue,
                dialogue_lengths=cumulative_lengths,
                object_categories=object_categories_repeated,
                object_bboxes=object_bboxes_repeated),
            dim=-1)
        object_beliefs = object_beliefs.detach()
        object_beliefs = object_beliefs.view(-1, max_que, max_obj)

        torch.enable_grad()
        # initiliaze hidden state
        hx = visual_features.new_zeros((1, batch_size,
                                        self.qgen.encoder.rnn.hidden_size))
        hx = hx, hx

        # initialize logits with zeros
        logits = visual_features.new_zeros(batch_size, max_que, max_qln,
                                           self.qgen.emb.num_embeddings)

        # loop over questions
        for qi in range(max_que):

            # create mask with examples that have more questions
            running = qi < num_questions

            # qgen forward pass for next question
            logits[running, qi, :torch.max(question_lengths[running, qi])] = \
                self.qgen(
                    source_questions[running, qi],
                    question_lengths[running, qi],
                    visual_features[running],
                    (hx[0][:, running], hx[1][:, running]) if qi > 0 else None,
                    flatten_output=False)

            hx[0][:, running], hx[1][:, running] = self.qgen.last_hidden

        return logits.view(-1, self.qgen.emb.num_embeddings)

    def save(self, file='bin/qgen_belief.pt'):
        self.qgen.save(file)
