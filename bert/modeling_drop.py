from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from bert.modeling import BertModel, BERTLayerNorm, ACT2FN


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).byte(), replace_with)


def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()


def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()


def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]


def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask


def gather_indexes(input, positions):
    '''
    :param input: [N, L, D]
    :param positions: [N, J]
    :return: [N*J, D]
    '''
    batch_size = input.size()[0]
    seq_length = input.size()[1]

    offsets = torch.arange(batch_size).to(input.device).unsqueeze(1) * seq_length
    flat_positions = flatten(offsets + positions)
    flat_input = flatten(input)
    return flat_input[flat_positions,:]


def get_self_att_representation(input, input_score, input_mask, dim=1):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=dim)
    return output


def get_weighted_att_representation(query, input, input_mask):
    '''
    :param query: [N, D]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    attention_score = torch.matmul(query.unsqueeze(1), input.transpose(-1, -2)) # [N, 1, L]
    attention_score = attention_score.squeeze(1)
    input_mask = input_mask.to(dtype=attention_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    attention_score = attention_score + input_mask
    attention_prob = nn.Softmax(dim=-1)(attention_score)
    attention_prob = attention_prob.unsqueeze(-1)
    output = torch.sum(attention_prob * input, dim=1)
    return output


def distant_cross_entropy(logits, labels):
    '''
    :param logits: [N, L]
    :param labels: [N, L]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_likelihood = log_softmax(logits)

    log_likelihood = replace_masked_values(log_likelihood, labels, -1e7)
    log_marginal_likelihood = logsumexp(log_likelihood)
    return log_marginal_likelihood


def tile_for_conditional_span_end(tensor, nbest_size):
    ndims = len(tensor.size())

    if ndims == 2:
        return tensor.unsqueeze(1).repeat(1, nbest_size, 1).view(tensor.size()[0]*nbest_size, tensor.size()[1])
    elif ndims == 3:
        return tensor.unsqueeze(1).repeat(1, nbest_size, 1, 1).view(tensor.size()[0]*nbest_size, tensor.size()[1], tensor.size()[2])
    else:
        raise Exception()


def negative_log_likelihood(log_probs, labels):
    '''
    :param log_probs: (batch_size, # of classes)
    :param labels: (batch_size, # of gold indices)
    :return: loss: (1, ), losses{batch_size, }
    '''

    glod_mask = (labels != -1).long()
    clamped_labels = replace_masked_values(labels, glod_mask, 0)

    # (batch_size, # of gold indices)
    log_likelihood = torch.gather(log_probs, 1, clamped_labels)

    # For those padded abilities, we set their log probabilities to be very small negative value
    log_likelihood = replace_masked_values(log_likelihood, glod_mask, -1e7)

    # Take logsumexp across the last dimension to sum the probs
    # Shape: (batch_size, )
    log_likelihood = logsumexp(log_likelihood)

    # Set the log likelihood to be 0 for the instance that has no valid gold spans
    # Shape: (batch_size, )
    summed_gold_mask = (glod_mask.sum(-1) != 0).long()
    log_likelihood = replace_masked_values(log_likelihood, summed_gold_mask, 0)

    # Calculate the averaged negative log likelihood across mini-batches
    # When all instances of the batch have no valid gold spans, we add an offset to avoid nan loss
    # Shape: (1, )
    offset = (summed_gold_mask.sum() == 0).float()
    loss = - log_likelihood.sum() / (summed_gold_mask.sum().float() + offset)

    return loss, log_likelihood


def gather_log_likelihood(log_probs, labels):
    glod_mask = (labels != -1).long()
    clamped_labels = replace_masked_values(labels, glod_mask, 0)

    # (batch_size, # of gold indices)
    log_likelihood = torch.gather(log_probs, 1, clamped_labels)
    return log_likelihood


def gather_representations(output, indices):
    ndims = len(indices.size())
    if ndims == 1:
        indices = indices.unsqueeze(-1)
    index_mask = (indices != -1).long()
    clamped_indices = replace_masked_values(indices, index_mask, 0)
    # Shape: (batch_size, # of indices, hidden_size)
    gathered_output = torch.gather(output, 1,
                                   clamped_indices.unsqueeze(-1).expand(-1, -1, output.size(-1)))
    return gathered_output


def masked_logsumexp(log_likelihood, labels):
    glod_mask = (labels != -1).long()
    log_likelihood = replace_masked_values(log_likelihood, glod_mask, -1e7)
    log_marginal_likelihood = logsumexp(log_likelihood)
    return log_marginal_likelihood


def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                          device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)


def pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(0)
    return sequence


def convert_crf_output(outputs, sequence_length, device):
    predictions = []
    for output in outputs:
        pred = pad_sequence(output[0], sequence_length)
        predictions.append(torch.tensor(pred, dtype=torch.long))
    predictions = torch.stack(predictions, dim=0)
    if device is not None:
        predictions = predictions.to(device)
    return predictions


class BertFeedForward(nn.Module):
    def __init__(self, config, input_size, intermediate_size, output_size):
        super(BertFeedForward, self).__init__()
        self.dense = nn.Linear(input_size, intermediate_size)
        self.affine = nn.Linear(intermediate_size, output_size)
        self.act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BERTLayerNorm(config)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.affine(hidden_states)
        return hidden_states


class MTMSN(nn.Module):
    def __init__(self, config, answering_abilities=None, max_number_of_answer=8):
        super(MTMSN, self).__init__()
        self.bert = BertModel(config)
        self._passage_affine = nn.Linear(config.hidden_size, 1)
        self._question_affine = nn.Linear(config.hidden_size, 1)

        if answering_abilities is None:
            self.answering_abilities = ["span_extraction", "addition_subtraction", "counting", "negation"]
        else:
            self.answering_abilities = answering_abilities

        if len(self.answering_abilities) >= 1:
            self._answer_ability_predictor = BertFeedForward(config, 3 * config.hidden_size, config.hidden_size,
                                                             len(self.answering_abilities))

        self.number_pos = -1
        self.base_pos = -2
        self.end_pos = -3
        self.start_pos = -4

        if "span_extraction" in self.answering_abilities:
            self._span_extraction_index = self.answering_abilities.index("span_extraction")
            self._base_predictor = BertFeedForward(config, config.hidden_size, config.hidden_size, 1)
            self._start_predictor = BertFeedForward(config, config.hidden_size, config.hidden_size, 1)
            self._end_predictor = BertFeedForward(config, config.hidden_size, config.hidden_size, 1)
            self._start_affine = nn.Linear(4 * config.hidden_size, 1)
            self._end_affine = nn.Linear(4 * config.hidden_size, 1)
            self._span_number_predictor = BertFeedForward(config, 3 * config.hidden_size, config.hidden_size,
                                                          max_number_of_answer)

        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = BertFeedForward(config, 5 * config.hidden_size, config.hidden_size, 3)
            self._sign_embeddings = nn.Embedding(3, 2 * config.hidden_size)
            self._sign_rerank_affine = nn.Linear(2 * config.hidden_size, 1)
            self._sign_rerank_predictor = BertFeedForward(config, 5 * config.hidden_size, config.hidden_size, 1)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._number_count_affine = nn.Linear(2 * config.hidden_size, 1)
            self._number_count_predictor = BertFeedForward(config, 5 * config.hidden_size, config.hidden_size, 10)

        if "negation" in self.answering_abilities:
            self._negation_index = self.answering_abilities.index("negation")
            self._number_negation_predictor = BertFeedForward(config, 5 * config.hidden_size, config.hidden_size, 2)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, input_ids, token_type_ids, attention_mask, number_indices,
                answer_as_span_starts=None, answer_as_span_ends=None, answer_as_span_numbers=None,
                answer_as_counts=None, answer_as_add_sub_expressions=None, answer_as_negations=None,
                number_indices2=None, sign_indices=None, sign_labels=None, encoded_numbers_input=None,
                passage_input=None, question_input=None, pooled_input=None):
        if mode == "rerank_inference":
            assert number_indices2 is not None and sign_indices is not None and encoded_numbers_input is not None and \
                   passage_input is not None and question_input is not None and pooled_input is not None
            sign_encoded_numbers = encoded_numbers_input.unsqueeze(1).repeat(1, number_indices2.size(1), 1, 1)

            sign_mask = (number_indices2 != -1).long()
            clamped_number_indices2 = replace_masked_values(number_indices2, sign_mask, 0)
            sign_output = torch.gather(sign_encoded_numbers, 2,
                                       clamped_number_indices2.unsqueeze(-1).expand(-1, -1, -1, sign_encoded_numbers.size(-1)))

            clamped_sign_indices = replace_masked_values(sign_indices, sign_mask, 0)
            sign_embeddings = self._sign_embeddings(clamped_sign_indices)
            sign_output += sign_embeddings

            sign_weights = self._sign_rerank_affine(sign_output).squeeze(-1)
            sign_pooled_output = get_self_att_representation(sign_output, sign_weights, sign_mask, dim=2)

            sign_pooled_output = torch.cat(
                [sign_pooled_output, passage_input.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1),
                 question_input.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1),
                 pooled_input.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1)], -1)

            sign_rerank_logits = self._sign_rerank_predictor(sign_pooled_output).squeeze(-1)
            return sign_rerank_logits

        elif mode == "normal":
            all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

            passage_weights = self._passage_affine(all_encoder_layers[self.base_pos]).squeeze(-1)
            passage_vector = get_self_att_representation(all_encoder_layers[self.base_pos], passage_weights, token_type_ids)

            question_weights = self._question_affine(all_encoder_layers[self.base_pos]).squeeze(-1)
            question_vector = get_self_att_representation(all_encoder_layers[self.base_pos], question_weights, (1-token_type_ids))

            if len(self.answering_abilities) >= 1:
                # Shape: (batch_size, number_of_abilities)
                answer_ability_logits = \
                    self._answer_ability_predictor(torch.cat([passage_vector, question_vector, pooled_output], -1))
                answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
                best_answer_ability = torch.argmax(answer_ability_log_probs, -1)

                # Shape: (batch_size, # of numbers in the passage)
                number_indices = number_indices.squeeze(-1)
                number_mask = (number_indices != -1).long()

                if "counting" in self.answering_abilities:
                    # Shape: (batch_size, # of numbers in the passage, 2*hidden_size)
                    encoded_passage_for_numbers = torch.cat(
                        [all_encoder_layers[self.base_pos], all_encoder_layers[self.number_pos]], dim=-1)
                    encoded_numbers = gather_representations(encoded_passage_for_numbers, number_indices)

                    # Shape: (batch_size, hidden_size)
                    count_weights = self._number_count_affine(encoded_numbers).squeeze(-1)
                    count_pooled_output = get_self_att_representation(encoded_numbers, count_weights, number_mask)

                    # Shape: (batch_size, 10)
                    count_number_logits = self._number_count_predictor(
                        torch.cat([count_pooled_output, passage_vector, question_vector, pooled_output], -1))
                    count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)

                    # Info about the best count number prediction
                    # Shape: (batch_size,)
                    best_count_number = torch.argmax(count_number_log_probs, -1)

                if "span_extraction" in self.answering_abilities:
                    base_weights = self._base_predictor(all_encoder_layers[self.base_pos]).squeeze(-1)
                    base_q_pooled_output = get_self_att_representation(all_encoder_layers[self.base_pos],
                                                                       base_weights, (1 - token_type_ids))

                    start_weights = self._start_predictor(all_encoder_layers[self.start_pos]).squeeze(-1)
                    start_q_pooled_output = get_self_att_representation(all_encoder_layers[self.start_pos],
                                                                        start_weights, (1 - token_type_ids))

                    end_weights = self._end_predictor(all_encoder_layers[self.end_pos]).squeeze(-1)
                    end_q_pooled_output = get_self_att_representation(all_encoder_layers[self.end_pos],
                                                                      end_weights, (1 - token_type_ids))

                    start_output = torch.cat((all_encoder_layers[self.base_pos], all_encoder_layers[self.start_pos],
                                              base_q_pooled_output.unsqueeze(1) * all_encoder_layers[self.base_pos],
                                              start_q_pooled_output.unsqueeze(1) * all_encoder_layers[self.start_pos]), -1)

                    end_output = torch.cat((all_encoder_layers[self.base_pos], all_encoder_layers[self.end_pos],
                                            base_q_pooled_output.unsqueeze(1) * all_encoder_layers[self.base_pos],
                                            end_q_pooled_output.unsqueeze(1) * all_encoder_layers[self.end_pos]), -1)

                    span_start_logits = self._start_affine(start_output).squeeze(-1)
                    span_end_logits = self._end_affine(end_output).squeeze(-1)

                    span_start_log_probs = torch.nn.functional.log_softmax(span_start_logits, -1)
                    span_end_log_probs = torch.nn.functional.log_softmax(span_end_logits, -1)

                    # Shape: (batch_size, 8)
                    span_number_logits = self._span_number_predictor(torch.cat([passage_vector, question_vector, pooled_output], -1))
                    span_number_log_probs = torch.nn.functional.log_softmax(span_number_logits, -1)

                    # Info about the best count number prediction
                    # Shape: (batch_size,)
                    best_span_number = torch.argmax(span_number_log_probs, -1)

                if "addition_subtraction" in self.answering_abilities:
                    # Shape: (batch_size, # of numbers in the passage, 2*hidden_size)
                    encoded_passage_for_numbers = torch.cat(
                        [all_encoder_layers[self.base_pos], all_encoder_layers[self.number_pos]], dim=-1)
                    encoded_numbers = gather_representations(encoded_passage_for_numbers, number_indices)

                    # Shape: (batch_size, # of numbers in the passage, 5*hidden_size)
                    concat_encoded_numbers = torch.cat(
                        [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                         question_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                         pooled_output.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

                    # Shape: (batch_size, # of numbers in the passage, 3)
                    number_sign_logits = self._number_sign_predictor(concat_encoded_numbers)

                    number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

                    # Rerank
                    if number_indices2 is not None and sign_indices is not None:
                        # Shape: (batch_size, beam_size, # of numbers in the passage, 2*hidden_size)
                        sign_encoded_numbers = encoded_numbers.unsqueeze(1).repeat(1, number_indices2.size(1), 1, 1)

                        # Shape: (batch_size, beam_size, max_count)
                        sign_mask = (number_indices2 != -1).long()
                        clamped_number_indices2 = replace_masked_values(number_indices2, sign_mask, 0)
                        # Shape: (batch_size, beam_size, max_count, 2*hidden_size)
                        sign_output = torch.gather(sign_encoded_numbers, 2,
                                                   clamped_number_indices2.unsqueeze(-1).expand(-1, -1, -1, sign_encoded_numbers.size(-1)))

                        # Shape: (batch_size, beam_size, max_count, 2*hidden_size)
                        clamped_sign_indices = replace_masked_values(sign_indices, sign_mask, 0)
                        sign_embeddings = self._sign_embeddings(clamped_sign_indices)
                        sign_output += sign_embeddings

                        # Shape: (batch_size, beam_size, 2*hidden_size)
                        sign_weights = self._sign_rerank_affine(sign_output).squeeze(-1)
                        sign_pooled_output = get_self_att_representation(sign_output, sign_weights, sign_mask, dim=2)

                        sign_pooled_output = torch.cat(
                            [sign_pooled_output, passage_vector.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1),
                             question_vector.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1),
                             pooled_output.unsqueeze(1).repeat(1, sign_pooled_output.size(1), 1)], -1)

                        # Shape: (batch_size, beam_size)
                        sign_rerank_logits = self._sign_rerank_predictor(sign_pooled_output).squeeze(-1)

                if "negation" in self.answering_abilities:
                    # Shape: (batch_size, # of numbers in the passage, 2*hidden_size)
                    encoded_passage_for_numbers = torch.cat(
                        [all_encoder_layers[self.base_pos], all_encoder_layers[self.number_pos]], dim=-1)
                    encoded_numbers = gather_representations(encoded_passage_for_numbers, number_indices)

                    # Shape: (batch_size, # of numbers in the passage, 5*hidden_size)
                    concat_encoded_numbers = torch.cat(
                        [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                         question_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                         pooled_output.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

                    # Shape: (batch_size, # of numbers in the passage, 2)
                    number_negation_logits = self._number_negation_predictor(concat_encoded_numbers)

                    number_negation_log_probs = torch.nn.functional.log_softmax(number_negation_logits, -1)

                    # Shape: (batch_size, # of numbers in passage).
                    best_negations_for_numbers = torch.argmax(number_negation_log_probs, -1)
                    # For padding numbers, the best sign masked as 0 (not included).
                    best_negations_for_numbers = replace_masked_values(best_negations_for_numbers, number_mask, 0)

            # If answer is given, compute the loss.
            if (answer_as_span_starts is not None and answer_as_span_ends is not None and answer_as_span_numbers is not None) \
                or answer_as_counts is not None or answer_as_add_sub_expressions is not None or answer_as_negations is not None \
                or (number_indices2 is not None and sign_indices is not None and sign_labels is not None):

                log_marginal_likelihood_list = []

                for answering_ability in self.answering_abilities:
                    if answering_ability == "span_extraction":
                        # Shape: (batch_size, # of answer spans)
                        log_likelihood_span_starts = gather_log_likelihood(span_start_log_probs, answer_as_span_starts)
                        log_likelihood_span_ends = gather_log_likelihood(span_end_log_probs, answer_as_span_ends)
                        log_likelihood_span_starts = masked_logsumexp(log_likelihood_span_starts, answer_as_span_starts)
                        log_likelihood_span_ends = masked_logsumexp(log_likelihood_span_ends, answer_as_span_ends)

                        log_likelihood_span_numbers = gather_log_likelihood(span_number_log_probs, answer_as_span_numbers)
                        log_likelihood_span_numbers = masked_logsumexp(log_likelihood_span_numbers, answer_as_span_numbers)

                        log_marginal_likelihood_for_span = log_likelihood_span_starts + log_likelihood_span_ends + \
                                                           log_likelihood_span_numbers
                        # Shape: (batch_size, )
                        log_marginal_likelihood_list.append(log_marginal_likelihood_for_span)

                    elif answering_ability == "addition_subtraction":
                        # The padded add-sub combinations use -1 as the signs for all numbers, and we mask them here.
                        # Shape: (batch_size, # of combinations, # of numbers in the passage)
                        gold_add_sub_mask = (answer_as_add_sub_expressions != -1).long()
                        clamped_gold_add_sub_signs = replace_masked_values(answer_as_add_sub_expressions, gold_add_sub_mask, 0)
                        # Shape: (batch_size, # of numbers in the passage, # of combinations)
                        gold_add_sub_signs = clamped_gold_add_sub_signs.transpose(1, 2)
                        # Shape: (batch_size, # of numbers in the passage, # of combinations)
                        log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs)
                        # the log likelihood of the masked positions should be 0
                        # so that it will not affect the joint probability
                        log_likelihood_for_number_signs = \
                            replace_masked_values(log_likelihood_for_number_signs, number_mask.unsqueeze(-1), 0)
                        # Shape: (batch_size, # of combinations)
                        log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
                        # For those padded combinations, we set their log probabilities to be very small negative value
                        # Shape: (batch_size, # of combinations)
                        gold_combination_mask = (gold_add_sub_mask.sum(-1) != 0).long()
                        log_likelihood_for_add_subs = \
                            replace_masked_values(log_likelihood_for_add_subs, gold_combination_mask, -1e7)
                        # Shape: (batch_size, )
                        log_likelihood_for_add_sub = logsumexp(log_likelihood_for_add_subs)
                        # Shape: (batch_size, )
                        log_likelihood_sign_rerank = distant_cross_entropy(sign_rerank_logits, sign_labels)

                        log_marginal_likelihood_for_add_sub = log_likelihood_for_add_sub + log_likelihood_sign_rerank
                        log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)

                    elif answering_ability == "counting":
                        # Shape: (batch_size, # of count answers)
                        log_likelihood_for_counts = gather_log_likelihood(count_number_log_probs, answer_as_counts)
                        # Shape: (batch_size, )
                        log_marginal_likelihood_for_count = masked_logsumexp(log_likelihood_for_counts, answer_as_counts)
                        log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)

                    elif answering_ability == "negation":
                        # The padded add-sub combinations use -1 as the signs for all numbers, and we mask them here.
                        # Shape: (batch_size, # of combinations, # of numbers in the passage)
                        gold_negation_mask = (answer_as_negations != -1).long()
                        clamped_gold_negations = replace_masked_values(answer_as_negations, gold_negation_mask, 0)
                        # Shape: (batch_size, # of numbers in the passage, # of combinations)
                        gold_negations = clamped_gold_negations.transpose(1, 2)
                        # Shape: (batch_size, # of numbers in the passage, # of combinations)
                        log_likelihood_for_negations = torch.gather(number_negation_log_probs, 2, gold_negations)
                        # the log likelihood of the masked positions should be 0
                        # so that it will not affect the joint probability
                        log_likelihood_for_negations = \
                            replace_masked_values(log_likelihood_for_negations, number_mask.unsqueeze(-1), 0)
                        # Shape: (batch_size, # of combinations)
                        log_likelihood_for_negations = log_likelihood_for_negations.sum(1)
                        # For those padded combinations, we set their log probabilities to be very small negative value
                        # Shape: (batch_size, # of combinations)
                        gold_combination_mask = (gold_negation_mask.sum(-1) != 0).long()
                        log_likelihood_for_negations = \
                            replace_masked_values(log_likelihood_for_negations, gold_combination_mask, -1e7)
                        # Shape: (batch_size, )
                        log_marginal_likelihood_for_negation = logsumexp(log_likelihood_for_negations)
                        log_marginal_likelihood_list.append(log_marginal_likelihood_for_negation)

                    else:
                        raise ValueError(f"Unsupported answering ability: {answering_ability}")

                if len(self.answering_abilities) > 1:
                    # Add the ability probabilities if there are more than one abilities
                    all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                    all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                    marginal_log_likelihood = logsumexp(all_log_marginal_likelihoods)
                else:
                    marginal_log_likelihood = log_marginal_likelihood_list[0]
                return - marginal_log_likelihood.mean()

            else:
                output_dict = {}
                if len(self.answering_abilities) >= 1:
                    output_dict["best_answer_ability"] = best_answer_ability
                if "counting" in self.answering_abilities:
                    output_dict["best_count_number"] = best_count_number
                if "negation" in self.answering_abilities:
                    output_dict["best_negations_for_numbers"] = best_negations_for_numbers
                if "span_extraction" in self.answering_abilities:
                    output_dict["span_start_logits"] = span_start_logits
                    output_dict["span_end_logits"] = span_end_logits
                    output_dict["best_span_number"] = best_span_number
                if "addition_subtraction" in self.answering_abilities:
                    output_dict["number_sign_logits"] = number_sign_logits
                    output_dict["number_mask"] = number_mask
                    output_dict["encoded_numbers_output"] = encoded_numbers
                    output_dict["passage_output"] = passage_vector
                    output_dict["question_output"] = question_vector
                    output_dict["pooled_output"] = pooled_output
                return output_dict
        else:
            raise Exception
