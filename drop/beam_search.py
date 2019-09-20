import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def reduce_mul(l):
    out = 1.0
    for x in l:
        out *= x
    return out


def decode_step(step, encoder_logits):
    words_prob = encoder_logits[step]
    words_prob = softmax(words_prob)
    ouput_step = [(idx, prob) for idx, prob in enumerate(words_prob)]
    ouput_step = sorted(ouput_step, key=lambda x: x[1], reverse=True)
    return ouput_step


def check_exceed(seq, mask, max_count):
    count = 0
    for i, word in enumerate(seq):
        sign_index = word[0]
        if sign_index > 0 and mask[i]:
            count += 1
    if count > max_count:
        return False
    else:
        return True


def beam_search_step(step, encoder_logits, top_seqs, mask, k, max_count):
    all_seqs = []
    for seq in top_seqs:
        seq_score = reduce_mul([_score for _, _score in seq])
        # get current step using encoder_context & seq
        current_step = decode_step(step, encoder_logits)
        for i, word in enumerate(current_step):
            if i >= k:
                break
            word_index, word_score = word
            score = seq_score * word_score
            rs_seq = seq + [word]
            all_seqs.append((rs_seq, score))
    all_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)
    # Expression constraint
    filtered_seqs = [seq for seq, _ in all_seqs if check_exceed(seq, mask, max_count)]
    # topk_seqs = [seq for seq, _ in all_seqs[:k]]
    topk_seqs = [seq for seq in filtered_seqs[:k]]
    return topk_seqs


def beam_search(encoder_logits, mask, beam_size, max_count):
    max_len = sum(mask)
    # START
    top_seqs = [[(0, 1.0)]]
    # loop
    for i in range(1, max_len + 1):
        top_seqs = beam_search_step(i, encoder_logits, top_seqs, mask, beam_size, max_count)

    number_indices_list, sign_indices_list, scores_list = [], [], []
    for seq in top_seqs:
        number_indices, sign_indices = [], []
        for i, word in enumerate(seq):
            sign_index, score = word
            if sign_index > 0 and mask[i]:
                number_indices.append(i)
                sign_indices.append(sign_index)
        if number_indices == [] and sign_indices == []:
            continue
        number_indices_list.append(number_indices)
        sign_indices_list.append(sign_indices)
        seq_score = reduce_mul([_score for _, _score in seq])
        scores_list.append(seq_score)
    if scores_list != []:
        scores_list = softmax(np.array(scores_list))
    return number_indices_list, sign_indices_list, scores_list.tolist()

