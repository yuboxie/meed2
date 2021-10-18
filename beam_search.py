import numpy as np
import tensorflow as tf
from scipy.special import logsumexp


def beam_search(iter_func, beam_width, max_length, SOS_ID, EOS_ID, alpha, n_gram):
    """A simple implementation of beam search.
    This function assumes that the batch size is 1.
    iter_func should return numpy values.
    alpha is the length normalization coefficient.
    """
    dec_inp = tf.expand_dims([SOS_ID], 0)
    pred = iter_func(dec_inp, 1)  # (1, 1, vocab_size)
    pred = pred.reshape(-1)  # (vocab_size,)
    pred_id = pred.argsort()[-beam_width:][::-1]

    result_seqs = [[SOS_ID, i] for i in pred_id]  # (beam_width, seq_len)
    log_probs = pred[pred_id] - logsumexp(pred)  # (beam_width,)
    result_seqs, log_probs, new_beam_width = _reorganize(result_seqs, log_probs, EOS_ID, n_gram)

    vocab_size = pred.shape[0]

    for _ in range(max_length - 1):
        if new_beam_width == 0:
            break

        dec_inp = tf.constant(result_seqs[:new_beam_width])
        pred = iter_func(dec_inp, new_beam_width)  # (new_beam_width, seq_len, vocab_size)

        pred = pred[:,-1,:]  # (new_beam_width, vocab_size)
        current_log_probs = pred - np.expand_dims(logsumexp(pred, 1), 1)
        current_log_probs += np.expand_dims(log_probs[:new_beam_width], 1)
        current_log_probs = current_log_probs.reshape(-1)  # (new_beam_width * vocab_size,)
        pred_id = current_log_probs.argsort()[-new_beam_width:][::-1]

        beam_id = pred_id // vocab_size
        vocab_id = pred_id % vocab_size

        log_probs[:new_beam_width] = current_log_probs[pred_id]

        new_seqs = []
        for beam, token in zip(beam_id, vocab_id):
            new_seqs.append(result_seqs[beam] + [token])
        result_seqs[:new_beam_width] = new_seqs

        result_seqs, log_probs, new_beam_width = _reorganize(result_seqs, log_probs, EOS_ID, n_gram)

    # Add EOS token to unfinished sequences.
    if new_beam_width > 0:
        dec_inp = tf.constant(result_seqs[:new_beam_width])
        pred = iter_func(dec_inp, new_beam_width)[:,-1,:]  # (new_beam_width, vocab_size)
        eos_log_probs = pred[:,EOS_ID] - logsumexp(pred, 1)
        log_probs[:new_beam_width] += eos_log_probs
        for i in range(new_beam_width):
            result_seqs[i].append(EOS_ID)

    # Sort the resulted sequences by their log probabilities.
    result_seqs, log_probs = _sort_by_log_probs(result_seqs, log_probs, alpha)

    return result_seqs, log_probs

def _n_gram_overlap(seq, n_gram):
    if len(seq) <= n_gram + 1:
        return False
    seq_str = [str(t) for t in seq[1:]]
    n_gram_set = set()
    for i in range(len(seq_str) - n_gram + 1):
        n_gram_str = ','.join(seq_str[i:(i+n_gram)])
        if not n_gram_str in n_gram_set:
            n_gram_set.add(n_gram_str)
        else:
            return True
    return False

def _reorganize(result_seqs, log_probs, EOS_ID, n_gram):
    """Reorganize result_seqs so that finished sequences are at the end.
    """
    continue_id = []
    finished_id = []
    continue_seqs = []
    finished_seqs = []
    for i, seq in enumerate(result_seqs):
        if seq[-1] == EOS_ID:
            finished_id.append(i)
            finished_seqs.append(seq)
        else:
            continue_id.append(i)
            continue_seqs.append(seq)
            if _n_gram_overlap(seq, n_gram):
                log_probs[i] = -1e20
    new_beam_width = len(continue_id)
    reordered_id = continue_id + finished_id
    result_seqs = continue_seqs + finished_seqs
    return result_seqs, log_probs[reordered_id], new_beam_width

def _sort_by_log_probs(result_seqs, log_probs, alpha):
    seq_lens = np.array([len(seq) - 1 for seq in result_seqs])
    len_norm = seq_lens ** alpha
    sorted_id = (log_probs / len_norm).argsort()[::-1]
    new_log_probs = (log_probs / len_norm)[sorted_id]
    new_result_seqs = [result_seqs[i] for i in sorted_id]
    return new_result_seqs, new_log_probs
