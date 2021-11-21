import time
import numpy as np
import pandas as pd
import tensorflow as tf
from os import mkdir
from os.path import exists
from math import ceil
from pytorch_transformers import RobertaTokenizer

from model_utils import *
from model import MEED
from datasets import *
from beam_search import beam_search


# Some hyper-parameters
num_layers = 4
d_model = 300
num_heads = 6
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 102
type_vocab_size = 2  # Segments

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size
SOS_ID = tokenizer.encode('<s>')[0]
EOS_ID = tokenizer.encode('</s>')[0]

beam_width = 32
alpha = 1.0  # Decoding length normalization coefficient
n_gram = 4  # n-gram repeat blocking in beam search

num_emotions = 41
max_length = 100  # Maximum number of tokens
buffer_size = 300000
batch_size = 1  # For prediction, we always use batch size 1.
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

model_optimal_epoch = {'meed_os': 50, 'meed_os_edos': 6, 'meed_os_ed': 10}


def evaluate(meed, inp, inp_seg, inp_emot, pred_tar_emot, tar_seg):
    enc_padding_mask = create_padding_mask(inp)
    enc_output = meed.encode(inp, inp_seg, inp_emot, False, enc_padding_mask)

    def iter_func(dec_inp, bw):
        enc_output_tiled = tf.tile(enc_output, [bw, 1, 1])
        dec_padding_mask = tf.tile(enc_padding_mask, [bw, 1, 1, 1])

        look_ahead_mask = create_look_ahead_mask(tf.shape(dec_inp)[1])
        dec_target_padding_mask = create_padding_mask(dec_inp)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        dec_inp_seg = tf.ones_like(dec_inp) * tar_seg[0,0]
        pred_tar_emot_tiled = tf.constant([pred_tar_emot] * dec_inp.shape[0])

        pred, attention_weights = meed.decode(enc_output_tiled, pred_tar_emot_tiled, dec_inp,
            dec_inp_seg, False, combined_mask, dec_padding_mask)
        return pred.numpy()

    result_seqs, log_probs = beam_search(iter_func, beam_width, max_length - 1, SOS_ID, EOS_ID, alpha, n_gram)

    return result_seqs, log_probs

def main(model_name, dataset):
    optimal_epoch = model_optimal_epoch[model_name]
    checkpoint_path = 'checkpoints/{}'.format(model_name)
    pred_emot_path = 'prediction/{}/emo_pred_{}.csv'.format(dataset, model_name[5:])
    save_path = 'prediction/{}/{}.csv'.format(dataset, model_name)

    index_path = 'data/{}/test_2000_index.npy'.format(dataset)
    index = np.load(index_path)

    if dataset == 'os' or dataset == 'edos':
        data_path = 'data/{}/test_human'.format(dataset)
        test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length, index)
    elif dataset == 'ed':
        data_path = 'data/ed'
        _, _, test_dataset, _, N = create_ed_datasets(tokenizer, data_path, buffer_size, batch_size, max_length, index)


    # Define the model.
    meed = MEED(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)

    # Build the model.
    build_meed_model(meed, max_length, vocab_size)
    print('Model has been built.')

    # Define optimizer and metrics.
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = meed, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    # Restore from the optimal_epoch.
    ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
    print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))


    pred_emot_df = pd.read_csv(pred_emot_path).iloc[index]
    print('pred_emot_df.shape = {}'.format(pred_emot_df.shape))

    contexts = []
    pred_ys = []
    pred_emots = []
    tar_ys = []
    tar_emots = []

    for (i, inputs) in tqdm(enumerate(test_dataset), total = ceil(N / batch_size)):
        inp, inp_seg, inp_emot, _, tar_real, tar_seg, _ = inputs
        pred_emot = pred_emot_df.iloc[i]['y_pred']
        tar_emot = pred_emot_df.iloc[i]['y_true']
        pred_emots.append(pred_emot)
        tar_emots.append(tar_emot)

        context = tokenizer.decode(inp[0].numpy().tolist())
        context = ['- {}'.format(u.strip()) for u in context if '<pad>' not in u]
        context = '\n'.join(context)
        contexts.append(context)

        tar_preds, log_probs = evaluate(meed, inp, inp_seg, inp_emot, pred_emot, tar_seg)
        tar_pred_dec = tokenizer.decode(tar_preds[0])  # top candidate of beam search
        pred_y = tar_pred_dec[0].strip() if len(tar_pred_dec) > 0 else ''
        pred_ys.append(pred_y)

        tar_y = tokenizer.decode([SOS_ID] + tar_real[0].numpy().tolist())[0].strip()
        tar_ys.append(tar_y)

    print('Saving the prediction results...')
    data = {'context': contexts, 'pred_y': pred_ys, 'pred_emot': pred_emots,
            'tar_y': tar_ys, 'tar_emot': tar_emots}
    pd.DataFrame(data).to_csv(save_path)


if __name__ == '__main__':
    for model_name in ['meed_os', 'meed_os_edos', 'meed_os_ed']:
        for dataset in ['os', 'edos', 'ed']:
            main(model_name, dataset)
