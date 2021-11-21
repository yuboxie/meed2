import time
import numpy as np
import pandas as pd
import tensorflow as tf
from os import mkdir
from os.path import exists
from tqdm import tqdm
from math import ceil
from sklearn.metrics import precision_recall_fscore_support
from pytorch_transformers import RobertaTokenizer

from model_utils import *
from model_emo_pred import EmotionPredictor, loss_function
from datasets import *


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

num_emotions = 41
max_length = 100  # Maximum number of tokens
buffer_size = 300000
batch_size = 256
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

model_optimal_epoch = {'emo_pred_os': 3, 'emo_pred_os_edos': 2, 'emo_pred_os_ed': 4}
log_path = 'prediction/emo_pred.log'


def main(model_name, dataset, f_out):
    f_out.write('{} predicting on {}...\n'.format(model_name, dataset))

    optimal_epoch = model_optimal_epoch[model_name]
    checkpoint_path = 'checkpoints/{}'.format(model_name)
    save_path = 'prediction/{}/{}.csv'.format(dataset, model_name)

    if dataset == 'os' or dataset == 'edos':
        data_path = 'data/{}/test_human'.format(dataset)
        test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length)
    elif dataset == 'ed':
        data_path = 'data/ed'
        _, _, test_dataset, _, N = create_ed_datasets(tokenizer, data_path, buffer_size, batch_size, max_length)

    # Define the model.
    emotion_predictor = EmotionPredictor(num_layers, d_model, num_heads, dff, hidden_act,
        dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)

    # Build the model.
    build_emo_pred_model(emotion_predictor, max_length, vocab_size)
    f_out.write('Model has been built.\n')

    # Define optimizer and metrics.
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emotion_predictor, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    # Restore from the optimal epoch.
    ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
    f_out.write('Checkpoint {} restored.\n'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))

    y_true = []
    y_pred = []
    for inputs in tqdm(test_dataset, total = ceil(N / batch_size)):
        inp, inp_seg, inp_emot, _, tar_real, _, tar_emot = inputs
        enc_padding_mask = create_padding_mask(inp)
        pred_emot = emotion_predictor(inp, inp_seg, inp_emot, False, enc_padding_mask)
        pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
        y_true += tar_emot.numpy().tolist()
        y_pred += pred_emot.tolist()

    f_out.write('Number of testing examples: {}\n'.format(len(y_true)))
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    f_out.write('Accuracy\t{:.4f}\n'.format(acc))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average = 'macro')
    f_out.write('Macro\tP: {:.4f}, R: {:.4f}, F: {:.4f}\n'.format(p, r, f))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
    f_out.write('Weighted\tP: {:.4f}, R: {:.4f}, F: {:.4f}\n'.format(p, r, f))

    f_out.write('Saving the prediction results...\n\n')
    prediction = {'y_pred': y_pred, 'y_true': y_true}
    pd.DataFrame(prediction).to_csv(save_path)


if __name__ == '__main__':
    f_out = open(log_path, 'w')
    for model_name in ['emo_pred_os', 'emo_pred_os_edos', 'emo_pred_os_ed']:
        for dataset in ['os', 'edos', 'ed']:
            main(model_name, dataset, f_out)
    f_out.close()
