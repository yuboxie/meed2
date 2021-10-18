import tensorflow as tf
from model_basics import *


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True, reduction = 'none')

    # To be consistent with RoBERTa, the padding index is set to 1.
    mask = tf.math.logical_not(tf.math.equal(real, 1))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype = loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_, 1) / tf.reduce_sum(mask, 1)


class MEEDEmbedder(tf.keras.Model):
    def __init__(self, d_model, dropout_rate, layer_norm_eps,
                 max_position_embed, type_vocab_size, vocab_size, num_emotions):
        super().__init__(name = 'meed_embedder')

        self.padding_idx = 1

        self.word_embeddings = tf.keras.layers.Embedding(vocab_size, d_model, name = 'word_embed')
        self.pos_embeddings = tf.keras.layers.Embedding(max_position_embed, d_model, name = 'pos_embed')
        self.seg_embeddings = tf.keras.layers.Embedding(type_vocab_size, d_model, name = 'seg_embed')
        self.emot_embeddings = tf.keras.layers.Embedding(num_emotions, d_model, name = 'emot_embed')

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_embed')
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_embed')

    def call(self, x, seg, emot, training):
        # x.shape == (batch_size, seq_len)
        # seg.shape == (batch_size, seq_len)
        # emot.shape == (batch_size, seq_len), emoji ids

        seq_len = tf.shape(x)[1]

        # Add word embedding and position embedding.
        pos = tf.range(self.padding_idx + 1, seq_len + self.padding_idx + 1)
        pos = tf.broadcast_to(pos, tf.shape(x))
        x = self.word_embeddings(x)  # (batch_size, seq_len, d_model)
        x += self.pos_embeddings(pos)
        x += self.seg_embeddings(seg)
        x += self.emot_embeddings(emot)

        x = self.layernorm(x)
        x = self.dropout(x, training = training)

        return x  # (batch_size, seq_len, d_model)

class PlainEncoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act,
                 dropout_rate, layer_norm_eps):
        super().__init__(name = 'plain_encoder')
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, i)
            for i in range(num_layers)
        ]

    def call(self, x, training, mask):
        # x.shape == (batch_size, input_seq_len, d_model)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)

class PlainDecoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act,
                 dropout_rate, layer_norm_eps):
        super().__init__(name = 'plain_decoder')
        self.num_layers = num_layers
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, i)
            for i in range(num_layers)
        ]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # x.shape == (batch_size, target_seq_len, d_model)

        attention_weights = {}
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

# All history utterances are concatenated into one sequence.
class MEED(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
                 layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions):
        super().__init__()

        self.embedder = MEEDEmbedder(d_model, dropout_rate, layer_norm_eps,
            max_position_embed, type_vocab_size, vocab_size, num_emotions)

        self.encoder = PlainEncoder(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps)

        self.decoder = PlainDecoder(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps)

        self.final_layer = tf.keras.layers.Dense(vocab_size, name = 'final_layer')

    def call(self, inp, inp_seg, inp_emot, tar, tar_seg, tar_emot, training,
             enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # inp_embed.shape == (batch_size, input_seq_len, d_model)
        inp_embed = self.embedder(inp, inp_seg, inp_emot, training)

        # enc_output.shape == (batch_size, input_seq_len, d_model)
        enc_output = self.encoder(inp_embed, training, enc_padding_mask)

        # tar_embed.shape == (batch_size, target_seq_len, d_model)
        tar_embed = self.embedder(tar, tar_seg, tf.expand_dims(tar_emot, 1), training)

        # dec_output.shape == (batch_size, target_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar_embed,
            enc_output, training, look_ahead_mask, dec_padding_mask)

        # final_output.shape == (batch_size, target_seq_len, vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def encode(self, inp, inp_seg, inp_emot, training, enc_padding_mask):
        # inp_embed.shape == (batch_size, input_seq_len, d_model)
        inp_embed = self.embedder(inp, inp_seg, inp_emot, training)

        # enc_output.shape == (batch_size, input_seq_len, d_model)
        enc_output = self.encoder(inp_embed, training, enc_padding_mask)

        return enc_output

    def decode(self, enc_output, pred_tar_emot, tar, tar_seg, training, look_ahead_mask, dec_padding_mask):
        # tar_embed.shape == (batch_size, target_seq_len, d_model)
        tar_embed = self.embedder(tar, tar_seg, tf.expand_dims(pred_tar_emot, 1), training)

        # dec_output.shape == (batch_size, target_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar_embed,
            enc_output, training, look_ahead_mask, dec_padding_mask)

        # final_output.shape == (batch_size, target_seq_len, vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
