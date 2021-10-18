import tensorflow as tf
from model_basics import *
from model import MEEDEmbedder, PlainEncoder


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True, reduction = 'none')
    loss_ = loss_object(real, pred)
    return loss_

def loss_function_weighted(real, pred, class_weights):
    # real.shape & pred.shape == (batch_size, num_classes)
    # real is assumed to be one-hot
    # pred is assumed to be logits
    log_y_pred = pred - tf.expand_dims(tf.math.reduce_logsumexp(pred, 1), 1)
    return -tf.reduce_sum(real * log_y_pred * class_weights, 1)  # (batch_size,)


# MEED encoder pooler, a response emotion classifier based on the encoder outputs.
class MEEDEncoderPooler(tf.keras.Model):
    def __init__(self, d_model, num_emotions):
        super().__init__(name = 'encoder_pooler')
        self.hidden_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'hidden_layer')
        self.output_layer = tf.keras.layers.Dense(num_emotions, name = 'output_layer')

        self.attention_v = tf.keras.layers.Dense(1, use_bias = False, name = 'attention_v')
        self.attention_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'attention_layer')

    def call(self, x, mask):
        # x.shape == (batch_size, seq_len, d_model)
        # mask.shape == (batch_size, 1, 1, seq_len)

        # Compute the attention scores
        projected = self.attention_layer(x)  # (batch_size, seq_len, d_model)
        logits = tf.squeeze(self.attention_v(projected), 2)  # (batch_size, seq_len)
        logits += (tf.squeeze(mask) * -1e9)  # Mask out the padding positions
        scores = tf.expand_dims(tf.nn.softmax(logits), 1)  # (batch_size, 1, seq_len)

        # x.shape == (batch_size, d_model)
        x = tf.squeeze(tf.matmul(scores, x), 1)

        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x  # (batch_size, num_emotions)

# All history utterances are concatenated into one sequence.
class EmotionPredictor(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
                 layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions):
        super().__init__()

        self.embedder = MEEDEmbedder(d_model, dropout_rate, layer_norm_eps,
            max_position_embed, type_vocab_size, vocab_size, num_emotions)

        self.encoder = PlainEncoder(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps)

        self.encoder_pooler = MEEDEncoderPooler(d_model, num_emotions)

    def call(self, inp, inp_seg, inp_emot, training, enc_padding_mask):
        # inp_embed.shape == (batch_size, input_seq_len, d_model)
        inp_embed = self.embedder(inp, inp_seg, inp_emot, training)

        # enc_output.shape == (batch_size, input_seq_len, d_model)
        enc_output = self.encoder(inp_embed, training, enc_padding_mask)

        # pred_tar_emot.shape == (batch_size, num_emotions)
        pred_tar_emot = self.encoder_pooler(enc_output, enc_padding_mask)

        return pred_tar_emot
