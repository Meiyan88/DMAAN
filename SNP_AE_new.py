# Import APIs
import tensorflow as tf
import numpy as np
import utils


class engine(tf.keras.Model):
    tf.keras.backend.set_floatx('float32')

    def __init__(self, N_o):
        super(engine, self).__init__()
        self.N_o = N_o  # the number of classification outputs

        """SNP Representation Module"""
        # Encoder network, Q
        self.encoder_SNP = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(2722,)),  # F_SNP = 2839
                tf.keras.layers.Dense(units=500, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=50, activation=None, kernel_regularizer='L1L2'),  # 2 * dim(z_SNP) = 100
            ]
        )

        # Decoder network, P
        self.decoder_SNP = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)), # dim(z_SNP) = 50
                tf.keras.layers.Dense(units=500, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=2722, activation=None, kernel_regularizer='L1L2'), # F_SNP = 2839
            ]
        )

    # @tf.function

    def encode_SNP(self, x_SNP):
        zb_SNP = self.encoder_SNP(x_SNP)
        return zb_SNP

    def decode_SNP(self, z_SNP, apply_sigmoid=False):
        logits = self.decoder_SNP(z_SNP)
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits


if __name__ == '__main__':
    #### test
    X_SNP = tf.cast(np.random.normal(size=(3, 2098)), tf.float32)
    X_MRI = tf.cast(np.random.normal(size=(3, 90)), tf.float32)
    X_PET = tf.cast(np.random.normal(size=(3, 90)), tf.float32)
    model = engine(N_o=3)
    mu, log_sigma_square = model.encode(x_SNP=X_SNP)
    zb_SNP = model.reparameterize(mean=mu, logvar=log_sigma_square)
    eb_SNP_hat_logit = model.decode(z_SNP=zb_SNP)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=eb_SNP_hat_logit, labels=X_SNP)
    log_prob_eb_SNP_given_zb_SNP = -tf.math.reduce_sum(cross_ent, axis=1)
    log_prob_zb_SNP = utils.log_normal_pdf2(sample=zb_SNP, mean=0., logvar=0.)
    log_q_zb_given_eb_SNP = utils.log_normal_pdf2(sample=zb_SNP, mean=mu, logvar=log_sigma_square)
    L_rec = -tf.math.reduce_mean(log_prob_eb_SNP_given_zb_SNP + log_prob_zb_SNP - log_q_zb_given_eb_SNP)
    s = 0
