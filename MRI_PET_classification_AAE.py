"""
Script for ENGINE: Enhancing Neuroimaging and Genetic Information by Neural Embedding framework
Written in Tensorflow 2.1.0
"""

# Import APIs
import tensorflow as tf
import numpy as np
import utils


class engine(tf.keras.Model):
    tf.keras.backend.set_floatx('float32')
    """ENGINE framework"""

    def __init__(self, N_o):
        super(engine, self).__init__()
        self.N_o = N_o  # the number of classification outputs

        """MRI Representation Module"""
        # Encoder network, MRI share
        self.encoder_MRI_share = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(90,)),  # F_MRI = 90
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=50, activation='elu', kernel_regularizer='L1L2'),  # 2 * dim(z_MRI) = 100
            ]
        )

        # Encoder network, MRI
        self.encoder_MRI_disentangled = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)),  # dim(f) = 25
                tf.keras.layers.Dense(units=50, activation=None, kernel_regularizer='L1L2'),  # 2 * dim(z_MRI) = 100
            ]
        )

        # MRI Decoder network
        self.decoder_MRI = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)),  # dim(z_MRI_com) + dim(z_MRI_spe) = 50
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=90, activation=None, kernel_regularizer='L1L2'),  # F_MRI = 90
            ]
        )

        """PET Representation Module"""
        # Encoder network, PET share
        self.encoder_PET_share = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(90,)),  # F_PET = 90
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=50, activation='elu', kernel_regularizer='L1L2'),  # 2 * dim(z_PET) = 100
            ]
        )
        # Encoder network, PET
        self.encoder_PET_disentangled = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)),  # dim(f) = 25
                tf.keras.layers.Dense(units=50, activation=None, kernel_regularizer='L1L2'),  # 2 * dim(z_PET) = 100
            ]
        )

        # PET Decoder network,
        self.decoder_PET = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)),  # dim(z_PET_com) + dim(z_PET_spe) = 50
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=90, activation=None, kernel_regularizer='L1L2'),  # F_PET = 90
            ]
        )


        """Diagnostician Module"""
        # Diagnostician network, C
        self.diagnostician_share = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(75,)),  # dim(Concat(a, x_MRI)) = 90
                tf.keras.layers.Dense(units=25, activation='elu', kernel_regularizer='L1L2'),
            ]
        )

        self.diagnostician_clf = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(25,)),  # dim(f) = 25
                tf.keras.layers.Dense(units=self.N_o, activation=None, kernel_regularizer='L1L2'),  # |N_o|
            ]
        )

        self.diagnostician_reg = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(25,)),  # dim(f) = 25
                tf.keras.layers.Dense(units=1, activation=None, kernel_regularizer='L1L2'),  # 1
            ]
        )

        """Discriminator Module"""
        # Discriminator network,
        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)), # F_MRI = 93
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=100, activation='relu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer='L1L2'), # real or fake
            ]
        )

    # @tf.function
    def drop_out(self, input, keep_prob):
        return tf.nn.dropout(input, keep_prob)

    # Represent mu and sigma from the input MRI
    def encode_MRI1(self, x_MRI):
        zb_MRI = self.encoder_MRI_share(x_MRI)
        return zb_MRI

    def encode_MRI2(self, zb_MRI):
        zb_MRI_com, zb_MRI_spe = tf.split(self.encoder_MRI_disentangled(zb_MRI), num_or_size_splits=2, axis=1)
        return zb_MRI_com, zb_MRI_spe
    

    # Represent mu and sigma from the input PET
    def encode_PET1(self, x_PET):
        zb_PET = self.encoder_PET_share(x_PET)
        return zb_PET

    def encode_PET2(self, zb_PET):
        zb_PET_com, zb_PET_spe = tf.split(self.encoder_PET_disentangled(zb_PET), num_or_size_splits=2, axis=1)
        return zb_PET_com, zb_PET_spe


    # Reconstruct the input MRI
    def decode_MRI(self, z_com, z_spe, apply_sigmoid=False):
        logits = self.decoder_MRI(tf.concat([z_com, z_spe], axis=-1))
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits

    # Reconstruct the input MRI
    def decode_PET(self, z_com, z_spe, apply_sigmoid=False):
        logits = self.decoder_PET(tf.concat([z_com, z_spe], axis=-1))
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits


    # Downstream tasks; brain disease diagnosis and cognitive score prediction
    def diagnose(self, x_MRI_com, x_MRI_spe, x_PET_com, x_PET_spe, apply_logistic_activation=False):
        com = (x_MRI_com + x_PET_com) / 2
        feature_in = tf.concat([com, x_MRI_spe, x_PET_spe], axis=-1)
        # feature_in = tf.concat([x_MRI_com, x_PET_com, x_MRI_spe, x_PET_spe], axis=-1)
        feature = self.diagnostician_share(feature_in)  # Hadamard production of the attentive vector
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            y_hat = tf.math.softmax(logit_clf)
            s_hat = tf.math.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

    def discriminate(self, z):
        return self.discriminator(z)


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