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


        """Association Module"""
        # Diagnostician network, C
        self.asso_MRI = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50 + 4,)),  # dim(Concat(a, x_MRI)) = 90
                tf.keras.layers.Dense(units=100, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=25 * 4, activation=None, kernel_regularizer='L1L2'),
            ]
        )

        self.asso_PET = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50 + 4,)),  # dim(Concat(a, x_MRI)) = 90
                tf.keras.layers.Dense(units=100, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=25 * 4, activation=None, kernel_regularizer='L1L2'),
            ]
        )

        """Diagnostician Module"""
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

    # @tf.function
    # Reconstructed SNPs sampling
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(10, 50))
        return self.decode(eps, apply_sigmoid=True)

    # Construct latent distribution
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.math.exp(logvar * .5) + mean

    # Association analysis
    def asso_SNP2MRI(self, z_SNP, c_demo):
        z = tf.concat([c_demo, z_SNP], axis=-1)
        z_MRI_com_fake, z_MRI_spe_fake, mask_MRI_com, mask_MRI_spe =\
            tf.split(self.asso_MRI(z), num_or_size_splits=4, axis=1)
        return z_MRI_com_fake, z_MRI_spe_fake, mask_MRI_com, mask_MRI_spe

    def asso_SNP2PET(self, z_SNP, c_demo):
        z = tf.concat([c_demo, z_SNP], axis=-1)
        z_PET_com_fake, z_PET_spe_fake, mask_PET_com, mask_PET_spe = \
            tf.split(self.asso_PET(z), num_or_size_splits=4, axis=1)
        return z_PET_com_fake, z_PET_spe_fake, mask_PET_com, mask_PET_spe

    # Downstream tasks; brain disease diagnosis and cognitive score prediction
    def diagnose(self, x_MRI_com, x_MRI_spe, x_PET_com, x_PET_spe, mask_MRI_com, mask_MRI_spe,
                 mask_PET_com, mask_PET_spe, apply_logistic_activation=False):
        x_MRI_com = tf.multiply(x_MRI_com, mask_MRI_com)
        x_MRI_spe = tf.multiply(x_MRI_spe, mask_MRI_spe)
        x_PET_com = tf.multiply(x_PET_com, mask_PET_com)
        x_PET_spe = tf.multiply(x_PET_spe, mask_PET_spe)
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
