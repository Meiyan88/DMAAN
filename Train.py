# import APIs
import utils
import MRI_PET_classification_AAE
import prior
import SNP_AE_new
import DistModel
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import shutil

from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, accuracy_score

gpu_id = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
physical_gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1800)]
)
logical_gpus = tf.config.list_logical_devices("GPU")

import torch

class experiment():
    def __init__(self, fold_idx, task):
        self.fold_idx = fold_idx
        self.task = task

        # Learning schedules
        self.num_epochs = 200  # 100
        self.num_batches = 64
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=1000,
                                                                 decay_rate=.96, staircase=False)  # init_lr: 1e-3

        # Loss control hyperparameter
        self.alpha_rec = 1.0  # reconstruction
        self.alpha_dist = 1.0
        # self.alpha_gen = .5  # generation
        # self.alpha_dis = 1  # discrimination
        self.alpha_clf = 1.0  # classification
        self.alpha_reg = 1.0  # regression
        self.alpha_SNP = 1.0
        self.alpha_G = 1.0
        self.alpha_D = 1.0

    def training(self):
        best_test_metric = 0
        print(f' Start Training, Fold {self.fold_idx}')

        ADNI1_id_tr1, ADNI1_id_vl1, ADNI1_id_ts1, \
        X_MRI_tr_ADNI1, X_PET_tr_ADNI1, X_SNP_tr_ADNI1, C_dmg_tr_ADNI1, Y_dis_tr_ADNI1, S_cog_tr_ADNI1, \
        X_MRI_vl_ADNI1, X_PET_vl_ADNI1, X_SNP_vl_ADNI1, C_dmg_vl_ADNI1, Y_dis_vl_ADNI1, S_cog_vl_ADNI1, \
        X_MRI_ts_ADNI1, X_PET_ts_ADNI1, X_SNP_ts_ADNI1, C_dmg_ts_ADNI1, Y_dis_ts_ADNI1, S_cog_ts_ADNI1 = \
            utils.load_dataset_pair_id("Image_ADNI1.csv", "SNP_ADNI1.csv",
                                       self.fold_idx, self.task, SNP_mapping=True, divide_ICV=False,
                                       random_state=123)

        ADNI1_id_tr2, ADNI1_id_vl2, ADNI1_id_ts2, \
        _X_MRI_tr_ADNI1, _X_PET_tr_ADNI1, _X_SNP_tr_ADNI1, _C_dmg_tr_ADNI1, _Y_dis_tr_ADNI1, _S_cog_tr_ADNI1, \
        _X_MRI_vl_ADNI1, _X_PET_vl_ADNI1, _X_SNP_vl_ADNI1, _C_dmg_vl_ADNI1, _Y_dis_vl_ADNI1, _S_cog_vl_ADNI1, \
        _X_MRI_ts_ADNI1, _X_PET_ts_ADNI1, _X_SNP_ts_ADNI1, _C_dmg_ts_ADNI1, _Y_dis_ts_ADNI1, _S_cog_ts_ADNI1 = \
            utils.load_dataset_noPET_id("Incomplete_Image_ADNI1.csv", "Incomplete_SNP_ADNI1.csv",
                                        self.fold_idx, self.task, SNP_mapping=True, divide_ICV=False,
                                        random_state=123)

        ADNI2_id_tr, ADNI2_id_vl, ADNI2_id_ts, \
        X_MRI_tr_ADNI2, X_PET_tr_ADNI2, X_SNP_tr_ADNI2, C_dmg_tr_ADNI2, Y_dis_tr_ADNI2, S_cog_tr_ADNI2, \
        X_MRI_vl_ADNI2, X_PET_vl_ADNI2, X_SNP_vl_ADNI2, C_dmg_vl_ADNI2, Y_dis_vl_ADNI2, S_cog_vl_ADNI2, \
        X_MRI_ts_ADNI2, X_PET_ts_ADNI2, X_SNP_ts_ADNI2, C_dmg_ts_ADNI2, Y_dis_ts_ADNI2, S_cog_ts_ADNI2 = \
            utils.load_dataset_pair_id("Image_ADNI2.csv", "SNP_ADNI2.csv", self.fold_idx,
                                       self.task, SNP_mapping=True, divide_ICV=False, random_state=123)

        ID_train = np.concatenate((ADNI1_id_tr1, ADNI1_id_tr2, ADNI1_id_vl2, ADNI1_id_ts2), axis=0)
        ID_valid = np.concatenate((ADNI1_id_vl1, ADNI1_id_ts1), axis=0)
        ID_test = np.concatenate((ADNI2_id_tr, ADNI2_id_vl, ADNI2_id_ts), axis=0)

        X_MRI_train, X_PET_train, E_SNP_train, C_demo_train, Y_train, S_train, \
            = np.concatenate((X_MRI_tr_ADNI1, _X_MRI_tr_ADNI1, _X_MRI_vl_ADNI1, _X_MRI_ts_ADNI1), axis=0), \
              np.concatenate((X_PET_tr_ADNI1, _X_PET_tr_ADNI1, _X_PET_vl_ADNI1, _X_PET_ts_ADNI1), axis=0), \
              np.concatenate((X_SNP_tr_ADNI1, _X_SNP_tr_ADNI1, _X_SNP_vl_ADNI1, _X_SNP_ts_ADNI1), axis=0), \
              np.concatenate((C_dmg_tr_ADNI1, _C_dmg_tr_ADNI1, _C_dmg_vl_ADNI1, _C_dmg_ts_ADNI1), axis=0), \
              np.concatenate((Y_dis_tr_ADNI1, _Y_dis_tr_ADNI1, _Y_dis_vl_ADNI1, _Y_dis_ts_ADNI1), axis=0), \
              np.concatenate((S_cog_tr_ADNI1, _S_cog_tr_ADNI1, _S_cog_vl_ADNI1, _S_cog_ts_ADNI1), axis=0),

        X_MRI_vl, X_PET_vl, X_SNP_vl, C_dmg_vl, Y_dis_vl, S_cog_vl \
            = np.concatenate((X_MRI_vl_ADNI1, X_MRI_ts_ADNI1), axis=0), \
              np.concatenate((X_PET_vl_ADNI1, X_PET_ts_ADNI1), axis=0), \
              np.concatenate((X_SNP_vl_ADNI1, X_SNP_ts_ADNI1), axis=0), \
              np.concatenate((C_dmg_vl_ADNI1, C_dmg_ts_ADNI1), axis=0), \
              np.concatenate((Y_dis_vl_ADNI1, Y_dis_ts_ADNI1), axis=0), \
              np.concatenate((S_cog_vl_ADNI1, S_cog_ts_ADNI1), axis=0),

        X_MRI_ts, X_PET_ts, X_SNP_ts, C_dmg_ts, Y_dis_ts, S_cog_ts \
            = np.concatenate((X_MRI_tr_ADNI2, X_MRI_vl_ADNI2, X_MRI_ts_ADNI2), axis=0), \
              np.concatenate((X_PET_tr_ADNI2, X_PET_vl_ADNI2, X_PET_ts_ADNI2), axis=0), \
              np.concatenate((X_SNP_tr_ADNI2, X_SNP_vl_ADNI2, X_SNP_ts_ADNI2), axis=0), \
              np.concatenate((C_dmg_tr_ADNI2, C_dmg_vl_ADNI2, C_dmg_ts_ADNI2), axis=0), \
              np.concatenate((Y_dis_tr_ADNI2, Y_dis_vl_ADNI2, Y_dis_ts_ADNI2), axis=0), \
              np.concatenate((S_cog_tr_ADNI2, S_cog_vl_ADNI2, S_cog_ts_ADNI2), axis=0),



        N_o = len(self.task)

        # Call optimizers
        opt_rec_SNP = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_rec_ebMRI = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_rec_ebPET = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_rec_MRI = tf.keras.optimizers.Adam(learning_rate=self.lr)
        _opt_rec_MRI = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_rec_PET = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_dist = tf.keras.optimizers.Adam(learning_rate=self.lr)
        _opt_dist = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_clf = tf.keras.optimizers.Adam(learning_rate=self.lr)
        _opt_clf = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_disc = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_G_SNP = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_G_MRI = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_G_PET = tf.keras.optimizers.Adam(learning_rate=self.lr)


        base_model1 = MRI_PET_classification_AAE.engine(N_o=N_o)
        # base_model1.load_weights(r'model_classification_baseline_AE/model_best{}'.format(fold))
        base_model2 = SNP_AE_new.engine(N_o=N_o)
        # base_model2.load_weights(r'model_Asso_SNP_AE_new/model_best{}'.format(fold))
        final_model = DistModel.engine(N_o=N_o)
        # final_model.load_weights(r'model_Asso_Asso_AE/model_best{}'.format(fold))

        report = []
        final_acc = 0
        final_auc = 0
        final_metric = 0
        final_epoch = 0

        for epoch in range(self.num_epochs):

            num_iters = int(Y_train.shape[0] / self.num_batches)
            L_rec_ebMRI_per_epoch = 0
            L_rec_ebPET_per_epoch = 0
            L_rec_SNP_per_epoch = 0
            L_rec_MRI_per_epoch = 0
            L_rec_PET_per_epoch = 0
            L_dist_per_epoch = 0
            L_clf_per_epoch = 0
            L_reg_per_epoch = 0
            L_disc_per_epoch = 0
            L_G_MRI_per_epoch = 0
            L_G_PET_per_epoch = 0
            L_G_SNP_per_epoch = 0

            # Randomize the training dataset
            rand_idx = np.random.permutation(Y_train.shape[0])
            ID_train = ID_train[rand_idx, ...]
            X_MRI_train = X_MRI_train[rand_idx, ...]
            X_PET_train = X_PET_train[rand_idx, ...]
            E_SNP_train = E_SNP_train[rand_idx, ...]
            C_demo_train = C_demo_train[rand_idx, ...]
            Y_train = Y_train[rand_idx, ...]
            S_train = S_train[rand_idx, ...]

            for batch in range(num_iters):
                # Sample a minibatch
                xb_MRI = X_MRI_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(
                    np.float32)
                xb_PET = X_PET_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(
                    np.float32)
                eb_SNP = E_SNP_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(
                    np.float32)
                cb_demo = C_demo_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                yb_clf = Y_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                sb_reg = S_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                ID_trb = ID_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]

                idx_no_pet = np.sort(
                    np.where((xb_PET[:, 0] == 0) & (xb_PET[:, 4] == 0) & (xb_PET[:, 11] == 0))[0]).tolist()
                idx_have_pet = np.sort(
                    np.where((xb_PET[:, 0] != 0) | (xb_PET[:, 4] != 0) | (xb_PET[:, 11] != 0))[0]).tolist()
                with tf.GradientTape() as tape_disc:
                    zb_SNP = base_model2.encode_SNP(x_SNP=eb_SNP)
                    zb_MRI = base_model1.encode_MRI1(x_MRI=xb_MRI)
                    zb_PET = base_model1.encode_PET1(x_PET=xb_PET[idx_have_pet])

                    # Discriminate feature
                    real_gaussian_MRI = prior.gaussian(batch_size=zb_MRI.shape[0], n_dim=50, mean=0, var=1)
                    real_gaussian_PET = prior.gaussian(batch_size=zb_PET.shape[0], n_dim=50, mean=0, var=1)
                    real_gaussian_SNP = prior.gaussian(batch_size=zb_SNP.shape[0], n_dim=50, mean=0, var=1)

                    discriminator_MRI_fake = base_model1.discriminate(zb_MRI)
                    discriminator_PET_fake = base_model1.discriminate(zb_PET)
                    discriminator_SNP_fake = base_model1.discriminate(zb_SNP)
                    discriminator_MRI_real = base_model1.discriminate(real_gaussian_MRI)
                    discriminator_PET_real = base_model1.discriminate(real_gaussian_PET)
                    discriminator_SNP_real = base_model1.discriminate(real_gaussian_SNP)

                    # discriminator loss
                    D_loss_real_MRI = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_MRI_real,
                                                                labels=tf.ones_like(discriminator_MRI_real)))
                    D_loss_real_PET = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_PET_real,
                                                                labels=tf.ones_like(discriminator_PET_real)))
                    D_loss_real_SNP = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_SNP_real,
                                                                labels=tf.ones_like(discriminator_SNP_real)))
                    D_loss_fake_MRI = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_MRI_fake,
                                                                labels=tf.zeros_like(discriminator_MRI_fake)))
                    D_loss_fake_PET = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_PET_fake,
                                                                labels=tf.zeros_like(discriminator_PET_fake)))
                    D_loss_fake_SNP = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_SNP_fake,
                                                                labels=tf.zeros_like(discriminator_SNP_fake)))

                    D_loss = D_loss_real_MRI + D_loss_real_PET + D_loss_real_SNP + \
                             D_loss_fake_MRI + D_loss_fake_PET + D_loss_fake_SNP

                    D_loss *= self.alpha_D

                var1 = base_model1.trainable_variables
                theta_Discriminator = [var1[26], var1[27], var1[28], var1[29]]

                grad_disc = tape_disc.gradient(D_loss, theta_Discriminator)
                opt_disc.apply_gradients(zip(grad_disc, theta_Discriminator))
                L_disc_per_epoch += np.mean(D_loss)

                # Estimate gradient and loss - Generate
                with tf.GradientTape() as tape_rec_MRI, tf.GradientTape() as tape_rec_PET, \
                        tf.GradientTape() as tape_dist, tf.GradientTape() as tape_clf, \
                        tf.GradientTape() as _tape_dist, tf.GradientTape() as tape_rec_SNP, \
                        tf.GradientTape() as tape_rec_ebMRI, tf.GradientTape() as tape_rec_ebPET, \
                        tf.GradientTape() as tape_G_MRI, tf.GradientTape() as tape_G_PET, \
                        tf.GradientTape() as tape_G_SNP, tf.GradientTape() as _tape_clf, \
                        tf.GradientTape() as _tape_rec_MRI:

                    ### Step1 Have PET

                    zb_MRI = base_model1.encode_MRI1(x_MRI=xb_MRI[idx_have_pet])
                    zb_MRI_com, zb_MRI_spe = base_model1.encode_MRI2(zb_MRI=zb_MRI)

                    zb_PET = base_model1.encode_PET1(x_PET=xb_PET[idx_have_pet])
                    zb_PET_com, zb_PET_spe = base_model1.encode_PET2(zb_PET=zb_PET)

                    zb_SNP = base_model2.encode_SNP(x_SNP=eb_SNP[idx_have_pet])
                    eb_SNP_hat_logit = base_model2.decode_SNP(z_SNP=zb_SNP)

                    zb_MRI_com_hat, zb_MRI_spec_hat, mask_MRI_com, mask_MRI_spec = \
                        final_model.asso_SNP2MRI(z_SNP=zb_SNP, c_demo=cb_demo[idx_have_pet])
                    zb_PET_com_hat, zb_PET_spec_hat, mask_PET_com, mask_PET_spec = \
                        final_model.asso_SNP2PET(z_SNP=zb_SNP, c_demo=cb_demo[idx_have_pet])

                    xb_MRI_hat_logit1 = base_model1.decode_MRI(z_com=zb_MRI_com, z_spe=zb_MRI_spe)
                    xb_MRI_hat_logit2 = base_model1.decode_MRI(z_com=zb_PET_com, z_spe=zb_MRI_spe)
                    xb_PET_hat_logit1 = base_model1.decode_PET(z_com=zb_MRI_com, z_spe=zb_PET_spe)
                    xb_PET_hat_logit2 = base_model1.decode_PET(z_com=zb_PET_com, z_spe=zb_PET_spe)

                    yb_clf_hat, sb_reg_hat = final_model.diagnose(
                        x_MRI_com=zb_MRI_com, x_MRI_spe=zb_MRI_spe,
                        x_PET_com=zb_PET_com, x_PET_spe=zb_PET_spe,
                        mask_MRI_com=mask_MRI_com, mask_MRI_spe=mask_MRI_spec,
                        mask_PET_com=mask_PET_com, mask_PET_spe=mask_PET_spec, apply_logistic_activation=True)

                    ### Step2 dont Have PET
                    if len(idx_no_pet) != 0:
                        _zb_MRI = base_model1.encode_MRI1(x_MRI=xb_MRI[idx_no_pet])
                        _zb_MRI_com, _zb_MRI_spe = base_model1.encode_MRI2(zb_MRI=_zb_MRI)

                        _zb_SNP = base_model2.encode_SNP(x_SNP=eb_SNP[idx_no_pet])
                        _eb_SNP_hat_logit = base_model2.decode_SNP(z_SNP=_zb_SNP)

                        _zb_MRI_com_hat, _zb_MRI_spec_hat, _mask_MRI_com, _mask_MRI_spec = \
                            final_model.asso_SNP2MRI(z_SNP=_zb_SNP, c_demo=cb_demo[idx_no_pet])
                        _zb_PET_com_hat, _zb_PET_spec_hat, _mask_PET_com, _mask_PET_spec = \
                            final_model.asso_SNP2PET(z_SNP=_zb_SNP, c_demo=cb_demo[idx_no_pet])

                        _xb_MRI_hat_logit1 = base_model1.decode_MRI(z_com=_zb_MRI_com, z_spe=_zb_MRI_spe)
                        _xb_MRI_hat_logit2 = base_model1.decode_MRI(z_com=_zb_PET_com_hat, z_spe=_zb_MRI_spe)
                        _xb_PET_hat_logit1 = base_model1.decode_PET(z_com=_zb_MRI_com, z_spe=_zb_PET_spec_hat)
                        _xb_PET_hat_logit2 = base_model1.decode_PET(z_com=_zb_PET_com_hat, z_spe=_zb_PET_spec_hat)

                        _yb_clf_hat, _sb_reg_hat = final_model.diagnose(
                            x_MRI_com=_zb_MRI_com, x_MRI_spe=_zb_MRI_spe,
                            x_PET_com=_zb_PET_com_hat, x_PET_spe=_zb_PET_spec_hat,
                            mask_MRI_com=_mask_MRI_com, mask_MRI_spe=_mask_MRI_spec,
                            mask_PET_com=_mask_PET_com, mask_PET_spe=_mask_PET_spec, apply_logistic_activation=True)

                    # MRI/PET Reconstruction loss
                    reconstruction_loss_MRI1 = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(xb_MRI[idx_have_pet], xb_MRI_hat_logit1)))
                    reconstruction_loss_MRI2 = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(xb_MRI[idx_have_pet], xb_MRI_hat_logit2)))
                    reconstruction_loss_PET1 = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(xb_PET[idx_have_pet], xb_PET_hat_logit1)))
                    reconstruction_loss_PET2 = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(xb_PET[idx_have_pet], xb_PET_hat_logit2)))
                    if len(idx_no_pet) != 0:
                        _reconstruction_loss_MRI1 = tf.reduce_mean(
                            tf.sqrt(tf.keras.losses.mean_squared_error(xb_MRI[idx_no_pet], _xb_MRI_hat_logit1)))
                        _reconstruction_loss_MRI2 = tf.reduce_mean(
                            tf.sqrt(tf.keras.losses.mean_squared_error(xb_MRI[idx_no_pet], _xb_MRI_hat_logit2)))
                        _L_rec_MRI = _reconstruction_loss_MRI1 + _reconstruction_loss_MRI2
                        _L_rec_MRI *= self.alpha_rec
                    # else:
                    L_rec_MRI = reconstruction_loss_MRI1 + reconstruction_loss_MRI2
                    L_rec_PET = reconstruction_loss_PET1 + reconstruction_loss_PET2

                    L_rec_MRI *= self.alpha_rec
                    L_rec_PET *= self.alpha_rec

                    # SNP Reconstruction loss
                    reconstruction_loss_SNP = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(eb_SNP[idx_have_pet], eb_SNP_hat_logit)))
                    # kl_loss_SNP = -0.5 * tf.reduce_mean(1 + log_sigma_square - tf.square(mu) - tf.exp(log_sigma_square))
                    if len(idx_no_pet) != 0:
                        _reconstruction_loss_SNP = tf.reduce_mean(
                            tf.sqrt(tf.keras.losses.mean_squared_error(eb_SNP[idx_no_pet], _eb_SNP_hat_logit)))
                        L_rec_SNP = reconstruction_loss_SNP + _reconstruction_loss_SNP
                    else:
                        L_rec_SNP = reconstruction_loss_SNP
                    L_rec_SNP *= self.alpha_SNP

                    # embedding Reconstruction loss
                    L_rec_ebMRI = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(zb_MRI_com, zb_MRI_com_hat))) \
                                  + tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(zb_MRI_spe, zb_MRI_spec_hat)))
                    L_rec_ebPET = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(zb_PET_com, zb_PET_com_hat))) \
                                  + tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(zb_PET_spe, zb_PET_spec_hat)))
                    if len(idx_no_pet) != 0:
                        _L_rec_ebMRI = tf.reduce_mean(
                            tf.sqrt(tf.keras.losses.mean_squared_error(_zb_MRI_com, _zb_MRI_com_hat))) \
                                       + tf.reduce_mean(
                            tf.sqrt(tf.keras.losses.mean_squared_error(_zb_MRI_spe, _zb_MRI_spec_hat)))
                        x_L_rec_ebMRI = L_rec_ebMRI + _L_rec_ebMRI
                    else:
                        x_L_rec_ebMRI = L_rec_ebMRI

                    # Distance loss
                    L_dist = tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(zb_MRI_com, zb_PET_com))) / \
                             tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(zb_MRI_spe, zb_PET_spe)))
                    if len(idx_no_pet) != 0:
                        _L_dist = tf.reduce_mean(
                            tf.sqrt(tf.keras.losses.mean_squared_error(_zb_MRI_com, _zb_PET_com_hat))) / \
                                  tf.reduce_mean(
                                      tf.sqrt(tf.keras.losses.mean_squared_error(_zb_MRI_spe, _zb_PET_spec_hat)))
                        _L_dist *= self.alpha_dist
                        # x_L_dist = L_dist + _L_dist
                    # else:
                    #     x_L_dist = L_dist
                    L_dist *= self.alpha_dist

                    # Classification loss
                    L_clf = tfa.losses.sigmoid_focal_crossentropy(yb_clf[idx_have_pet], yb_clf_hat)
                    if len(idx_no_pet) != 0:
                        # print(len(yb_clf[idx_no_pet]), len(_yb_clf_hat))
                        _L_clf = tfa.losses.sigmoid_focal_crossentropy(yb_clf[idx_no_pet], _yb_clf_hat)
                        _L_clf *= self.alpha_clf
                    L_clf *= self.alpha_clf

                    # generator loss
                    discriminator_MRI_fake = base_model1.discriminate(zb_MRI)
                    discriminator_PET_fake = base_model1.discriminate(zb_PET)
                    discriminator_SNP_fake = base_model1.discriminate(zb_SNP)

                    G_loss_MRI = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_MRI_fake,
                                                                labels=tf.ones_like(discriminator_MRI_fake)))
                    G_loss_PET = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_PET_fake,
                                                                labels=tf.ones_like(discriminator_PET_fake)))
                    G_loss_SNP = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_SNP_fake,
                                                                labels=tf.ones_like(discriminator_SNP_fake)))
                    G_loss_MRI *= self.alpha_G
                    G_loss_PET *= self.alpha_G
                    G_loss_PET *= self.alpha_G

                # Apply gradients
                var1 = base_model1.trainable_variables
                var2 = base_model2.trainable_variables
                var3 = final_model.trainable_variables
                theta_MRI_Encoder = [var1[0], var1[1], var1[2], var1[3], var1[4], var1[5]]
                theta_MRI_Encoder_G = [var1[0], var1[1], var1[2], var1[3]]
                theta_MRI_Decoder = [var1[6], var1[7], var1[8], var1[9]]
                theta_PET_Encoder = [var1[10], var1[11], var1[12], var1[13], var1[14], var1[15]]
                theta_PET_Encoder_G = [var1[10], var1[11], var1[12], var1[13]]
                theta_PET_Decoder = [var1[16], var1[17], var1[18], var1[19]]
                theta_SNP_Encoder = [var2[0], var2[1], var2[2], var2[3]]
                theta_SNP_Decoder = [var2[4], var2[5], var2[6], var2[7]]
                theta_SNP2MRI_ASSO = [var3[0], var3[1], var3[2], var3[3]]
                theta_SNP2PET_ASSO = [var3[4], var3[5], var3[6], var3[7]]
                theta_C_share = [var3[8], var3[9]]
                theta_C_clf = [var3[10], var3[11]]

                grad_G_SNP = tape_G_SNP.gradient(G_loss_SNP, theta_SNP_Encoder)
                opt_G_SNP.apply_gradients(zip(grad_G_SNP, theta_SNP_Encoder))
                L_G_SNP_per_epoch += np.mean(G_loss_SNP)

                grad_rec_SNP = tape_rec_SNP.gradient(L_rec_SNP, theta_SNP_Encoder + theta_SNP_Decoder)
                opt_rec_SNP.apply_gradients(zip(grad_rec_SNP, theta_SNP_Encoder + theta_SNP_Decoder))
                L_rec_SNP_per_epoch += np.mean(L_rec_SNP)

                grad_rec_ebMRI = tape_rec_ebMRI.gradient(x_L_rec_ebMRI, theta_SNP_Encoder + theta_SNP2MRI_ASSO)
                opt_rec_ebMRI.apply_gradients(zip(grad_rec_ebMRI, theta_SNP_Encoder + theta_SNP2MRI_ASSO))
                L_rec_ebMRI_per_epoch += np.mean(x_L_rec_ebMRI)

                grad_rec_ebPET = tape_rec_ebPET.gradient(L_rec_ebPET, theta_SNP_Encoder + theta_SNP2PET_ASSO)
                opt_rec_ebPET.apply_gradients(zip(grad_rec_ebPET, theta_SNP_Encoder + theta_SNP2PET_ASSO))
                L_rec_ebPET_per_epoch += np.mean(L_rec_ebPET)

                grad_G_MRI = tape_G_MRI.gradient(G_loss_MRI, theta_MRI_Encoder_G)
                opt_G_MRI.apply_gradients(zip(grad_G_MRI, theta_MRI_Encoder_G))
                L_G_MRI_per_epoch += np.mean(G_loss_MRI)

                grad_rec_MRI = tape_rec_MRI.gradient(L_rec_MRI,
                                                     theta_MRI_Encoder + theta_PET_Encoder + theta_MRI_Decoder)
                opt_rec_MRI.apply_gradients(
                    zip(grad_rec_MRI, theta_MRI_Encoder + theta_PET_Encoder + theta_MRI_Decoder))
                L_rec_MRI_per_epoch += np.mean(L_rec_MRI)

                grad_G_PET = tape_G_PET.gradient(G_loss_PET, theta_PET_Encoder_G)
                opt_G_PET.apply_gradients(zip(grad_G_PET, theta_PET_Encoder_G))
                L_G_PET_per_epoch += np.mean(G_loss_PET)

                grad_rec_PET = tape_rec_PET.gradient(L_rec_PET,
                                                     theta_MRI_Encoder + theta_PET_Encoder + theta_PET_Decoder)
                opt_rec_PET.apply_gradients(
                    zip(grad_rec_PET, theta_MRI_Encoder + theta_PET_Encoder + theta_PET_Decoder))
                L_rec_PET_per_epoch += np.mean(L_rec_PET)

                grad_dist = tape_dist.gradient(L_dist, theta_MRI_Encoder + theta_PET_Encoder)
                opt_dist.apply_gradients(zip(grad_dist, theta_MRI_Encoder + theta_PET_Encoder))
                L_dist_per_epoch += np.mean(L_dist)

                grad_clf = tape_clf.gradient(L_clf, theta_MRI_Encoder + theta_PET_Encoder + theta_SNP_Encoder +
                                             theta_SNP2MRI_ASSO + theta_SNP2PET_ASSO + theta_C_share + theta_C_clf)
                opt_clf.apply_gradients(zip(grad_clf, theta_MRI_Encoder + theta_PET_Encoder + theta_SNP_Encoder +
                                            theta_SNP2MRI_ASSO + theta_SNP2PET_ASSO + theta_C_share + theta_C_clf))
                L_clf_per_epoch += np.mean(L_clf)

                if len(idx_no_pet) != 0:
                    _grad_clf = _tape_clf.gradient(_L_clf, theta_MRI_Encoder + theta_SNP_Encoder + theta_C_share +
                                                   theta_C_clf + theta_SNP2MRI_ASSO + theta_SNP2PET_ASSO)
                    _opt_clf.apply_gradients(zip(_grad_clf, theta_MRI_Encoder + theta_SNP_Encoder + theta_C_share +
                                                 theta_C_clf + theta_SNP2MRI_ASSO + theta_SNP2PET_ASSO))
                    L_clf_per_epoch += np.mean(_L_clf)

                    _grad_dist = _tape_dist.gradient(_L_dist,
                                                     theta_MRI_Encoder + theta_SNP_Encoder + theta_SNP2PET_ASSO)
                    _opt_dist.apply_gradients(
                        zip(_grad_dist, theta_MRI_Encoder + theta_SNP_Encoder + theta_SNP2PET_ASSO))
                    L_dist_per_epoch += np.mean(_L_dist)

                    _grad_rec_MRI = _tape_rec_MRI.gradient(_L_rec_MRI,
                                                           theta_MRI_Encoder + theta_SNP_Encoder + theta_SNP2PET_ASSO + theta_MRI_Decoder)
                    _opt_rec_MRI.apply_gradients(zip(_grad_rec_MRI,
                                                     theta_MRI_Encoder + theta_SNP_Encoder + theta_SNP2PET_ASSO + theta_MRI_Decoder))
                    L_rec_MRI_per_epoch += np.mean(_L_rec_MRI)

            L_rec_MRI_per_epoch /= num_iters
            L_rec_PET_per_epoch /= num_iters
            L_rec_SNP_per_epoch /= num_iters
            L_dist_per_epoch /= num_iters
            L_clf_per_epoch /= num_iters
            L_reg_per_epoch /= num_iters

            # Loss report
            print(f'Epoch: {epoch + 1}, LMRI: {L_rec_MRI_per_epoch:>.4f}, LPET: {L_rec_PET_per_epoch:>.4f}, '
                  f'Ldis: {L_dist_per_epoch:>.4f}, Lclf: {L_clf_per_epoch:>.4f}, LSNP: {L_rec_SNP_per_epoch:>.4f}, '
                  f'G_MRI:{L_G_MRI_per_epoch:>.4f}, G_PET:{L_G_PET_per_epoch:>.4f}, G_SNP:{L_G_SNP_per_epoch:>.4f}, '
                  f'Disc:{L_disc_per_epoch:>.4f}, L_rec_ebMRI:{L_rec_ebMRI_per_epoch:>.4f}, '
                  f'L_rec_ebPET:{L_rec_ebPET_per_epoch:>.4f}')


            # Results


            idx_no_pet_vl = np.sort(
                np.where((X_PET_vl[:, 0] == 0) & (X_PET_vl[:, 4] == 0) & (X_PET_vl[:, 11] == 0))[0])
            idx_have_pet_vl = np.sort(
                np.where((X_PET_vl[:, 0] != 0) | (X_PET_vl[:, 4] != 0) | (X_PET_vl[:, 11] != 0))[0])
            idx_no_pet_ts = np.sort(
                np.where((X_PET_ts[:, 0] == 0) & (X_PET_ts[:, 4] == 0) & (X_PET_ts[:, 11] == 0))[0])
            idx_have_pet_ts = np.sort(
                np.where((X_PET_ts[:, 0] != 0) | (X_PET_ts[:, 4] != 0) | (X_PET_ts[:, 11] != 0))[0])

            zb_MRI_vl = base_model1.encode_MRI1(x_MRI=X_MRI_vl)
            zb_MRI_com_vl, zb_MRI_spe_vl = base_model1.encode_MRI2(zb_MRI=zb_MRI_vl)

            zb_PET_vl = base_model1.encode_PET1(x_PET=X_PET_vl[idx_have_pet_vl])
            zb_PET_com_vl, zb_PET_spe_vl = base_model1.encode_PET2(zb_PET=zb_PET_vl)

            zb_SNP_vl = base_model2.encode_SNP(x_SNP=X_SNP_vl)

            zb_MRI_com_hat_vl, zb_MRI_spec_hat_vl, mask_MRI_com_vl, mask_MRI_spec_vl = \
                final_model.asso_SNP2MRI(z_SNP=zb_SNP_vl, c_demo=C_dmg_vl)
            zb_PET_com_hat_vl, zb_PET_spec_hat_vl, mask_PET_com_vl, mask_PET_spec_vl = \
                final_model.asso_SNP2PET(z_SNP=zb_SNP_vl, c_demo=C_dmg_vl)

            Y_vl_hat, S_vl_hat = final_model.diagnose(
                x_MRI_com=zb_MRI_com_vl, x_MRI_spe=zb_MRI_spe_vl,
                x_PET_com=zb_PET_com_vl, x_PET_spe=zb_PET_spe_vl,
                mask_MRI_com=mask_MRI_com_vl, mask_MRI_spe=mask_MRI_spec_vl,
                mask_PET_com=mask_PET_com_vl, mask_PET_spe=mask_PET_spec_vl, apply_logistic_activation=True)

            vl_label = np.argmax(Y_dis_vl, axis=-1)
            vl_hat = np.argmax(Y_vl_hat.numpy(), axis=-1)
            vl_eq = np.equal(vl_hat, vl_label).sum().item()
            auc = roc_auc_score(Y_dis_vl, Y_vl_hat)
            acc = float(vl_eq) / float(vl_label.shape[0])

            # metric = acc
            print(f'Valid AUC: {auc:>.4f}, '
                  f'Valid ACC: {acc:>.4f}')

            zb_MRI_ts = base_model1.encode_MRI1(x_MRI=X_MRI_ts)
            zb_MRI_com_ts, zb_MRI_spe_ts = base_model1.encode_MRI2(zb_MRI=zb_MRI_ts)

            zb_PET_ts = base_model1.encode_PET1(x_PET=X_PET_ts[idx_have_pet_ts])
            zb_PET_com_ts, zb_PET_spe_ts = base_model1.encode_PET2(zb_PET=zb_PET_ts)

            zb_SNP_ts = base_model2.encode_SNP(x_SNP=X_SNP_ts)

            zb_MRI_com_hat_ts, zb_MRI_spec_hat_ts, mask_MRI_com_ts, mask_MRI_spec_ts = \
                final_model.asso_SNP2MRI(z_SNP=zb_SNP_ts, c_demo=C_dmg_ts)
            zb_PET_com_hat_ts, zb_PET_spec_hat_ts, mask_PET_com_ts, mask_PET_spec_ts = \
                final_model.asso_SNP2PET(z_SNP=zb_SNP_ts, c_demo=C_dmg_ts)

            Y_ts_hat, S_ts_hat = final_model.diagnose(
                x_MRI_com=zb_MRI_com_ts, x_MRI_spe=zb_MRI_spe_ts,
                x_PET_com=zb_PET_com_ts, x_PET_spe=zb_PET_spe_ts,
                mask_MRI_com=mask_MRI_com_ts, mask_MRI_spe=mask_MRI_spec_ts,
                mask_PET_com=mask_PET_com_ts, mask_PET_spe=mask_PET_spec_ts, apply_logistic_activation=True)

            ts_label = np.argmax(Y_dis_ts, axis=-1)
            ts_hat = np.argmax(Y_ts_hat.numpy(), axis=-1)
            ts_eq = np.equal(ts_hat, ts_label).sum().item()
            tsauc = roc_auc_score(Y_dis_ts, Y_ts_hat)
            tsacc = float(ts_eq) / float(ts_label.shape[0])
            print(f'Test AUC: {tsauc:>.4f}, '
                  f'Test ACC: {tsacc:>.4f}')
            metric = auc
            if metric > final_metric and (epoch + 1) > 20:
                base_model1.save_weights(r'AD_model_temp/IMG/model_best{}'.format(fold))
                base_model2.save_weights(r'AD_model_temp/SNP/model_best{}'.format(fold))
                final_model.save_weights(r'AD_model_temp/Asso/model_best{}'.format(fold))
                final_acc = acc
                final_auc = auc
                final_epoch = epoch
                final_metric = metric
            print('**', final_epoch + 1)

        print('=====================')
        print(f'Valid best AUC: {final_auc:>.4f}, '
              f'Valid best ACC: {final_acc:>.4f}, '
              f'Valid best epoch: {final_epoch + 1} ')

        model1_best = MRI_PET_classification_AAE.engine(N_o=N_o)
        model2_best = SNP_AE_new.engine(N_o=N_o)
        model3_best = DistModel.engine(N_o=N_o)
        model1_best.load_weights(r'AD_model_temp/IMG/model_best{}'.format(fold))
        model2_best.load_weights(r'AD_model_temp/SNP/model_best{}'.format(fold))
        model3_best.load_weights(r'AD_model_temp/Asso/model_best{}'.format(fold))
        zb_MRI_ts = model1_best.encode_MRI1(x_MRI=X_MRI_ts)
        zb_MRI_com_ts, zb_MRI_spe_ts = model1_best.encode_MRI2(zb_MRI=zb_MRI_ts)

        zb_PET_ts = model1_best.encode_PET1(x_PET=X_PET_ts[idx_have_pet_ts])
        zb_PET_com_ts, zb_PET_spe_ts = model1_best.encode_PET2(zb_PET=zb_PET_ts)

        zb_SNP_ts = model2_best.encode_SNP(x_SNP=X_SNP_ts)

        zb_MRI_com_hat_ts, zb_MRI_spec_hat_ts, mask_MRI_com_ts, mask_MRI_spec_ts = \
            model3_best.asso_SNP2MRI(z_SNP=zb_SNP_ts, c_demo=C_dmg_ts)
        zb_PET_com_hat_ts, zb_PET_spec_hat_ts, mask_PET_com_ts, mask_PET_spec_ts = \
            model3_best.asso_SNP2PET(z_SNP=zb_SNP_ts, c_demo=C_dmg_ts)

        Y_ts_hat, S_ts_hat = model3_best.diagnose(
            x_MRI_com=zb_MRI_com_ts, x_MRI_spe=zb_MRI_spe_ts,
            x_PET_com=zb_PET_com_ts, x_PET_spe=zb_PET_spe_ts,
            mask_MRI_com=mask_MRI_com_ts, mask_MRI_spe=mask_MRI_spec_ts,
            mask_PET_com=mask_PET_com_ts, mask_PET_spe=mask_PET_spec_ts, apply_logistic_activation=True)

        ts_label = np.argmax(Y_dis_ts, axis=-1)
        ts_hat = np.argmax(Y_ts_hat.numpy(), axis=-1)
        ts_eq = np.equal(ts_hat, ts_label).sum().item()

        print(f'ts AUC: {roc_auc_score(Y_dis_ts, Y_ts_hat):>.4f}, '
              f'ts ACC: {float(ts_eq) / float(ts_label.shape[0]):>.4f}')
        return


task = ['CN', 'AD']  # ['CN', 'MCI'], ['sMCI', 'pMCI'], ['CN', 'MCI', 'AD'], ['CN', 'sMCI', 'pMCI', 'AD']
for fold in range(5):  # five-fold cross-validation
    if fold == 0:
        print('-----fold {}-----'.format(fold + 1))
        exp = experiment(fold + 1, ['CN', 'AD'])
        exp.training()
