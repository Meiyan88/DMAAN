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
from utils import calcBCA
import shutil

from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, accuracy_score, roc_curve

gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
physical_gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
)
logical_gpus = tf.config.list_logical_devices("GPU")


class experiment():
    def __init__(self, fold_idx, task):
        self.fold_idx = fold_idx
        self.task = task

        # Learning schedules
        self.num_epochs = 200  # 100
        self.num_batches = 5
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=1000,
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

    def testing(self):
        best_test_metric = 0


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

        N_o = Y_train.shape[-1]

        model1_best = MRI_PET_classification_AAE.engine(N_o=N_o)
        model2_best = SNP_AE_new.engine(N_o=N_o)
        model3_best = DistModel.engine(N_o=N_o)
        if os.path.exists(r'AD_model_best/IMG/model_best{}.index'.format(fold)) == True:
            model1_best.load_weights(r'AD_model_best/IMG/model_best{}'.format(fold))
            model2_best.load_weights(r'AD_model_best/SNP/model_best{}'.format(fold))
            model3_best.load_weights(r'AD_model_best/Asso/model_best{}'.format(fold))
        else:
            print('No such file')
            return

        zb_MRI_ts = model1_best.encode_MRI1(x_MRI=X_MRI_ts)
        zb_MRI_com_ts, zb_MRI_spe_ts = model1_best.encode_MRI2(zb_MRI=zb_MRI_ts)

        zb_PET_ts = model1_best.encode_PET1(x_PET=X_PET_ts)
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
        ts_hat = Y_ts_hat.numpy()[:, 1]
        auc_ts = roc_auc_score(ts_label, ts_hat)
        ts_best_threshold = 0.5
        ts_pred = np.where(ts_hat <= ts_best_threshold, 0, 1)
        ts_eq = np.equal(ts_pred, ts_label).sum().item()
        acc_ts = float(ts_eq) / float(ts_label.shape[0])
        ts_BCA = calcBCA(ts_pred, ts_label, 2)
        TN, FP, FN, TP = confusion_matrix(ts_label, ts_pred).ravel()
        best_SEN, best_SPE = TP / (TP + FN), TN / (FP + TN)

        print(f'Test ACC: {acc_ts:>.4f}, '
              f'Test AUC: {auc_ts:>.4f}, '
              f'Test SEN: {best_SEN:>.4f}, '
              f'Test SPE: {best_SPE:>.4f}, '
              f'Test BCA: {ts_BCA:>.4f}')
        return


task = ['CN', 'AD']  # ['CN', 'MCI'], ['sMCI', 'pMCI']
for fold in range(5):  # five-fold cross-validation
    if fold == 0:
        print('Time {} -----fold {}-----'.format(time, fold + 1))
        exp = experiment(fold + 1, ['CN', 'AD'])
        exp.testing()
