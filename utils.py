import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, train_test_split


def SNP_encoder(X_SNP_tr, X_SNP_ts):
    # Based on population, this encoder transforms the discrete SNP vectors to be numerical.
    # The encoder is fit by the training SNP data and applied to the testing SNP data.

    # Fit the encoding table
    encoder = np.empty(shape=(3, X_SNP_tr.shape[1]))
    for i in range(X_SNP_tr.shape[1]):
        for j in [0, 1, 2]:
            encoder[j, i] = np.array(X_SNP_tr[:, i] == j).sum()

    encoder /= X_SNP_tr.shape[0]  # (3, 1275)

    X_E_SNP_tr = np.empty(shape=X_SNP_tr.shape)
    X_E_SNP_ts = np.empty(shape=X_SNP_ts.shape)

    # Map the SNP values
    for sbj in range(X_SNP_tr.shape[0]):
        for dna in range(X_SNP_tr.shape[-1]):

            X_E_SNP_tr[sbj, dna] = encoder[..., dna][int(X_SNP_tr[sbj, dna])]

    for sbj in range(X_SNP_ts.shape[0]):
        for dna in range(X_SNP_ts.shape[-1]):
            X_E_SNP_ts[sbj, dna] = encoder[..., dna][int(X_SNP_ts[sbj, dna])]

    return X_E_SNP_tr, X_E_SNP_ts
    
def load_dataset_pair(path, SNP_path, fold, task, SNP_mapping=True, divide_ICV=False, random_state=5930):
    data_all = np.loadtxt(path, delimiter=',', dtype=str)
    data_all = data_all[1:, :]

    mytask = []
    is_MCI = False
    for t in task:
        if t == 'CN':
            # mytask.append(0)
            mytask.append(1)

        if t == 'AD':
            # mytask.append(3)
            mytask.append(3)

        if t == 'MCI':  # append sMCI and pMCI samples
            # mytask.append(1)
            mytask.append(2)
            # is_MCI = True

    mytask = np.array(mytask)

    # SNP_match_ID = np.loadtxt("SNP_match_AD_ID.csv",
    #                           delimiter=',', dtype=str)[1:, :][:, 1:].astype(np.int)
    # X_SNP_ori = loadmat('SNPdata.mat')['SNP']  # SNP features (N, 2098)
    # X_SNP_ori = np.concatenate((SNP_match_ID, X_SNP_ori), axis=-1)
    # X_SNP_ori = X_SNP_ori[X_SNP_ori[:, 0].argsort()]
    # X_SNP_ori = X_SNP_ori[:, 1:]
    SNP_match = np.loadtxt(SNP_path, delimiter=',', dtype=str)
    SNP_match = SNP_match[1:, :].astype(np.int)
    SNP_match_ID = SNP_match[:, 0]
    X_SNP_ori = SNP_match[:, 1:]

    subject_id = data_all[:, :1]
    idx_SNP = []
    SNP_new = []
    for num, ID in enumerate(np.sort(SNP_match_ID)):
        idx = np.where(subject_id == str(ID))
        SNP_new.append(np.repeat(X_SNP_ori[num][None, :], len(idx[0]), axis=0))
        idx_SNP.append(idx[0])
    SNP_new = np.concatenate(SNP_new, axis=0)
    idx_SNP = np.concatenate(idx_SNP)
    subject_id = subject_id[idx_SNP, :]
    data_all = data_all[idx_SNP, :]


    # Define the data path
    Y_dis = data_all[:, 2].astype(np.int)  # Disease labels, CN: 1, MCI: 2, AD: 3
    # Extract appropriate indicies for the given task
    task_idx = np.zeros(shape=Y_dis.shape[0])

    for t in range(len(mytask)):
        task_idx += np.array(Y_dis == mytask[t])
    task_idx = task_idx.astype(bool)

    Y_dis = Y_dis[task_idx]
    subject_id = subject_id[task_idx]
    X_ICV = data_all[:, 10:11][task_idx, :].astype(np.float)  #
    X_MRI = data_all[:, 11:101][task_idx, :].astype(np.float)
    X_PET = data_all[:, 101:][task_idx, :].astype(np.float)  # MRI volume features (N, 93)
    C_sex = data_all[:, 4:5][task_idx]  # Sex codes 'F', 'M' (N, )
    C_edu = data_all[:, 5:6][task_idx].astype(np.float)  # Education codes (N, )
    C_age = data_all[:, 3:4][task_idx].astype(np.float)  # Age codes (N, )
    S_MMSE = data_all[:, 8:9][task_idx].astype(np.float)  # Cognitive scores (MMSE), (N, )
    X_SNP = SNP_new[task_idx, :].astype(np.float)

    idx_have_PET = np.where(X_PET[:, 0] != 0)
    subject_id = subject_id[idx_have_PET]
    Y_dis = Y_dis[idx_have_PET]
    X_ICV = X_ICV[idx_have_PET]
    X_MRI = X_MRI[idx_have_PET]
    X_PET = X_PET[idx_have_PET]
    C_sex = C_sex[idx_have_PET]
    C_edu = C_edu[idx_have_PET]
    C_age = C_age[idx_have_PET]
    S_MMSE = S_MMSE[idx_have_PET]
    X_SNP = X_SNP[idx_have_PET]

    # Normalization
    C_age /= 100
    C_edu /= 20
    S_MMSE /= 30

    # Categorical encoding for the sex code
    for i in range(np.unique(C_sex).shape[0]):
        C_sex[C_sex == np.unique(C_sex)[i]] = i
    C_sex = C_sex.astype(np.int)
    C_sex = np.eye(np.unique(C_sex).shape[0])[C_sex]

    # Demographic information concatenation
    C_dmg = np.concatenate((C_sex, C_age[:, None], C_edu[:, None]), -1)  # (737, 4)


    # Fold dividing
    unique_id = np.unique(subject_id)
    split_Y = []
    for ID in unique_id:
        split_Y.append(Y_dis[np.where(subject_id == ID)[0]][0])
    split_Y = np.array(split_Y)

    fold_now = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, valid_test_index in kf.split(unique_id, split_Y):
        fold_now = fold_now + 1
        if fold_now == fold:
            select_tr = unique_id[train_index]
            select_vl_ts = valid_test_index
    select_vl, select_ts, _, _ = train_test_split(unique_id[select_vl_ts], split_Y[select_vl_ts], test_size=0.5,
                                                  random_state=random_state)
    _select_tr, _select_vl, _select_ts = np.sort(select_tr.astype(np.int)), np.sort(select_vl.astype(np.int)), \
                                        np.sort(select_ts.astype(np.int))
    # np.savetxt('fold_id/pari_fold{}_train'.format(fold), _select_tr, fmt='%s', delimiter=',')
    # np.savetxt('fold_id/pari_fold{}_valid'.format(fold), _select_vl, fmt='%s', delimiter=',')
    # np.savetxt('fold_id/pari_fold{}_test'.format(fold), _select_ts, fmt='%s', delimiter=',')
    tr_idx = []
    for ID in select_tr:
        idx = np.where(subject_id == ID)
        tr_idx.append(idx[0])
    tr_idx = np.concatenate(tr_idx)

    vl_idx = []
    for ID in select_vl:
        idx = np.where(subject_id == ID)
        vl_idx.append(idx[0])
    vl_idx = np.concatenate(vl_idx)

    ts_idx = []
    for ID in select_ts:
        idx = np.where(subject_id == ID)
        ts_idx.append(idx[0])
    ts_idx = np.concatenate(ts_idx)

    # One-hot encoding for the disease label
    for i in range(np.unique(Y_dis).shape[0]):
        Y_dis[Y_dis == np.unique(Y_dis)[i]] = i

    Y_dis = np.eye(np.unique(Y_dis).shape[0])[Y_dis]

    X_ICV_tr, X_ICV_vl, X_ICV_ts = X_ICV[tr_idx, :], X_ICV[vl_idx, :], X_ICV[ts_idx, :]
    X_MRI_tr, X_MRI_vl, X_MRI_ts = X_MRI[tr_idx, :], X_MRI[vl_idx, :], X_MRI[ts_idx, :]
    X_PET_tr, X_PET_vl, X_PET_ts = X_PET[tr_idx, :], X_PET[vl_idx, :], X_PET[ts_idx, :]
    X_SNP_tr, X_SNP_vl, X_SNP_ts = X_SNP[tr_idx, :], X_SNP[vl_idx, :], X_SNP[ts_idx, :]
    C_dmg_tr, C_dmg_vl, C_dmg_ts = C_dmg[tr_idx, :], C_dmg[vl_idx, :], C_dmg[ts_idx, :]
    Y_dis_tr, Y_dis_vl, Y_dis_ts = Y_dis[tr_idx, :], Y_dis[vl_idx, :], Y_dis[ts_idx, :]
    S_cog_tr, S_cog_vl, S_cog_ts = S_MMSE[tr_idx], S_MMSE[vl_idx], S_MMSE[ts_idx, :]

    # MRI/PET normalization
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    if divide_ICV == True:
        X_MRI_tr = X_MRI_tr / X_ICV_tr
        X_MRI_vl = X_MRI_vl / X_ICV_vl
        X_MRI_ts = X_MRI_ts / X_ICV_ts
    X_MRI_tr = scaler1.fit_transform(X_MRI_tr)
    X_MRI_vl = scaler1.transform(X_MRI_vl)
    X_MRI_ts = scaler1.transform(X_MRI_ts)
    X_PET_tr = scaler2.fit_transform(X_PET_tr)
    X_PET_vl = scaler2.transform(X_PET_vl)
    X_PET_ts = scaler2.transform(X_PET_ts)

    if SNP_mapping:
        # SNP encoding
        X_SNP_tr_ori = X_SNP_tr.copy()
        X_SNP_tr, X_SNP_vl = SNP_encoder(X_SNP_tr=X_SNP_tr, X_SNP_ts=X_SNP_vl)
        _, X_SNP_ts = SNP_encoder(X_SNP_tr=X_SNP_tr_ori, X_SNP_ts=X_SNP_ts)

    C_dmg_tr = C_dmg_tr[:, 0, :]
    C_dmg_vl = C_dmg_vl[:, 0, :]
    C_dmg_ts = C_dmg_ts[:, 0, :]


    Y = np.concatenate([Y_dis_tr, Y_dis_vl, Y_dis_ts], axis=0)[:, 1]
    AD_idx = np.where(Y==1)[0]
    NC_idx = np.where(Y==0)[0]
    C = np.concatenate([C_dmg_tr, C_dmg_vl, C_dmg_ts], axis=0)
    MMSE = np.concatenate([S_cog_tr, S_cog_vl, S_cog_ts], axis=0)[:, 0] * 30
    MMSE_AD = MMSE[AD_idx]
    MMSE_NC = MMSE[NC_idx]

    C_sex = C[:, :2][:, 1]
    C_age = C[:, 2] * 100
    C_edu = C[:, 3] * 20
    C_sex_AD = C_sex[AD_idx]
    xx1 = np.where(C_sex_AD==0)[0]
    xx2 = np.where(C_sex_AD==1)[0]
    C_sex_NC = C_sex[NC_idx]
    xx3 = np.where(C_sex_NC==0)[0]
    xx4 = np.where(C_sex_NC==1)[0]
    C_age_AD = C_age[AD_idx]
    C_age_NC = C_age[NC_idx]
    C_edu_AD = C_edu[AD_idx]
    C_edu_NC = C_edu[NC_idx]
    C_age_AD_mean = np.mean(C_age_AD)
    C_age_NC_mean = np.mean(C_age_NC)
    C_edu_AD_mean = np.mean(C_edu_AD)
    C_edu_NC_mean = np.mean(C_edu_NC)
    MMSE_AD_mean = np.mean(MMSE_AD)
    MMSE_NC_mean = np.mean(MMSE_NC)
    C_age_AD_std = np.std(C_age_AD)
    C_age_NC_std = np.std(C_age_NC)
    C_edu_AD_std = np.std(C_edu_AD)
    C_edu_NC_std = np.std(C_edu_NC)
    MMSE_AD_std = np.std(MMSE_AD)
    MMSE_NC_std = np.std(MMSE_NC)


    return X_MRI_tr, X_PET_tr, X_SNP_tr, C_dmg_tr, Y_dis_tr, S_cog_tr, \
           X_MRI_vl, X_PET_vl, X_SNP_vl, C_dmg_vl, Y_dis_vl, S_cog_vl, \
           X_MRI_ts, X_PET_ts, X_SNP_ts, C_dmg_ts, Y_dis_ts, S_cog_ts


def load_dataset_noPET(path, SNP_path, fold, task, SNP_mapping=True, divide_ICV=False, random_state=5930):
    data_all = np.loadtxt(path, delimiter=',', dtype=str)
    data_all = data_all[1:, :]

    mytask = []
    is_MCI = False
    for t in task:
        if t == 'CN':
            # mytask.append(0)
            mytask.append(1)

        if t == 'AD':
            # mytask.append(3)
            mytask.append(3)

        if t == 'MCI':  # append sMCI and pMCI samples
            # mytask.append(1)
            mytask.append(2)
            # is_MCI = True

    mytask = np.array(mytask)

    # SNP_match_ID = np.loadtxt("SNP_match_AD_ID.csv",
    #                           delimiter=',', dtype=str)[1:, :][:, 1:].astype(np.int)
    # X_SNP_ori = loadmat('SNPdata.mat')['SNP']  # SNP features (N, 2098)
    # X_SNP_ori = np.concatenate((SNP_match_ID, X_SNP_ori), axis=-1)
    # X_SNP_ori = X_SNP_ori[X_SNP_ori[:, 0].argsort()]
    # X_SNP_ori = X_SNP_ori[:, 1:]
    SNP_match = np.loadtxt(SNP_path, delimiter=',', dtype=str)
    SNP_match = SNP_match[1:, :].astype(np.int)
    SNP_match_ID = SNP_match[:, 0]
    X_SNP_ori = SNP_match[:, 1:]

    subject_id = data_all[:, :1]
    idx_SNP = []
    SNP_new = []
    for num, ID in enumerate(np.sort(SNP_match_ID)):
        idx = np.where(subject_id == str(ID))
        SNP_new.append(np.repeat(X_SNP_ori[num][None, :], len(idx[0]), axis=0))
        idx_SNP.append(idx[0])
    SNP_new = np.concatenate(SNP_new, axis=0)
    idx_SNP = np.concatenate(idx_SNP)
    subject_id = subject_id[idx_SNP, :]
    data_all = data_all[idx_SNP, :]


    # Define the data path
    Y_dis = data_all[:, 2].astype(np.int)  # Disease labels, CN: 1, MCI: 2, AD: 3
    # Extract appropriate indicies for the given task
    task_idx = np.zeros(shape=Y_dis.shape[0])

    for t in range(len(mytask)):
        task_idx += np.array(Y_dis == mytask[t])
    task_idx = task_idx.astype(bool)

    Y_dis = Y_dis[task_idx]
    subject_id = subject_id[task_idx]
    X_ICV = data_all[:, 10:11][task_idx, :].astype(np.float)  #
    X_MRI = data_all[:, 11:101][task_idx, :].astype(np.float)
    X_PET = data_all[:, 101:][task_idx, :].astype(np.float)  # MRI volume features (N, 93)
    C_sex = data_all[:, 4:5][task_idx]  # Sex codes 'F', 'M' (N, )
    C_edu = data_all[:, 5:6][task_idx].astype(np.float)  # Education codes (N, )
    C_age = data_all[:, 3:4][task_idx].astype(np.float)  # Age codes (N, )
    S_MMSE = data_all[:, 8:9][task_idx].astype(np.float)  # Cognitive scores (MMSE), (N, )
    X_SNP = SNP_new[task_idx, :].astype(np.float)

    idx_have_noPET = np.where(X_PET[:, 0] == 0)
    subject_id = subject_id[idx_have_noPET]
    Y_dis = Y_dis[idx_have_noPET]
    X_ICV = X_ICV[idx_have_noPET]
    X_MRI = X_MRI[idx_have_noPET]
    X_PET = X_PET[idx_have_noPET]
    C_sex = C_sex[idx_have_noPET]
    C_edu = C_edu[idx_have_noPET]
    C_age = C_age[idx_have_noPET]
    S_MMSE = S_MMSE[idx_have_noPET]
    X_SNP = X_SNP[idx_have_noPET]

    # Normalization
    C_age /= 100
    C_edu /= 20
    S_MMSE /= 30

    # Categorical encoding for the sex code
    for i in range(np.unique(C_sex).shape[0]):
        C_sex[C_sex == np.unique(C_sex)[i]] = i
    C_sex = C_sex.astype(np.int)
    C_sex = np.eye(np.unique(C_sex).shape[0])[C_sex]

    # Demographic information concatenation
    C_dmg = np.concatenate((C_sex, C_age[:, None], C_edu[:, None]), -1)  # (737, 4)


    # Fold dividing
    unique_id = np.unique(subject_id)
    split_Y = []
    for ID in unique_id:
        split_Y.append(Y_dis[np.where(subject_id == ID)[0]][0])
    split_Y = np.array(split_Y)

    fold_now = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, valid_test_index in kf.split(unique_id, split_Y):
        fold_now = fold_now + 1
        if fold_now == fold:
            select_tr = unique_id[train_index]
            select_vl_ts = valid_test_index
    select_vl, select_ts, _, _ = train_test_split(unique_id[select_vl_ts], split_Y[select_vl_ts], test_size=0.5,
                                                  random_state=random_state)
    _select_tr, _select_vl, _select_ts = np.sort(select_tr.astype(np.int)), np.sort(select_vl.astype(np.int)), \
                                         np.sort(select_ts.astype(np.int))
    tr_idx = []
    for ID in select_tr:
        idx = np.where(subject_id == ID)
        tr_idx.append(idx[0])
    tr_idx = np.concatenate(tr_idx)

    vl_idx = []
    for ID in select_vl:
        idx = np.where(subject_id == ID)
        vl_idx.append(idx[0])
    vl_idx = np.concatenate(vl_idx)

    ts_idx = []
    for ID in select_ts:
        idx = np.where(subject_id == ID)
        ts_idx.append(idx[0])
    ts_idx = np.concatenate(ts_idx)

    # One-hot encoding for the disease label
    for i in range(np.unique(Y_dis).shape[0]):
        Y_dis[Y_dis == np.unique(Y_dis)[i]] = i

    Y_dis = np.eye(np.unique(Y_dis).shape[0])[Y_dis]

    X_ICV_tr, X_ICV_vl, X_ICV_ts = X_ICV[tr_idx, :], X_ICV[vl_idx, :], X_ICV[ts_idx, :]
    X_MRI_tr, X_MRI_vl, X_MRI_ts = X_MRI[tr_idx, :], X_MRI[vl_idx, :], X_MRI[ts_idx, :]
    X_PET_tr, X_PET_vl, X_PET_ts = X_PET[tr_idx, :], X_PET[vl_idx, :], X_PET[ts_idx, :]
    X_SNP_tr, X_SNP_vl, X_SNP_ts = X_SNP[tr_idx, :], X_SNP[vl_idx, :], X_SNP[ts_idx, :]
    C_dmg_tr, C_dmg_vl, C_dmg_ts = C_dmg[tr_idx, :], C_dmg[vl_idx, :], C_dmg[ts_idx, :]
    Y_dis_tr, Y_dis_vl, Y_dis_ts = Y_dis[tr_idx, :], Y_dis[vl_idx, :], Y_dis[ts_idx, :]
    S_cog_tr, S_cog_vl, S_cog_ts = S_MMSE[tr_idx], S_MMSE[vl_idx], S_MMSE[ts_idx, :]

    # MRI/PET normalization
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    if divide_ICV == True:
        X_MRI_tr = X_MRI_tr / X_ICV_tr
        X_MRI_vl = X_MRI_vl / X_ICV_vl
        X_MRI_ts = X_MRI_ts / X_ICV_ts
    X_MRI_tr = scaler1.fit_transform(X_MRI_tr)
    X_MRI_vl = scaler1.transform(X_MRI_vl)
    X_MRI_ts = scaler1.transform(X_MRI_ts)
    X_PET_tr = scaler2.fit_transform(X_PET_tr)
    X_PET_vl = scaler2.transform(X_PET_vl)
    X_PET_ts = scaler2.transform(X_PET_ts)

    if SNP_mapping:
        # SNP encoding
        X_SNP_tr_ori = X_SNP_tr.copy()
        X_SNP_tr, X_SNP_vl = SNP_encoder(X_SNP_tr=X_SNP_tr, X_SNP_ts=X_SNP_vl)
        _, X_SNP_ts = SNP_encoder(X_SNP_tr=X_SNP_tr_ori, X_SNP_ts=X_SNP_ts)

    C_dmg_tr = C_dmg_tr[:, 0, :]
    C_dmg_vl = C_dmg_vl[:, 0, :]
    C_dmg_ts = C_dmg_ts[:, 0, :]

    Y = np.concatenate([Y_dis_tr, Y_dis_vl, Y_dis_ts], axis=0)[:, 1]
    AD_idx = np.where(Y==1)[0]
    NC_idx = np.where(Y==0)[0]
    C = np.concatenate([C_dmg_tr, C_dmg_vl, C_dmg_ts], axis=0)
    MMSE = np.concatenate([S_cog_tr, S_cog_vl, S_cog_ts], axis=0)[:, 0] * 30
    MMSE_AD = MMSE[AD_idx]
    MMSE_NC = MMSE[NC_idx]

    C_sex = C[:, :2][:, 1]
    C_age = C[:, 2] * 100
    C_edu = C[:, 3] * 20
    C_sex_AD = C_sex[AD_idx]
    xx1 = np.where(C_sex_AD==0)[0]
    xx2 = np.where(C_sex_AD==1)[0]
    C_sex_NC = C_sex[NC_idx]
    xx3 = np.where(C_sex_NC==0)[0]
    xx4 = np.where(C_sex_NC==1)[0]
    C_age_AD = C_age[AD_idx]
    C_age_NC = C_age[NC_idx]
    C_edu_AD = C_edu[AD_idx]
    C_edu_NC = C_edu[NC_idx]
    C_age_AD_mean = np.mean(C_age_AD)
    C_age_NC_mean = np.mean(C_age_NC)
    C_edu_AD_mean = np.mean(C_edu_AD)
    C_edu_NC_mean = np.mean(C_edu_NC)
    MMSE_AD_mean = np.mean(MMSE_AD)
    MMSE_NC_mean = np.mean(MMSE_NC)
    C_age_AD_std = np.std(C_age_AD)
    C_age_NC_std = np.std(C_age_NC)
    C_edu_AD_std = np.std(C_edu_AD)
    C_edu_NC_std = np.std(C_edu_NC)
    MMSE_AD_std = np.std(MMSE_AD)
    MMSE_NC_std = np.std(MMSE_NC)

    return X_MRI_tr, X_PET_tr, X_SNP_tr, C_dmg_tr, Y_dis_tr, S_cog_tr, \
           X_MRI_vl, X_PET_vl, X_SNP_vl, C_dmg_vl, Y_dis_vl, S_cog_vl, \
           X_MRI_ts, X_PET_ts, X_SNP_ts, C_dmg_ts, Y_dis_ts, S_cog_ts
