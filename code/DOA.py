import os
import pandas as pd
import numpy as np
import argparse
import numpy as np
import sys
sys.path.append('/')
from utility.DatasetProcessor import DatasetProcessor
from numba import jit

@jit
def compute_kc_user(name):
    fileName = name + "_responses.csv"
    f = open(fileName, "r")
    lines = f.readlines()
    kc_user = []
    kc_user_val = []
    dico_u = []
    old_k = -1
    num_kc = 0
    for line in lines:
        r = line.split(',')
        if (int(r[0]) != old_k):
            num_kc = num_kc + 1
            if (old_k != -1):
                kc_user.append(new_kc_user)
                kc_user_val.append(new_kc_user_val)
                dico_u.append(dico_uk)
            new_kc_user = []
            new_kc_user_val = []
            dico_uk = {}
            old_k = int(r[0])
        user = int(r[1])
        if user not in dico_uk:
            dico_uk[user] = len(dico_uk)
            new_kc_user.append([])
            new_kc_user_val.append([])
        # add item and values
        for i in range(2, len(r)):
            try:
                q, rep = parse_it(r[i])
                new_kc_user[dico_uk[user]].append(int(q))
                new_kc_user_val[dico_uk[user]].append(rep)
            except:
                continue
    kc_user.append(new_kc_user)
    kc_user_val.append(new_kc_user_val)
    dico_u.append(dico_uk)
    # sort files ?
    for k in range(len(kc_user)):
        for u in range(len(kc_user[k])):
            kc_user[k][u], kc_user_val[k][u] = zip(*sorted(zip(kc_user[k][u], kc_user_val[k][u])))
    return kc_user, kc_user_val, dico_u, num_kc

@jit
def parse_it(it):
    # Extract question/answer from label
    p = it.find('_')
    q = it[:p]
    r = it[p + 1:]
    r = int(float(r))
    return q, r

@jit
def dao_creuse(F, kc_user, kc_user_val, dico_u):
    doa = []
    user_embed = F
    for k in range(num_kc):
        Z1 = 0.0
        v = 0.0
        users = list(dico_u[k])
        for a in range(len(kc_user[k])):
            for b in range(len(kc_user[k])):  
                Z2 = 0.0
                Z0 = 0.0
                ua = users[a]
                ub = users[b]
                try:
                    user_embed[ua][k] > user_embed[ub][k]
                except:
                    print((ua))
                    print((ub))
                    print((k))
                if (user_embed[ua][k] > user_embed[ub][k]):
                    ia = a  
                    ib = b  
                    ma = 0
                    mb = 0
                    while (ma < len(kc_user[k][ia])) and (mb < len(kc_user[k][ib])):
                        if (kc_user[k][ia][ma] < kc_user[k][ib][mb]):
                            ma = ma + 1
                        else:
                            if (kc_user[k][ia][ma] > kc_user[k][ib][mb]):
                                mb = mb + 1
                            else:
                                # same item
                                if (kc_user_val[k][ia][ma] > kc_user_val[k][ib][mb]):
                                    Z2 = Z2 + 1
                                    Z0 = Z0 + 1
                                else:
                                    if (kc_user_val[k][ia][ma] < kc_user_val[k][ib][mb]):
                                        Z0 = Z0 + 1
                                ma = ma + 1
                                mb = mb + 1
                    if (Z0 > 0):
                        Z1 = Z1 + 1
                        v = v + Z2 / Z0
        if (Z1 > 0):
            v = v / Z1
        # print("dao k : ", k, v)
        doa.append(v)
    return doa

@jit
def fromDFtoArray(name, vector, type_value):
    # Read dataframe and generate a matrix or
    # a vector of appropriate type
    df = pd.read_csv(name, index_col=None, header=None)
    cols = df.columns
    if (type_value == "f"):
        for col in cols:
            df[col] = df[col].astype(float)
    if (type_value == 'i'):
        for col in cols:
            df[col] = df[col].astype(int)
    r = df.values
    if (vector):
        r = r.reshape(-1, )
    return r

@jit
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--datasetName", help="data file")
    args = parser.parse_args()
    dataset_name = args.datasetName

    xp_folder_name = "8-Preprocessing_pipeline"
    rel_data_path = "../data/raw_format"
    rel_xp_path = "../experiments_logs"
    abs_xp_path = "../experiments_logs"
    embs_path = "../results/table_2/users/"

    exp = DatasetProcessor("3", data_path=rel_data_path, experiment_path=rel_xp_path)

    dataset_file= dataset_name+'.csv'
    metadata_file = "metadata_"+dataset_name+".json"

    exp.import_dataset(dataset_file_name=dataset_file, metadata_file_name="metadata/"+metadata_file,
                   dataset_name=dataset_name)
    exp.shuffle(dataset_name=dataset_name, attributes=["start_time"], group_attributes=["user_id"], rgn_seed=1)



    print("dataset ",dataset_name)

    dao_mean_train = {"DINA":[],"MCD":[],"NCDM":[],"MIRT":[],"CD-BPR":[]}
    dao_mean_test = {"DINA":[],"MCD":[],"NCDM":[],"MIRT":[],"CD-BPR":[]}
    reo_mean = {"DINA":[],"MCD":[],"NCDM":[],"MIRT":[],"CD-BPR":[]}

    for i_fold in range(5):
        print("fold", i_fold)
        for model in dao_mean_train.keys() :
            print("model ", model)

            exp.train_test_split(dataset_name, test_proportion=0.2, valid_proportion=0.2, n_folds=5, i_fold=i_fold)

            # train
            data_train = exp.get_train_valid_dataset(dataset_name)
            data_train.insert(1, 'q_r', data_train['item_id'].astype(str) + '_' + data_train['correct'].astype(str))
            data_train.insert(1, 'q_r_count', data_train.groupby(['skill_id', 'user_id']).cumcount() + 1)
            pivot_df = data_train.pivot_table(index=['skill_id', 'user_id'], columns='q_r_count', values='q_r',
                                              aggfunc=lambda x: x).reset_index()
            data = pivot_df.sort_values('skill_id')
            data.to_csv('dataTrain_responses.csv', header=False, index=False, na_rep='')

            # test
            data_test = exp.get_test_dataset(dataset_name)
            data_test.insert(1, 'q_r', data_test['item_id'].astype(str) + '_' + data_test['correct'].astype(str))
            data_test.insert(1, 'q_r_count', data_test.groupby(['skill_id', 'user_id']).cumcount() + 1)
            pivot_df = data_test.pivot_table(index=['skill_id', 'user_id'], columns='q_r_count', values='q_r',
                                             aggfunc=lambda x: x).reset_index()
            data = pivot_df.sort_values('skill_id')
            data.to_csv('dataTest_responses.csv', header=False, index=False, na_rep='')

            F = fromDFtoArray(embs_path + dataset_name +"_"+str(i_fold)+"_"+model+".csv", False, 'f')
            # print(F)
            kc_user, kc_user_val, dico_u, num_kc = compute_kc_user("dataTrain")
            print("num_kc", num_kc)
            r_train = dao_creuse(F, kc_user, kc_user_val, dico_u)

            kc_user, kc_user_val, dico_u, num_kc = compute_kc_user("dataTest")
            r_test = dao_creuse(F, kc_user, kc_user_val, dico_u)

            dao_mean_train[model].append(np.mean(r_train))
            dao_mean_test[model].append(np.mean(r_test))
            reo_mean[model].append(1 - np.mean(np.array(r_test)) / np.mean(np.array(r_train)))

    print("dao train", dao_mean_train)
    print("dao test", dao_mean_test)
    print("reo", reo_mean)
    for model in dao_mean_train.keys():
        print("model",model)
        print("dao train", np.mean(np.array(dao_mean_train[model])),"+-",np.std(np.array(dao_mean_train[model])))
        print("dao test", np.mean(np.array(dao_mean_test[model])),"+-",np.std(np.array(dao_mean_test[model])))
        print("reo test", np.mean(np.array(reo_mean[model])), "+-", np.std(np.array(reo_mean[model])))
