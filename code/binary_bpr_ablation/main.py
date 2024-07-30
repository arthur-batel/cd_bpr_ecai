import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import pandas as pd
from scipy import stats
import csv
import json 
import random
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics.cluster import contingency_matrix
from utils import *
from BPR_model import *

dtype = torch.float32
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  


def read_file(dataTrain, dataTest):
    # Compute dictionaries
    df = pd.read_csv(dataTrain, names=['user_id', 'item_id','correct','knowledge'])
    dfTest = pd.read_csv(dataTest, names=['user_id', 'item_id','correct','knowledge'])
    # dico kc
    kc = df['knowledge']
    kcT = dfTest['knowledge']
    kc = flattern_arrays(kc.values, kcT.values)
    num_kc = len(kc)
    dico_kc = { k:v for (k,v) in zip(kc, range(len(kc)))}
    # dico users
    users = df['user_id']
    usersT = dfTest['user_id']
    users = flattern_arrays(users.values, usersT.values)
    num_users = len(users)
    dico_users = { k:v for (k,v) in zip(users, range(num_users))}
    # dico items and their associated kc
    itemsDT = df['item_id']
    itemsT = dfTest['item_id']
    items = flattern_arrays(itemsDT.values, itemsT.values)
    num_items = len(items)
    dico_items = { k:v for (k,v) in zip(items, range(num_items))} 
    return dico_kc, dico_users, dico_items
    
def parse_dataframe(data, dico_kc, dico_users, dico_item, is_train = True):
    df = pd.read_csv(data, names=['user_id', 'item_id','correct','knowledge'])
    # Compute table of positive and negative items by KC and Users
    # and the dictionary that associate the KC to a question/answer
    num_kc = len(dico_kc)
    num_users = len(dico_users)
    # Find positive items for each kc/user
    triplets = []
    y_true = []
    for k in range(num_kc):
        triplets.append([])
        y_true.append([])
    k = 0
    if(is_train):
        item_users = {}
    grouped_kc = df.groupby('knowledge')
    for kc_name, df_kc in grouped_kc:
        # Find positive items for each user
        grouped = df_kc.groupby('user_id')
        for group_name, df_group in grouped:
            for row_index, row in df_group.iterrows():
                col = row['item_id']
                if col not in dico_items:
                    dico_items[col] = len(dico_items)
                q,r = parse_it(col)
                col_neg = q+'_'+str(1-int(r))
                if col_neg not in dico_items:
                    dico_items[col_neg] = len(dico_items)
                
                if(is_train):
                    triplets[dico_kc[int(kc_name)]].append([dico_users[group_name], dico_items[col], dico_items[col_neg]]) 
                    if(dico_items[col] not in item_users):
                        item_users[dico_items[col]] = []
                    item_users[dico_items[col]].append(dico_users[group_name])
                    if(r == 1):
                        y_true[dico_kc[int(kc_name)]].append(r)
                    else: 
                        y_true[dico_kc[int(kc_name)]].append(0)
                else:
                    # reponse q_1, q_0 with y the expected answer
                    if(r==1): 
                        triplets[dico_kc[int(kc_name)]].append([dico_users[group_name], dico_items[col], dico_items[col_neg]]) 
                    else:
                        triplets[dico_kc[int(kc_name)]].append([dico_users[group_name], dico_items[col_neg], dico_items[col]]) 
                    y_true[dico_kc[int(kc_name)]].append(r) 
    for k in range(num_kc):
        triplets[k] = np.array(triplets[k])
        y_true[k] = np.array(y_true[k])
        
    if(is_train):
        return dico_items, triplets, y_true, item_users 
    else:
        return dico_items, triplets, y_true

def generate_quad(dico_items, triplets, t_trainy, item_users):
    l = list(dico_items)
    quadriplets = []
    y = []
    for k in range(len(triplets)):
        t_quadriplets = []
        t_y = []
        for i in range(len(triplets[k])):
            t = triplets[k][i]
            q,r = parse_it(l[t[1]])
            if(t[2] in item_users) and (r == 1):
                u = random.randint(0,len(item_users[t[2]])-1)
                uu = item_users[t[2]][u]
                t_quadriplets.append([t[0], t[1], t[2], uu])
                t_y.append(t_trainy[k][i])
            else:
                t_quadriplets.append([t[0], t[1], t[2], t[0]])
                t_y.append(t_trainy[k][i])
        quadriplets.append(np.array(t_quadriplets))  
        y.append(np.array(t_y))
    return quadriplets, y

def init_frequencielle(train, n, p, dim, dico_items): 
    it = list(dico_items)
    cc_exp_row = np.random.rand(n,dim) 
    cc_exp_col = np.random.rand(p,dim)
    for k in range(len(train)):
        for i in range(len(train[k])):
            user = train[k][i][0]
            item = train[k][i][1]
            q,r = parse_it(it[item])
            if(r == 1):
                cc_exp_row[user][k] += 1
                cc_exp_col[item][k] += 1
            if(r == 0):
                cc_exp_row[user][k] -= 1
                cc_exp_col[item][k] -= 1
    for i in range(n):
        cc_exp_row[i] = cc_exp_row[i] / np.sum(cc_exp_row[i])
    for i in range(p):
        cc_exp_col[i] = cc_exp_col[i] / np.sum(cc_exp_col[i])
    return cc_exp_row, cc_exp_col 

def write_file_doa(FileName, embed, train, dico_kc, dico_users, dico_items, ablation):
    # write embeddings 
    it = list(dico_items)
    ut = list(dico_users)
    nom = FileName+"_embed_ablation_"+str(ablation)+".csv"
    f = open(nom, 'w')
    writer = csv.writer(f)
    for i in range(embed.shape[0]):
        row = embed[i]
        writer.writerow(row)
    f.close()
    # write R the responses
    # user followed by the list of item_resp groupes in block of k 
    nom = FileName+"_responses.csv"
    f = open(nom, 'w')
    writer = csv.writer(f)
    previous_user = -1
    for k in range(len(dico_kc)):
        for i in range(len(train[k])):
            if(previous_user != -1) and (train[k][i][0] != previous_user):
                positive_items = list(set(positive_items))
                row = row + positive_items
                writer.writerow(row)
            if(train[k][i][0] != previous_user):
                row = [k]+[train[k][i][0]]
                positive_items = []
                previous_user = train[k][i][0]
            positive_items.append(it[train[k][i][1]])
    # Remove duplicate positive_items
    positive_items = list(set(positive_items))
    row = row + positive_items
    writer.writerow(row)
    f.close()


#############################
#############################
# HP
batch_size = 512
ablation = 0
# 0 no ablation, 1 ablation L2, 2 ablation init, 3 both
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dtrain", "--dataTrain", help="data file")
    parser.add_argument("-dtest", "--dataTest", help="data file")
    parser.add_argument("-ab", "--ablation", help="int")
    parser.add_argument("-e", "--epochs", help="int",type=int)
    args = parser.parse_args()
    dataTrain = args.dataTrain
    dataTest = args.dataTest
    epochs = args.epochs
    if args.ablation:
        ablation = int(args.ablation)
    file = dataTrain[:-4]
    
    dico_kc, dico_users, dico_items = read_file(dataTrain, dataTest)
    embedding_size = len(dico_kc)
    
    dico_items, t_train, ty_train, item_users = parse_dataframe(dataTrain, dico_kc, dico_users, dico_items, True)
    train, y_train = generate_quad(dico_items, t_train, ty_train, item_users) 
    dico_items, test, y_test = parse_dataframe(dataTest, dico_kc, dico_users, dico_items, False)
    print("NB (question-answer) in the data", int(len(dico_items)/2))  
    num_users = len(dico_users)   
    num_items = len(dico_items)
    cc_exp_row, cc_exp_col = init_frequencielle(train, num_users, num_items, embedding_size, dico_items)
    bpr_model = BPRModel(batch_size, num_users, num_items, embedding_size, device, cc_exp_row, cc_exp_col, ablation)
    bpr_model = bpr_model.to(device)
    # Training loop
    acc = bpr_model.train(train, len(dico_kc), epochs, batch_size, y_train, ablation)
    # DOA 
    new_embedding_value = bpr_model.user_embeddings.weight.clone().detach().cpu().numpy()
    write_file_doa(file, new_embedding_value, train, dico_kc, dico_users, dico_items, ablation)
    doa = compute_doa(file)
    print("DOA:", doa)
    # Test
    acc, precision, recall,f1, auc, rmse = bpr_model.evaluate_model(test, len(dico_kc), y_test)

