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

from collections import Counter
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

def evaluate_all(dataTrain, dataTest, filename):
    dico_kc, dico_users, dico_items, d_quest_val, nb_item_train = read_file_cv(dataTrain, dataTest)
    embedding_size = len(dico_kc)
    dico_items, t_train, ty_train, item_users = parse_dataframe_cv(dataTrain, dico_kc, dico_users, dico_items, d_quest_val, True)
    train, y_train = generate_quad(dico_items, t_train, ty_train, item_users, d_quest_val) 
    dico_items, test, y_test = parse_dataframe_cv(dataTest, dico_kc, dico_users, dico_items,d_quest_val, False) 

    num_users = len(dico_users)   
    num_items = len(dico_items)
    cc_exp_row, cc_exp_col = init_frequencielle(train, num_users, num_items, embedding_size, dico_items, d_quest_val)
    bpr_model = BPRModel(batch_size, num_users, num_items, embedding_size, device, cc_exp_row, cc_exp_col)
    bpr_model = bpr_model.to(device)
    # Training loop
    acc = bpr_model.train(train, len(dico_kc), epochs, batch_size, y_train)
    new_embedding_value = bpr_model.user_embeddings.weight.clone().detach().cpu().numpy()
    write_file_doa(filename, new_embedding_value, train, dico_kc, dico_users, dico_items)  
    doa, doatot = compute_doa(filename)
    # write embed items
    new_embedding_items = bpr_model.item_embeddings.weight.clone().detach().cpu().numpy()
    write_file(filename+"embedding_items.csv", new_embedding_items[0:nb_item_train])
    # Test
    acc, precision, rappel, all_decisions, all_prefs = bpr_model.evaluate_model(test, len(dico_kc), y_test) 
    '''
    s = str(acc) +","+ str( precision)+ ","+str(rappel)+ ","+str(doa)
    for i in range(embedding_size):
        s = s + ','+ str(doatot[i])
    print(s)
    '''
    return acc, precision, rappel, doa

def cross_validation(data, kfolds, filename):
    n = data.shape[0]
    fold_size = n // kfolds

    indices = np.arange(n)
    np.random.shuffle(indices)
    doa = []
    acc = []
    precision = [] 
    rappel = []
    for i in range(kfolds):
        start = i * fold_size
        end = (i + 1) * fold_size if i != kfolds - 1 else n
        validation_indices = indices[start:end]
        training_indices = np.concatenate([indices[:start], indices[end:]])
        validation_data = data[validation_indices]
        training_data = data[training_indices]
        dataTrain = pd.DataFrame(data = training_data, columns=['user_id', 'item_id','correct','knowledge',"max_val"])
        dataTest = pd.DataFrame(data = validation_data, columns=['user_id', 'item_id','correct','knowledge',"max_val"])
        
        a, p, r, d = evaluate_all(dataTrain, dataTest, filename)
        acc.append(a)
        precision.append(p) 
        rappel.append(r)
        doa.append(d)
    acc = np.array(acc)
    precision = np.array(precision)
    rappel = np.array(rappel)
    doa = np.array(doa)
    print("avg values", np.mean(acc), np.mean(precision), np.mean(rappel), np.mean(doa))
#############################
#############################
# HP
epochs = 30
batch_size = 1024


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--data", help="data file")
    args = parser.parse_args()
    data = args.data

    filename = data[:-4]
    df = pd.read_csv(data)
    cross_validation(df.values, 5, filename)
     
   
