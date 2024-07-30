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
import os 
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics.cluster import contingency_matrix
from utils import *
from BPR_model import *
from sklearn.preprocessing import normalize

dtype = torch.float32
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  


def read_file(dataTrain, dataTest):
    # Compute dictionaries
    df = pd.read_csv(dataTrain, names=['user_id', 'item_id','correct','knowledge',"max_val"])
    dfTest = pd.read_csv(dataTest, names=['user_id', 'item_id','correct','knowledge',"max_val"])
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
    nb_item_train = len(flattern_array(itemsDT.values))
    itemsT = dfTest['item_id']
    items = flattern_arrays(itemsDT.values, itemsT.values)
    num_items = len(items)
    dico_items = { k:v for (k,v) in zip(items, range(num_items))} 
    d_quest_val = {}
    dico_items, d_quest_val = parse_item_values(df[['item_id', 'max_val']], dico_items, d_quest_val)
    dico_items, d_quest_val = parse_item_values(dfTest[['item_id', 'max_val']], dico_items, d_quest_val)
    
    for quest_val in d_quest_val:
        quest_val = flattern_array(np.array(quest_val))
    return dico_kc, dico_users, dico_items, d_quest_val, nb_item_train

def parse_dataframe(data, dico_kc, dico_users, dico_items, d_quest_val, nb_item_train, is_train = True):
    df = pd.read_csv(data, names=['user_id', 'item_id','correct','knowledge','max_val'])
    
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
    items_kc = np.full(len(dico_items),0)
    grouped_kc = df.groupby('knowledge')
    for kc_name, df_kc in grouped_kc:
        # Find positive items for each user
        grouped = df_kc.groupby('user_id')
        for group_name, df_group in grouped:
            for row_index, row in df_group.iterrows():
                col = row['item_id']
                q,r = parse_it(col)
                q = int(q)                
                #on en garde une aleatoirement pour le triplet
                if(is_train):
                    for r_kept in range(1):#len(d_quest_val[q])):
                        r_kept = r
                        while(r_kept == r) and (len(d_quest_val[q]) > 1):
                            r_kept = random.randint(0,len(d_quest_val[q])-1)
                        col_neg_kept = str(q)+'_'+str(d_quest_val[q][r_kept])     
                        #print(r_kept, r)
                        amplitude = 1*(np.abs(r-r_kept) + 1)
                        triplets[dico_kc[int(kc_name)]].append([dico_users[group_name], dico_items[col], dico_items[col_neg_kept], amplitude]) 
                        if(dico_items[col] not in item_users):
                            item_users[dico_items[col]] = []
                        item_users[dico_items[col]].append(dico_users[group_name])
                        y_true[dico_kc[int(kc_name)]].append(r > r_kept)
                        items_kc[dico_items[col]] = int(kc_name)
                else:
                    # a voir, dans la decision ils doivent etre tous plus loin...
                    # on fait toutes les paires
                    # a bien verifier !!!!
                    for j in range(len(d_quest_val[q])):
                        #r_kept = random.randint(0,len(d_quest_val[q])-1)
                        r_kept = d_quest_val[q][j]
                        col_neg_kept = str(q)+'_'+str(r_kept)    
                        amplitude = 1*(np.abs(r-r_kept) + 1 )
                        if(r >= r_kept): 
                            triplets[dico_kc[int(kc_name)]].append([dico_users[group_name], dico_items[col], dico_items[col_neg_kept], amplitude]) 
                            y_true[dico_kc[int(kc_name)]].append(r > r_kept) 
                        else:
                            triplets[dico_kc[int(kc_name)]].append([dico_users[group_name], dico_items[col_neg_kept], dico_items[col], amplitude]) 
                            y_true[dico_kc[int(kc_name)]].append(r > r_kept) 
    for k in range(num_kc):
        triplets[k] = np.array(triplets[k])
        y_true[k] = np.array(y_true[k])
    
    
    if(is_train):
        write_file_std(data[:-4]+"_kc.txt", items_kc[0:nb_item_train])
        return dico_items, triplets, y_true, item_users
    else:
        return dico_items, triplets, y_true

def compute_accuracy_multi_mod(all_preferences, dico_users, dico_items, dataTest):
    # Remove duplicate
    new_array = [tuple(row) for row in all_preferences]
    all_preferences = np.unique(new_array, axis=0)
    # Revert dictionaries
    l = list(dico_items)
    lu = list(dico_users)
    # Extract the questions
    list_quest = []
    for i in range(len(all_preferences)):
        q,r = parse_it(l[int(all_preferences[i,1])])
        list_quest.append(int(q))
    all_preferences = np.concatenate((all_preferences, np.array(list_quest).reshape(-1,1)), axis=1)   
    # Compute the predicted value
    responses = []
    list_user = flattern_array(all_preferences[:,0])
    for u in list_user:
        my_users = all_preferences[:,0] == u
        list_quest = flattern_array(all_preferences[my_users,3])
        for quest in list_quest: 
            rows = all_preferences[:,3] == quest
            my_rows = np.logical_and(rows,my_users)
            # we got the rows corresponding to a user and a question
            # and take the row with the maximum predicted values
            m = np.argmax(all_preferences[my_rows,2])
            # Get the modality
            item = int(all_preferences[my_rows][m][1])
            q1,modality = parse_it(l[item])
            responses.append([lu[int(u)], int(quest), int(modality)])   
    # Sort responses
    dfPred = pd.DataFrame(responses, columns=['user_id', 'question','modality'])
    dfPred = dfPred.sort_values(by=['user_id', 'question'])
    pred = dfPred['modality'].values
    # True data
    dfTrue = pd.read_csv(dataTest, names=['user_id', 'item_id','correct','knowledge',"question"])
    for row_index, row in dfTrue.iterrows():
        col = row['item_id']
        q,r = parse_it(col)
        dfTrue.at[row_index,'question'] = int(q)
    dfTrue = dfTrue.drop_duplicates(subset=['user_id', 'question'])  
    dfTrue = dfTrue.sort_values(by=['user_id', 'question'])
    print("Accuracy multi-modal", accuracy_score(dfTrue['correct'].values, pred), "RMSE", np.sqrt(mean_squared_error(dfTrue['correct'].values, pred)))
               

        
   
#############################
#############################
# HP
epochs = 20
batch_size = 1024
alpha = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dtrain", "--dataTrain", help="data file")
    parser.add_argument("-dtest", "--dataTest", help="data file")
    parser.add_argument("-a", "--alpha", help="float")
    args = parser.parse_args()
    dataTrain = args.dataTrain
    dataTest = args.dataTest
    filename = dataTrain[:-4]
    if args.alpha:
        alpha = int(args.alpha)
    
    dico_kc, dico_users, dico_items, d_quest_val, nb_item_train = read_file(dataTrain, dataTest)
    embedding_size = len(dico_kc)
    dico_items, t_train, ty_train, item_users = parse_dataframe(dataTrain, dico_kc, dico_users, dico_items, d_quest_val, nb_item_train, True)
    train, y_train = generate_quad(dico_items, t_train, ty_train, item_users, d_quest_val) 
    dico_items, test, y_test = parse_dataframe(dataTest, dico_kc, dico_users, dico_items,d_quest_val, nb_item_train, False)
    
    num_users = len(dico_users)   
    num_items = len(dico_items)
    cc_exp_row, cc_exp_col = init_frequencielle(train, num_users, num_items, embedding_size, dico_items, d_quest_val)
    bpr_model = BPRModel(batch_size, num_users, num_items, embedding_size, device, cc_exp_row, cc_exp_col)
    bpr_model = bpr_model.to(device)
    # Training loop
    acc = bpr_model.train(train, len(dico_kc), epochs, batch_size, y_train)
    
    new_embedding_value = bpr_model.user_embeddings.weight.clone().detach().cpu().numpy()
    write_file_doa(filename, new_embedding_value, train, dico_kc, dico_users, dico_items)  
    doa, rdoa = compute_doa(filename)
      
    # Test
    acc, precision, rappel, all_predictions, all_preferences = bpr_model.evaluate_model(test, len(dico_kc), y_test) 
    s = str(acc) +","+ str( precision)+ ","+str(rappel)+ ","+str(doa)
    print(s)
    compute_accuracy_multi_mod(all_preferences, dico_users, dico_items, dataTest)
    # coding users and kc from user_label.csv
    '''
    os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(dataTrain))
    file = dir_path+'/user_label.csv'
    dfQE = pd.read_csv(file, names=['user_id', 'knowledge',"value"])
    # create matrix
    questEval = np.full((num_users, embedding_size),-1)
    #fill the matrix
    uu = flattern_array(dfQE['user_id'].values)
    for t in dfQE.values:
        if(int(t[0]) in dico_users):
            ind = dico_users[int(t[0])]
            kc = dico_kc[int(t[1])]  
            val = int(t[2])
            questEval[ind, kc] = val  
    write_file(filename+"_user_quest_label.csv", questEval)
    '''
#the_predictions = compute_pred(all_predictions, test, y_test, dico_users, dico_items)
#file = "_test_predictions_"+str(alpha)+".csv"
#write_file(filename + file, np.array(the_predictions))