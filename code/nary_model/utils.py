import numpy as np
import pandas as pd
import csv
import random
from sklearn.preprocessing import normalize

def write_file_std(name, data):
    f = open(name, 'w')
    for i in range(len(data)):
        f.write(str(data[i])+"\n")   
    f.close()

def write_file(name, data):
    f = open(name, 'w')
    writer = csv.writer(f)
    for i in range(data.shape[0]):
        writer.writerow(data[i])
    f.close()

def fromDFtoArray(name, vector, type_value):
    # Read dataframe and generate a matrix or 
    # a vector of appropriate type
    df = pd.read_csv(name,index_col=None, header=None)
    cols = df.columns
    if(type_value == "f"):    
        for col in cols:
            df[col] = df[col].astype(float)
    if(type_value == 'i'):
        for col in cols:
            df[col] = df[col].astype(int)
    r = df.values
    if(vector):
        r = r.reshape(-1,)
    return r

def parse_it(it):
    # Extract question/answer from label
    p = it.find('_')
    q = it[:p]
    r = it[p+1:]
    r = int(float(r))
    return q,r  

def parse_item_values(d, dico_items, d_quest_val):
    for t in d.values:
        col = t[0]
        ma = t[1]
        if col not in dico_items:
            dico_items[col] = len(dico_items)
        # Warning, all user's answers are positives!
        q,r = parse_it(col)
        q = int(q)
        # on rajoute toutes les valeurs avant r jusqu'a 0
        if(q not in d_quest_val):
            d_quest_val[q] = []
       
        for val in range(ma):
            if(val not in d_quest_val[q]):
                d_quest_val[q].append(val)
            col_neg = str(q)+'_'+str(val)
            if col_neg not in dico_items:
                dico_items[col_neg] = len(dico_items)
    return dico_items, d_quest_val

def flattern_arrays(user_idx, user_idx_test):
    # Transform two arrays in an array of unique values
    liste1 = list(user_idx)
    liste2 = list(user_idx_test)
    new = list(set(liste2).difference(liste1))
    liste1.sort()
    liste1 = np.array(liste1).reshape((-1,1))
    liste1 = np.unique(liste1).tolist()
    new.sort()
    new = np.array(new).reshape((-1,1))
    new = np.unique(new).tolist()
    if(len(new) > 0):
        users = liste1 + new
    else:
        users = liste1
    return users

def flattern_array(user_idx):
    # Transform an array in an array of unique values
    users = list(user_idx.reshape((-1,1))) 
    users.sort()
    users = np.array(users)
    users = np.unique(users)
    return users

def parse_kc(s):
    d = []
    s = s[1:]
    while(s.find(',') > 0):
        t = s[:s.find(',')]
        d.append(int(t))
        s = s[s.find(',')+1:]
    s = s[:len(s)-1]
    d.append(int(s))
    return d

def parse(data):
    d = []
    for i in range(len(data)):
        s = data[i][1:]
        while(s.find(',') > 0):
            t = s[:s.find(',')]
            d.append(int(t))
            s = s[s.find(',')+1:]
        s = s[:len(s)-1]
        d.append(int(s))
    d = flattern_array(np.array(d))
    return d

def compute_kc_user(name):
    fileName = name+"_responses.csv"
    f = open(fileName, "r")
    lines = f.readlines()
    kc_user = []
    kc_user_val = []
    dico_u = []
    old_k = -1
    num_kc = 0
    for line in lines:
        r = line.split(',')
        if(int(r[0]) != old_k):
            num_kc = num_kc + 1
            if(old_k != -1):
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
        for i in range(2,len(r)):
            q,rep = parse_it(r[i])
            new_kc_user[dico_uk[user]].append(int(q))
            new_kc_user_val[dico_uk[user]].append(rep)
    kc_user.append(new_kc_user)
    kc_user_val.append(new_kc_user_val)
    dico_u.append(dico_uk)
    # sort files 
    for k in range(len(kc_user)):
        for u in range(len(kc_user[k])):
            kc_user[k][u], kc_user_val[k][u] = zip(*sorted(zip(kc_user[k][u], kc_user_val[k][u])))
    return kc_user, kc_user_val, dico_u, num_kc

def doa_creuse(F, kc_user, kc_user_val, dico_u):
    doa = []
    user_embed = F
    num_kc = len(kc_user)
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
                if(user_embed[ua][k] > user_embed[ub][k]):
                    ia = a
                    ib = b
                    ma = 0
                    mb = 0
                    while(ma < len(kc_user[k][ia])) and (mb < len(kc_user[k][ib])):
                        if(kc_user[k][ia][ma] < kc_user[k][ib][mb]):
                            ma = ma + 1
                        else:
                            if(kc_user[k][ia][ma] > kc_user[k][ib][mb]):
                                mb = mb + 1
                            else:
                                # meme item
                                if(kc_user_val[k][ia][ma] > kc_user_val[k][ib][mb]):
                                    Z2 = Z2 + 1
                                    Z0 = Z0 + 1
                                else:
                                    if(kc_user_val[k][ia][ma] < kc_user_val[k][ib][mb]): 
                                        Z0 = Z0 + 1 
                                ma = ma + 1
                                mb = mb + 1  
                    if(Z0 > 0):
                        Z1 = Z1 + 1
                        v = v + Z2/Z0
                
        if(Z1 > 0):
            v = v / Z1
        doa.append(v)  
    return doa

def read_file_cv(df, dfTest):
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
    
def parse_dataframe_cv(df, dico_kc, dico_users, dico_items, d_quest_val, is_train = True):    
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
                q,r = parse_it(col)
                q = int(q)
                if(is_train):
                    for r_kept in range(1):
                        r_kept = r
                        while(r_kept == r) and (len(d_quest_val[q]) > 1):
                            r_kept = random.randint(0,len(d_quest_val[q])-1)
                        col_neg_kept = str(q)+'_'+str(d_quest_val[q][r_kept])     
                        amplitude = 1*(np.abs(r-r_kept) + 1)
                        triplets[dico_kc[int(kc_name)]].append([dico_users[group_name], dico_items[col], dico_items[col_neg_kept], amplitude]) 
                        if(dico_items[col] not in item_users):
                            item_users[dico_items[col]] = []
                        item_users[dico_items[col]].append(dico_users[group_name])
                        y_true[dico_kc[int(kc_name)]].append(r > r_kept)
                else:
                    # a voir, dans la decision ils doivent etre tous plus loin...
                    # on fait toutes les paires
                    # a bien verifier !!!!
                    for j in range(len(d_quest_val[q])):
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
        return dico_items, triplets, y_true, item_users
    else:
        return dico_items, triplets, y_true

def generate_quad(dico_items, triplets, t_trainy, item_users, d_quest_val):
    l = list(dico_items)
    quadriplets = []
    y = []
    for k in range(len(triplets)):
        t_quadriplets = []
        t_y = []
        for i in range(len(triplets[k])):
            t = triplets[k][i]
            q,r = parse_it(l[t[1]])
            q = int(q)
            # on prend une valeur differente pour l'autre user a eloigner...
            # on prend user sur autre modalite
            q,val = parse_it(l[t[2]])
            q = int(q)
            col_neg = str(q)+'_'+str(val) 
            if(t[2] in item_users):
                u = random.randint(0,len(item_users[t[2]])-1)         
                uu = item_users[t[2]][u]
                t_quadriplets.append([t[0], t[1], t[2], uu, t[3]])
                t_y.append(t_trainy[k][i])
            else:
                t_quadriplets.append([t[0], t[1], t[2], t[0], 1])
                t_y.append(t_trainy[k][i])
        quadriplets.append(np.array(t_quadriplets))  
        y.append(np.array(t_y))
    return quadriplets, y

def init_frequencielle(train, n, p, dim, dico_items, d_quest_val): 
    it = list(dico_items)
    cc_exp_row = np.random.rand(n,dim) 
    cc_exp_col = np.random.rand(p,dim)
    for k in range(len(train)):
        for i in range(len(train[k])):
            user = train[k][i][0]
            item = train[k][i][1]
            q,r = parse_it(it[item])
            q = int(q)
            mv = int(len(d_quest_val[q])/2)
            middle_value = d_quest_val[q][mv]
            if(r > middle_value):
                cc_exp_row[user][k] += int(r) ## 1
                cc_exp_col[item][k] += int(r)
            if(r < middle_value - 1):
                cc_exp_row[user][k] -= int(r)
                cc_exp_col[item][k] -= int(r)
           
    cc_exp_row = normalize(cc_exp_row, axis=1, norm='l1')
    cc_exp_col = normalize(cc_exp_col, axis=1, norm='l1')
    
    return cc_exp_row, cc_exp_col 

def write_file_doa(FileName, embed, train, dico_kc, dico_users, dico_items):
    # write embeddings 
    it = list(dico_items)
    ut = list(dico_users)
    nom = FileName+"_embed.csv"
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
                positive_items = list(set(flattern_array(np.array(positive_items)))) 
                row = row + positive_items
                writer.writerow(row)
            if(train[k][i][0] != previous_user):
                row = [k]+[train[k][i][0]]
                positive_items = []
                previous_user = train[k][i][0]
            positive_items.append(it[train[k][i][1]])
    #remove duplicate positive_items
    positive_items = list(set(flattern_array(np.array(positive_items)))) 
    row = row + positive_items
    writer.writerow(row)
    f.close()

def compute_doa(filename):
    F = fromDFtoArray(filename+"_embed.csv",False,'f')
    kc_user, kc_user_val, dico_u, num_kc = compute_kc_user(filename)
    r = doa_creuse(F, kc_user, kc_user_val, dico_u)
    doa = np.mean(r)
    return doa,r

