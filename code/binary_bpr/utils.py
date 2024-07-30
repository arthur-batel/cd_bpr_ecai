import numpy as np
import pandas as pd

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
                                # same item
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
        #print("doa k : ", k, v)
        doa.append(v)  
    return doa
