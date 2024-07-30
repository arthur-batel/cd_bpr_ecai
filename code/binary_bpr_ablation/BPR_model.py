import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics.cluster import contingency_matrix

dtype = torch.float32
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

class BPRModel(nn.Module):
    def __init__(self, batch_size, num_users, num_items, embedding_size, device, cc_exp_row, cc_exp_col, ablation):
        self.device = device
        self.batch_size = batch_size
        
        super(BPRModel, self).__init__()
        # Model parameters
        self.user_embeddings = nn.Embedding(num_users, embedding_size).to(self.device)
        self.item_embeddings = nn.Embedding(num_items, embedding_size).to(self.device)
        
        # Initialization
        if(ablation != 2) and (ablation != 3):
            self.user_embeddings.weight.data.copy_(torch.from_numpy(cc_exp_row))
            self.item_embeddings.weight.data.copy_(torch.from_numpy(cc_exp_col))
       
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        
    def forward(self, user, item, k): 
        user_embedding = self.user_embeddings(user) 
        item_embedding = self.item_embeddings(item)
        
        if (k == -1):
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
            item_embedding = F.normalize(item_embedding, p=2, dim=1)
            return - torch.norm(user_embedding - item_embedding, p=2, dim=1)
            
        else:
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
            return torch.mean(user_embedding[:,k])

    def bpr_loss(self,positive_scores, negative_scores):
        return -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores))) 
       
    def train(self, triplets, num_kc, epochs, batch_size, y, ablation):
        for epoch in range(epochs):
            all_labels = None
            all_decisions = None
            first = True
            for k in range(num_kc):
                trips = triplets[k]
                y_true = y[k]
                for i in range(0, len(trips), self.batch_size):
                    users_batch = trips[i:i + self.batch_size, 0]
                    items_batch = trips[i:i + self.batch_size, 1]
                    negatives_batch = trips[i:i + self.batch_size, 2]
                    neg_users_batch = trips[i:i + self.batch_size, 3]
                    y_train = y_true[i:i + self.batch_size]
                    
                    # Convert the numpy.ndarray to tensor
                    users_batch = torch.from_numpy(users_batch).to(self.device)
                    items_batch = torch.from_numpy(items_batch).to(self.device)
                    negatives_batch = torch.from_numpy(negatives_batch).to(self.device)
                    neg_users_batch = torch.from_numpy(neg_users_batch).to(self.device)
                                   
                    # call forward
                    positive_scores = self(users_batch, items_batch, -1)
                    negative_scores = self(users_batch, negatives_batch, -1)
                    loss1 = self.bpr_loss(positive_scores, negative_scores)
                    
                    positive_scores_bis = self(users_batch, negatives_batch, k)
                    negative_scores_bis = self(neg_users_batch, negatives_batch, k)
                    loss2 = self.bpr_loss(positive_scores_bis, negative_scores_bis)
                    
                    if(ablation != 1) and (ablation != 3):
                        loss = loss1 + loss2
                    else:
                        loss = loss1

                    if (first == True):
                        all_labels = y_train
                        comp = negative_scores < positive_scores
                        comp = comp.cpu()
                        all_decisions = comp
                        first = False
                    else:
                        all_labels = np.concatenate((all_labels, y_train), axis=0)
                        comp = negative_scores < positive_scores
                        comp = comp.cpu()
                        all_decisions = np.concatenate((all_decisions, comp), axis=0)
                                            
                    # Backward 
                    loss.backward(retain_graph=True)  
                    # Optimizer step
                    self.optimizer.step()
                    # Clear the gradients for the next iteration
                    self.optimizer.zero_grad()

            if(epoch % 10 == 0):
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        acc = accuracy_score(all_labels, all_decisions)
        return acc
       
    # Evaluate the model
    def evaluate_model(self, test_triplets, num_kc, y_test):
        # Initialize counters for metrics
        all_predictions = None
        all_labels = None
        all_decisions = None
        first = True
        for k in range(num_kc):
            trips = test_triplets[k]
            y = y_test[k]
            for i in range(0, len(trips), self.batch_size):
                users_batch = trips[i:i + self.batch_size, 0]
                items_batch = trips[i:i + self.batch_size, 1]
                negatives_batch = trips[i:i + self.batch_size, 2]
                y_true = y[i:i + self.batch_size]
                # Convert the ndarray to tensor
                users_batch = torch.from_numpy(users_batch).to(self.device)
                items_batch = torch.from_numpy(items_batch).to(self.device)
                negatives_batch = torch.from_numpy(negatives_batch).to(self.device)
                # Compute distances
                positive_scores = self(users_batch, items_batch,-1).cpu()
                negative_scores = self(users_batch, negatives_batch,-1).cpu()
                # compute probabilities
                proba = torch.sigmoid(positive_scores - negative_scores ) 
                      
                if (first == True):
                    all_labels = y_true
                    all_predictions = proba.detach().cpu().numpy()
                    comp = negative_scores < positive_scores
                    comp = comp.cpu()
                    all_decisions = comp
                    first = False
                else:
                    all_labels = np.concatenate((all_labels, y_true), axis=0)
                    all_predictions = np.concatenate((all_predictions, proba.detach().cpu().numpy()), axis=0)
                    comp = negative_scores < positive_scores
                    comp = comp.cpu()
                    all_decisions = np.concatenate((all_decisions, comp), axis=0)
                               
        mse1 = mean_squared_error(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_predictions)
        acc = accuracy_score(all_labels, all_decisions)
        precision = precision_score(all_labels, all_decisions)
        recall = recall_score(all_labels, all_decisions)
        f1 = f1_score(all_labels, all_decisions)

        print(f"acc, precision, recall, f1, auc, rmse : {acc},{precision},{recall},{f1},{auc},{np.sqrt(mse1)}")
        return acc, precision, recall,f1, auc,np.sqrt(mse1)

