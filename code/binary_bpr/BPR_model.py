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

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

class BPRModel(nn.Module):
    def __init__(self, batch_size, num_users, num_items, embedding_size, device, cc_exp_row=None, cc_exp_col=None,lr=0.01):
        self.device = device
        self.batch_size = batch_size
        
        super(BPRModel, self).__init__()
        # Model parameters
        self.user_embeddings = nn.Embedding(num_users, embedding_size).to(self.device)
        self.item_embeddings = nn.Embedding(num_items, embedding_size).to(self.device)

        # Initialization
        self.user_embeddings.weight.data.copy_(torch.from_numpy(cc_exp_row))
        self.item_embeddings.weight.data.copy_(torch.from_numpy(cc_exp_col))

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.optimizer = optim.Adam(self.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def forward(self, user, item, k): 
        user_embedding = self.user_embeddings(user) 
        item_embedding = self.item_embeddings(item)
        
        if (k == -1):
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
            item_embedding = F.normalize(item_embedding, p=2, dim=1)
            return - torch.norm(user_embedding - item_embedding, p=2, dim=1)
            
        else:
            return torch.mean(user_embedding[:,k])

    def bpr_loss(self,positive_scores, negative_scores):
        return -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores))) 

    def train(self, triplets, num_kc, epochs, batch_size, y, valid_data=None,valid_target=None,quit_delta=20):

        best_acc = 0
        best_ite=0
        best_metrics = []
        acc=0

        for epoch in range(epochs):
            all_labels = None
            all_decisions = None
            first = True
            for k in range(num_kc):
                trips = triplets[k]
                # Set the mask to zero for specific indices (e.g., user 0)
                masked_indices = [k]
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
                    
                    # Create a mask with the same shape as the embeddings
                    current_batch_size = users_batch.shape[0]
                    
                    # call forward
                    positive_scores = self(users_batch, items_batch, -1)
                    negative_scores = self(users_batch, negatives_batch, -1)
                    loss1 = self.bpr_loss(positive_scores, negative_scores)
                    
                    positive_scores_bis = self(users_batch, negatives_batch, k)
                    negative_scores_bis = self(neg_users_batch, negatives_batch, k)
                    loss2 = self.bpr_loss(positive_scores_bis, negative_scores_bis)
                    
                    loss = loss1 + loss2

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
                        
                                        
                    loss.backward(retain_graph=True)  
                    
                    # Optimizer step
                    self.optimizer.step()
                    # Clear the gradients for the next iteration
                    self.optimizer.zero_grad()
                    
            if valid_data is not None and epoch % 5 == 0:
                correctness, acc, users, auc, rmse = self.evaluate_model(valid_data, num_kc, valid_target)
                if acc > best_acc:
                    best_acc = acc
                    best_ite = epoch
                    best_metrics = [correctness, users, auc, rmse]

                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, loss: %.6f, best_ite: %.6f" % (epoch, auc, acc, rmse, loss.item(), best_ite))

                if epoch - best_ite >= quit_delta:
                    break
            elif epoch % 5 == 0 :
                try :
                    print("[Epoch %d] loss: %.6f" % (epoch, loss.item()))
                except UnboundLocalError as e:
                    print(e)
        return acc
       
    # Evaluate the model
    def evaluate_model(self, test_triplets, num_kc, y_test):
        # Initialize counters for metrics
        precision = 0.0
        nb_evals = 0.0
        all_predictions = None
        all_labels = None
        all_decisions = None
        first = True
        users = []
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
                users.extend(users_batch.tolist())
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
                correct_ranking = sum(negative_scores < positive_scores)  
                nb_evals += len(users_batch)
                # Update metrics
                precision += correct_ranking         
        precision /= nb_evals 
        mse1 = mean_squared_error(all_labels, all_predictions)
        # Compute AUC for the entire dataset
        auc = roc_auc_score(all_labels, all_predictions)
        return (all_labels==all_decisions).astype(int),accuracy_score(all_labels, all_decisions) , users, auc, np.sqrt(mse1) ,precision_score(all_labels, all_decisions),recall_score(all_labels, all_decisions),f1_score(all_labels, all_decisions)
