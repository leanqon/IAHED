import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
import torch as T
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from imblearn.over_sampling import SMOTE
from torch.utils.data import Subset, DataLoader
import math
from sklearn import metrics
import torch.nn as nn
from torch import optim
import importlib
import torch.nn.functional as F
from torch.nn.functional import pad
import import_ipynb
import model_utils
import evaluation
import parameters
from parameters import *
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pickle import dump,load
from sklearn.model_selection import train_test_split
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution,LayerDeepLift,DeepLift
import argparse
from torch.autograd import Variable
from argparse import ArgumentParser
import matplotlib.pyplot as plt
importlib.reload(model_utils)
import model_utils
import mimic_model_copy4 as model
importlib.reload(parameters)
import parameters
from parameters import *
importlib.reload(evaluation)
import evaluation
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import IncrementalPCA
from imblearn.pipeline import Pipeline
import logging
from hmmlearn import hmm
from filterpy.monte_carlo import systematic_resample
import h5py
import torch.utils.data as data
from torch.utils.data import random_split
from encoder import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class ML_models():
    def __init__(self,model_type,data_name):
        self.model_type=model_type
        self.data_name=data_name
        dir = f"./data/ml_data/data{self.data_name}/{args.strategy}"
        os.makedirs(dir, exist_ok=True)
        if not os.path.exists(f"{dir}/train_data.npy"):
            if os.path.exists(f'./data/sparse/data{self.data_name}/{args.strategy}/dataset_{args.stride}.pt'):
                dataset=torch.load(f'./data/sparse/data{self.data_name}/{args.strategy}/dataset_{args.stride}.pt')
                self.train_loader, self.val_loader, self.test_loader, self.train_val_loader, self.train_val_subset, self.train_val_idx = DL_models.dataset_loader(self, dataset) 
                self.ml_train(model_type, True)
            else:
                return
        else: 
            self.ml_train(model_type, False)
        
        logging.basicConfig(filename='./train_records/ml_model_results.log', level=logging.INFO, filemode='a')

    def get_data_from_loader(self, loader):
        features_list = []
        labels_list = []

        for seqs in loader:
            features = torch.cat([seq.contiguous().view(seq.size(0), -1) for seq in seqs[0:7]], dim=-1)
            labels = seqs[-1].contiguous().view(-1)
            features_list.append(features)
            labels_list.append(labels)

        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        return all_features.cpu().numpy(), all_labels.cpu().numpy()
        
    def sample_data(self, features, labels, sample_ratio=0.1):
        data_size = len(features)
        sample_size = int(data_size * sample_ratio)

        indices = np.random.choice(data_size, sample_size, replace=False)
        sampled_features = features[indices]
        sampled_labels = labels[indices]

        return sampled_features, sampled_labels

    def ml_train(self, model_type, data_unprocessed):
        if data_unprocessed:
            train_data, y_train = self.get_data_from_loader(self.train_val_loader)
            test_data, y_test = self.get_data_from_loader(self.test_loader)
            np.save(f"./data/ml_data/data{self.data_name}/{args.strategy}/train_data.npy", train_data)
            np.save(f"./data/ml_data/data{self.data_name}/{args.strategy}/y_train.npy", y_train)
            np.save(f"./data/ml_data/data{self.data_name}/{args.strategy}/test_data.npy", test_data)
            np.save(f"./data/ml_data/data{self.data_name}/{args.strategy}/y_test.npy", y_test)
    
        else:
            train_data=np.load(f"./data/ml_data/data{self.data_name}/{args.strategy}/train_data.npy")
            y_train=np.load(f"./data/ml_data/data{self.data_name}/{args.strategy}/y_train.npy")
            test_data=np.load(f"./data/ml_data/data{self.data_name}/{args.strategy}/test_data.npy")
            y_test=np.load(f"./data/ml_data/data{self.data_name}/{args.strategy}/y_test.npy")
            
        if model_type == 'Random Forest':
            print("===================Random Forest=====================")
            train_data, y_train = self.sample_data(train_data, y_train, sample_ratio=1)
            test_data, y_test = self.sample_data(test_data, y_test, sample_ratio=1)
            model = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=42).fit(train_data, y_train)
            logits = model.predict_log_proba(test_data)
            prob = model.predict_proba(test_data)

        elif model_type == 'Logistic Regression':
            print("===================Logistic Regression=====================")
            train_data, y_train = self.sample_data(train_data, y_train, sample_ratio=1)
            test_data, y_test = self.sample_data(test_data, y_test, sample_ratio=1)
            model = LogisticRegression(C=0.01, penalty='none',max_iter=100,class_weight='balanced').fit(train_data, y_train) #0.001
            logits = model.predict_log_proba(test_data)
            prob = model.predict_proba(test_data)

        elif model_type == 'Xgboost':
            print("===================Xgboost=====================")
            train_data, y_train = self.sample_data(train_data, y_train, sample_ratio=1)
            test_data, y_test = self.sample_data(test_data, y_test, sample_ratio=1)
            model = xgb.XGBClassifier(learning_rate=0.01,n_estimators=6,max_depth=50, min_child_weight=8, gamma=0.5, colsample_bytree=0.01, objective="binary:logistic").fit(train_data, y_train)
            prob = model.predict_proba(test_data)
            logits = np.log2(prob[:, 1] / prob[:, 0])

        elif model_type == 'Gradient Boosting':
            print("===================Gradient Boosting=====================")
            train_data, y_train = self.sample_data(train_data, y_train, sample_ratio=1)
            test_data, y_test = self.sample_data(test_data, y_test, sample_ratio=1)
            model = HistGradientBoostingClassifier(max_iter=5,max_leaf_nodes=8,max_depth=10, min_samples_leaf=80, max_bins=3).fit(train_data, y_train)
            prob = model.predict_proba(test_data)
            logits = np.log2(prob[:, 1] / prob[:, 0])

        all_predictions = np.argmax(prob, axis=1)
        all_probabilities = prob[:, 1]

        tra_accuracy = accuracy_score(y_test, all_predictions)
        tra_precision = precision_score(y_test, all_predictions)
        tra_recall = recall_score(y_test, all_predictions)
        tra_f1 = f1_score(y_test, all_predictions)
        tra_roc = roc_auc_score(y_test, all_probabilities)
        tra_prc = average_precision_score(y_test, all_probabilities)

        print(f"Accuracy: {tra_accuracy:.4f}, Precision: {tra_precision:.4f}, Recall: {tra_recall:.4f}, F1 Score: {tra_f1:.4f}, AUC-ROC: {tra_roc:.4f}, AUC-PR: {tra_prc:.4f}")
        log_message = f"Model: {model_type}, Accuracy: {tra_accuracy:.4f}, Precision: {tra_precision:.4f}, Recall: {tra_recall:.4f}, F1 Score: {tra_f1:.4f}, AUC-ROC: {tra_roc:.4f}, AUC-PR: {tra_prc:.4f}"
        logging.info(log_message)

        # Save true labels and predicted probabilities
        save_dir=f"./images/data{self.data_name}/ml_model"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/{model_type}_{args.strategy}_true_labels.npy", y_test)
        np.save(f"{save_dir}/{model_type}_{args.strategy}_predicted_probabilities.npy", all_probabilities)

def calculate_positive_negative_ratio(loader):
        positive_count = 0
        negative_count = 0
        for data in loader:
            labels = data[-1].squeeze(-1)
            positive_count += (labels == 1).sum().item()
            negative_count += (labels == 0).sum().item()
        total_count = positive_count + negative_count
        return positive_count / total_count, negative_count / total_count

def save_load_data(train_loader, val_loader, test_loader, save=True):
    save_dir = f"./data/sparse/data{args.data_name}/{args.strategy}"
    if save:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(train_loader, f"{save_dir}/train_loader_{args.batch_size}.pt")
        torch.save(val_loader, f"{save_dir}/val_loader_{args.batch_size}.pt")
        torch.save(test_loader, f"{save_dir}/test_loader_{args.batch_size}.pt")
        print("Data loaders saved successfully.")
    else:
        train_loader = torch.load(f"{save_dir}/train_loader_{args.batch_size}.pt")
        val_loader = torch.load(f"{save_dir}/val_loader_{args.batch_size}.pt")
        test_loader = torch.load(f"{save_dir}/test_loader_{args.batch_size}.pt")
        print("Data loaders loaded successfully.")
    return train_loader, val_loader, test_loader

class TimeSeriesDataset(Dataset):
    def __init__(self, sequence1, sequence2, sequence3, sequence4, sequence5, sequence6, sequence7, labels):
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.sequence3 = sequence3
        self.sequence4 = sequence4
        self.sequence5 = sequence5
        self.sequence6 = sequence6
        self.sequence7 = sequence7
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq1 = self.sequence1[idx]
        seq2 = self.sequence2[idx]
        seq3 = self.sequence3[idx]
        seq4 = self.sequence4[idx]
        seq5 = self.sequence5[idx]
        seq6 = self.sequence6[idx]
        seq7 = self.sequence7[idx]
        label = self.labels[idx]

        seq1_tensor = torch.tensor(seq1, dtype=torch.float32)
        seq2_tensor = torch.tensor(seq2, dtype=torch.int64)
        seq3_tensor = torch.tensor(seq3, dtype=torch.int64)
        seq4_tensor = torch.tensor(seq4, dtype=torch.int64)
        seq5_tensor = torch.tensor(seq5, dtype=torch.int64)
        seq6_tensor = torch.tensor(seq6, dtype=torch.int64)
        seq7_tensor = torch.tensor(seq7, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.int64)

        return seq1_tensor, seq2_tensor, seq3_tensor, seq4_tensor, seq5_tensor, seq6_tensor, seq7_tensor, label_tensor

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self, filename)
        labels_filename = os.path.join(os.path.dirname(filename), f'labels_{args.stride}.npy')
        np.save(labels_filename, self.labels)

class TorchFileDataset(Dataset):
    def __init__(self, filename, labels_filename):
        self.dataset = torch.load(filename)
        if os.path.exists(labels_filename):
            self.labels = np.load(labels_filename)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class IAHEDAutoencoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10):
        super().__init__()
        self.encoder = TSEncoder(input_dims, output_dims, hidden_dims, depth)
        decoder_dims = 6 * output_dims
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(decoder_dims, decoder_dims//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(decoder_dims//2, input_dims, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded.transpose(1, 2))
        return encoded, decoded.transpose(1, 2)

class AutoencoderTrainer:
    def __init__(self, model, dataset, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.ae_criterion = nn.MSELoss()
        self.dataset = dataset

    def train(self, label_num, train_val_dataset, train_val_idx): #size_num,
        labels = np.load('./data/sparse/data{}/{}/labels_{}.npy'.format(args.data_name,args.strategy,args.stride))
        indices = np.where(labels == label_num)[0]
        train_val_idx = set(train_val_idx)
        indices = [i for i in indices if i in train_val_idx]
        train_val_dataset = Subset(self.dataset, indices)
        train_ratio=0.8
        num_epochs_ae = args.num_epochs_ae
        train_size = int(train_ratio * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)#, collate_fn=self.dataset.my_collate_fn, pin_memory=True
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)#, collate_fn=self.dataset.my_collate_fn, pin_memory=True

        best_loss = float('inf')
        for epoch in range(num_epochs_ae):
            for data in train_loader:
                seqs = tuple(d.to(self.device) for d in data[:3])
                dynamic_x = torch.cat (seqs,dim=2)
                _, decoded = self.model(dynamic_x)
                self.optimizer.zero_grad()
                loss = self.ae_criterion(decoded, dynamic_x)
                loss.backward()
                self.optimizer.step()
            total_val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    seqs = tuple(d.to(self.device) for d in data[:3])
                    dynamic_x = torch.cat (seqs,dim=2)
                    _, decoded = self.model(dynamic_x) 
                    loss = self.ae_criterion(decoded, dynamic_x)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs_ae}], Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_weights = self.model.state_dict()
        if label_num==0:
            torch.save(best_weights, "./data/sparse/data{}/{}/majority_best_weights_{}_{}_{}_{}.pth".format(args.data_name, args.strategy, args.batch_size, args.common_dim, args.num_epochs_ae, args.output_dim))
        else:   
            torch.save(best_weights, "./data/sparse/data{}/{}/minority_best_weights_{}_{}_{}_{}.pth".format(args.data_name, args.strategy, args.batch_size, args.common_dim, args.num_epochs_ae, args.output_dim))

class DL_models():
    def __init__(self,data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag,anti_flag,vent_flag,model_type,k_fold,data_name,sampling_first,undersampling,model_name,train,save_data=False,pre_train=False,train_test=False,test=False):
        self.save_path="saved_models/"+model_name+".tar"
        self.data_icu=data_icu
        self.diag_flag,self.proc_flag,self.out_flag,self.chart_flag,self.med_flag,self.anti_flag,self.vent_flag,self.lab_flag=diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag,anti_flag,vent_flag
        self.modalities=self.diag_flag+self.proc_flag+self.out_flag+self.chart_flag+self.med_flag+self.lab_flag+self.anti_flag+self.vent_flag
        self.k_fold=k_fold
        self.model_type=model_type
        self.sampling_first=sampling_first
        self.undersampling=undersampling
        self.data_name=data_name
        if torch.cuda.is_available():
            self.device='cuda:0'
        else:
            self.device='cpu'
        
        self.cond_vocab_size,self.proc_vocab_size,self.med_vocab_size,self.out_vocab_size,self.chart_vocab_size,self.lab_vocab_size,self.anti_vocab_size,self.vent_vocab_size,self.eth_vocab_size,self.gender_vocab_size,self.ins_vocab_size,self.cond_vocab,self.proc_vocab,self.med_vocab,self.out_vocab,self.chart_vocab,self.lab_vocab,self.anti_vocab,self.vent_vocab,self.eth_vocab,self.gender_vocab,self.age_vocab,self.ins_vocab=model_utils.init_read(data_name,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag,anti_flag,vent_flag)
        self.dynamic_dim = self.chart_vocab_size + self.anti_vocab_size + self.vent_vocab_size
        self.age_vocab_size = 1
        self.static_dim = self.cond_vocab_size + self.eth_vocab_size +self.gender_vocab_size+ self.age_vocab_size
        self.input_dim = self.dynamic_dim + self.static_dim
        self.loss=evaluation.Loss(self.device,True,True,True,True,True,True,True,True,True,True,True)
        if save_data:
            start = time.time()
            labels=pd.read_csv(f'./data/csv{self.data_name}/labels.csv', header=0)
            hids=labels.iloc[:,0] 
            meds,chart,out,proc,lab,anti,vent,stat,demo,Y=self.getXY_all2(hids,device=self.device)
            end = time.time()
            print(f"the running time is: {end - start} s")
            hids_all=hids.to_numpy()
            hids_all=torch.LongTensor(hids_all).to(self.device)
            if not os.path.exists(f"./data/tensor{self.data_name}"):
                os.makedirs(f"./data/tensor{self.data_name}")
            if len(vent)!=0:
                torch.save(vent.to(torch.device('cpu')), f"./data/tensor{self.data_name}/vent.pth")
            else:
                print("No Vent Data")
            if len(anti)!=0:
                torch.save(anti.to(torch.device('cpu')), f"./data/tensor{self.data_name}/anti.pth")
            else:
                print("No Anti Data")
            if len(lab)!=0:
                torch.save(lab.to(torch.device('cpu')), f"./data/tensor{self.data_name}/lab.pth")
            else:
                print("No Lab Data")
            if len(meds)!=0:
                torch.save(meds.to(torch.device('cpu')), f"./data/tensor{self.data_name}/meds.pth")
            else:
                print("No Meds Data")
            if len(chart)!=0:
                torch.save(chart.to(torch.device('cpu')), f"./data/tensor{self.data_name}/chart.pth")
            else:
                print("No Chart Data")
            if len(out)!=0:
                torch.save(out.to(torch.device('cpu')), f"./data/tensor{self.data_name}/out.pth")
            else:
                print("No Out Data")
            if len(proc)!=0:
                torch.save(proc.to(torch.device('cpu')), f"./data/tensor{self.data_name}/proc.pth")
            else:
                print("No Proc Data")
            if len(stat)!=0:
                torch.save(stat.to(torch.device('cpu')), f"./data/tensor{self.data_name}/stat.pth")
            else:
                print("No Stat Data")
            if len(demo)!=0:
                torch.save(demo.to(torch.device('cpu')), f"./data/tensor{self.data_name}/demo.pth")
            else:
                print("No Demo Data")
            if len(Y)!=0:
                torch.save(Y.to(torch.device('cpu')), f"./data/tensor{self.data_name}/Y.pth")
            else:
                print("No Y Data")
        else:
            print("=================================")
        if os.path.exists(f"./data/sparse/data{self.data_name}/{args.strategy}/dataset_{args.stride}.pt"):
            dataset = TorchFileDataset(f'./data/sparse/data{self.data_name}/{args.strategy}/dataset_{args.stride}.pt',f'./data/sparse/data{self.data_name}/{args.strategy}/labels_{args.stride}.npy')
        else: 
            dataset = self.build(binning=False, data_save=True)
            
        train_loader, val_loader, test_loader, train_val_loader, train_val_subset, train_val_idx = self.dataset_loader(dataset)
         
        if pre_train:
            pre_model = TSEncoderAutoencoder(self.dynamic_dim, args.output_dim).to(self.device)
            trainer = AutoencoderTrainer(pre_model, dataset, device = self.device)
            trainer.train(1, train_val_subset, train_val_idx) 
            trainer.train(0, train_val_subset, train_val_idx) 

        if train_test:
            self.train(train_loader, val_loader)
            self.test(test_loader)

        if test:
            self.test(test_loader)
       
    def dataset_loader(self, dataset):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(sss.split(np.zeros(len(dataset)), dataset.labels.squeeze()))
        filename = "./data/sparse/data{}/{}/train_val_idx_{}_{}_{}_{}{}.pkl".format(self.data_name, args.strategy, args.batch_size, args.common_dim, args.num_epochs_ae, args.output_dim,args.updated)
        if not os.path.exists(filename):
            with open(filename, 'wb') as file:
                pickle.dump(train_val_idx, file)
        train_val_subset = Subset(dataset, train_val_idx)
        test_subset = Subset(dataset, test_idx)
        sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, val_index in sss_inner.split(np.zeros(len(train_val_subset)), np.array(dataset.labels.squeeze())[train_val_idx]): 
            train_index_resample, train_index_remain = train_test_split(train_index, test_size=0.1, random_state=42)
            train_labels_resample = [dataset.labels.squeeze()[i] for i in train_index_resample]
            over = SMOTE(sampling_strategy=args.sample_over)
            under = RandomUnderSampler(sampling_strategy=args.sample_over)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            train_index_oversampled, train_labels_oversampled = pipeline.fit_resample(np.array(train_index_resample).reshape(-1, 1), train_labels_resample)
            train_index_oversampled = train_index_oversampled.squeeze()
            train_index_remain = np.array(train_index_remain)
            train_labels_remain = np.array([dataset.labels.squeeze()[i] for i in train_index_remain])
            train_index = np.concatenate((train_index_oversampled, train_index_remain), axis=0)
            train_labels = np.concatenate((train_labels_oversampled, train_labels_remain), axis=0)
            train_subset = Subset(dataset, train_index)
            val_subset = Subset(dataset, val_index)
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
            test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
            train_val_loader = DataLoader(train_val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
            train_positive_negative_ratio = calculate_positive_negative_ratio(train_loader)
            test_positive_negative_ratio = calculate_positive_negative_ratio(test_loader)
            print(f"Train Positive Negative Ratio: {train_positive_negative_ratio}, Test Positive Negative Ratio: {test_positive_negative_ratio}") 
        return train_loader, val_loader, test_loader, train_val_loader, train_val_subset, train_val_idx 

    def train(self, train_loader, val_loader):
        self.create_model(self.model_type)
        best_f1 = -0.1
        patience = 20
        stagnant_epochs = 0
        
        best_metric = -0.1
        improvement_threshold = 0.01  
        patience_increase = 5  
        max_patience = 15  
        dynamic_patience = 15 
        for epoch in range(args.num_epochs):
            all_predictions = []
            all_labels = []
            all_probabilities = []
            total_loss = 0.0
            self.net.train()
            for data in train_loader:
                seqs = data[:-1]
                labels = data[-1].to(self.device).squeeze(-1)
                loss, outputs = self.train_model(seqs,labels)
                total_loss += loss.item()
                predictions = (outputs > 0.1).float()#0.1
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs.detach().cpu().numpy())
                
            tra_accuracy = accuracy_score(all_labels, all_predictions)
            tra_precision = precision_score(all_labels, all_predictions)
            tra_recall = recall_score(all_labels, all_predictions)
            tra_f1 = f1_score(all_labels, all_predictions)
            tra_roc = roc_auc_score(all_labels, all_probabilities)
            tra_prc = average_precision_score(all_labels, all_probabilities) 
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {tra_accuracy:.4f}, Precision: {tra_precision:.4f}, Recall: {tra_recall:.4f}, F1 Score: {tra_f1:.4f}, AUC-ROC: {tra_roc:.4f}, AUC-PR: {tra_prc:.4f}")
            val_metrics = self.evaluate_model(self.net, val_loader, self.loss, is_plot=False)
            print(f"Validation {', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")
            self.scheduler.step(val_metrics['Loss'])
            save_dir = f'./data/sparse/data{self.data_name}/{args.strategy}'
            os.makedirs(save_dir, exist_ok=True)
            current_metric = val_metrics['F1 Score'] 
            
            if current_metric > best_metric + improvement_threshold:
                best_metric = current_metric
                torch.save(self.net.state_dict(), f"{save_dir}/best_model_{self.model_type}_{args.batch_size}_{args.use_pretrained}.pth")
                stagnant_epochs = 0
                dynamic_patience = min(dynamic_patience + patience_increase, max_patience)
            else:
                stagnant_epochs += 1
                if stagnant_epochs >= dynamic_patience:
                    print("Early stopping triggered.")
                    break
            
    def train_model(self, seqs, labels):
        self.optimizer.zero_grad()
        outputs, contrastive_loss, logits = self.net(seqs,labels)
        total_loss=self.loss(outputs,labels,logits,contrastive_loss,True,False,contrastive_weight=args.contrastive_weight)
        total_loss.backward()
        self.optimizer.step()
        return total_loss, outputs

    def test(self, test_loader):
        self.create_model(self.model_type) 
        save_dir = f'./data/sparse/data{self.data_name}/{args.strategy}'
        self.net.load_state_dict(torch.load(f"{save_dir}/best_model_{self.model_type}_{args.batch_size}_{args.use_pretrained}.pth"))
        test_metrics = self.evaluate_model(self.net, test_loader, self.loss, is_plot=True)
        print(f"Test {', '.join([f'{k}: {v:.4f}' for k, v in test_metrics.items()])}")

    def evaluate_model(self, model, data_loader, criterion, is_plot=False):
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = [] 
        
        with torch.no_grad():
            for data in data_loader:
                seqs = data[:-1]
                labels = data[-1].to(self.device).squeeze(-1)
                outputs, contrastive_loss, logits = model(seqs,labels)
                loss = criterion(outputs, labels, logits, contrastive_loss, True, False, contrastive_weight=args.contrastive_weight)
                total_loss += loss.item()
                
                predictions = (outputs > 0.1).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())

        if is_plot:
            predicted_save_dir = f'./images/data{self.data_name}'
            os.makedirs(predicted_save_dir, exist_ok=True)
            np.save(f'{predicted_save_dir}/{self.model_type}_{args.strategy}_{args.batch_size}_{args.use_pretrained}_true_labels.npy', np.array(all_labels))
            np.save(f'{predicted_save_dir}/{self.model_type}_{args.strategy}_{args.batch_size}_{args.use_pretrained}_predicted_probabilities.npy', np.array(all_probabilities))

            fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            dir = f'./images/data{args.data_name}/{args.strategy}'
            os.makedirs(dir, exist_ok=True)
            plt.savefig(f'{dir}/roc_curve{self.data_name}{self.model_type}_{args.use_pretrained}.png')  
            plt.show()

            precision, recall, _ = precision_recall_curve(all_labels, all_probabilities)

            plt.figure()
            plt.plot(recall, precision, color='darkorange', lw=2, label='AP = %0.2f' % average_precision_score(all_labels, all_probabilities))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.legend(loc="lower right")
            plt.savefig(f'{dir}/precision_recall_curve_givens{self.data_name}{self.model_type}_{args.use_pretrained}.png')  
            plt.show()

        metrics = {}
        metrics['Loss'] = total_loss / len(data_loader)
        metrics['Accuracy'] = accuracy_score(all_labels, all_predictions)
        metrics['Precision'] = precision_score(all_labels, all_predictions)
        metrics['Recall'] = recall_score(all_labels, all_predictions)
        metrics['F1 Score'] = f1_score(all_labels, all_predictions)
        metrics['AUC-ROC'] = roc_auc_score(all_labels, all_probabilities)
        metrics['AUC-PR'] = average_precision_score(all_labels, all_probabilities)
        
        return metrics
    
    def build(self, binning, data_save):
        chart_all,anti_all,vent_all,stat_all,demo_all,y_all=self.loading_data_all(False,True,False,False,False,True,True,True,True,True,device='cpu') 
        chart,anti,vent,stat,demo,mdr_label,y=self.Add_Window_Horizon(chart_all, anti_all, vent_all, stat_all, demo_all, y_all, args.window, args.horizon, args.stride)
        del chart_all,anti_all,vent_all,stat_all,demo_all,y_all
        chart,anti,vent,stat,demo,mdr_label,y=self.datamerge(chart, anti, vent, stat, demo, mdr_label, y, device='cpu')
        gender=demo[:,:,0].unsqueeze(2)
        eth=demo[:,:,1].unsqueeze(2)
        age=demo[:,:,3].unsqueeze(2)
        gender=self.restore_tensor(gender,self.gender_vocab)
        eth=self.restore_tensor(eth,self.eth_vocab)   
    
        if binning:
            tensor_discretizer = TensorDiscretizer(chart,self.chart_vocab,n_clusters=4,method='binning')
            new_chart, new_chart_vocab = tensor_discretizer.transform()
            age_vocab=['age']
            tensor_discretizer2 = TensorDiscretizer(age,age_vocab,n_clusters=4,method='binning')
            new_age, new_age_vocab = tensor_discretizer2.transform() 
        else:
            chart=chart.float()
            chart_mean = torch.mean(chart, dim=(0, 1), keepdim=True)
            chart_std = torch.std(chart, dim=(0, 1), keepdim=True)
            new_chart = (chart - chart_mean) / chart_std
            age=age.float()
            age_mean = torch.mean(age,dim=(0,1),keepdim=True)
            age_std = torch.std(age,dim=(0,1),keepdim=True)
            new_age = (age - age_mean) / age_std

        new_chart, new_anti, new_vent = self.dataclip(new_chart, anti, vent, mdr_label, strategy=args.strategy, threshold=12, fill_value=0)
        dataset = TimeSeriesDataset(new_chart, new_anti, new_vent, stat, gender, eth, new_age, y)
        if data_save:
            dataset.save(f'./data/sparse/data{self.data_name}/{args.strategy}/dataset_{args.stride}.pt')
        else:
            pass
        return dataset
    
    def dataclip(self, chart, anti, vent, mdr_label, strategy, threshold=12, fill_value=0):
        new_chart, new_anti, new_vent, new_mdr_label = [], [], [], []
        max_length = chart.shape[1]

        for i in range(chart.shape[0]):
            mdrb_occurrences = (mdr_label[i, :, 0] == 1).nonzero(as_tuple=True)[0]
            first_occurrence = mdrb_occurrences[0].item() if len(mdrb_occurrences) > 0 else chart.shape[1]

            if strategy == 'truncate':
                cutoff = min(first_occurrence, max_length)
            elif strategy == 'fill':
                cutoff = max_length
                chart[i, first_occurrence:, :] = fill_value
                anti[i, first_occurrence:, :] = fill_value
                vent[i, first_occurrence:, :] = fill_value
            elif strategy == 'threshold':
                if first_occurrence < threshold:
                    cutoff = first_occurrence
                else:
                    cutoff = chart.shape[1]
                    chart[i, first_occurrence:, :] = fill_value
                    anti[i, first_occurrence:, :] = fill_value
                    vent[i, first_occurrence:, :] = fill_value
            else:
                raise ValueError("Invalid strategy specified")
            
            padded_chart = pad(chart[i, :cutoff, :], (0, 0, 0, max_length - cutoff), value=fill_value)
            padded_anti = pad(anti[i, :cutoff, :], (0, 0, 0, max_length - cutoff), value=fill_value)
            padded_vent = pad(vent[i, :cutoff, :], (0, 0, 0, max_length - cutoff), value=fill_value)

            new_chart.append(padded_chart)
            new_anti.append(padded_anti)
            new_vent.append(padded_vent)

        new_chart = torch.stack(new_chart)
        new_anti = torch.stack(new_anti)
        new_vent = torch.stack(new_vent)

        return new_chart, new_anti, new_vent
 
    def create_model(self,model_type):
        if model_type=='lstm':
            self.net = model.LSTMModel(self.device, input_dim=self.input_dim, hidden_dim=args.common_dim).to('cuda:0')

        if model_type=='gru':
            self.net = model.GRUModel(self.device, input_dim=self.input_dim, hidden_dim=args.common_dim).to('cuda:0')

        if model_type=='cnn':
            self.net = model.CNNModel(self.device, num_channels=self.input_dim, num_filters=64, kernel_size=3, stride=1).to('cuda:0')

        if model_type=='main_model':
            self.net = model.MainModel(self.device, self.dynamic_dim, self.static_dim, common_dim=args.common_dim, hidden_dim=args.output_dim, use_pretrained=args.use_pretrained)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrn_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5)
        self.net.to(self.device)
    
    def save_output(self):
        reversed_eth = {self.eth_vocab[key]: key for key in self.eth_vocab}
        reversed_gender = {self.gender_vocab[key]: key for key in self.gender_vocab}
        reversed_age = {self.age_vocab[key]: key for key in self.age_vocab}
        reversed_ins = {self.ins_vocab[key]: key for key in self.ins_vocab}

        self.eth=list(pd.Series(self.eth).map(reversed_eth))
        self.gender=list(pd.Series(self.gender).map(reversed_gender))
        self.age=list(pd.Series(self.age).map(reversed_age))
        self.ins=list(pd.Series(self.ins).map(reversed_ins))

        output_df=pd.DataFrame()
        output_df['Labels']=self.truth
        output_df['Prob']=self.prob
        output_df['Logits']=self.logits
        output_df['ethnicity']=self.eth
        output_df['gender']=self.gender
        output_df['age']=self.age
        output_df['insurance']=self.ins
        with open(f'./data/output{self.data_name}/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)  

    def getXY_all2(self,ids,device):
        dyn_df=[]
        stat_df=[]
        demo_df=[]
        meds=torch.zeros(size=(0,0))
        chart=torch.zeros(size=(0,0))
        proc=torch.zeros(size=(0,0))
        out=torch.zeros(size=(0,0))
        lab=torch.zeros(size=(0,0))
        anti=torch.zeros(size=(0,0))
        vent=torch.zeros(size=(0,0))
        stat_df=torch.zeros(size=(1,0))
        demo_df=torch.zeros(size=(1,0))
        y_df=torch.zeros(size=(1,0))
        dyn=pd.read_csv(f'./data/csv{self.data_name}/'+str(ids[0])+'/dynamic.csv',header=[0,1])
        keys=dyn.columns.levels[0]
        for i in range(len(keys)):
            dyn_df.append(torch.zeros(size=(1,0)))
        for sample in ids:
            mdr=pd.read_csv(f'./data/csv{self.data_name}/'+str(sample)+'/mdr.csv')
            y=mdr.to_numpy()
            y=torch.tensor(y)
            y=y.unsqueeze(0)
            if y_df.nelement():
                y_df=torch.cat((y_df,y),0)
            else:
                y_df=y

            for key in range(len(keys)):
                dyn=pd.read_csv(f'./data/csv{self.data_name}/'+str(sample)+'/dynamic.csv',header=[0,1])
                dyn=dyn[keys[key]]
                dyn=dyn.to_numpy()
                dyn=torch.tensor(dyn)
                dyn=dyn.unsqueeze(0)
                dyn=torch.tensor(dyn)
                dyn=dyn.type(torch.LongTensor)
                
                if dyn_df[key].nelement():
                    dyn_df[key]=torch.cat((dyn_df[key],dyn),0)
                else:
                    dyn_df[key]=dyn
            
            stat=pd.read_csv(f'./data/csv{self.data_name}/'+str(sample)+'/static.csv',header=[0,1])
            stat=stat['COND']
            stat=stat.to_numpy()
            stat=torch.tensor(stat)

            if stat_df.nelement():
                stat_df=torch.cat((stat_df,stat),0)
            else:
                stat_df=stat
                
            demo=pd.read_csv(f'./data/csv{self.data_name}/'+str(sample)+'/demo.csv',header=0)
            demo["gender"].replace(self.gender_vocab, inplace=True)
            demo["ethnicity"].replace(self.eth_vocab, inplace=True)
            demo["insurance"].replace(self.ins_vocab, inplace=True)
            demo["Age"].replace(self.age_vocab, inplace=True)
            demo=demo[["gender","ethnicity","insurance","Age"]]
            demo=demo.values
            demo=torch.tensor(demo)

            if demo_df.nelement():
                demo_df=torch.cat((demo_df,demo),0)
            else:
                demo_df=demo
        
        for k in range(len(keys)):
            if keys[k]=='MEDS':
                meds=dyn_df[k]
            if keys[k]=='CHART':
                chart=dyn_df[k]
            if keys[k]=='OUT':
                out=dyn_df[k]
            if keys[k]=='PROC':
                proc=dyn_df[k]
            if keys[k]=='LAB':
                lab=dyn_df[k]
            if keys[k]=='ANTI':
                anti=dyn_df[k]
            if keys[k]=='VENT':
                vent=dyn_df[k]
            
        stat_df=torch.tensor(stat_df)
        stat_df=stat_df.type(torch.LongTensor).to(device)
        
        demo_df=torch.tensor(demo_df)
        demo_df=demo_df.type(torch.LongTensor).to(device)
        
        y_df=torch.tensor(y_df)
        y_df=y_df.type(torch.LongTensor).to(device)
        return meds,chart,out,proc,lab,anti,vent,stat_df, demo_df, y_df 

    def loading_data_all(self,med_all,chart_all,out_all,proc_all,lab_all,anti_all,vent_all,stat_all,demo_all,Y_all,device):
        if med_all:
            med_all=torch.load(f"./data/tensor{self.data_name}/meds.pth")
            med=torch.tensor(med_all).to(device)
        else:
            med=torch.zeros(size=(0,0)).to(device)
        if chart_all:
            chart_all = torch.load(f"./data/tensor{self.data_name}/chart.pth")
            chart=torch.tensor(chart_all).to(device)
        else:
            chart=torch.zeros(size=(0,0)).to(device)
        if out_all:
            out_all = torch.load(f"./data/tensor{self.data_name}/out.pth")
            out=torch.tensor(out_all).to(device)
        else:
            out=torch.zeros(size=(0,0)).to(device)
        if proc_all:
            proc_all = torch.load(f"./data/tensor{self.data_name}/proc.pth")
            proc=torch.tensor(proc_all).to(device)
        else:
            proc=torch.zeros(size=(0,0)).to(device)
        if lab_all:
            lab_all = torch.load(f"./data/tensor{self.data_name}/lab.pth")
            lab=torch.tensor(lab_all).to(device)
        else:
            lab=torch.zeros(size=(0,0)).to(device)
        if anti_all:
            anti_all = torch.load(f"./data/tensor{self.data_name}/anti.pth")
            anti=torch.tensor(anti_all).to(device)
        else:
            anti=torch.zeros(size=(0,0)).to(device)
        if vent_all:
            vent_all = torch.load(f"./data/tensor{self.data_name}/vent.pth")
            vent=torch.tensor(vent_all).to(device)
        else:
            vent=torch.zeros(size=(0,0)).to(device)
        if stat_all:
            stat_all=torch.load(f"./data/tensor{self.data_name}/stat.pth")
            stat=torch.tensor(stat_all).to(device)
        else:
            stat=torch.zeros(size=(0,0)).to(device)
        if demo_all:
            demo_all=torch.load(f"./data/tensor{self.data_name}/demo.pth")
            demo=torch.tensor(demo_all).to(device)
        else:
            demo=torch.zeros(size=(0,0)).to(device)
        if Y_all:
            Y_all=torch.load(f"./data/tensor{self.data_name}/Y.pth")
            y=torch.tensor(Y_all).to(device)
        else:
            y=torch.zeros(size=(0,0)).to(device)
        stat=stat.unsqueeze(1).expand(-1,chart.shape[1],-1)
        demo=demo.unsqueeze(1).expand(-1,chart.shape[1],-1)
        return chart,anti,vent,stat,demo,y

    def Add_Window_Horizon(self,data1,data2,data3,data4,data5,data_y,window,horizon,stride):
        length=data1.shape[1]
        end_index = length - horizon - window + 1
        X_chart = []      #windows
        X_anti = []
        X_vent = []
        X_stat = []
        X_demo = []
        label = []
        Y = []      #horizon
        index = 0
        while index < end_index:
            X_chart.append(data1[:,index:index+window,:])
            X_anti.append(data2[:,index:index+window,:])
            X_vent.append(data3[:,index:index+window,:])
            X_stat.append(data4[:,index:index+window,:])
            X_demo.append(data5[:,index:index+window,:])
            label.append(data_y[:,index:index+window,:])
            Y.append(data_y[:,index+window+horizon-1:index+window+horizon,:])
            index = index + stride #1
        X_chart = torch.stack(X_chart)
        X_anti = torch.stack(X_anti)
        X_vent = torch.stack(X_vent)
        X_stat = torch.stack(X_stat)
        X_demo = torch.stack(X_demo)
        label = torch.stack(label)
        Y = torch.stack(Y)
        del data1,data2,data3,data4,data5,data_y
        return X_chart, X_anti, X_vent, X_stat, X_demo, label, Y

    def datamerge(self,chart,anti,vent,stat,demo,mdr_label,y,device):
        chart=chart.reshape(chart.shape[0]*chart.shape[1],chart.shape[2],chart.shape[3]).to(device)
        anti=anti.reshape(anti.shape[0]*anti.shape[1],anti.shape[2],anti.shape[3]).to(device)
        vent=vent.reshape(vent.shape[0]*vent.shape[1],vent.shape[2],vent.shape[3]).to(device)
        stat=stat.reshape(stat.shape[0]*stat.shape[1],stat.shape[2],stat.shape[3]).to(device)
        demo=demo.reshape(demo.shape[0]*demo.shape[1],demo.shape[2],demo.shape[3]).to(device)
        mdr_label=mdr_label.reshape(mdr_label.shape[0]*mdr_label.shape[1],mdr_label.shape[2],mdr_label.shape[3]).to(device)
        y=y.reshape(y.shape[0]*y.shape[1],y.shape[2],y.shape[3]).to(device)
        return chart,anti,vent,stat,demo,mdr_label,y

    def restore_tensor(self,input_tensor, category_dict):
        num_categories = len(category_dict)
        num_batch = input_tensor.shape[0]
        output_tensor = torch.zeros(num_batch, args.window, num_categories).to(input_tensor.device)   
        for category_name,category_id in category_dict.items():
            category_mask = (input_tensor.squeeze(dim=2) == category_id).float()
            output_tensor[:, :, category_id] = category_mask
        return output_tensor

class TensorDiscretizer:
    def __init__(self, tensor, feature_list, n_clusters=4, method="kmeans"):
        self.device = tensor.device  
        self.tensor = tensor.cpu().numpy() if torch.cuda.is_available() else tensor.numpy()
        self.feature_list = feature_list
        self.n_clusters = n_clusters
        self.method = method
        self.discretized_tensor = None
        self.updated_list = None
        
    def discretize_tensor_by_timesteps(self):
        
        tensor_shape = self.tensor.shape
        discretized_tensor = np.zeros_like(self.tensor)

        if self.method == "kmeans":
            for i in range(tensor_shape[2]):
                data = self.tensor[:, :, i].reshape(-1, 1)
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(data)
                labels = kmeans.predict(data)
                discretized_tensor[:, :, i] = labels.reshape(tensor_shape[0], tensor_shape[1])
        if self.method == "binning":  
            for i in range(tensor_shape[2]):
                bin_edges = np.linspace(self.tensor[:, :, i].min(), self.tensor[:, :, i].max(), self.n_clusters + 1)
                binned_feature = np.digitize(self.tensor[:, :, i], bin_edges) - 1
                discretized_tensor[:, :, i] = binned_feature

        self.discretized_tensor = discretized_tensor
        
    def update_feature_list(self):
        new_list = []
        for value in self.feature_list:
            for i in range(self.n_clusters):
                new_value = f"{value}_cluster_{i}"
                new_list.append(new_value)
        self.updated_list = new_list
        
    def transform(self):
        self.discretize_tensor_by_timesteps()
        self.update_feature_list()
        num_updated_features = len(self.updated_list)
        updated_tensor = np.zeros((self.tensor.shape[0], self.tensor.shape[1], num_updated_features))

        for i, feature_name in enumerate(self.feature_list):
            for j in range(self.n_clusters):
                updated_feature_name = f"{feature_name}_cluster_{j}"
                feature_indices = [self.updated_list.index(updated_feature_name)]
                cluster_indices = np.where(self.discretized_tensor[:, :, i] == j)
                updated_tensor[cluster_indices[0], cluster_indices[1], feature_indices] = 1

        self.discretized_tensor = torch.tensor(updated_tensor).to(self.device)
        
        return self.discretized_tensor, self.updated_list
