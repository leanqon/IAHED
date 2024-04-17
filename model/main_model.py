import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
import importlib
import numpy as np
import parameters
from parameters import *
importlib.reload(parameters)

import iahed_train as dl_train
from encoder import *
from losses import *
Tensor = torch.Tensor

class LSTMModel(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)  
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        x = torch.cat(x, dim=2).to(self.device)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(lstm_out))
        output = self.sigmoid(out)
        contrastive_loss = 0
        return output, contrastive_loss, out
    
class GRUModel(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.5):
        super(GRUModel, self).__init__()
        self.device = device
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)  
        self.fc2 = nn.Linear(64, 1) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        x = torch.cat(x, dim=2).to(self.device)
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        x = self.relu(self.fc1(gru_out))
        out = self.fc2(x)
        output = self.sigmoid(out)
        contrastive_loss = 0
        return output, contrastive_loss, out

class CNNModel(nn.Module):
    def __init__(self, device, num_channels, num_filters=64, kernel_size=3, stride=1):
        super(CNNModel, self).__init__()
        self.device = device
        self.conv1 = nn.Conv1d(num_channels, num_filters, kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=kernel_size, stride=stride)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.flattened_size = None
    
    def forward(self, seqs,labels):
        x = torch.cat(seqs, dim=2).to(self.device)
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.flattened_size is None:
            self.flattened_size = x.size(1) * x.size(2)
        fc1 = nn.Linear(self.flattened_size, 128).to(self.device)
        x = x.view(-1, self.flattened_size)
        x = F.relu(fc1(x))
        out = self.fc2(x)
        output = self.sigmoid(out)
        contrastive_loss = 0
        return output, contrastive_loss, out

class MainModel(nn.Module): 
    def __init__(self, device, dynamic_dim, static_dim, common_dim, hidden_dim, use_pretrained):
        super(MainModel, self).__init__()
        self.device=device
        self.use_pretrained = use_pretrained
        self.sigmoid = nn.Sigmoid()
        self.common_dim = common_dim
        self.hidden_dim = hidden_dim
        self.dynamic_dim = dynamic_dim
        self.temporal_unit = 0
        self.featureattention = FeatureAttention(static_dim, dynamic_dim, hidden_dim)                                                                          
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.ts_encoder =dl_train.IAHEDAutoencoder(input_dims=dynamic_dim, output_dims=hidden_dim).to(self.device)
        self.encoder_maj = dl_train.IAHEDAutoencoder(input_dims=self.dynamic_dim, output_dims=self.hidden_dim).to(self.device)
        self.encoder_min = dl_train.IAHEDAutoencoder(input_dims=self.dynamic_dim, output_dims=self.hidden_dim).to(self.device)
        if self.use_pretrained:
            majority_weight_filename = "./data/sparse/data{}/{}/majority_best_weights_{}_{}_{}_{}.pth".format(args.data_name,args.strategy,args.batch_size, args.common_dim, args.num_epochs_ae, args.output_dim)
            minority_weight_filename = "./data/sparse/data{}/{}/minority_best_weights_{}_{}_{}_{}.pth".format(args.data_name,args.strategy,args.batch_size, args.common_dim, args.num_epochs_ae, args.output_dim)
            if os.path.exists(majority_weight_filename):
                self.encoder_maj.load_state_dict(torch.load(majority_weight_filename))
            if os.path.exists(minority_weight_filename):
                self.encoder_min.load_state_dict(torch.load(minority_weight_filename))
            self.contrastive_loss_func = ContrastiveLoss(margin=1.0, pos_weight=0.8, neg_weight=0.2)

    def autoencoder(self, seqs):
        dynamic_x = torch.cat(seqs[0:3], dim=2).to(self.device)
        majority_features = self.encoder_maj(dynamic_x.clone())[1]
        minority_features = self.encoder_min(dynamic_x.clone())[1]  
        return majority_features, minority_features

    def take_per_row(self, A, indx, num_elem):
        all_indx = indx[:,None] + np.arange(num_elem)
        return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

    def temporal_crop_and_rotate(self, x1, x2, F):
        ts_l = x1.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x1.size(0))
        x1_cropped = self.take_per_row(x1, crop_offset + crop_eleft, crop_right - crop_eleft)
        x2_cropped = self.take_per_row(x2, crop_offset + crop_left, crop_eright - crop_left)
        x1_rotated = self.apply_givens_rotation(x1_cropped, F)
        x2_rotated = self.apply_givens_rotation(x2_cropped, F)
        return x1_rotated, x2_rotated, crop_l
    
    def givens_rotation_matrix(self, i, j, theta, F):
        G = torch.eye(F, device=self.device)
        G[i, i] = G[j, j] = torch.cos(theta)
        G[i, j] = -torch.sin(theta)
        G[j, i] = torch.sin(theta)
        return G

    def apply_givens_rotation(self, x, F):
        i, j = np.random.choice(F, 2, replace=False)
        theta = np.random.uniform(-np.pi, np.pi)
        G = self.givens_rotation_matrix(i, j, torch.tensor(theta, device=self.device), F)
        return torch.matmul(x, G)

    def forward(self, seqs, labels): 
        if self.use_pretrained:
            majority_features, minority_features = self.autoencoder(seqs)
            contrastive_loss = self.contrastive_loss_func(majority_features, minority_features, labels)
        else:
            majority_features = torch.cat(seqs[0:3], dim=2).to(self.device)
            minority_features = majority_features.clone()  
            contrastive_loss = 0.0

        F = self.dynamic_dim
        cropped_dynamic_x1, cropped_dynamic_x2, crop_l = self.temporal_crop_and_rotate(majority_features, minority_features, F)
        dynamic_repr1 = self.ts_encoder(cropped_dynamic_x1)[1]
        dynamic_repr1 = dynamic_repr1[:,-crop_l:]
        dynamic_repr2 = self.ts_encoder(cropped_dynamic_x2)[1]
        dynamic_repr2 = dynamic_repr2[:,:crop_l]
        combined_dynamic_repr = torch.cat((dynamic_repr1, dynamic_repr2), dim=-1)
        hierarchical_loss = hierarchical_contrastive_loss(dynamic_repr1, dynamic_repr2, temporal_unit=self.temporal_unit)
        out = combined_dynamic_repr[:,-1,:].squeeze(1)
        static_x = torch.cat(seqs[3:7],dim=2).to(self.device)
        static_x = static_x[:,-1,:].squeeze(1)
        static_x = self.dropout(static_x)
        combined_data = self.featureattention(out, static_x)
        combined_data = self.dropout(combined_data)
        out1 = self.fc1(combined_data)
        logits = self.fc2(out1)
        predictions = self.sigmoid(logits)
        total_loss = 0.5*contrastive_loss + hierarchical_loss 

        return predictions, total_loss, logits

class FeatureAttention(nn.Module):
    def __init__(self, time_invariant_dim, time_variant_dim, feature_dim):
        super(FeatureAttention, self).__init__()
        self.time_invariant_embedding = nn.Linear(time_invariant_dim, feature_dim*2)
        self.fc1 = nn.Linear(time_variant_dim*2, feature_dim*2)
        self.attention = nn.Linear(feature_dim*4, 1)  
        self.batchnorm = nn.BatchNorm1d(feature_dim*2)

    def forward(self, time_variant, time_invariant):
        time_invariant_embedded = self.time_invariant_embedding(time_invariant)
        time_invariant_embedded = self.batchnorm(time_invariant_embedded)
        time_variant = self.fc1(time_variant)
        time_variant = self.batchnorm(time_variant)
        combined_features = torch.cat((time_variant, time_invariant_embedded), dim=1)
        attention_weights = self.attention(combined_features)
        attention_weights = torch.sigmoid(attention_weights)
        combined_features = attention_weights * time_variant + (1 - attention_weights) * time_invariant_embedded
     
        return combined_features        

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=0.8, neg_weight=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * self.neg_weight * torch.pow(euclidean_distance, 2).clone() +
                                      (label) * self.pos_weight * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0).clone(), 2))
        return loss_contrastive

