import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
import argparse
from argparse import ArgumentParser


ARG_PARSER = ArgumentParser()

ARG_PARSER.add_argument('--alpha', default=2, type=int)
ARG_PARSER.add_argument('--gamma', default=2, type=int)
ARG_PARSER.add_argument('--focal_weight', default=0.7, type=int)
ARG_PARSER.add_argument('--contrastive_weight', default=0.3, type=int)
ARG_PARSER.add_argument('--gat_indim', default=32, type=int)
ARG_PARSER.add_argument('--nheads', default=4, type=int)
ARG_PARSER.add_argument('--trans_layers', default=2, type=int)
ARG_PARSER.add_argument('--common_dim', default=32, type=int) #32 64
ARG_PARSER.add_argument('--output_dim', default=32, type=int) #32
ARG_PARSER.add_argument('--num_workers', default=1, type=int)
ARG_PARSER.add_argument('--enc_contrastive_weight', default=0.2, type=int)
ARG_PARSER.add_argument('--dynamic_contrastive_weight', default=0.8, type=int)
ARG_PARSER.add_argument('--step_size', default=24, type=int)
ARG_PARSER.add_argument('--num_epochs_ae', default=20, type=int)

ARG_PARSER.add_argument('--batch_size', default=400, type=int)#main_model 400 lstm gru cnn 200 lr 0.001
ARG_PARSER.add_argument('--test_size', default=0.2, type=int)
ARG_PARSER.add_argument('--val_size', default=0.1, type=int)
ARG_PARSER.add_argument('--data_name', default='_168_12_2', type=str) # '_168_12_2' '_336_24_2'
ARG_PARSER.add_argument('--sample_over', default=1, type=int)
ARG_PARSER.add_argument('--sample_under', default=1, type=int)

ARG_PARSER.add_argument('--num_epochs', default=100, type=int)
ARG_PARSER.add_argument('--patience', default=2, type=int)

ARG_PARSER.add_argument('--rnnLayers', default=2, type=float)
#ARG_PARSER.add_argument('--embedding_size', default=40, type=float)
#ARG_PARSER.add_argument('--feature_size', default=120, type=float)
ARG_PARSER.add_argument('--latent_size', default=20, type=float)#in_channels fc embedding_size
ARG_PARSER.add_argument('--embed_size', default=20, type=int)
ARG_PARSER.add_argument('--rnn_size', default=20, type=float)
ARG_PARSER.add_argument('--lrn_rate', default=0.001, type=float)
ARG_PARSER.add_argument('--lambda1', default=0.001, type=float)
ARG_PARSER.add_argument('--window', default=84, type=int)#24 84
ARG_PARSER.add_argument('--horizon', default=24, type=int)#12 24
ARG_PARSER.add_argument('--stride', default=1, type=int)
ARG_PARSER.add_argument('--updated', default='_moved_icu_mdrb', type=str)
ARG_PARSER.add_argument('--strategy', default='threshold', type=str) #'truncate' 'threshold' 'fill'
ARG_PARSER.add_argument('--use_pretrained', default='False', type=str) #False True

#ARG_PARSER.add_argument('--input_dim', default=40, type=int)
#ARG_PARSER.add_argument('--output_dim', default=1, type=int)
#ARG_PARSER.add_argument('--rnn_units', default=20, type=int)# dim_out
#ARG_PARSER.add_argument('--num_layers', default=2, type=int)
ARG_PARSER.add_argument('--cheb_k', default=3, type=int)
ARG_PARSER.add_argument('--embed_dim', default=20, type=int) #node embedding dim latent_size

ARG_PARSER.add_argument('--in_channels', default=20, type=int) #latent_size*6
ARG_PARSER.add_argument('--K', default=2, type=int)
ARG_PARSER.add_argument('--nb_chev_filter', default=4, type=int)#10
ARG_PARSER.add_argument('--nb_time_filter', default=4, type=int)#10
ARG_PARSER.add_argument('--time_strides', default=1, type=int)
ARG_PARSER.add_argument('--len_input', default=84, type=int) #time_step window 24 84
ARG_PARSER.add_argument('--num_of_vertices', default=20, type=int) #batch_size node_num
ARG_PARSER.add_argument('--num_for_predict', default=84, type=int) #time_step



# ARG_PARSER.add_argument('--cond_seq_len', default=39, type=int)#473
# ARG_PARSER.add_argument('--proc_seq_len', default=24, type=int)#490
# ARG_PARSER.add_argument('--med_seq_len', default=280, type=int)#565
# ARG_PARSER.add_argument('--out_seq_len', default=146, type=int)
# ARG_PARSER.add_argument('--chart_seq_len', default=118, type=int)#54

# ARG_PARSER.add_argument('--cond_vocab_size', default=1424, type=int)#607#625
# ARG_PARSER.add_argument('--proc_vocab_size', default=152, type=int)#355#347
# ARG_PARSER.add_argument('--med_vocab_size', default=274, type=int)#340#378
# ARG_PARSER.add_argument('--out_vocab_size', default=72, type=int)
# ARG_PARSER.add_argument('--chart_vocab_size', default=76, type=int)#74,75
args = ARG_PARSER.parse_args(args=[])