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
ARG_PARSER.add_argument('--common_dim', default=32, type=int) 
ARG_PARSER.add_argument('--output_dim', default=32, type=int) 
ARG_PARSER.add_argument('--num_workers', default=1, type=int)
ARG_PARSER.add_argument('--step_size', default=24, type=int)
ARG_PARSER.add_argument('--num_epochs_ae', default=20, type=int)

ARG_PARSER.add_argument('--batch_size', default=400, type=int)
ARG_PARSER.add_argument('--test_size', default=0.2, type=int)
ARG_PARSER.add_argument('--val_size', default=0.1, type=int)
ARG_PARSER.add_argument('--data_name', default='_168_12_2', type=str) # '_168_12_2' '_336_24_2'
ARG_PARSER.add_argument('--sample_over', default=1, type=int)
ARG_PARSER.add_argument('--sample_under', default=1, type=int)
ARG_PARSER.add_argument('--num_epochs', default=400, type=int)
ARG_PARSER.add_argument('--patience', default=2, type=int)

ARG_PARSER.add_argument('--lrn_rate', default=0.001, type=float)
ARG_PARSER.add_argument('--window', default=84, type=int)#24 84
ARG_PARSER.add_argument('--horizon', default=24, type=int)#12 24
ARG_PARSER.add_argument('--stride', default=1, type=int)
ARG_PARSER.add_argument('--strategy', default='threshold', type=str) #'truncate' 'threshold' 'fill'
ARG_PARSER.add_argument('--use_pretrained', default='False', type=str) #False True

args = ARG_PARSER.parse_args(args=[])