import ipywidgets as widgets
import sys
from pathlib import Path
import os
import importlib
import sklearn.ensemble
module_path='preprocessing/day_intervals_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)

module_path='utils'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='preprocessing/hosp_module_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='model'
if module_path not in sys.path:
    sys.path.append(module_path)

root_dir = os.path.dirname(os.path.abspath('UserInterface.ipynb'))

import day_intervals_cohort
from day_intervals_cohort import *

import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

import data_generation_icu

import data_generation
import evaluation



import feature_selection_icu
from feature_selection_icu import *

import dl_train_copy4
from dl_train_copy4 import *

import behrt_train
from behrt_train import *

import fairness

import callibrate_output

import evaluation

from parameters import *

disease_label=""
time=0
label='Mortality'

data_icu="ICU"
#data_mort="Mortality"
#data_adm='Readmission'
#data_los='Length of Stay'
#icd_code='No Disease Filter'
diag_flag=True 
proc_flag=False
out_flag=False
chart_flag=True 
med_flag=False
anti_flag=True 
vent_flag=True
k_fold=5
model_type = 'Xgboost'
#'Random Forest','Logistic Regression','Gradient Boosting','Xgboost'
#'rgsl','Time-series LSTM','Timthresholdiiiie-series CNN','Hybrid LSTM','Hybrid CNN','main_model2', 'lstm'
cohort_output='pts_3'
data_mort=True
data_admn=False
data_los=Falseimpute='Mean'# 'Mean','Median'
include=168 #168 336
bucket=2
#predW=12
#is_data_all=False
save_data=False
data_name= args.data_name #'_336_24_2' #'_168_12_2'

#gen=data_generation_icu.Generator(cohort_output,data_mort,data_admn,data_los,diag_flag,proc_flag,out_flag,chart_flag,med_flag,anti_flag,vent_flag,impute,include,bucket)
#predW,,is_data_all=is_data_all

ml_train=dl_train_copy4.ML_models(model_type,data_name)

#dl_model=dl_train_copy4.DL_models(data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,False,anti_flag,vent_flag,model_type,k_fold,data_name,
#        sampling_first=False,undersampling=True,model_name='icu_read',train=True,save_data=False,pre_train=False,train_test=True,test=False)
