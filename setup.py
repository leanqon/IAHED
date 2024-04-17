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

from feature_selection_icu import *

from iahed_train import *

from parameters import *

disease_label=""
time=0
label='Mortality'

data_icu="ICU"

diag_flag=True 
proc_flag=False
out_flag=False
chart_flag=True 
med_flag=False
anti_flag=True 
vent_flag=True
k_fold=5
model_type = 'Xgboost' #'Random Forest','Logistic Regression','Gradient Boosting','Xgboost','Time-series LSTM','Time-series CNN','main_model2', 'lstm'
cohort_output='pts_3'
data_mort=True
data_admn=False
data_los=Falseimpute='Mean'# 'Mean','Median'
include=168 #168 336
bucket=2
save_data=False
data_name= args.data_name #'_336_24_2' #'_168_12_2'

gen=data_generation_icu.Generator(cohort_output,data_mort,data_admn,data_los,diag_flag,proc_flag,out_flag,chart_flag,med_flag,anti_flag,vent_flag,impute,include,bucket)

ml_train=iahed_train.ML_models(model_type,data_name)

dl_model=iahed_train.DL_models(data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,False,anti_flag,vent_flag,model_type,k_fold,data_name,
        sampling_first=False,undersampling=True,model_name='icu_read',train=True,save_data=False,pre_train=False,train_test=True,test=False)
