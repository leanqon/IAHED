import sys
from pathlib import Path
import os
module_path='model'
if module_path not in sys.path:
    sys.path.append(module_path)
root_dir = os.path.dirname(os.path.abspath('UserInterface.ipynb'))
from feature_selection_icu import *
from train import *
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
model_type = 'main_model' #'Random Forest','Logistic Regression','Gradient Boosting','lstm','cnn'
cohort_output='pts_3'
data_mort=True
data_admn=False
data_los=False
impute='Mean'# 'Mean','Median'
include=168 #168 336
bucket=2
save_data=False
data_name= args.data_name 

data_generation=data_generation_icu.Generator(cohort_output,data_mort,data_admn,data_los,diag_flag,proc_flag,out_flag,chart_flag,med_flag,anti_flag,vent_flag,impute,include,bucket)

ml_train=train.ML_models(model_type,data_name)

dl_model=train.DL_models(data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,False,anti_flag,vent_flag,model_type,k_fold,data_name,sampling_first=False,undersampling=True,save_data=False,pre_train=False,train_test=True,test=False)
