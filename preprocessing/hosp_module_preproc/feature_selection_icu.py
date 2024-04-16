import os
import pickle
import glob
import importlib
#print(os.getcwd())
#os.chdir('../../')
#print(os.getcwd())
import utils.icu_preprocess_util
from utils.icu_preprocess_util import * 
importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *# module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *  
importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *  
importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *


if not os.path.exists("./data/features"):
    os.makedirs("./data/features")
if not os.path.exists("./data/features/chartevents"):
    os.makedirs("./data/features/chartevents")
if not os.path.exists("./data/summary/subset"):
    os.makedirs("./data/summary/subset")
if not os.path.exists("./data/features/subset"):
    os.makedirs("./data/features/subset")


def feature_icu(cohort_output, version_path, diag_flag=True,out_flag=True,chart_flag=True,proc_flag=True,med_flag=True,anti_flag=True,vent_flag=True):
    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag = preproc_icd_module("~/MIMIC-IV-Data-Pipeline/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
        diag[['subject_id', 'hadm_id', 'stay_id', 'icd_code','root_icd10_convert','root']].to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if out_flag:  
        print("[EXTRACTING OUPTPUT EVENTS DATA]")
        out = preproc_out("~/MIMIC-IV-Data-Pipeline/physionet.org/files/mimiciv/2.2/icu/outputevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=None)
        out[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    
    if chart_flag:
        print("[EXTRACTING CHART EVENTS DATA]")
        chart=preproc_chart("~/MIMIC-IV-Data-Pipeline/physionet.org/files/mimiciv/2.2/icu/chartevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=['stay_id','charttime','itemid','valuenum','valueuom'])
        chart = drop_wrong_uom(chart, 0.95)
        chart[['stay_id', 'itemid','event_time_from_admit','valuenum']].to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
    
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc("~/MIMIC-IV-Data-Pipeline/physionet.org/files/mimiciv/2.2/icu/procedureevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'starttime', dtypes=None, usecols=['stay_id','starttime','itemid'])
        proc[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    
    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds("~/MIMIC-IV-Data-Pipeline/physionet.org/files/mimiciv/2.2/icu/inputevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz')
        med[['subject_id', 'hadm_id', 'stay_id', 'itemid' ,'starttime','endtime', 'start_hours_from_admit', 'stop_hours_from_admit','rate','amount','orderid']].to_csv('./data/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    
    if anti_flag:
        print("[EXTRACTING ANTIBIOTICS DATA]")
        anti = preproc_anti("~/MIMIC-IV-Data-Pipeline/data/features/antibiotic.csv", './data/cohort/'+cohort_output+'.csv.gz')
        anti[['subject_id', 'hadm_id', 'stay_id', 'antibiotic','starttime','stoptime', 'start_hours_from_admit', 'stop_hours_from_admit']].to_csv('./data/features/preproc_anti_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED ANTIBIOTICS DATA]")

    if vent_flag:
        print("[EXTRACTING VENTILATION DATA]")
        vent = preproc_vent("~/MIMIC-IV-Data-Pipeline/data/features/ventilation_full.csv", './data/cohort/'+cohort_output+'.csv.gz')
        vent[['subject_id', 'hadm_id', 'stay_id','ventilation_status','starttime','endtime', 'start_hours_from_admit', 'stop_hours_from_admit']].to_csv('./data/features/preproc_vent_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED VENTILATION DATA]")

def preprocess_features_icu(cohort_output, diag_flag, group_diag,chart_flag,clean_chart,clean_again_chart,impute_outlier_chart,thresh,left_thresh):
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
        if(group_diag=='Keep both ICD-9 and ICD-10 codes'):
            diag['new_icd_code']=diag['icd_code']
        if(group_diag=='Convert ICD-9 to ICD-10 codes'):
            diag['new_icd_code']=diag['root_icd10_convert']
        if(group_diag=='Convert ICD-9 to ICD-10 and group ICD-10 codes'):
            diag['new_icd_code']=diag['root']

        diag=diag[['subject_id', 'hadm_id', 'stay_id', 'new_icd_code']].dropna()
        print("Total number of rows",diag.shape[0])
        diag.to_csv("./data/features/preproc_diag_icu2.csv.gz", compression='gzip', index=False) #########
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
        
    if chart_flag:
        if clean_chart:   
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv("./data/features/subset/preproc_chart_icu.csv.gz", compression='gzip',header=0)
            #chart = outlier_imputation(chart, 'itemid', 'valuenum', thresh,left_thresh,impute_outlier_chart)
            if clean_again_chart:
                chart = chart[chart['itemid'].isin([220045,225309,225310,225312,220050,220051,220052,220179,220180,220181,220210,224690,220277,220621,226537,223762,223761])]#225664,224642  
                
            print("Total number of rows",chart.shape[0])
            chart.to_csv("./data/features/subset/preproc_chart_icu3.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
            
        
        
def generate_summary_icu(diag_flag,proc_flag,med_flag,anti_flag,out_flag,chart_flag,vent_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv("./data/features/preproc_diag_icu2.csv.gz", compression='gzip',header=0) #######
        freq=diag.groupby(['stay_id','new_icd_code']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['new_icd_code'])['mean_frequency'].mean().reset_index()
        total=diag.groupby('new_icd_code').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='new_icd_code',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/diag_summary.csv',index=False)
        summary['new_icd_code'].to_csv('./data/summary/diag_features.csv',index=False)

    if vent_flag:
        vent = pd.read_csv("./data/features/preproc_vent_icu.csv.gz", compression='gzip',header=0)
        freq=vent.groupby(['stay_id','ventilation_status']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['ventilation_status'])['mean_frequency'].mean().reset_index()
        total=vent.groupby('ventilation_status').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='ventilation_status',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/vent_summary.csv',index=False)
        summary['ventilation_status'].to_csv('./data/summary/vent_features.csv',index=False)

    if med_flag:
        med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
        freq=med.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        missing=med[med['amount']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=med.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/med_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/med_features.csv',index=False)

    if anti_flag:
        anti = pd.read_csv("./data/features/preproc_anti_icu.csv.gz", compression='gzip',header=0)
        freq=anti.groupby(['stay_id','antibiotic']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['antibiotic'])['mean_frequency'].mean().reset_index()
        missing=anti[anti['antibiotic']=='None'].groupby('antibiotic').size().reset_index(name="missing_count")
        total=anti.groupby('antibiotic').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='antibiotic',how='right')
        summary=pd.merge(freq,summary,on='antibiotic',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/anti_summary.csv',index=False)
        summary['antibiotic'].to_csv('./data/summary/anti_features.csv',index=False)
    
    if proc_flag:
        proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
        freq=proc.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=proc.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/proc_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/proc_features.csv',index=False)

        
    if out_flag:
        out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
        freq=out.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=out.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/out_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/out_features.csv',index=False)
        
    if chart_flag:
        chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
        freq=chart.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()

        missing=chart[chart['valuenum']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=chart.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing_perc']=100*(summary['missing_count']/summary['total_count'])
        #summary=summary.fillna(0)

#         final.groupby('itemid')['missing_count'].sum().reset_index()
#         final.groupby('itemid')['total_count'].sum().reset_index()
#         final.groupby('itemid')['missing%'].mean().reset_index()
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/chart_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/chart_features.csv',index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")
    
def features_selection_icu(cohort_output, diag_flag,proc_flag,med_flag,anti_flag,vent_flag,out_flag,chart_flag,group_diag,group_med,group_anti,group_vent,group_proc,group_out,group_chart):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv("./data/features/preproc_diag_icu2.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/diag_features.csv",header=0)
            summary = pd.read_csv("./data/summary/diag_summary.csv",header=0)
            summary = summary[summary['total_count']>= 10]
            features_new = features[features['new_icd_code'].isin(summary['new_icd_code'])]
            #diag=diag[diag['new_icd_code'].isin(features['new_icd_code'].unique())]
            diag=diag[diag['new_icd_code'].isin(summary['new_icd_code'].unique())]

            print("Total number of rows",diag.shape[0])
            features_new.to_csv("./data/summary/subset/diag_features.csv",index=False)
            diag.to_csv("./data/features/subset/preproc_diag_icu.csv.gz", compression='gzip', index=False)#######
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:       
        if group_med:   
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/med_features.csv",header=0)
            summary = pd.read_csv("./data/summary/med_summary.csv",header=0)
            summary = summary[summary['total_count']>= 10]
            features_new = features[features['itemid'].isin(summary['itemid'])]

            med=med[med['itemid'].isin(summary['itemid'].unique())]
            print("Total number of rows",med.shape[0])
            features_new.to_csv("./data/summary/subset/med_features.csv",index=False)
            med.to_csv('./data/features/subset/preproc_med_icu.csv.gz', compression='gzip', index=False)#######
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

    if anti_flag:
        if group_anti:
            print("[FEATURE SELECTION ANTIBIOTICS DATA]")
            anti = pd.read_csv("./data/features/preproc_anti_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/anti_features.csv",header=0)
            summary = pd.read_csv("./data/summary/anti_summary.csv",header=0)
            summary = summary[summary['total_count']>= 10]
            features_new = features[features['antibiotic'].isin(summary['antibiotic'])]
            anti=anti[anti['antibiotic'].isin(summary['antibiotic'].unique())]
            print("Total number of rows",anti.shape[0])
            features_new.to_csv("./data/summary/subset/anti_features.csv",index=False)
            anti.to_csv("./data/features/subset/preproc_anti_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED ANTIBIOTICS DATA]")
    
    if vent_flag:
        if group_vent:
            print("[FEATURE SELECTION VENTILATION DATA]")
            vent = pd.read_csv("./data/features/preproc_vent_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/vent_features.csv",header=0)
            summary = pd.read_csv("./data/summary/vent_summary.csv",header=0)
            summary = summary[summary['total_count']>= 10]
            features_new = features[features['ventilation_status'].isin(summary['ventilation_status'])]
            vent=vent[vent['ventilation_status'].isin(summary['ventilation_status'].unique())]
            print("Total number of rows",vent.shape[0])
            features_new.to_csv("./data/summary/subset/vent_features.csv",index=False)
            vent.to_csv("./data/features/subset/preproc_vent_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED VENTILATION DATA]")
            

    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/proc_features.csv",header=0)
            summary = pd.read_csv("./data/summary/proc_summary.csv",header=0)
            summary = summary[summary['total_count']>= 10]
            features_new = features[features['itemid'].isin(summary['itemid'])]
            proc=proc[proc['itemid'].isin(summary['itemid'].unique())]
            print("Total number of rows",proc.shape[0])
            features_new.to_csv("./data/summary/subset/proc_features.csv",index=False)
            proc.to_csv("./data/features/subset/preproc_proc_icu.csv.gz", compression='gzip', index=False)#########
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
        
    if out_flag:
        if group_out:            
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/out_features.csv",header=0)
            summary = pd.read_csv("./data/summary/out_summary.csv",header=0)
            summary = summary[summary['total_count']>= 1000]
            features_new = features[features['itemid'].isin(summary['itemid'])]
            out=out[out['itemid'].isin(summary['itemid'].unique())]
            print("Total number of rows",out.shape[0])
            features_new.to_csv("./data/summary/subset/out_features.csv",index=False)
            out.to_csv("./data/features/subset/preproc_out_icu.csv.gz", compression='gzip', index=False)########
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")
            
    if chart_flag:
        if group_chart:            
            print("[FEATURE SELECTION CHART EVENTS DATA]")
            
            chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0, index_col=None)
            
            features=pd.read_csv("./data/summary/chart_features.csv",header=0)
            summary = pd.read_csv("./data/summary/chart_summary.csv",header=0)
            summary = summary[summary['total_count']>= 1000]
            features_new = features[features['itemid'].isin(summary['itemid'])]
            chart=chart[chart['itemid'].isin(summary['itemid'].unique())]
            print("Total number of rows",chart.shape[0])
            features_new.to_csv("./data/summary/subset/chart_features.csv",index=False)
            chart.to_csv("./data/features/subset/preproc_chart_icu.csv.gz", compression='gzip', index=False)########
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")