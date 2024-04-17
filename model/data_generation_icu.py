import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import datetime
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/dict"):
    os.makedirs("./data/dict")
if not os.path.exists("./data/csv"):
    os.makedirs("./data/csv")
if not os.path.exists("./data/dict_all"):
    os.makedirs("./data/dict_all")
if not os.path.exists("./data/csv_all"):
    os.makedirs("./data/csv_all")
    
class Generator():
    def __init__(self,cohort_output,if_mort,if_admn,if_los,feat_cond,feat_proc,feat_out,feat_chart,feat_med,feat_anti,feat_vent,impute,include_time,bucket):
        self.feat_cond = feat_cond
        self.feat_proc = feat_proc
        self.feat_out = feat_out
        self.feat_chart = feat_chart
        self.feat_med = feat_med
        self.feat_anti = feat_anti
        self.feat_vent = feat_vent  
        self.cohort_output=cohort_output
        self.impute=impute
        self.include_time=include_time
        self.bucket=bucket
        self.data = self.generate_adm()
        print("[ READ COHORT ]")
        
        self.generate_feat()
        print("[ READ ALL FEATURES ]")
        
        if if_mort:
            self.mortality_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_admn:
            self.readmission_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_los:
            self.los_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        
        self.smooth_meds(bucket)
        print("[ SUCCESSFULLY SAVED DATA DICTIONARIES ]")
    
    def generate_feat(self):
        if(self.feat_cond):
            print("[ ======READING DIAGNOSIS ]")
            self.generate_cond()
        if(self.feat_proc):
            print("[ ======READING PROCEDURES ]")
            self.generate_proc()
        if(self.feat_out):
            print("[ ======READING OUT EVENTS ]")
            self.generate_out()
        if(self.feat_chart):
            print("[ ======READING CHART EVENTS ]")
            self.generate_chart()
        if(self.feat_med):
            print("[ ======READING MEDICATIONS ]")
            self.generate_meds()
        if(self.feat_anti):
            print("[ ======READING ANTIMICROBIALS ]")
            self.generate_anti()
        if(self.feat_vent):
            print("[ ======READING VENTILATION ]")
            self.generate_vent()

    def generate_adm(self):
        data=pd.read_csv(f"./data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
        data=data.drop(['mdr_los'],axis=1)
        data['charttime']=np.where(data['charttime']=='0',data['intime'],data['charttime'])
        data['intime'] = pd.to_datetime(data['intime'])
        data['outtime'] = pd.to_datetime(data['outtime'])
        data['charttime'] = pd.to_datetime(data['charttime'],errors='ignore')
        data['los']=pd.to_timedelta(data['outtime']-data['intime'],unit='h')
        data['los']=data['los'].astype(str)
        data[['days', 'dummy','hours']] = data['los'].str.split(' ', expand=True)
        data[['hours','min','sec']] = data['hours'].str.split(':',  expand=True)
        data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])
        data=data[data['los']>0]

        data['mdr_los']=pd.to_timedelta(data['charttime']-data['intime'],unit='h')
        data['mdr_los']=data['mdr_los'].astype(str)
        data[['days', 'dummy','hours']] = data['mdr_los'].str.split(' ',  expand=True)
        data[['hours','min','sec']] = data['hours'].str.split(':',  expand=True)
        data['mdr_los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])

        data['Age']=data['Age'].astype(int)
        data['icu_mdr']=data['icu_mdr'].astype(int)
        data['non_icu_mdr']=data['non_icu_mdr'].astype(int)

        return data
    
    def generate_cond(self):
        cond=pd.read_csv("./data/features/subset/preproc_diag_icu.csv.gz", compression='gzip', header=0, index_col=None)
        cond=cond[cond['stay_id'].isin(self.data['stay_id'])]
        cond_per_adm = cond.groupby('stay_id').size().max()
        self.cond, self.cond_per_adm = cond, cond_per_adm
    
    def generate_proc(self):
        proc=pd.read_csv("./data/features/subset/preproc_proc_icu.csv.gz", compression='gzip', header=0, index_col=None)
        proc=proc[proc['stay_id'].isin(self.data['stay_id'])]
        proc[['start_days', 'dummy','start_hours']] = proc['event_time_from_admit'].str.split(' ', expand=True)
        proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', expand=True)
        proc['start_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
        proc=proc.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        proc=proc[proc['start_time']>=0]
        proc=pd.merge(proc,self.data[['stay_id','los','mdr_los','icu_mdr','non_icu_mdr','charttime']],on='stay_id',how='left') 
        proc['sanity']=proc['los']-proc['start_time']
        proc=proc[proc['sanity']>0]
        del proc['sanity']
        
        self.proc=proc
    
    def generate_anti(self):
        anti=pd.read_csv("./data/features/subset/preproc_anti_icu.csv.gz", compression='gzip', header=0, index_col=None)
        anti=anti[anti['stay_id'].isin(self.data['stay_id'])]
        anti[['start_days', 'dummy','start_hours']] = anti['start_hours_from_admit'].str.split(' ', expand=True)
        anti[['start_hours','min','sec']] = anti['start_hours'].str.split(':', expand=True)
        anti['start_time']=pd.to_numeric(anti['start_days'])*24+pd.to_numeric(anti['start_hours'])
        anti[['start_days', 'dummy','start_hours']] = anti['stop_hours_from_admit'].str.split(' ', expand=True)
        anti[['start_hours','min','sec']] = anti['start_hours'].str.split(':', expand=True)
        anti['stop_time']=pd.to_numeric(anti['start_days'])*24+pd.to_numeric(anti['start_hours'])
        anti=anti.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        anti=anti[anti['start_time']>=0]
        anti['sanity']=anti['stop_time']-anti['start_time']
        anti=anti[anti['sanity']>0]
        del anti['sanity']
        anti=pd.merge(anti,self.data[['stay_id','los','mdr_los','icu_mdr','non_icu_mdr','charttime']],on='stay_id',how='left')
        anti['sanity']=anti['los']-anti['start_time']
        anti=anti[anti['sanity']>0]
        del anti['sanity']
        anti.loc[anti['stop_time'] > anti['los'],'stop_time']=anti.loc[anti['stop_time'] > anti['los'],'los']
        del anti['los']

        self.anti=anti

    def generate_vent(self):
        vent=pd.read_csv("./data/features/subset/preproc_vent_icu.csv.gz", compression='gzip', header=0, index_col=None)
        vent=vent[vent['stay_id'].isin(self.data['stay_id'])]
        vent[['start_days', 'dummy','start_hours']] = vent['start_hours_from_admit'].str.split(' ', expand=True)
        vent[['start_hours','min','sec']] = vent['start_hours'].str.split(':', expand=True)
        vent['start_time']=pd.to_numeric(vent['start_days'])*24+pd.to_numeric(vent['start_hours'])
        vent[['start_days', 'dummy','start_hours']] = vent['stop_hours_from_admit'].str.split(' ', expand=True)
        vent[['start_hours','min','sec']] = vent['start_hours'].str.split(':', expand=True)
        vent['stop_time']=pd.to_numeric(vent['start_days'])*24+pd.to_numeric(vent['start_hours'])
        vent=vent.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        vent=vent[vent['start_time']>=0]
        vent['sanity']=vent['stop_time']-vent['start_time']
        vent=vent[vent['sanity']>0]
        del vent['sanity']
        vent=pd.merge(vent,self.data[['stay_id','los','mdr_los','icu_mdr','non_icu_mdr','charttime']],on='stay_id',how='left')
        vent['sanity']=vent['los']-vent['start_time']
        vent=vent[vent['sanity']>0]
        del vent['sanity']
        vent.loc[vent['stop_time'] > vent['los'],'stop_time']=vent.loc[vent['stop_time'] > vent['los'],'los']
        del vent['los']
        
        self.vent=vent

    def generate_out(self):
        out=pd.read_csv("./data/features/subset/preproc_out_icu.csv.gz", compression='gzip', header=0, index_col=None)
        out=out[out['stay_id'].isin(self.data['stay_id'])]
        out[['start_days', 'dummy','start_hours']] = out['event_time_from_admit'].str.split(' ', expand=True)
        out[['start_hours','min','sec']] = out['start_hours'].str.split(':', expand=True)
        out['start_time']=pd.to_numeric(out['start_days'])*24+pd.to_numeric(out['start_hours'])
        out=out.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        out=out[out['start_time']>=0]
        out=pd.merge(out,self.data[['stay_id','los','mdr_los','icu_mdr','non_icu_mdr','charttime']],on='stay_id',how='left')
        out['sanity']=out['los']-out['start_time']
        out=out[out['sanity']>0]
        del out['sanity']
        
        self.out=out      
        
    def generate_chart(self):
        chunksize = 5000000
        final=pd.DataFrame()
        for chart in tqdm(pd.read_csv("./data/features/subset/preproc_chart_icu3.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
            chart=chart[chart['stay_id'].isin(self.data['stay_id'])]
            chart[['start_days', 'dummy','start_hours']] = chart['event_time_from_admit'].str.split(' ', expand=True)
            chart[['start_hours','min','sec']] = chart['start_hours'].str.split(':', expand=True)
            chart['start_time']=pd.to_numeric(chart['start_days'])*24+pd.to_numeric(chart['start_hours'])
            chart=chart.drop(columns=['start_days', 'dummy','start_hours','min','sec','event_time_from_admit'])
            chart=chart[chart['start_time']>=0]
            chart=pd.merge(chart,self.data[['stay_id','los','mdr_los','icu_mdr','non_icu_mdr','charttime']],on='stay_id',how='left')
            chart['sanity']=chart['los']-chart['start_time']
            chart=chart[chart['sanity']>0]
            del chart['sanity']
            del chart['los']
            
            if final.empty:
                final=chart
            else:
                final=pd.concat([final,chart], ignore_index=True)
        
        self.chart=final
        
    def generate_meds(self):
        meds=pd.read_csv("./data/features/subset/preproc_med_icu.csv.gz", compression='gzip', header=0, index_col=None)
        meds[['start_days', 'dummy','start_hours']] = meds['start_hours_from_admit'].str.split(' ', expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', expand=True)
        meds['start_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds[['start_days', 'dummy','start_hours']] = meds['stop_hours_from_admit'].str.split(' ', expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', expand=True)
        meds['stop_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds=meds.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        meds['sanity']=meds['stop_time']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        meds=meds[meds['stay_id'].isin(self.data['stay_id'])]
        meds=pd.merge(meds,self.data[['stay_id','los','mdr_los','icu_mdr','non_icu_mdr','charttime']],on='stay_id',how='left')
        meds['sanity']=meds['los']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
        del meds['los']
        meds['rate']=meds['rate'].apply(pd.to_numeric, errors='coerce')
        meds['amount']=meds['amount'].apply(pd.to_numeric, errors='coerce')
        
        self.meds=meds
    
    def mortality_length(self,include_time):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                     
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]

        ###ANTIBIOTICS
        if(self.feat_anti):
            self.anti=self.anti[self.anti['stay_id'].isin(self.data['stay_id'])]
            self.anti=self.anti[self.anti['start_time']<=include_time]
            self.anti.loc[self.anti.stop_time >include_time, 'stop_time']=include_time

        ###VENTILATION
        if(self.feat_vent):
            self.vent=self.vent[self.vent['stay_id'].isin(self.data['stay_id'])]
            self.vent=self.vent[self.vent['start_time']<=include_time]
            self.vent.loc[self.vent.stop_time >include_time, 'stop_time']=include_time
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=self.out[self.out['start_time']<=include_time]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=self.chart[self.chart['start_time']<=include_time]

    def los_length(self,include_time):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        
        self.data['los']=include_time

        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                    
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=self.out[self.out['start_time']<=include_time]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=self.chart[self.chart['start_time']<=include_time]
            
    def readmission_length(self,include_time):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=pd.merge(self.meds,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.meds['stop_time']=self.meds['stop_time']-self.meds['select_time']
            self.meds['start_time']=self.meds['start_time']-self.meds['select_time']
            self.meds=self.meds[self.meds['stop_time']>=0]
            self.meds.loc[self.meds.start_time <0, 'start_time']=0
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=pd.merge(self.proc,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.proc['start_time']=self.proc['start_time']-self.proc['select_time']
            self.proc=self.proc[self.proc['start_time']>=0]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=pd.merge(self.out,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.out['start_time']=self.out['start_time']-self.out['select_time']
            self.out=self.out[self.out['start_time']>=0]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=pd.merge(self.chart,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.chart['start_time']=self.chart['start_time']-self.chart['select_time']
            self.chart=self.chart[self.chart['start_time']>=0]
            
    def smooth_meds(self,bucket):
        final_meds=pd.DataFrame()
        final_proc=pd.DataFrame()
        final_out=pd.DataFrame()
        final_chart=pd.DataFrame()
        final_anti=pd.DataFrame()
        final_vent=pd.DataFrame()
        final_data=pd.DataFrame()
        if(self.feat_med):
            self.meds=self.meds.sort_values(by=['start_time'])
        if(self.feat_proc):
            self.proc=self.proc.sort_values(by=['start_time'])
        if(self.feat_out):
            self.out=self.out.sort_values(by=['start_time'])
        if(self.feat_chart):
            self.chart=self.chart.sort_values(by=['start_time'])
        if(self.feat_anti):
            self.anti=self.anti.sort_values(by=['start_time'])
        if(self.feat_vent):
            self.vent=self.vent.sort_values(by=['start_time'])
        h=0
        t=0
        for i in tqdm(range(48,self.los,bucket)):
            ###DATA
            sub_data=self.data[self.data['mdr_los']!=0]
            sub_data=sub_data[sub_data['mdr_los']<=i+bucket].groupby(['stay_id']).agg({'subject_id':'max','icu_mdr':'max','non_icu_mdr':'max','charttime':'max','intime':'max','outtime':'max','mdr_los':'max'})
            sub_data=sub_data.reset_index()
            sub_data['seq']=int((i+bucket)/bucket -1)
            if final_data.empty:
                final_data=sub_data
            else:
                final_data=pd.concat([final_data,sub_data])
            h=h+1

        for i in tqdm(range(0,self.los,bucket)): 
            ###MEDS
            if(self.feat_med):
                sub_meds=self.meds[(self.meds['start_time']>=i) & (meds['start_time']<i+bucket)].groupby(['stay_id','itemid','orderid']).agg({'start_time':'max','stop_time':'max','subject_id':'max','mdr_los':'min','rate':np.nanmean,'amount':np.nanmean})
                sub_meds=sub_meds.reset_index()
                sub_meds['mdr_label']=np.where((sub_meds['mdr_los']>i)&(sub_meds['mdr_los']<=i+bucket),1,0)
                sub_meds['start_time']=t
                sub_meds['stop_time']=sub_meds['stop_time']/bucket
                sub_meds['seq']=int((i+bucket)/bucket -1)
                if final_meds.empty:
                    final_meds=sub_meds
                else:
                    final_meds=pd.concat([final_meds,sub_meds])
        
            if(self.feat_proc):
                sub_proc=self.proc[(self.proc['start_time']>=i) & (self.proc['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'start_time':'max','subject_id':'max','mdr_los':'min'})
                sub_proc=sub_proc.reset_index()
                sub_proc['mdr_label']=np.where((sub_proc['mdr_los']>i)&(sub_proc['mdr_los']<=i+bucket),1,0)
                sub_proc['start_time']=t
                sub_proc['seq']=int((i+bucket)/bucket -1)
                if final_proc.empty:
                    final_proc=sub_proc
                else:    
                    final_proc=pd.concat([final_proc,sub_proc])
        
            ###ANTI
            if(self.feat_anti):
                sub_anti=self.anti[(self.anti['start_time']>=i) & (self.anti['start_time']<i+bucket)].groupby(['stay_id','antibiotic']).agg({'start_time':'max','subject_id':'max','mdr_los':'min'})
                sub_anti=sub_anti.reset_index()
                sub_anti['mdr_label']=np.where((sub_anti['mdr_los']>i)&(sub_anti['mdr_los']<=i+bucket),1,0)
                sub_anti['start_time']=t
                sub_anti['seq']=int((i+bucket)/bucket -1)
                if final_anti.empty:
                    final_anti=sub_anti
                else:
                    final_anti=pd.concat([final_anti,sub_anti])
        
            ###VENT
            if(self.feat_vent):
                sub_vent=self.vent[(self.vent['start_time']>=i) & (self.vent['start_time']<i+bucket)].groupby(['stay_id','ventilation_status']).agg({'start_time':'max','subject_id':'max','mdr_los':'min'})
                sub_vent=sub_vent.reset_index()
                sub_vent['mdr_label']=np.where((sub_vent['mdr_los']>i)&(sub_vent['mdr_los']<=i+bucket),1,0)
                sub_vent['start_time']=t
                sub_vent['seq']=int((i+bucket)/bucket -1)
                if final_vent.empty:
                    final_vent=sub_vent
                else:
                    final_vent=pd.concat([final_vent,sub_vent])
        
            ###OUT
            if(self.feat_out):
                sub_out=self.out[(self.out['start_time']>=i) & (self.out['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'start_time':'max','subject_id':'max','mdr_los':'min'})
                sub_out=sub_out.reset_index()
                sub_out['start_time']=t
                sub_out['seq']=int((i+bucket)/bucket -1)
                if final_out.empty:
                    final_out=sub_out
                else:    
                    final_out=pd.concat([final_out,sub_out])
                
            ###CHART
            if(self.feat_chart):
                sub_chart=self.chart[(self.chart['start_time']>=i) & (self.chart['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'start_time':'max','mdr_los':'min','valuenum':np.nanmean})
                sub_chart=sub_chart.reset_index()
                sub_chart['start_time']=t
                sub_chart['seq']=int((i+bucket)/bucket -1)
                if final_chart.empty:
                    final_chart=sub_chart
                else:    
                    final_chart=pd.concat([final_chart,sub_chart])
            t=t+1

        print("bucket",bucket)
        los=int(self.los/bucket)
        
        ###MEDS
        if(self.feat_med):
            f2_meds=final_meds.groupby(['stay_id','itemid','orderid']).size()
            self.med_per_adm=f2_meds.groupby('stay_id').sum().reset_index()[0].max()                 
            self.medlength_per_adm=final_meds.groupby('stay_id').size().max()
        
        ###PROC
        if(self.feat_proc):
            f2_proc=final_proc.groupby(['stay_id','itemid']).size()
            self.proc_per_adm=f2_proc.groupby('stay_id').sum().reset_index()[0].max()       
            self.proclength_per_adm=final_proc.groupby('stay_id').size().max()
            
        ###OUT
        if(self.feat_out):
            f2_out=final_out.groupby(['stay_id','itemid']).size()
            self.out_per_adm=f2_out.groupby('stay_id').sum().reset_index()[0].max() 
            self.outlength_per_adm=final_out.groupby('stay_id').size().max()

        ###ANTI
        if(self.feat_anti):
            f2_anti=final_anti.groupby(['stay_id','antibiotic']).size()
            self.anti_per_adm=f2_anti.groupby('stay_id').sum().reset_index()[0].max() 
            self.antilength_per_adm=final_anti.groupby('stay_id').size().max()    
        
        ###VENT
        if(self.feat_vent):
            f2_vent=final_vent.groupby(['stay_id','ventilation_status']).size()
            self.vent_per_adm=f2_vent.groupby('stay_id').sum().reset_index()[0].max() 
            self.ventlength_per_adm=final_vent.groupby('stay_id').size().max()

        ###chart
        if(self.feat_chart):
            f2_chart=final_chart.groupby(['stay_id','itemid']).size()
            self.chart_per_adm=f2_chart.groupby('stay_id').sum().reset_index()[0].max()             
            self.chartlength_per_adm=final_chart.groupby('stay_id').size().max()
        
        print("[ PROCESSED TIME SERIES TO EQUAL TIME INTERVAL ]")

        self.create_Dict(final_data,final_meds,final_proc,final_out,final_anti,final_vent,final_chart,los)
                   
    def create_Dict(self,data,meds,proc,out,anti,vent,chart,los):
        dataDic={}
        print(los)
        labels_csv=pd.DataFrame(columns=['stay_id','icu_mdr'])
        labels_csv['stay_id']=pd.Series(self.hids)
        labels_csv['icu_mdr']=0

        for hid in self.hids:
            grp=self.data[self.data['stay_id']==hid]
            dataDic[hid]={'Cond':{},'Proc':{},'Med':{},'Out':{},'Anti':{},'Vent':{},'Chart':{},'ethnicity':grp['ethnicity'].iloc[0],'age':int(grp['Age']),'gender':grp['gender'].iloc[0],'icu_mdr':int(grp['icu_mdr'])}
            labels_csv.loc[labels_csv['stay_id']==hid,'icu_mdr']=int(grp['icu_mdr'])

        for hid in tqdm(self.hids):
            grp=self.data[self.data['stay_id']==hid]
            demo_csv=grp[['Age','gender','ethnicity','insurance']]
            if not os.path.exists("./data/csv"+str(hid)):
                os.makedirs("./data/csv/"+str(hid))
            demo_csv.to_csv('./data/csv/'+str(hid)+'/demo.csv',index=False)
            
            dyn_csv=pd.DataFrame()
            mdr_csv=pd.DataFrame()

            ###DATA
            df3=data[data['stay_id']==hid]
            if df3.shape[0]==0:
                df3=pd.DataFrame(np.zeros([los,1]))
                df3=df3.fillna(0)
                df3.columns=['mdr_label']
                df3['mdr_label']=df3['mdr_label'].astype(int)
            else:
                df3=df3.pivot_table(index='seq',columns='stay_id',values='icu_mdr')
                df3.columns=['mdr_label']
                add_indices = pd.Index(range(los)).difference(df3.index)
                add_df = pd.DataFrame(index=add_indices, columns=df3.columns).fillna(np.nan)
                df3=pd.concat([df3, add_df])
                df3=df3.sort_index()
                df3=df3.ffill()
                df3=df3.fillna(0)
                df3['mdr_label']=df3['mdr_label'].astype(int)
            if(mdr_csv.empty):
                mdr_csv=df3
            else:
                mdr_csv=pd.concat([mdr_csv,df3],axis=1)

            ###MEDS
            if(self.feat_med):
                feat=meds['itemid'].unique()
                df2=meds[meds['stay_id']==hid]
                if df2.shape[0]==0:
                    amount=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    amount=amount.fillna(0)
                    amount.columns=pd.MultiIndex.from_product([["MEDS"], amount.columns])
                else:
                    rate=df2.pivot_table(index='start_time',columns='itemid',values='rate')
                    amount=df2.pivot_table(index='start_time',columns='itemid',values='amount')
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='stop_time')
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.ffill()
                    df2=df2.fillna(0)

                    rate=pd.concat([rate, add_df])
                    rate=rate.sort_index()
                    rate=rate.ffill()
                    rate=rate.fillna(-1)

                    amount=pd.concat([amount, add_df])
                    amount=amount.sort_index()
                    amount=amount.ffill()
                    amount=amount.fillna(-1)
                    df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0)
                    df2[df2>0]=1
                    df2[df2<0]=0
                    rate.iloc[:,0:]=df2.iloc[:,0:]*rate.iloc[:,0:]
                    amount.iloc[:,0:]=df2.iloc[:,0:]*amount.iloc[:,0:]
                    dataDic[hid]['Med']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Med']['rate']=rate.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Med']['amount']=amount.iloc[:,0:].to_dict(orient="list")


                    feat_df=pd.DataFrame(columns=list(set(feat)-set(amount.columns)))

                    amount=pd.concat([amount,feat_df],axis=1)

                    amount=amount[feat]
                    amount=amount.fillna(0)

                    amount.columns=pd.MultiIndex.from_product([["MEDS"], amount.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=amount
                else:
                    dyn_csv=pd.concat([dyn_csv,amount],axis=1)
                            
            ###PROCS
            if(self.feat_proc):
                feat=proc['itemid'].unique()
                df2=proc[proc['stay_id']==hid]
                if df2.shape[0]==0:
                    df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
                else:
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1
                    dataDic[hid]['Proc']=df2.to_dict(orient="list")


                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=df2
                else:
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)
                                  
            ###OUT
            if(self.feat_out):
                feat=out['itemid'].unique()
                df2=out[out['stay_id']==hid]
                if df2.shape[0]==0:
                    df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["OUT"], df2.columns])
                else:
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1
                    dataDic[hid]['Out']=df2.to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["OUT"], df2.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=df2
                else:
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)
                
            ###ANTI
            if(self.feat_anti): 
                feat=anti['antibiotic'].unique()
                df2=anti[anti['stay_id']==hid]
                if df2.shape[0]==0:
                    df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["ANTI"], df2.columns])
                else:
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='antibiotic',values='val')
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1
                    dataDic[hid]['Anti']=df2.to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["ANTI"], df2.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=df2
                else:   
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)

            ###VENT
            if(self.feat_vent):
                feat=vent['ventilation_status'].unique()
                df2=vent[vent['stay_id']==hid]
                if df2.shape[0]==0:
                    df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["VENT"], df2.columns])
                else:
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='ventilation_status',values='val')
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1
                    dataDic[hid]['Vent']=df2.to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["VENT"], df2.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=df2
                else:
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)
                
            ###CHART
            if(self.feat_chart):
                feat=chart['itemid'].unique()
                df2=chart[chart['stay_id']==hid]
                if df2.shape[0]==0:
                    val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    val=val.fillna(0)
                    val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
                else:
                    val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)

                    val=pd.concat([val, add_df])
                    val=val.sort_index()
                    if self.impute=='Mean':
                        val=val.ffill()
                        val=val.bfill()
                        val=val.fillna(val.mean())
                    elif self.impute=='Median':
                        val=val.ffill()
                        val=val.bfill()
                        val=val.fillna(val.median())
                    val=val.fillna(0)


                    df2[df2>0]=1
                    df2[df2<0]=0
                    dataDic[hid]['Chart']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Chart']['val']=val.iloc[:,0:].to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                    val=pd.concat([val,feat_df],axis=1)

                    val=val[feat]
                    val=val.fillna(0)
                    val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=val
                else:
                    dyn_csv=pd.concat([dyn_csv,val],axis=1)
            
            #Save temporal data to csv
            dyn_csv.to_csv('./data/csv/'+str(hid)+'/dynamic.csv',index=False)
            mdr_csv.to_csv('./data/csv/'+str(hid)+'/mdr.csv',index=False)
            
            ##########COND#########
            if(self.feat_cond):
                feat=self.cond['new_icd_code'].unique()
                grp=self.cond[self.cond['stay_id']==hid]
                if(grp.shape[0]==0):
                    dataDic[hid]['Cond']={'fids':list(['<PAD>'])}
                    feat_df=pd.DataFrame(np.zeros([1,len(feat)]),columns=feat)
                    grp=feat_df.fillna(0)
                    grp.columns=pd.MultiIndex.from_product([["COND"], grp.columns])
                else:
                    dataDic[hid]['Cond']={'fids':list(grp['new_icd_code'])}
                    grp['val']=1
                    grp=grp.drop_duplicates()
                    grp=grp.pivot(index='stay_id',columns='new_icd_code',values='val').reset_index(drop=True)
                    feat_df=pd.DataFrame(columns=list(set(feat)-set(grp.columns)))
                    grp=pd.concat([grp,feat_df],axis=1)
                    grp=grp.fillna(0)
                    grp=grp[feat]
                    grp.columns=pd.MultiIndex.from_product([["COND"], grp.columns])
            grp.to_csv('./data/csv/'+str(hid)+'/static.csv',index=False)   
            labels_csv.to_csv('./data/csv/labels.csv',index=False)    
            
                
        ######SAVE DICTIONARIES##############
        metaDic={'Cond':{},'Proc':{},'Med':{},'Out':{},'Chart':{},'Anti':{},'Vent':{},'LOS':{}}
        metaDic['LOS']=los
        with open("./data/dict/dataDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open("./data/dict/hadmDic", 'wb') as fp:
            pickle.dump(self.hids, fp)
        
        with open("./data/dict/ethVocab", 'wb') as fp:
            pickle.dump(list(self.data['ethnicity'].unique()), fp)
            self.eth_vocab = self.data['ethnicity'].nunique()
            
        with open("./data/dict/ageVocab", 'wb') as fp:
            pickle.dump(list(self.data['Age'].unique()), fp)
            self.age_vocab = self.data['Age'].nunique()
            
        with open("./data/dict/insVocab", 'wb') as fp:
            pickle.dump(list(self.data['insurance'].unique()), fp)
            self.ins_vocab = self.data['insurance'].nunique()
            
        if(self.feat_med):
            with open("./data/dict/medVocab", 'wb') as fp:
                pickle.dump(list(meds['itemid'].unique()), fp)
            self.med_vocab = meds['itemid'].nunique()
            metaDic['Med']=self.med_per_adm
            
        if(self.feat_out):
            with open("./data/dict/outVocab", 'wb') as fp:
                pickle.dump(list(out['itemid'].unique()), fp)
            self.out_vocab = out['itemid'].nunique()
            metaDic['Out']=self.out_per_adm
            
        if(self.feat_chart):
            with open("./data/dict/chartVocab", 'wb') as fp:
                pickle.dump(list(chart['itemid'].unique()), fp)
            self.chart_vocab = chart['itemid'].nunique()
            metaDic['Chart']=self.chart_per_adm
        
        if(self.feat_cond):
            with open("./data/dict/condVocab", 'wb') as fp:
                pickle.dump(list(self.cond['new_icd_code'].unique()), fp)
            self.cond_vocab = self.cond['new_icd_code'].nunique()
            metaDic['Cond']=self.cond_per_adm
        
        if(self.feat_proc):    
            with open("./data/dict/procVocab", 'wb') as fp:
                pickle.dump(list(proc['itemid'].unique()), fp)
            self.proc_vocab = proc['itemid'].nunique()
            metaDic['Proc']=self.proc_per_adm

        if(self.feat_anti):
            with open("./data/dict/antiVocab", 'wb') as fp:
                pickle.dump(list(anti['antibiotic'].unique()), fp)
            self.anti_vocab = anti['antibiotic'].nunique()
            metaDic['Anti']=self.anti_per_adm
        
        if(self.feat_vent):
            with open("./data/dict/ventVocab", 'wb') as fp:
                pickle.dump(list(vent['ventilation_status'].unique()), fp)
            self.vent_vocab = vent['ventilation_status'].nunique()
            metaDic['Vent']=self.vent_per_adm
            
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
            