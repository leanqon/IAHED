import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
import pickle
import torch
import pandas as pd

def loading_data_all(data_name, med_flag, chart_flag, out_flag, proc_flag, lab_flag, anti_flag, vent_flag, stat_flag, demo_flag, y_flag, device):
    if med_flag:
        med_all = torch.load(f"./data/tensor{data_name}/meds.pth")
        med = med_all.clone().detach().to(device)
    else:
        med = torch.zeros(size=(0, 0)).to(device)
    if chart_flag:
        chart_all = torch.load(f"./data/tensor{data_name}/chart.pth")
        chart = chart_all.clone().detach().to(device)
    else:
        chart = torch.zeros(size=(0, 0)).to(device)
    if out_flag:
        out_all = torch.load(f"./data/tensor{data_name}/out.pth")
        out = out_all.clone().detach().to(device)
    else:
        out = torch.zeros(size=(0, 0)).to(device)
    if proc_flag:
        proc_all = torch.load(f"./data/tensor{data_name}/proc.pth")
        proc = proc_all.clone().detach().to(device)
    else:
        proc = torch.zeros(size=(0, 0)).to(device)
    if lab_flag:
        lab_all = torch.load(f"./data/tensor{data_name}/lab.pth")
        lab = lab_all.clone().detach().to(device)
    else:
        lab = torch.zeros(size=(0, 0)).to(device)
    if anti_flag:
        anti_all = torch.load(f"./data/tensor{data_name}/anti.pth")
        anti = anti_all.clone().detach().to(device)
    else:
        anti = torch.zeros(size=(0, 0)).to(device)
    if vent_flag:
        vent_all = torch.load(f"./data/tensor{data_name}/vent.pth")
        vent = vent_all.clone().detach().to(device)
    else:
        vent = torch.zeros(size=(0, 0)).to(device)
    if stat_flag:
        stat_all = torch.load(f"./data/tensor{data_name}/stat.pth")
        stat = stat_all.clone().detach().to(device)
    else:
        stat = torch.zeros(size=(0, 0)).to(device)
    if demo_flag:
        demo_all = torch.load(f"./data/tensor{data_name}/demo.pth")
        demo = demo_all.clone().detach().to(device)
    else:
        demo = torch.zeros(size=(0, 0)).to(device)
    if y_flag:
        Y_all = torch.load(f"./data/tensor{data_name}/Y.pth")
        y = Y_all.clone().detach().to(device)
    else:
        y = torch.zeros(size=(0, 0)).to(device)
    stat = stat.unsqueeze(1).expand(-1, chart.shape[1], -1)
    demo = demo.unsqueeze(1).expand(-1, chart.shape[1], -1)
    return chart, anti, vent, stat, demo, y

def init_read(data_name, diag_flag, proc_flag, out_flag, chart_flag, med_flag, lab_flag, anti_flag, vent_flag):
    condVocabDict = {}
    procVocabDict = {}
    medVocabDict = {}
    outVocabDict = {}
    chartVocabDict = {}
    labVocabDict = {}
    ethVocabDict = {}
    ageVocabDict = {}
    genderVocabDict = {}
    insVocabDict = {}
    antiVocabDict = {}
    ventVocabDict = {}

    with open(f'./data/dict{data_name}/' + 'ethVocabDict', 'rb') as fp:
        ethVocabDict = pickle.load(fp)

    with open(f'./data/dict{data_name}/' + 'ageVocabDict', 'rb') as fp:
        ageVocabDict = pickle.load(fp)

    with open(f'./data/dict{data_name}/' + 'genderVocabDict', 'rb') as fp:
        genderVocabDict = pickle.load(fp)

    with open(f'./data/dict{data_name}/' + 'insVocabDict', 'rb') as fp:
        insVocabDict = pickle.load(fp)

    if diag_flag:
        file = 'condVocab'
        with open(f'./data/dict{data_name}/' + file, 'rb') as fp:
            condVocabDict = pickle.load(fp)
    if proc_flag:
        file = 'procVocab'
        with open(f'./data/dict{data_name}/' + file, 'rb') as fp:
            procVocabDict = pickle.load(fp)
    if med_flag:
        file = 'medVocab'
        with open(f'./data/dict{data_name}/' + file, 'rb') as fp:
            medVocabDict = pickle.load(fp)
    if out_flag:
        file = 'outVocab'
        with open(f'./data/dict{data_name}/' + file, 'rb') as fp:
            outVocabDict = pickle.load(fp)
    if chart_flag:
        file = 'chartVocab'
        with open(f'./data/dict{data_name}/' + file, 'rb') as fp:
            chartVocabDict = pickle.load(fp)
    if lab_flag:
        file = 'labsVocab'
        with open(f'./data/dict{data_name}/' + file, 'rb') as fp:
            labVocabDict = pickle.load(fp)
    if anti_flag:
        file = 'antiVocab'
        with open(f'./data/dict{data_name}/' + file, 'rb') as fp:
            antiVocabDict = pickle.load(fp)
    if vent_flag:
        file = 'ventVocab'
        with open(f'./data/dict{data_name}/' + file, 'rb') as fp:
            ventVocabDict = pickle.load(fp)

    return len(condVocabDict), len(procVocabDict), len(medVocabDict), len(outVocabDict), len(chartVocabDict), len(
        labVocabDict), len(antiVocabDict), len(ventVocabDict), len(ethVocabDict), len(genderVocabDict), len(
        insVocabDict), condVocabDict, procVocabDict, medVocabDict, outVocabDict, chartVocabDict, labVocabDict, antiVocabDict, ventVocabDict, ethVocabDict, genderVocabDict, ageVocabDict, insVocabDict

def restore_tensor(input_tensor, category_dict):
    num_categories = len(category_dict)
    num_batch = input_tensor.shape[0]
    output_tensor = torch.zeros(num_batch, args.window, num_categories).to(input_tensor.device)   
    for category_name,category_id in category_dict.items():
        category_mask = (input_tensor.squeeze(dim=2) == category_id).float()
        output_tensor[:, :, category_id] = category_mask
    return output_tensor

data_name = "_168_12_2" #"_336_24_2" 
diag_flag = True
proc_flag = False
out_flag = False
chart_flag = True
med_flag = False
lab_flag = False
anti_flag = True
vent_flag = True
stat_flag = True
demo_flag = True
y_flag = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

chart, anti, vent, stat, demo, y = loading_data_all(data_name, med_flag, chart_flag, out_flag, proc_flag, lab_flag, anti_flag, vent_flag, stat_flag, demo_flag, y_flag, device)
cond_vocab_len, proc_vocab_len, med_vocab_len, out_vocab_len, chart_vocab_len, lab_vocab_len, anti_vocab_len, vent_vocab_len, eth_vocab_len, gender_vocab_len, ins_vocab_len, cond_vocab_dict, proc_vocab_dict, med_vocab_dict, out_vocab_dict, chart_vocab_dict, lab_vocab_dict, anti_vocab_dict, vent_vocab_dict, eth_vocab_dict, gender_vocab_dict, age_vocab_dict, ins_vocab_dict = init_read(data_name, diag_flag, proc_flag, out_flag, chart_flag, med_flag, lab_flag, anti_flag, vent_flag)

print("condVocabDict length:", cond_vocab_len)
print("procVocabDict length:", proc_vocab_len)
print("medVocabDict length:", med_vocab_len)
print("outVocabDict length:", out_vocab_len)
print("chartVocabDict length:", chart_vocab_len)
print("labVocabDict length:", lab_vocab_len)
print("antiVocabDict length:", anti_vocab_len)
print("ventVocabDict length:", vent_vocab_len)
print("ethVocabDict length:", eth_vocab_len)
print("genderVocabDict length:", gender_vocab_len)
print("insVocabDict length:", ins_vocab_len)

gender=demo[:,0,0]
eth=demo[:,0,1]
age=demo[:,0,3]
ins=demo[:,0,2]
stat=stat[:,0,:]

reversed_age_dict = {v: k for k, v in age_vocab_dict.items()}
age_groups = {
    "less_than_18": 0,
    "18_to_30": 0,
    "31_to_60": 0,
    "61_and_above": 0
}

for encoded_age in age:
    actual_age = reversed_age_dict.get(encoded_age.item(), None)
    
    if actual_age is not None:
        if actual_age < 18:
            age_groups["less_than_18"] += 1
        elif 18 <= actual_age <= 30:
            age_groups["18_to_30"] += 1
        elif 31 <= actual_age <= 60:
            age_groups["31_to_60"] += 1
        else:
            age_groups["61_and_above"] += 1

for group, count in age_groups.items():
    print(f"{group}: {count}")

actual_ages = [reversed_age_dict[encoded_age.item()] for encoded_age in age if encoded_age.item() in reversed_age_dict]
actual_ages_tensor = torch.tensor(actual_ages, dtype=torch.float)

# Calculate mean and standard deviation
age_mean = torch.mean(actual_ages_tensor).item()
age_std = torch.std(actual_ages_tensor, unbiased=True).item()  # 'unbiased=True' uses (N-1) in the denominator

print(f"Mean age: {age_mean:.2f}")
print(f"Age SD: {age_std:.2f}")

def categorize_and_count(tensor, vocab_dict, categories):
    reversed_dict = {v: k for k, v in vocab_dict.items()}
    category_counts = {category: 0 for category in categories}
    total_count = 0

    for encoded_value in tensor:
        actual_value = reversed_dict.get(encoded_value.item(), None)

        for category, values in categories.items():
            if actual_value in values:
                category_counts[category] += 1
                total_count += 1
                break

    category_percentages = {category: count / total_count * 100 for category, count in category_counts.items()}
    return category_counts, category_percentages

def transform_dict(input_dict):
    output_dict = {}
    for key in input_dict.keys():
        output_dict[key] = [key]
    return output_dict

gender_categories = transform_dict(gender_vocab_dict)
gender_counts, gender_percentages = categorize_and_count(gender, gender_vocab_dict, gender_categories)
print(gender_counts)
print(gender_percentages)

ins_categories = transform_dict(ins_vocab_dict)
ins_counts, ins_percentages = categorize_and_count(ins, ins_vocab_dict, ins_categories)
print(ins_counts)
print(ins_percentages)

eth_categories = transform_dict(eth_vocab_dict)
eth_counts, eth_percentages = categorize_and_count(eth, eth_vocab_dict, eth_categories)
print(eth_counts)
print(eth_percentages)

def create_summary_table(counts_dict, percentages_dict):
    data = {}
    for category in counts_dict.keys():
        data[category] = {
            'Count': counts_dict[category],
            'Percentage': percentages_dict[category]
        }
    df = pd.DataFrame(data).transpose()
    return df

dfs = []
dfs.append(create_summary_table(gender_counts, gender_percentages))
dfs.append(create_summary_table(ins_counts, ins_percentages))
dfs.append(create_summary_table(eth_counts, eth_percentages))
df = pd.concat(dfs)
print(df)

condition_categories_old = {
    'Myocardial infarction': ['410', '412', 'I21', 'I22', 'I252'],
    'Congestive heart failure': ['428', '39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4254', '4259', 'I43', 'I50', 'I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'P290'],
    'Peripheral vascular disease': ['440', '441', '0930', '4373', '4471', '5571', '5579', 'V434', '4431', '4439', 'I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959'],
    'Cerebrovascular disease': ['430', '438', '36234', 'G45', 'G46', 'I60', 'I69', 'H340'],
    'Dementia': ['290', '2941', '3312', 'F00', 'F01', 'F02', 'F03', 'G30', 'F051', 'G311'],
    'Chronic pulmonary disease': ['490', '505', '4168', '4169', '5064', '5081', '5088', 'J40', 'J47', 'J60', 'J67', 'I278', 'I279', 'J684', 'J701', 'J703'],
    'Rheumatic disease': ['725', '4465', '7100', '7101', '7102', '7103', '7104', '7140', '7141', '7142', '7148', 'M05', 'M06', 'M32', 'M33', 'M34', 'M315', 'M351', 'M353', 'M360'],
    'Peptic ulcer disease': ['531', '532', '533', '534', 'K25', 'K26', 'K27', 'K28'],
    'Mild liver disease': ['570', '571', '0706', '0709', '5733', '5734', '5738', '5739', 'V427', '07022', '07023', '07032', '07033', '07044', '07054', 'B18', 'K73', 'K74', 'K700', 'K701', 'K702', 'K703', 'K709', 'K713', 'K714', 'K715', 'K717', 'K760', 'K762', 'K763', 'K764', 'K768', 'K769', 'Z944'],
    'Diabetes without chronic complication': ['E10','E11','2500', '2501', '2502', '2503', '2508', '2509', 'E100', 'E10l', 'E106', 'E108', 'E109', 'E110', 'E111', 'E116', 'E118', 'E119', 'E120', 'E121', 'E126', 'E128', 'E129', 'E130', 'E131', 'E136', 'E138', 'E139', 'E140', 'E141', 'E146', 'E148', 'E149'],
    'Diabetes with chronic complication': ['E12','E13','2504', '2505', '2506', '2507', 'E102', 'E103', 'E104', 'E105', 'E107', 'E112', 'E113', 'E114', 'E115', 'E117', 'E122', 'E123', 'E124', 'E125', 'E127', 'E132', 'E133', 'E134', 'E135', 'E137', 'E142', 'E143', 'E144', 'E145', 'E147'],
    'Hemiplegia or paraplegia': ['342', '343', '3341', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449', 'G81', 'G82', 'G041', 'G114', 'G801', 'G802', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839'],
    'Renal disease': ['582', '585', '586', 'V56', '5880', 'V420', 'V451', '5830', '5837', '40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493', 'N18', 'N19', 'I120', 'I131', 'N032', 'N033', 'N034', 'N035', 'N036', 'N037', 'N052', 'N053', 'N054', 'N055', 'N056', 'N057', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992'],
    'Malignant cancer, including lymphoma and leukemia, except malignant neoplasm of skin': ['140', '172', '1740', '1958', '200', '208', '2386', 'C43', 'C88', 'C00', 'C26', 'C30', 'C34', 'C37', 'C41', 'C45', 'C58', 'C60', 'C76', 'C81', 'C85', 'C90', 'C97'],
    'Severe liver disease': ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77', 'K83', 'K76.0', 'K76.1', 'K76.8','4560', '4561', '4562', '5722', '5728', 'I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765', 'K766', 'K767'],
    'Metastatic solid tumor': ['196', '197', '198', '199', 'C77', 'C78', 'C79', 'C80'],
    'AIDS/HIV': ['042', '043', '044', 'B20', 'B21', 'B22', 'B24']
}

condition_categories = {
    'Myocardial infarction': ['410', '412', 'I21', 'I22', 'I252', 'I20', 'I23', 'I24', 'I25'],
    'Congestive heart failure': ['I50', '428', '39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4254', '4259', 'I43', 'I50', 'I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'P290'],
    'Peripheral vascular disease': ['I72', 'I73', 'I74', 'I77', 'I79', '440', '441', '0930', '4373', '4471', '5571', '5579', 'V434', '4431', '4439', 'I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959'],
    'Cerebrovascular disease': ['I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', '430', '438', '36234', 'G45', 'G46', 'I60', 'I69', 'H340'],
    'Dementia': ['F04', 'F05', 'F06', 'F07', 'F08', 'F09', '290', '2941', '3312', 'F00', 'F01', 'F02', 'F03', 'G30', 'F051', 'G311'],
    'Chronic pulmonary disease': ['J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67', 'J68', 'J69', 'J70', 'J71', 'J72', 'J73', 'J74', 'J75', 'J76', 'J77', 'J78', 'J79', '490', '505', '4168', '4169', '5064', '5081', '5088', 'J40', 'J47', 'J60', 'J67', 'I278', 'I279', 'J684', 'J701', 'J703'],
    'Rheumatic disease': ['M30', 'M31', 'M35', 'M36', 'M45', 'M46', 'M47', 'M48', 'M49', 'M50', 'M51', 'M52', 'M53', 'M54', 'M60', 'M61', 'M62', 'M70', 'M71', 'M72', 'M73', 'M74', 'M75', 'M76', 'M77', 'M78', 'M79', 'M80', 'M81', 'M82', 'M83', 'M84', 'M85', 'M86', 'M87', 'M88', 'M89', '725', '4465', '7100', '7101', '7102', '7103', '7104', '7140', '7141', '7142', '7148', 'M05', 'M06', 'M32', 'M33', 'M34', 'M315', 'M351', 'M353', 'M360'],
    'Peptic ulcer disease': ['K29', 'K30', 'K31', 'K58', 'K59', 'K62', 'K63', 'K64', 'K65', 'K66', 'K67', 'K68', 'K69', 'K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77', 'K80', 'K81', 'K82', 'K83', 'K84', 'K85', 'K86', 'K87', 'K88', 'K90', 'K91', 'K92', 'K93', 'K94', 'K95', '531', '532', '533', '534', 'K25', 'K26', 'K27', 'K28'],
    'Mild liver disease': ['B18', 'B19', 'B95', 'B96', '570', '571', '0706', '0709', '5733', '5734', '5738', '5739', 'V427', '07022', '07023', '07032', '07033', '07044', '07054', 'B18', 'K73', 'K74', 'K700', 'K701', 'K702', 'K703', 'K709', 'K713', 'K714', 'K715', 'K717', 'K760', 'K762', 'K763', 'K764', 'K768', 'K769', 'Z944'],
    'Diabetes without chronic complication': ['E10', 'E11', 'E12', 'E13', '2500', '2501', '2502', '2503', '2508', '2509', 'E100', 'E10l', 'E106', 'E108', 'E109', 'E110', 'E111', 'E116', 'E118', 'E119', 'E120', 'E121', 'E126', 'E128', 'E129', 'E130', 'E131', 'E136', 'E138', 'E139', 'E140', 'E141', 'E146', 'E148', 'E149'],
    'Diabetes with chronic complication': ['E10.0', 'E10.1', 'E10.2', 'E10.3', 'E10.4', 'E10.5', 'E10.6', 'E10.7', 'E10.8', 'E10.9', 'E11.0', 'E11.1', 'E11.2', 'E11.3', 'E11.4', 'E11.5', 'E11.6', 'E11.7', 'E11.8', 'E11.9', 'E12','E13','2504', '2505', '2506', '2507', 'E102', 'E103', 'E104', 'E105', 'E107', 'E114', 'E115', 'E117', 'E122', 'E123', 'E124', 'E125', 'E127', 'E132', 'E133', 'E134', 'E135', 'E137', 'E142', 'E143', 'E144', 'E145', 'E147'],
    'Hemiplegia or paraplegia': ['G81', 'G82', '342', '343', '3341', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449', 'G041', 'G114', 'G801', 'G802', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839'],
    'Renal disease': ['N17', 'N28', 'N29', 'N30', 'N31', 'N38', 'N39', 'N40', 'N41', 'N42', 'N43', 'N44', 'N45', 'N46', 'N47', 'N48', 'N49', 'N72', 'N73', 'N80', 'N81', 'N82', 'N83', 'N84', 'N90', 'N91', 'N92', 'N93', 'N99', '582', '585', '586', 'V56', '5880', 'V420', 'V451', '5830', '5837', '40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493', 'N18', 'N19', 'I120', 'I131', 'N032', 'N033', 'N034', 'N035', 'N036', 'N037', 'N052', 'N053', 'N054', 'N055', 'N056', 'N057', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992'],
    'Malignant cancer, including lymphoma and leukemia, except malignant neoplasm of skin': ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C59', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C77', 'C78', 'C79', 'C80', 'C81', 'C82', 'C83', 'C84', 'C85', 'C86', 'C87', 'C88', 'C89', 'C90', 'C91', 'C92', 'C93', 'C94', 'C95', 'C96', 'C97', 'C98', 'C99', '140', '172', '1740', '1958', '200', '208', '2386'],
    'Severe liver disease': ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77', 'K78', 'K79', 'K80', 'K81', 'K82', 'K83', 'K84', 'K85', 'K86', 'K87', 'K88', 'K89', 'K90', 'K91', 'K92', 'K93', 'K94', 'K95', 'K96', 'K97', 'K98', 'K99', 'K76.0', 'K76.1', 'K76.8', '4560', '4561', '4562', '5722', '5728', 'I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765', 'K766', 'K767'],
    'Metastatic solid tumor': ['C76', 'C77', 'C78', 'C79', 'C80', 'C81', 'C82', 'C83', 'C84', 'C85', 'C86', 'C87', 'C88', 'C89', 'C90', 'C91', 'C92', 'C93', 'C94', 'C95', 'C96', 'C97', 'C98', 'C99', '196', '197', '198', '199'],
    'AIDS/HIV': ['B20', 'B21', 'B22', 'B23', 'B24', '042', '043', '044']
}

def calculate_category_counts(stat, cond_vocab_dict, condition_categories):
    category_counts = {category: 0 for category in condition_categories}
    total_count = len(stat)  # Update total_count to be the length of stat

    index_to_category = {}
    for category, conditions in condition_categories.items():
        for condition in conditions:
            if condition in cond_vocab_dict:
                index = cond_vocab_dict.index(condition)
                index_to_category[index] = category

    patient_conditions = set()  # Track the conditions already encountered in a patient

    for patient in stat:
        patient_conditions.clear()  # Clear the set for each patient
        for index, has_condition in enumerate(patient):
            if has_condition.item() == 1 and index in index_to_category:
                category = index_to_category[index]
                if category not in patient_conditions:
                    category_counts[category] += 1
                    patient_conditions.add(category)

    category_percentages = {category: count / total_count * 100 for category, count in category_counts.items()}

    return category_counts, category_percentages

# Use the function
cond_counts, cond_percentages = calculate_category_counts(stat, cond_vocab_dict, condition_categories)

# Output the results
#print("Counts:", category_counts)
#print("Percentages:", category_percentages)
dfs.append(create_summary_table(cond_counts, cond_percentages))
df = pd.concat(dfs)
df = df.round().astype(int)  # Round all columns and convert to integer type
print(df)
df.to_csv(f"./data/sparse/data{data_name}/df{data_name}.csv", index=True)