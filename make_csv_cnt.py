# 환자별 설문조사 항목에 몇번을 얼마나 체크했는지를 엑셀로 변환하는 코드
import pandas as pd

file_path = 'raw_data1.csv'
data = pd.read_csv(file_path)

survey_columns = ['menti_seq', 'check1_value', 'check2_value', 'check3_value', 'check4_value', 'check5_value', 'check6_value',
                  'service1', 'service2', 'service3', 'service4', 'service5', 'service6', 'service7', 'service8',
                  'service9', 'service10', 'service11', 'service12', 'service13', 'service14', 'service15', 'service16']
survey_data = data[survey_columns]

columns = [
    'check1_val_0', 'check1_val_1', 'check1_val_2',
    'check2_val_0', 'check2_val_1',
    'check3_val_0', 'check3_val_1',
    'check4_val_0', 'check4_val_1',
    'check5_val_1', 'check5_val_2', 'check5_val_3',
    'check6_val_0', 'check6_val_1', 'check6_val_2', 'check6_val_3',
    'service1_0', 'service1_1',
    'service2_0', 'service2_1',
    'service3_0', 'service3_1',
    'service4_0', 'service4_1',
    'service5_0', 'service5_1',
    'service6_0', 'service6_1',
    'service7_0', 'service7_1',
    'service8_0', 'service8_1',
    'service9_0', 'service9_1',
    'service10_0', 'service10_1',
    'service11_0', 'service11_1',
    'service12_0', 'service12_1',
    'service13_0', 'service13_1',
    'service14_0', 'service14_1',
    'service15_0', 'service15_1',
    'service16_0', 'service16_1',
]
result = pd.DataFrame(columns=columns)

for menti_seq, group in survey_data.groupby('menti_seq'):
    counts = {}
    counts['check1_val_0'] = (group['check1_value'] == 0).sum()
    counts['check1_val_1'] = (group['check1_value'] == 1).sum()
    counts['check1_val_2'] = (group['check1_value'] == 2).sum()
    counts['check2_val_0'] = (group['check2_value'] == 0).sum()
    counts['check2_val_1'] = (group['check2_value'] == 1).sum()
    counts['check3_val_0'] = (group['check3_value'] == 0).sum()
    counts['check3_val_1'] = (group['check3_value'] == 1).sum()
    counts['check4_val_0'] = (group['check4_value'] == 0).sum()
    counts['check4_val_1'] = (group['check4_value'] == 1).sum()
    counts['check5_val_1'] = (group['check5_value'] == 1).sum()
    counts['check5_val_2'] = (group['check5_value'] == 2).sum()
    counts['check5_val_3'] = (group['check5_value'] == 3).sum()
    counts['check6_val_0'] = (group['check6_value'] == 0).sum()
    counts['check6_val_1'] = (group['check6_value'] == 1).sum()
    counts['check6_val_2'] = (group['check6_value'] == 2).sum()
    counts['check6_val_3'] = (group['check6_value'] == 3).sum()
    counts['service1_0'] = (group['service1'] == 0).sum()
    counts['service1_1'] = (group['service1'] == 1).sum()
    counts['service2_0'] = (group['service2'] == 0).sum()
    counts['service2_1'] = (group['service2'] == 1).sum()
    counts['service3_0'] = (group['service3'] == 0).sum()
    counts['service3_1'] = (group['service3'] == 1).sum()
    counts['service4_0'] = (group['service4'] == 0).sum()
    counts['service4_1'] = (group['service4'] == 1).sum()
    counts['service5_0'] = (group['service5'] == 0).sum()
    counts['service5_1'] = (group['service5'] == 1).sum()
    counts['service6_0'] = (group['service6'] == 0).sum()
    counts['service6_1'] = (group['service6'] == 1).sum()
    counts['service7_0'] = (group['service7'] == 0).sum()
    counts['service7_1'] = (group['service7'] == 1).sum()
    counts['service8_0'] = (group['service8'] == 0).sum()
    counts['service8_1'] = (group['service8'] == 1).sum()
    counts['service9_0'] = (group['service9'] == 0).sum()
    counts['service9_1'] = (group['service9'] == 1).sum()
    counts['service10_0'] = (group['service10'] == 0).sum()
    counts['service10_1'] = (group['service10'] == 1).sum()
    counts['service11_0'] = (group['service11'] == 0).sum()
    counts['service11_1'] = (group['service11'] == 1).sum()
    counts['service12_0'] = (group['service12'] == 0).sum()
    counts['service12_1'] = (group['service12'] == 1).sum()
    counts['service13_0'] = (group['service13'] == 0).sum()
    counts['service13_1'] = (group['service13'] == 1).sum()
    counts['service14_0'] = (group['service14'] == 0).sum()
    counts['service14_1'] = (group['service14'] == 1).sum()
    counts['service15_0'] = (group['service15'] == 0).sum()
    counts['service15_1'] = (group['service15'] == 1).sum()
    counts['service16_0'] = (group['service16'] == 0).sum()
    counts['service16_1'] = (group['service16'] == 1).sum()
    
    result = pd.concat([result, pd.DataFrame(counts, index=[menti_seq])])

output_file_path = 'survey_val_counts.csv'
result.to_csv(output_file_path, index_label='menti_seq')