# 환자별 설문 총 횟수를 엑셀로 변환하는 코드
import pandas as pd

data1_path = 'raw_data1.csv'
data2_path = 'raw_data2.csv'

data1 = pd.read_csv(data1_path, sep=",", encoding="utf-8")
data2 = pd.read_csv(data2_path, sep=",", encoding="utf-8")

data1_cnt = data1['menti_seq'].value_counts() #data1의 환자별 설문조사 횟수를 가져옴
data1_idx = sorted(set(data1['menti_seq'].index))

data2_cnt = data2['menti_seq'].value_counts() #data2의 환자별 phq-9, p4, loneliness 설문 횟수를 각각 가져옴
data2_cnt_phq = data2[(data2['srvy_name'] == 'PHQ-9')]['menti_seq'].value_counts()
data2_cnt_p4 = data2[(data2['srvy_name'] == 'P4')]['menti_seq'].value_counts()
data2_cnt_lone = data2[(data2['srvy_name'] == 'Loneliness')]['menti_seq'].value_counts()
data2_idx = sorted(set(data2['menti_seq'].index))

# 테이블 제작
table1 = pd.DataFrame({
    'menti_seq': data1_idx,
    'srvy_cnt': [data1_cnt.get(i, 0) for i in data1_idx]
})

table2 = pd.DataFrame({
    'menti_seq': data2_idx,
    'PHQ-9': [data2_cnt_phq.get(i, 0) for i in data2_idx],
    'P4': [data2_cnt_p4.get(i, 0) for i in data2_idx],
    'Loneliness': [data2_cnt_lone.get(i, 0) for i in data2_idx],
})

# 0인 행 제거
table1 = table1[table1['srvy_cnt'] != 0]
table2 = table2[(table2['PHQ-9'] != 0) | (table2['P4'] != 0) | (table2['Loneliness'] != 0)]

# 엑셀로 변환
with pd.ExcelWriter('survey_all.xlsx') as writer:
    table1.to_excel(writer, sheet_name='raw_data1', index=False)
    table2.to_excel(writer, sheet_name='raw_data2', index=False)