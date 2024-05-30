import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로 설정
file1_path = "phq_results.csv"  # menti_seq, Predicted, Actual 파일 경로
file2_path = "survey_val_counts.csv"  # menti_seq, service1, ..., service16 파일 경로

# 데이터 불러오기
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# 두 데이터프레임 합치기
merged_df = pd.merge(df1, df2, on="menti_seq")

# 실제 값을 기준으로 정상과 비정상(PHQ-9) 그룹으로 나누기
normal_df = merged_df[merged_df['Actual'] == 0]
phq9_df = merged_df[merged_df['Actual'] == 1]

# 서비스 항목들만 선택하여 새로운 데이터프레임 생성
service_columns = [col for col in merged_df.columns if 'service' in col]

# 서비스별 0과 1의 비율 계산
normal_counts = normal_df[service_columns].apply(pd.Series.value_counts, normalize=True).fillna(0).stack().unstack(level=1)
phq9_counts = phq9_df[service_columns].apply(pd.Series.value_counts, normalize=True).fillna(0).stack().unstack(level=1)

# 서비스별로 0과 1에 대한 열 이름 변경
normal_counts.columns = [f"{col}_Normal" for col in normal_counts.columns]
phq9_counts.columns = [f"{col}_PHQ-9" for col in phq9_counts.columns]

# 항목 이름 매핑
column_names_mapping = {
    "service1": "일상생활지원 물품지원",
    "service2": "일상생활지원 식사지원",
    "service3": "일상생활지원 가사지원",
    "service4": "일상생활지원 위생지원",
    "service5": "활동 및 프로그램지원 병원동행",
    "service6": "활동 및 프로그램지원 차량지원",
    "service7": "활동 및 프로그램지원 여가활동",
    "service8": "활동 및 프로그램지원",
    "service9": "정서지원 안부인사",
    "service10": "정서지원 지지상담",
    "service11": "정서지원 응급개입",
    "service12": "정서지원 센터연계",
    "service13": "정서지원 의료기관연계",
    "service14": "정서지원 복지서비스연계",
    "service15": "정서지원 민간지원연계",
    "service16": "정서지원 기타서비스내용"
}

# 컬럼 이름 변경
normal_counts.index = [f"{column_names_mapping[col]}" for col in normal_counts.index]
phq9_counts.index = [f"{column_names_mapping[col]}" for col in phq9_counts.index]

# 필요한 컬럼만 선택하여 새로운 데이터프레임 생성
combined_df = pd.concat([normal_counts, phq9_counts], axis=1).stack().unstack(level=1)

# 그래프 생성
fig, ax = plt.subplots(figsize=(14, 10))

width = 0.35  # bar width
x = np.arange(len(combined_df.index))

bars1 = ax.bar(x - width/2, combined_df['0_Normal'], width, label='Normal - 지원 안함', color='skyblue')
bars2 = ax.bar(x + width/2, combined_df['0_PHQ-9'], width, label='PHQ-9 - 지원 안함', color='coral')
bars3 = ax.bar(x - width/2, combined_df['1_Normal'], width, bottom=combined_df['0_Normal'], color='blue', label='Normal - 지원 함')
bars4 = ax.bar(x + width/2, combined_df['1_PHQ-9'], width, bottom=combined_df['0_PHQ-9'], color='red', label='PHQ-9 - 지원 함')

# 제목과 레이블 설정
ax.set_title('Service Support Distribution')
ax.set_xlabel('Service Category')
ax.set_ylabel('Proportion')
ax.set_xticks(x)
ax.set_xticklabels(combined_df.index, rotation=45, ha='right')
ax.legend()

# 바 라벨 추가
def add_labels(bars):
    for bar in bars:
        height = bar.get_height() + bar.get_y()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

plt.tight_layout()
plt.show()
