import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
raw_data1 = pd.read_csv('raw_data1.csv')
raw_data2 = pd.read_csv('raw_data2.csv')
"""
# Filter 'PHQ-9' data
phq9_data = raw_data2[raw_data2['srvy_name'] == 'PHQ-9']

# PHQ-9 test counts per patient
phq9_test_counts = phq9_data['menti_seq'].value_counts()

# Total mentions in raw_data1
total_mention_counts = raw_data1['menti_seq'].value_counts()

# Plot PHQ-9 test count distribution
plt.figure(figsize=(10, 6))
plt.hist(phq9_test_counts, bins=10, color='skyblue', edgecolor='black', range=(0, 10))
plt.title('PHQ-9 Test Counts Per Patient (0-10)')
plt.xlabel('Number of PHQ-9 Tests')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Plot total mention count distribution (0-10)
plt.figure(figsize=(10, 6))
plt.hist(total_mention_counts, bins=10, color='orange', edgecolor='black', range=(0, 20))
plt.title('Total Mentions Per Patient in raw_data1 (0-10)')
plt.xlabel('Number of Mentions')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Plot total mention count distribution (0-50)
plt.figure(figsize=(10, 6))
plt.hist(total_mention_counts, bins=30, color='green', edgecolor='black', range=(0, 50))
plt.title('Total Mentions Per Patient in raw_data1 (20-50)')
plt.xlabel('Number of Mentions')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
"""
# Filter and clip 'PHQ-9' data
phq9_data = raw_data2[raw_data2['srvy_name'] == 'PHQ-9']
phq9_data['srvy_result'] = phq9_data['srvy_result'].clip(upper=1)  # Clip values to 1 if they are greater than 1

# Define criteria functions
def has_disease(df):
    return df['srvy_result'].iloc[0] >= 1 and df['srvy_result'].iloc[-1] >= 1

def avg_disease(df, threshold):
    return df['srvy_result'].mean() >= threshold

# Group patients based on the criteria
grouped = phq9_data.groupby('menti_seq')
patients_start_end_disease = grouped.apply(has_disease)
patients_avg_disease_1 = grouped.apply(lambda df: avg_disease(df, 1.0))
patients_avg_disease_09 = grouped.apply(lambda df: avg_disease(df, 0.9))
patients_avg_disease_08 = grouped.apply(lambda df: avg_disease(df, 0.8))
patients_avg_disease_07 = grouped.apply(lambda df: avg_disease(df, 0.7))
patients_avg_disease_06 = grouped.apply(lambda df: avg_disease(df, 0.6))
patients_avg_disease_05 = grouped.apply(lambda df: avg_disease(df, 0.5))
patients_avg_disease_04 = grouped.apply(lambda df: avg_disease(df, 0.4))

# Classify patients based on start and end criteria
normal_start_end = patients_start_end_disease[~patients_start_end_disease].index
abnormal_start_end = patients_start_end_disease[patients_start_end_disease].index

# Classify patients based on average criteria thresholds
normal_avg_1 = patients_avg_disease_1[~patients_avg_disease_1].index
abnormal_avg_1 = patients_avg_disease_1[patients_avg_disease_1].index

normal_avg_09 = patients_avg_disease_09[~patients_avg_disease_09].index
abnormal_avg_09 = patients_avg_disease_09[patients_avg_disease_09].index

normal_avg_08 = patients_avg_disease_08[~patients_avg_disease_08].index
abnormal_avg_08 = patients_avg_disease_08[patients_avg_disease_08].index

normal_avg_07 = patients_avg_disease_07[~patients_avg_disease_07].index
abnormal_avg_07 = patients_avg_disease_07[patients_avg_disease_07].index

normal_avg_06 = patients_avg_disease_06[~patients_avg_disease_06].index
abnormal_avg_06 = patients_avg_disease_06[patients_avg_disease_06].index

normal_avg_05 = patients_avg_disease_05[~patients_avg_disease_05].index
abnormal_avg_05 = patients_avg_disease_05[patients_avg_disease_05].index

normal_avg_04 = patients_avg_disease_04[~patients_avg_disease_04].index
abnormal_avg_04 = patients_avg_disease_04[patients_avg_disease_04].index

# Total number of patients who underwent the PHQ-9 test
total_patients = phq9_data['menti_seq'].nunique()

# Print counts
print(f"Total patients: {total_patients}")
print(f"Start-End Criteria: Normal: {len(normal_start_end)}, Abnormal: {len(abnormal_start_end)}")
print(f"Average Criteria (1.0): Normal: {len(normal_avg_1)}, Abnormal: {len(abnormal_avg_1)}")
print(f"Average Criteria (0.9): Normal: {len(normal_avg_09)}, Abnormal: {len(abnormal_avg_09)}")
print(f"Average Criteria (0.8): Normal: {len(normal_avg_08)}, Abnormal: {len(abnormal_avg_08)}")
print(f"Average Criteria (0.7): Normal: {len(normal_avg_07)}, Abnormal: {len(abnormal_avg_07)}")
print(f"Average Criteria (0.6): Normal: {len(normal_avg_06)}, Abnormal: {len(abnormal_avg_06)}")
print(f"Average Criteria (0.5): Normal: {len(normal_avg_05)}, Abnormal: {len(abnormal_avg_05)}")
print(f"Average Criteria (0.4): Normal: {len(normal_avg_04)}, Abnormal: {len(abnormal_avg_04)}")

# Visualize distribution with bar plot
criteria = ['Start-End Criteria', 'Average >= 1.0', 'Average >= 0.9', 'Average >= 0.8', 'Average >= 0.7', 'Average >= 0.6', 'Average >= 0.5', 'Average >= 0.4']
normal_counts = [len(normal_start_end), len(normal_avg_1), len(normal_avg_09), len(normal_avg_08), len(normal_avg_07), len(normal_avg_06), len(normal_avg_05), len(normal_avg_04)]
abnormal_counts = [len(abnormal_start_end), len(abnormal_avg_1), len(abnormal_avg_09), len(abnormal_avg_08), len(abnormal_avg_07), len(abnormal_avg_06), len(abnormal_avg_05), len(abnormal_avg_04)]

plt.figure(figsize=(15, 6))
bar_width = 0.35
x = range(len(criteria))

plt.bar(x, normal_counts, width=bar_width, label='Normal', color='skyblue')
plt.bar([i + bar_width for i in x], abnormal_counts, width=bar_width, label='Abnormal', color='orange')

plt.xlabel('Criteria')
plt.ylabel('Count of Patients')
plt.title('Distribution of Patients by Disease Classification')
plt.xticks([i + bar_width / 2 for i in x], criteria)
plt.legend()

# Add annotations for abnormal counts
for i, abnormal in enumerate(abnormal_counts):
    plt.text(i + bar_width, abnormal + 5, str(abnormal), ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add the total patients count in the upper left corner
plt.text(0, plt.ylim()[1], f'Total patients: {total_patients}', ha='left', va='top', fontsize=12, fontweight='bold')

plt.show()