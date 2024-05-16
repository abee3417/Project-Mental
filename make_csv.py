import pandas as pd

# Load the dataset
file_path = 'raw_data1.csv'
data = pd.read_csv(file_path)

# Extract only the relevant columns for the survey responses
survey_columns = ['menti_seq', 'check1_value', 'check2_value', 'check3_value', 'check4_value', 'check5_value', 'check6_value']
survey_data = data[survey_columns]

# Create an empty DataFrame to store the counts
columns = [
    'check1_val_0', 'check1_val_1', 'check1_val_2',
    'check2_val_0', 'check2_val_1',
    'check3_val_0', 'check3_val_1',
    'check4_val_0', 'check4_val_1',
    'check5_val_1', 'check5_val_2', 'check5_val_3',
    'check6_val_0', 'check6_val_1', 'check6_val_2', 'check6_val_3'
]
result = pd.DataFrame(columns=columns)

# Iterate through each menti_seq and calculate the counts
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
    
    # Add the counts to the result DataFrame
    result = pd.concat([result, pd.DataFrame(counts, index=[menti_seq])])

# Save the results to a CSV file
output_file_path = 'survey_val_counts.csv'
result.to_csv(output_file_path, index_label='menti_seq')