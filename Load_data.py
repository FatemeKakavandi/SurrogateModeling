import pandas as pd
import os
from fsutils import resource_file_path, get_all_files_with_extension
import numpy as np
import matplotlib.pyplot as plt
def load_data():
    path = resource_file_path('./data_source/DoE/V11-V19_Force-Displacement.xlsx')
    data = pd.read_excel(path)
    return data

def load_set_table():
    path = resource_file_path('./data_source/DoE/DOE.xlsx')
    data = pd.read_excel(path)
    return data


def load_rest_of_DoEs():
    path = resource_file_path('./data_source/DoE/V20_V27')
    files = get_all_files_with_extension(path,extension='.csv')
    final_df = pd.DataFrame()
    files.sort()
    for file in files:
        label = file.split('/')[-1].split('.')[0]
        temp = pd.read_csv(file)
        temp.drop(columns=temp.columns[-1], inplace=True)
        temp.columns=[f'{label}-time', f'{label}-force']
        final_df = pd.concat([final_df,temp],axis=1)
    return final_df

setting_table = load_set_table()
runs = setting_table['Run']

rem = load_rest_of_DoEs()

doe_data = load_data()

final_data = pd.concat([doe_data,rem], axis=1)
t0 = 0
tend = 0.0025
tstep = 0.0025/500

t_temp = list(np.arange(t0,tend,tstep))
eps = tstep*0.5

unit_data = pd.DataFrame()
for elm in runs:
    data_label = elm+'-force'
    time_label = elm+'-time'
    new_force = []
    for i in t_temp:
        new_force = new_force + [
            np.mean(final_data[(i - eps < final_data[time_label]) & (final_data[time_label] <= i + eps)][data_label])]
    unit_data[data_label]=new_force
save_path = resource_file_path('./data_source/DoE')
unit_data.to_csv(os.path.join(save_path,'final_data.csv'),sep=',')
