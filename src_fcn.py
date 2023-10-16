import pandas as pd
import os
from fsutils import resource_file_path, get_all_files_with_extension
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def load_set_table():
    path = resource_file_path('./data_source/DoE/DOE.xlsx')
    data = pd.read_excel(path)
    data.drop(columns='Run', inplace=True)
    return data

def load_all_doe_data():
    file_path = resource_file_path('./data_source/DoE/final_data.csv')
    output_data = pd.read_csv(file_path, sep=',')
    output_data.drop(columns=output_data.columns[0], inplace=True)
    return output_data


def return_x_y():
    x = load_set_table()
    y = load_all_doe_data().transpose()
    return np.array(x), np.array(y)


def return_train_test_data(x,y):
    #x = load_set_table()
    #y = load_all_doe_data().transpose()
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

    return X_train, X_test, y_train, y_test


def normalization(y):
    new_y = np.zeros(np.shape(y))
    y_max = y.max(axis=1)
    for i in range(len(y)):
        new_y[i] = y[i] / y_max[i]
    return new_y, y_max


def denormalize(y,max):
    new_y = np.zeros(np.shape(y))
    for i in range(len(y)):
        new_y[i] = y[i]*max[i]
    return new_y

def load_new_input_samples():
    path = resource_file_path('./data_source/normal_setting.xlsx')
    data = pd.read_excel(path)
    data.drop(columns='Run', inplace=True)
    return data