import matplotlib.pyplot as plt
from fsutils import resource_file_path
import pandas as pd
import numpy as np
from src_fcn import normalization,return_x_y
'''
test_predicted_data = np.array(pd.read_csv('output_data.csv'))

for i in range(len(test_predicted_data)):
    plt.plot(test_predicted_data[i])
    plt.show()
    
'''
x,y = return_x_y()
a,b = normalization(y)
ss = 2