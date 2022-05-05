import pandas as pd
import numpy as np

def write_excel(H_dir, country_list, output_dir):
    '''This function creates an Excel file summarising
    the H matrix.
    `H_dir` is the location of H
    `country_list` is the list of countries analysed
    `output_dir` is where to store the file
    '''
    H = np.load(H_dir)
    row_length, column_length = H.shape

    try:
        assert column_length == len(country_list)
    except AssertionError:
        print("Number of countries does not match the number of rows of H")
        raise
    
    
    column_names = ["Pattern {0}".format(i + 1) for i in range(row_length)]

    df = pd.DataFrame(data = H.T, index = country_list, columns = column_names)
    df.to_excel(output_dir)