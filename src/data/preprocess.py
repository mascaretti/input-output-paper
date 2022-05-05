import os
import numpy as np
import pandas as pd
from src.data.utils import get_eu_countries

def _read_country_year(year, input_file, output_file):
    '''This function returns the matrix A of the technical coefficients
    for a given country in the dataset and for a fixed year.
    The sectors whose output is identically equal to zero are eliminated.
    '''

    # We set the directory of the Excel files
    # WIOD Tables were obtained at http://www.wiod.org/database/niots16

    print("Reading: {0}".format(input_file))

    # We read from the Excel file
    df = pd.read_excel(input_file, "National IO-tables")
    print("Dimension of the dataframe: {}".format(df.shape))

    # PREPROCESSING
    year_index = df['Year'] == year #Selecting the relevant year
    domestic_index = df['Origin'] == 'Domestic' # Selecting the relevant part of the table
    df_year_domestic = df[year_index & domestic_index]

    # COMPUTING THE TRANSACTION MATRIX
    Z = df_year_domestic.iloc[:, 4:4+df_year_domestic.shape[0]]
    assert Z.shape[0] == Z.shape[1]
    print("Dimension of Z: {}".format(Z.shape))

    # OUTPUT
    output = df_year_domestic['GO'].to_numpy(np.float64)
    print("Dimension of the output: {}".format(output.size))

    Z = Z.to_numpy(dtype=np.float64)
    A = np.where(output != 0, Z / output, np.array([0.])) 

    print("Removing last two sectors...")
    N_SEC = A.shape[0]
    A_new = A[0:N_SEC-2][:, 0:N_SEC-2]
    assert A_new.shape[0] == A_new.shape[1]
    assert A.shape[0] - 2 == A_new.shape[0]

    print("Shape of A: {}".format(A_new.shape))
    np.save(output_file, A_new)

    return A_new


def preprocess(year, eu, country_list, input_dir, output_dir):
    '''This function takes as input the Year for which we wish to 
    conduct the analysis, the eu bool the specifies which country or,
    alternatively, a set of countries. The input_dir is to know where
    to read the data from. The output_dir is used to store data.
    '''
    # Get country list
    preprocess_list = get_eu_countries() if eu == True else country_list
    try:
        assert preprocess_list
    except AssertionError:
        print("\n")
        print("** List to preprocess is empty! **")
        print("\n")
        raise

    # Lexicographic Order
    preprocess_list.sort()
    
    # Get matrix dimension
    first_input_file = os.path.abspath(os.path.join(output_dir, preprocess_list[0] + ".npy"))
    print("Getting the dimension of the matrices from {0} at {1}".format(preprocess_list[0], first_input_file))
    if os.path.isfile(first_input_file) == True:
        print("File found...")
        first_matrix = np.load(first_input_file)
    else:
        print("File not found, creating it...")
        first_output_file = os.path.abspath(os.path.join(output_dir, preprocess_list[0] + ".npy"))
        first_input_file = os.path.abspath(os.path.join(input_dir, preprocess_list[0] + '.xlsx'))
        first_matrix = _read_country_year(year, first_input_file, first_output_file)

    # Postponing tests for later
    try:
        assert first_matrix.shape[0] == first_matrix.shape[1]
    except AssertionError:
        print("I/O Matrix is not square!")
        raise

    num_sectors = first_matrix.shape[0]
    print("Number of sectors: {}".format(num_sectors))
    print("WARNING: We remove two sectors!")

    # Removing last two sectors
    X = np.zeros((np.power(num_sectors, 2), len(preprocess_list)))
    print("Checking X: {}".format(X.shape))

    for col_index, country in enumerate(preprocess_list):
        curr_input_file = os.path.abspath(os.path.join(output_dir, country + ".npy"))
        print("Extracting {0} from {1}...".format(country, curr_input_file))

        if os.path.isfile(curr_input_file) == True:
            print("File found...\n")
            curr_country = np.load(curr_input_file)
        
        else:
            print("File not found, creating it...\n")
            curr_input_file = os.path.abspath(os.path.join(input_dir, country + ".xlsx"))
            curr_output_file = os.path.abspath(os.path.join(output_dir, country + ".npy"))
            curr_country = _read_country_year(year=year, input_file=curr_input_file, output_file=curr_output_file)

        try:
            assert curr_country.shape[0] == curr_country.shape[1]
        except AssertionError:
            print("I/O Matrix is not square!")
            
        X[:, col_index] = curr_country.reshape(np.power(num_sectors, 2), )
 
    # Saving results
    output_path = os.path.join(output_dir, "dataframe_" + str(year) + ".npy")
    print("Storing results at {0}".format(output_path))
    np.save(output_path, X)

    return X
