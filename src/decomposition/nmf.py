import os
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from progress.bar import Bar


def nmf(input_file, output_directory, rank, alpha, l1_ratio, rnd_seed):
    '''This function perfomrs the Non-negative Matrix Factorisation on the
    dataframe. It takes as input a file, it outputs the two matrices. Rank,
    penalisation and seed are also to be specified.
    '''

    # SETTING THE SEED
    np.random.seed(rnd_seed)

    # READING THE DATA
    X = np.load(input_file)

    # COMPUTING NMF
    nmf = NMF(n_components=rank, random_state=rnd_seed, alpha=alpha, l1_ratio=l1_ratio, verbose=True, max_iter=int(1e+5))
    W = nmf.fit_transform(X)
    H = nmf.components_

    # STORING RESULTS
    output_directory_W = os.path.abspath(os.path.join(output_directory, "W.npy"))
    output_directory_H = os.path.abspath(os.path.join(output_directory, "H.npy"))
    np.save(output_directory_W, W)
    np.save(output_directory_H, H)

    return W, H

def svd(input_file, output_directory, rnd_seed):
    '''This function computes the Singular Value Decomposition
    of the dataframe. The dataframe is to be found at `input_file`.
    Results are stored in the `output_directory`. A random seed is to
    be set by using `rnd_seed`.
    '''
    # Setting the seed
    np.random.seed(rnd_seed)

    # Reading the data
    X = np.load(input_file)

    # SVD
    u, s, vh = np.linalg.svd(X)

    # Storing results
    output_directory_u = os.path.abspath(os.path.join(output_directory, "u.npy"))
    output_directory_s = os.path.abspath(os.path.join(output_directory, "s.npy"))
    output_directory_vh = os.path.abspath(os.path.join(output_directory, "vh.npy"))

    np.save(output_directory_u, u)
    np.save(output_directory_s, s)
    np.save(output_directory_vh, vh)


def cross_val(input_file, output_directory, alpha, l1_ratio, rnd_seed):
    '''This function computes a Leave-One-Out cross validation on the data frame.
    For each column, a reduced nmf is computed. Using the obtained components, we
    then compute the reconstruction error. The zero model is the norm of the left out
    column. This is iterated for each column.
    The input of the function are `input_file` with the data frame, the `output_directory`,
    where we store data, `alpha` and `l1_ratio` are the penalisation terms and `rnd_seed`
    is the seed.
    '''
    # Setting the seed
    np.random.seed(rnd_seed)
    
    # Reading the data
    X = np.load(input_file)

    # Setting the maximum rank
    n, m = X.shape
    max_rank = min(n, m) - 1

    # Set the bar
    bar = Bar('CV: Processing', max = max_rank + 1)
    
    # Allocatin the vector
    cv_error = np.zeros((max_rank + 1,))

    # Computing the Cross Validation
    for k in np.arange(0, max_rank + 1):

        for i in np.arange(m):

            # Zero dimensional model
            if k == 0:
                curr_err = np.linalg.norm(X[:, i])
                cv_error[0] += curr_err
                continue
            
            # Leaving out one column
            SKIP_I = np.concatenate([np.arange(0, i), np.arange(i+1, m)], axis=None)
            X_curr = X[:, SKIP_I]

            # NMF on reduced matrix
            nmf = NMF(n_components=k, random_state=rnd_seed, alpha=alpha, l1_ratio=l1_ratio)
            W_curr = nmf.fit_transform(X_curr)

            # Computing the activation for the left-out column
            h = nnls(W_curr, X[:, i])[0]

            # Computing the error
            curr_err = np.linalg.norm((X[:, i] - (W_curr @ h)))
            cv_error[k] += curr_err

        # Advancing bar
        bar.next()

    # Closing bar
    bar.finish()

    # SAVING THE OUTPUT
    output_directory = os.path.join(output_directory, "cross_validation.npy")
    np.save(output_directory, cv_error)

    return cv_error



def bi_cross_val(input_file, output_file, num_folds, max_rank, alpha, l1_ratio, rnd_seed):
    '''This function computes a Generalised-Inverse BiCrossValidation. It is based on
    (http://dx.doi.org/10.1214/08-AOAS227).
    The input of the function are `input_file` with the data frame, the `output_file`,
    where we store the obtained vector, `alpha` and `l1_ratio` are the penalisation terms and `rnd_seed`
    is the seed. Values `num_fold` and `max_rank` define, respectively, the number of times
    a submatrix is sampled and the maximum rank of the decomposition to test (it is inlcuded).
    '''
    # Setting the seed
    np.random.seed(rnd_seed)

    # Reading the data
    X = np.load(input_file)
    n, m = X.shape

    # Setting the maximum rank
    try:
        assert max_rank >= 1
    except AssertionError:
        print("\n*** The maximum rank for the Bi-Cross-Validation must be greater than zero! ***\n")
        raise
    
    try:
        assert max_rank <= min(n, m)
    except AssertionError:
        print("\n*** The maximum rank for the Bi-Cross-Validation must be smaller than both the rows and columns of the data frame ***\n")
        raise

    # Vector to store BCV
    bcv = np.zeros((max_rank + 1, num_folds)) # max_rank + 1 because it is included
    bcv_index = 0

    # Size of the submatrices we will sample from the data matrix
    submatrix_size = 3

    for curr_rank in range(max_rank + 1):

        fold_index = 0
        
        # Creating fold bar
        fold_bar = Bar("Folds of rank {0} of {1}".format(curr_rank, max_rank), max = num_folds)

        for curr_folds in range(num_folds):

            # Selecting random subsample of the rows
            row_index = np.full(n, False)
            row_index[np.random.permutation(n)[:submatrix_size]] = True

            # Selecting random subsample of the columns
            column_index = np.full(m, False)
            column_index[np.random.permutation(m)[:submatrix_size]] = True

            # Decomposing X into blocks
            X_I_J = X[row_index,:][:, column_index]
            X_mI_J  = X[~row_index,:][:, column_index]
            X_I_mJ  = X[row_index,:][:, ~column_index]
            X_mI_mJ = X[~row_index,:][:, ~column_index]

            if curr_rank == 0:
                bcv[bcv_index, fold_index] = np.linalg.norm((X_I_J), ord='fro')
                fold_index += 1
                # Updating bar
                fold_bar.next()
                continue

            # NMF on reducing matrix
            nmf = NMF(n_components=curr_rank, random_state=rnd_seed, alpha=alpha, l1_ratio=l1_ratio)

            W_mI_mJ = nmf.fit_transform(X_mI_mJ)
            H_mI_mJ = nmf.components_
            
            # Generalised inverse
            X_hat_I_J = np.dot(np.dot(np.dot(X_I_mJ, np.linalg.pinv(H_mI_mJ)), np.linalg.pinv(W_mI_mJ)), X_mI_J)

            # Updating BCV
            bcv[bcv_index, fold_index] = np.linalg.norm((X_I_J - X_hat_I_J), ord='fro')
            fold_index += 1

            # Updating bar
            fold_bar.next()

        fold_bar.finish()
        bcv_index += 1

    assert bcv.shape[0] == max_rank + 1
    assert bcv.shape[1] == num_folds

    mean_err = np.mean(bcv, axis=1)
    
    np.save(output_file, mean_err)

    return mean_err
