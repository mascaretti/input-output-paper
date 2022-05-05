# Loading the required libraries ----------------------------------------------------------------------------
from src.data.utils import get_eu_countries
from src.data.download import download_data, check_data, check_country_list
from src.data.preprocess import preprocess
from src.decomposition.nmf import nmf, svd, cross_val, bi_cross_val
from src.centrality.network import page_rank, rwc
from src.visualisation.plots import barplot, lineplot, heatmap, radar
from src.report.table import write_excel
import os
import numpy as np
import pandas as pd
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# Loading the settings ----------------------------------------------------------------------------
with open("settings.yaml", "r") as f:
    settings = load(f, Loader=Loader)


# Downloading the data ----------------------------------------------------------------------------

## Download Behaviour
download_behaviour = settings['download']['behaviour']
try:
    assert download_behaviour in ['default', 'overwrite', 'skip']
    if download_behaviour == 'default':
        print("Only downloading missing data...")
    if download_behaviour == 'overwrite':
        print("Overwrite data...")
    if download_behaviour == 'skip':
        print("Skipping download altogether...")
except AssertionError:
    print("****\nERROR: Invalid value. Please, select either \"default\", \"overwrite\" or \"skip\".\n****")
    raise


# Scope of the analysis ----------------------------------------------------------------------------
try:
    assert settings['scope']['countries'] in ['EU', 'OTHER']
    if settings['scope']['countries'] == 'EU':
        eu = True
        country_list = get_eu_countries()
        print("Country list is {0}".format(country_list))
    elif settings['scope']['countries'] == 'OTHER':
        eu = False
        country_list = list(settings['scope']['country_list'])
except AssertionError:
    print("****\nERROR: Invalid value. Please, select either \"EU\" or \"OTHER\".\n****")
    raise

if country_list and eu:
    print('''*** Warning: you have provided a non-empty country list.
    The list will be ignored because EU countries were selected ***''')
    country_list = get_eu_countries()

print("Checking the provided list of country...")
check_country_list(country_list)

output_directory = os.path.join(os.getcwd(), "data/raw")
print("Files ought to be saved in the directory: {0}".format(output_directory))
print("The directory will be created if it does NOT exist.")

## Creating directory
if os.path.isdir(output_directory) == False:
    os.makedirs(output_directory)

if download_behaviour == 'skip':
    check_data(eu, output_directory, country_list)

else:
    ## Calling the function
    download_data(eu=eu,
                  country_list=country_list,
                  output_directory=output_directory,
                  download_behaviour=download_behaviour)


# Preprocessing ----------------------------------------------------------------------------
year = settings['scope']['year']
try:
    assert year in range(2000, 2014 + 1)
except AssertionError:
    print("The year must be between 2000 and 2014. Check the setting file")
    raise

dataframe_directory = os.path.join(os.getcwd(), "data/processed/" + str(year))
if os.path.isdir(dataframe_directory) == False:
    os.makedirs(dataframe_directory)

preprocess(year=year, eu=eu, country_list=country_list, input_dir=output_directory, output_dir=dataframe_directory)


# Non-Negative Matrix Factorisation ----------------------------------------------------------------------------

dataframe_file_path = os.path.join(dataframe_directory, "dataframe_" + str(year) + ".npy")

num_components = settings['nmf']['num_components']
tolerance = settings['nmf']['tolerance']
num_iterations = settings['nmf']['num_iterations']
alpha = settings['nmf']['alpha']
rho = settings['nmf']['rho']
seed = settings['nmf']['seed']

try:
    assert num_components >= 1
except AssertionError:
    print("\n*** The number of components must be greater than zero! ***\n")
    raise

try:
    assert num_components < len(country_list)
except AssertionError:
    print("\n*** The number of components must be less than the number of countries! ***\n")
    raise

output_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "nmf")
if os.path.isdir(output_directory) == False:
    os.makedirs(output_directory)

print("\n****")
print("Computing NMF with the following settings:")
print("\tInput file: {0}".format(dataframe_file_path))
print("\tOutput directory: {0}".format(output_directory))
print("\tNumber of components: {0}".format(num_components))
print("\tTolerance: {0}".format(tolerance))
print("\tMaximum Iterations: {0}".format(num_iterations))
print("\tAlpha: {0}".format(alpha))
print("\tL1 Ratio: {0}".format(rho))
print("\tRandom Seed: {0}".format(seed))
print("\n")
nmf(input_file=dataframe_file_path, output_directory=output_directory, rank=num_components, alpha=alpha, l1_ratio=rho, rnd_seed=seed)


 # SVD ----------------------------------------------------------------------------
do_svd = settings['validation']['svd']['compute']
try:
     assert do_svd in [True, False]
except AssertionError:
    print("SVD flag set to {0}".format(do_svd))
    print("\n*** SVD flag should be True or False. ***\n")
    raise

if do_svd:
    print("\n****")
    print("Computing SVD")
    svd_output_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "svd")
    print("Saving results in {0}\n".format(svd_output_directory))
    if os.path.isdir(svd_output_directory) == False:
        os.makedirs(svd_output_directory)
    svd(input_file=dataframe_file_path, output_directory=svd_output_directory, rnd_seed=seed)

else:
    print("\n****")
    print("Skipping SVD\n")


 # Cross Validation ----------------------------------------------------------------------------
do_cv = settings['validation']['cross_validation']['compute']
try:
    assert  do_cv in [True, False]
except AssertionError:
    print("Cross Validation flag set to {0}".format(do_svd))
    print("\n*** Cross Validation flag should be True or False. ***\n")
    raise

if do_cv:
    print("\n****")
    print("Computing Cross-Validation")
    cv_output_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "cv")
    print("Saving results in {0}\n".format(cv_output_directory))
    if os.path.isdir(cv_output_directory) is False:
        os.makedirs(cv_output_directory)
    cross_val(input_file=dataframe_file_path, output_directory=cv_output_directory,alpha=alpha, l1_ratio=rho, rnd_seed=seed)

else:
    print("\n****")
    print("Skipping Cross Validation\n")


# Bi - Cross Validation ----------------------------------------------------------------------------
## Final Loss Function -----------
do_bi_cv_flf = settings['validation']['bi_cross_validation']['final_loss_fun']['compute']
try:
    assert do_bi_cv_flf in [True, False]
except AssertionError:
    print("BiCrossValidation Flag set to {0}".format(do_bi_cv_flf))
    raise

if do_bi_cv_flf:

    num_folds_flf = settings['validation']['bi_cross_validation']['final_loss_fun']['num_folds']
    max_rank_flf = settings['validation']['bi_cross_validation']['final_loss_fun']['max_rank']
    
    print("\n****")
    print("Computing BiCV for final loss function")
    bi_cv_output_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "bicv")
    print("Saving results in {0}\n".format(bi_cv_output_directory))
    if os.path.isdir(bi_cv_output_directory) is False:
        os.makedirs(bi_cv_output_directory)
    bi_cv_filepath_flf = os.path.join(bi_cv_output_directory, "bicv_flf.npy")
    bi_cross_val(input_file=dataframe_file_path, output_file=bi_cv_filepath_flf, num_folds=num_folds_flf, max_rank=max_rank_flf, alpha=alpha, l1_ratio=rho, rnd_seed=seed)    

## Different seeds ---------------------------
do_bi_cv_s = settings['validation']['bi_cross_validation']['seeds']['compute']

try:
    assert do_bi_cv_s in [True, False]
except AssertionError:
    print("BiCrossValidation Flag set to {0}".format(do_bi_cv_s))
    raise


if do_bi_cv_s:
    
    num_folds_s = settings['validation']['bi_cross_validation']['seeds']['num_folds']
    max_rank_s = settings['validation']['bi_cross_validation']['seeds']['max_rank']
    seed_list = settings['validation']['bi_cross_validation']['seeds']['list_seed']

    try:
        assert seed_list
    except AssertionError:
        print("Empty seed list")
        raise

    if seed not in seed_list:
        seed_list.append(seed)

    print("\n****")
    print("Computing BiCV for different seeds")
    bi_cv_output_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "bicv")
    print("Saving results in {0}\n".format(bi_cv_output_directory))
    if os.path.isdir(bi_cv_output_directory) is False:
        os.makedirs(bi_cv_output_directory)

    for seed in seed_list:
        print("Seed: {0}".format(seed))
        bi_cv_filepath_s = os.path.join(bi_cv_output_directory, "bicv_seed_{0}.npy".format(seed))
        bi_cross_val(input_file=dataframe_file_path, output_file=bi_cv_filepath_s, num_folds=num_folds_s, max_rank=max_rank_s, alpha=alpha, l1_ratio=rho, rnd_seed=seed)   


## Different penalisations ---------------------------
do_bi_cv_p = settings['validation']['bi_cross_validation']['penalisations']['compute']

try:
    assert do_bi_cv_p in [True, False]
except AssertionError:
    print("BiCrossValidation Flag set to {0}".format(do_bi_cv_p))
    raise


if do_bi_cv_p:
    num_folds_p = settings['validation']['bi_cross_validation']['penalisations']['num_folds']
    max_rank_p = settings['validation']['bi_cross_validation']['penalisations']['max_rank']
    pen_list = settings['validation']['bi_cross_validation']['penalisations']['list_penalisations']

    try:
        assert pen_list
    except AssertionError:
        print("\n*** List of penalistations is empty. ***\n")
        raise

    print("\n****")
    print("Computing BiCV for different penalisation terms")
    bi_cv_output_directory = os.path.join(os.getcwd(), "data/processed", str(year), "bicv")
    print("Saving results in {0}\n".format(bi_cv_output_directory))
    if os.path.isdir(bi_cv_output_directory) is False:
        os.makedirs(bi_cv_output_directory)

    if [alpha, rho] not in pen_list:
        pen_list.append([alpha, rho])

    for pen in pen_list:
        alpha = pen[0]
        rho = pen[1]
        print(pen)
        print("Alpha: {0}, L1 Ratio: {1}".format(alpha, rho))
        bi_cv_filepath_p = os.path.join(bi_cv_output_directory, "bicv_alpha_{0}_rho_{1}.npy".format(alpha, rho))
        bi_cross_val(input_file=dataframe_file_path, output_file=bi_cv_filepath_p, num_folds=num_folds_p, max_rank=max_rank_p, alpha=alpha, l1_ratio=rho, rnd_seed=seed)


# Centrality ----------------------------------------------------------------------------
country_input_dir = os.path.join(os.getcwd(), "data/processed", str(year))
country_output_dir = os.path.join(os.getcwd(), "data/processed", str(year), "centrality")

hasWeights = settings['rwc']['weights']

if hasWeights == True:
    print("Loading weights...")
    df = pd.read_csv("data/external/shocks.csv")

    df.loc[df['shutdown'] == 0.0, "shutdown"] = 100.0
    df.loc[df['shutdown'] == 1.0, "shutdown"] = 1.0
    df.loc[df['shutdown'] == 0.5, "shutdown"] = 50.0
    df.loc[df['shutdown'] == 0.25, "shutdown"] = 25.0

    weights = df['shutdown'].to_numpy()

elif hasWeights == False:
    print("Weights set to None")
    weights = None
else:
    raise ValueError

if os.path.isdir(country_output_dir) == False:
    os.makedirs(country_output_dir)

print("\n****")
for country in country_list:
    print("PageRank/RWC for {0}".format(country))
    pr_country_input_path = os.path.join(country_input_dir, "{0}.npy".format(country))
    pr_country_output_path = os.path.join(country_output_dir, "pr_{0}".format(country))

    rwc_country_input_path = os.path.join(country_input_dir, "{0}.npy".format(country))
    rwc_country_output_path = os.path.join(country_output_dir, "rwc_{0}.npy".format(country))
    rwc_country_weighted_output_path = os.path.join(country_output_dir, "rwc_weighted_{0}.npy".format(country))
    rwc_country_difference_path = os.path.join(country_output_dir, "rwc_relative_{0}.npy".format(country))
    
    page_rank(pr_country_input_path, pr_country_output_path)
    shocks = rwc(rwc_country_input_path, rwc_country_output_path, rnd_seed=seed, weights=None)
    weighted_shocks = rwc(rwc_country_input_path, rwc_country_weighted_output_path, rnd_seed=seed, weights=weights)
    percentage_difference = (weighted_shocks - shocks) / shocks
    np.save(rwc_country_difference_path, arr=percentage_difference)

print("\n****")
W_path = os.path.join(os.getcwd(), "data/processed/", str(year), "nmf", "W.npy")
W = np.load(W_path)

for i in np.arange(num_components):
    print("PageRank/RWC for pattern {0}".format(i+1))
    pr_pattern_output_path = os.path.join(country_output_dir, "pr_pattern_{0}.npy".format(i))
    rwc_pattern_output_path = os.path.join(country_output_dir, "rwc_pattern_{0}.npy".format(i))
    page_rank(W[:, i], pr_pattern_output_path)
    rwc(W[: , i], rwc_pattern_output_path, seed, weights=weights)
    if hasWeights == True:
        weighted_shocks = rwc(W[: , i], os.path.join(country_output_dir, "comparison", "weighted_rwc_pattern_{0}.npy".format(i)), seed, weights=weights)
        unweighted_shocks = rwc(W[: , i], os.path.join(country_output_dir, "comparison", "rwc_pattern_{0}.npy".format(i)), seed, weights=None)
        percentage_difference = (weighted_shocks - unweighted_shocks) / unweighted_shocks
        np.save(file=os.path.join(country_output_dir, "rwc_relative_pattern_{0}.npy".format(i)), arr=percentage_difference)

# Visualisation ----------------------------------------------------------------------------
## Barplot Shock Centrality
colours = ['rgb(239,138,98)', 'rgb(153,153,153)', 'rgb(103,169,207)']
pr_paths = [os.path.join(country_output_dir, "pr_pattern_{0}.npy".format(i)) for i in range(num_components)]

graph_path = os.path.abspath(os.path.join(os.getcwd(), "reports", str(year), "figures"))
if os.path.isdir(graph_path) is False:
    os.makedirs(graph_path)

print("\n****")
for component in range(num_components):
    print("PageRank Barplot {0}".format(component + 1))
    pr_centr = np.load(pr_paths[component])
    pr_out = os.path.join(graph_path, "pr_pattern_{0}.".format(component)) # omit the extension, that I add in the fuction
    pr_title = "Page Rank Centrality | Pattern {0}".format(component + 1)
    barplot(centrality_values=pr_centr, title=pr_title, colour=colours[component], file_dir=pr_out)

rwc_paths = [os.path.join(country_output_dir, "rwc_pattern_{0}.npy".format(i)) for i in range(num_components)]
rwc_differences_paths = [os.path.join(country_output_dir, "rwc_relative_pattern_{0}.npy".format(i)) for i in range(num_components)]
for component in range(num_components):
    print("Random Walk Centrality Barplot {0}".format(component + 1))
    rwc_centr = np.load(rwc_paths[component])
    rwc_out = os.path.join(graph_path, "rwc_pattern_{0}.".format(component)) # omit the extension, that I add in the function

    if hasWeights == True:
        rwc_title = "Weighted Random Walk Centrality | Pattern {0}".format(component + 1)
        rwc_relative_title = "Percentage Differences | Pattern {0}".format(component + 1)
        rwc_relative_centr = np.load(rwc_differences_paths[component])
        barplot(centrality_values=rwc_relative_centr, title=rwc_relative_title, colour=colours[component], file_dir=os.path.join(graph_path, "rwc_relative_pattern_{0}.".format(component)))
    else:
        rwc_title = "Random Walk Centrality | Pattern {0}".format(component + 1)

    
    barplot(centrality_values=rwc_centr, title=rwc_title, colour=colours[component], file_dir=rwc_out)

## SVD
print("\n****\nSVD")

s_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "svd", "s.npy")
s = np.load(s_directory)
s_plot_values = np.array([[i+1, val] for i, val in enumerate(s)]).T
s_name = "SVD | Year: {0}".format(year)
s_path = os.path.join(graph_path, "svd.") # omit the extension, that I add later in the function

lineplot(input_values=s_plot_values, name_list=[s_name], title=s_name, xaxis_title="Component", yaxis_title="Singular Value", file_dir=s_path)



## Cross Validation
print("\n****\nCrossValidation")
cv_output_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "cv", "cross_validation.npy")
cv = np.load(cv_output_directory)
cv_plot_values = np.array([[i, val] for i, val in enumerate(cv)]).T 
cv_name = "Cross-Validation | Year: {0}".format(year)
cv_path = os.path.join(graph_path, "cv.") # omit the extension, that I add later in the function

lineplot(input_values=cv_plot_values, name_list=[cv_name], title=cv_name, xaxis_title="Number of components", yaxis_title="Error", file_dir=cv_path)

## BiCrossValidation
### Final Loss Function
print("\n***\nBiCrossValidation\n")
print("\tFinal")
bi_cv_output_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "bicv")
bi_cv_filepath_flf = os.path.join(bi_cv_output_directory, "bicv_flf.npy")

bicv = np.load(bi_cv_filepath_flf)
bicv_plot_values = np.array([[i, val] for i, val in enumerate(bicv)]).T
bicv_name = "Bi-Cross-Validation | Year: {0}".format(year)
bicv_path = os.path.join(graph_path, "bicv.") # omit the extension, that I add later in the function

lineplot(input_values=bicv_plot_values, name_list=bicv_name, title=bicv_name, xaxis_title="Number of components", yaxis_title="Error", file_dir=bicv_path)

### Seeds
print("\tSeeds")
bi_cv_s_names = []
bi_cv_s_values = []
bicv_path_s = os.path.join(graph_path, "bicv_seeds.") # omit the extension, that I add later in the function

seed_list = settings['validation']['bi_cross_validation']['seeds']['list_seed']

try:
    assert seed_list
except AssertionError:
    print("Empty seed list")
    raise

if seed not in seed_list:
    seed_list.append(seed)



for seed in seed_list:
    bi_cv_filepath_s = os.path.join(bi_cv_output_directory, "bicv_seed_{0}.npy".format(seed))
    bi_cv_s = np.load(bi_cv_filepath_s)
    bi_cv_s_plot_values = np.array([[i, val] for i, val in enumerate(bi_cv_s)]).T
    bi_cv_s_values.append(bi_cv_s_plot_values)

    bicv_names_s = "Seed: {0}".format(seed)
    bi_cv_s_names.append(bicv_names_s)

lineplot(input_values=bi_cv_s_values, name_list=bi_cv_s_names, title="Bi-Cross-Validation | Year: {0} | Different Seed".format(year), xaxis_title = "Number of components", yaxis_title = "Error", file_dir=bicv_path_s)


### Different Penalisations
print("\tPenalisations")
bicv_val_p = []
bicv_names_p = []
bicv_path_p = os.path.join(graph_path, "bicv_penalised.") # omit the extension, that I add later in the function


pen_list = settings['validation']['bi_cross_validation']['penalisations']['list_penalisations']
try:
    assert pen_list
except AssertionError:
    print("\n*** List of penalistations is empty. ***\n")
    raise

if [alpha, rho] not in pen_list:
    pen_list.append([alpha, rho])

for [alpha, rho] in pen_list:
    bi_cv_filepath_p = os.path.join(bi_cv_output_directory, "bicv_alpha_{0}_rho_{1}.npy".format(alpha, rho))
    bi_cv_p = np.load(bi_cv_filepath_p)
    bi_cv_p_plot_values = np.array([[i, val] for i, val in enumerate(bi_cv_p)]).T
    bicv_val_p.append(bi_cv_p_plot_values)

    bicv_name_p = "A.: {0}, R.: {1}".format(alpha, rho)
    bicv_names_p.append(bicv_name_p)

lineplot(input_values=bicv_val_p, name_list=bicv_names_p, title="Bi-Cross-Validation | Year: {0} | Different Penalisations".format(year), xaxis_title="Number of components", yaxis_title="Error", file_dir=bicv_path_p)


## Heatmap
print("\n****\nHeatmaps")
W = np.load(W_path)
print(W.shape)
for component in np.arange(num_components):
    n = np.int(np.sqrt(W.shape[0]))
    W_component = np.reshape(W[:,component], (n,n))
    title_component = "Heatmap for Pattern {0}".format(component + 1)
    file_dir = os.path.join(graph_path, "heatmap_{0}.".format(component)) # omit the extension, that I add later in the function
    heatmap(A = W_component, title=title_component, file_dir=file_dir)


    
## Radar Plot
colours = ['rgb(239,138,98)', 'rgb(153,153,153)', 'rgb(103,169,207)']
pr_paths = [os.path.join(country_output_dir, "pr_pattern_{0}.npy".format(i)) for i in range(num_components)]

graph_path = os.path.abspath(os.path.join(os.getcwd(), "reports", str(year), "figures"))
if os.path.isdir(graph_path) is False:
    os.makedirs(graph_path)

print("\n****")
for component in range(num_components):
    print("PageRank Radar Plot {0}".format(component + 1))
    pr_centr = np.load(pr_paths[component])
    pr_out = os.path.join(graph_path, "pr_rp_pattern_{0}.".format(component)) # omit the extension, that I add later in the function
    pr_title = "Page Rank Centrality | Radar Plot | Pattern {0}".format(component + 1)
    radar(centrality_values=pr_centr, title=pr_title, colour=colours[component], file_dir=pr_out)

rwc_paths = [os.path.join(country_output_dir, "rwc_pattern_{0}.npy".format(i)) for i in range(num_components)]
for component in range(num_components):
    print("Random Walk Centrality Radar Plot {0}".format(component + 1))
    rwc_centr = np.load(rwc_paths[component])
    rwc_out = os.path.join(graph_path, "rwc_rp_pattern_{0}.".format(component)) # omit the extension, that I add later in the function
    rwc_title = "Random Walk Centrality | Radar Plot | Pattern {0}".format(component + 1)
    radar(centrality_values=rwc_centr, title=rwc_title, colour=colours[component], file_dir=rwc_out)


# Reports
print("\n****\nGenerate Excel Report")
nmf_directory = os.path.join(os.getcwd(), "data/processed/", str(year), "nmf")
H_dir = os.path.join(nmf_directory, "H.npy")

report_path = os.path.abspath(os.path.join(os.getcwd(), "reports", str(year), "tables"))
if os.path.isdir(report_path) == False:
    os.makedirs(report_path)
excel_path = os.path.join(report_path, "report.xlsx")
write_excel(H_dir=H_dir, country_list=country_list, output_dir = excel_path)
