This git repository contains the code used for the experiments in the ICML 2019 paper "Robust Learning from Untrusted Sources". It contains the functions used, the scripts for running the large-scale experiments and Jupyter Notebooks for creating the plots and tables from the paper.

THE CODE IS NOT DIRECTLY EXECUTABLE, since it requires the availability of the datasets (and the extracted features for the Animals with Attributes 2 experiments) in appropriate directories. However, by using the main.py files it should be easy to run experiments like ours.

1. code_products contains the code for the Amazon Products experiments

	1.1. functions.py contains the main part of our implementation, in particular the functions for preparing the data splits, learning the logistic regression models, minimizing the bound, etc.
	1.2. main.py is an example script for running the experiments described in the paper.
	1.3. get_results_products.ipynb is the script for creating the plots and tables from the paper (once the experiments have been run).

2. code_animals contains the code for the Animals with Attributes 2 experiments

	2.1. functions.py contains the main part of our implementation, in particular the functions for preparing the data splits, corrupting the data, learning the logistic regression models, minimizing the 	bound, etc.
	2.2. main.py is an example script for running the experiments described in the paper.
	2.3. get_results_animals.ipynb is the script for creating the plots and tables from the paper (once the experiments have been run).

