# Robust Learning from Untrusted Sources

## Introduction
This git repository contains the code used for the experiments in the ICML 2019 paper ["Robust Learning from Untrusted Sources"](https://arxiv.org/abs/1901.10310). In particular, the functions used, the scripts for running the large-scale experiments and Jupyter Notebooks for creating the plots and tables from the paper and included.

**THE CODE IS NOT DIRECTLY EXECUTABLE**, since it requires the availability of the datasets (and the extracted features for the Animals with Attributes 2 experiments) in appropriate directories. However, by using the main.py files it should be easy to run experiments like ours.

## Description
1. code_products contains the code for the Amazon Products experiments
- functions.py contains the main part of our implementation, in particular the functions for preparing the data splits, learning the logistic regression models, minimizing the bound, etc.
- main.py is an example script for running the experiments described in the paper.
- get_results_products.ipynb is the script for creating the plots and tables from the paper (once the experiments have been run).

2. code_animals contains the code for the Animals with Attributes 2 experiments
- functions.py contains the main part of our implementation, in particular the functions for preparing the data splits, corrupting the data, learning the logistic regression models, minimizing the 	bound, etc.
- main.py is an example script for running the experiments described in the paper.
- get_results_animals.ipynb is the script for creating the plots and tables from the paper (once the experiments have been run).

## Data

The experiments from the paper are performed on the following datasets:

1. [Multitask Dataset of Product Reviews [1]](http://cvml.ist.ac.at/productreviews/) - A dataset of reviews for various Amazon products. The target task is sentiment analysis - is a review positive or negative?
2. [Animals with Attributes 2 [2]](https://cvml.ist.ac.at/AwA2/) - A dataset of images of various animals, together with 85 binary attributes.

For the Product Reviews, we used the readily available feature representations. For the Animals with Attributes data, one needs to extract appropriate features bofore applying our algorithm. In our experiments we used a ResNet50 pretrained network from the [Tensornets package](https://github.com/taehoonlee/tensornets) to obtain features for this dataset and for the [ImageNet](http://www.image-net.org/) data. Then a PCA projection was learned on ImageNet and applied to the Animals with Attributes 2 data. See the experiments section in the paper for more information.

## Running the experiments
With the relevant features extracted, the main.py scripts can be used to run experiments like those in the paper. Because of the large number of independent runs required, the experiments are best performed on a scientific cluster with multiple CPU cores, in parallel.

For the Product Reviews, run:

```
specific=0 # Binary label, indicating the type of experiment (books and non-books, or the experiment on all products)
ind=0 # the index of the independent run or the product ID, depending on the type of experiment
python main.py $specific $ind
```

For the Animals with Attributes run:

```
first_attr=0 # The first attribute of interest (the provided script runs tests for 7 consecutive attributes)
ind=0 # Index of the experiment - used to loop over different attacks and values of n
python main.py $first_attr $ind
```

## Citing this repository/paper

If you find our paper/experiments exciting, please consider citing us:

```
@inproceedings{konstantinov2019robust,
      title={Robust Learning from Untrusted Sources},
      author={Konstantinov, Nikola and Lampert, Christoph H.},
      booktitle={International Conference on Machine Learning ({ICML})},
      year={2019}
    }
```

## References

[1] A. Pentina and C. H. Lampert. [*"Multi-task Learning with Labeled and Unlabeled Tasks”*](http://proceedings.mlr.press/v70/pentina17a.html), International Conference on Machine Learning (ICML), 2017

[2] Y. Xian, C. H. Lampert, B. Schiele, Z. Akata. [*"Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly"*](https://ieeexplore.ieee.org/document/8413121), IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI) 40(8), 2018. ([arXiv:1707.00600 [cs.CV]](https://arxiv.org/abs/1707.00600))
