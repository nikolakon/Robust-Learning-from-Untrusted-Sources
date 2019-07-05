# Robust Learning from Untrusted Sources

## Introduction
This git repository contains the code used for the experiments in the ICML 2019 paper ["Robust Learning from Untrusted Sources"](https://arxiv.org/abs/1901.10310). In particular, the functions used, the scripts for running the large-scale experiments and Jupyter Notebooks for creating the plots and tables from the paper and included. The code is readily executable, provided that the datasets and the extracted features for the Animals with Attributes 2 experiments are stored in appropriate directories.

Our paper provides a framework for learning from multiple sources that are unreliable in terms of the data they provide. Assuming access to multiple batches of data of different quality and a small trusted dataset, our algorithm automatically assigns approprite weights to the batches, based on an appropriate measure of distance to the clean dataset. The algorithm then proceeds to find a predictor based on weighted empirical risk minimization. More information can be found in the paper, which is also provided here.

## Data

The experiments from the paper are performed on the following datasets:

1. [Multitask Dataset of Product Reviews](http://cvml.ist.ac.at/productreviews/) [1] - A dataset of reviews for various Amazon products. The target task is sentiment analysis - is a review positive or negative?
2. [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/) [2] - A dataset of images of various animals, together with 85 binary attributes. The used feature representations can be found [here](https://cvml.ist.ac.at/AwA2/dataset/AwA2-features-ICML2019.zip).

For the Product Reviews, we used the readily available feature representations.

For the Animals with Attributes data, we provide a [link to the used features](https://cvml.ist.ac.at/AwA2/dataset/AwA2-features-ICML2019.zip). We used a ResNet50 pretrained network from the [Tensornets package](https://github.com/taehoonlee/tensornets) to obtain features for this dataset and for the [ImageNet](http://www.image-net.org/) data. Then a PCA projection was learned on ImageNet and applied to the Animals with Attributes 2 data. See the experiments section in the paper for more information.

For completeness, we also provide an example script showing how to extract the ResNet50 features of the images, as well as code for the PCA projection.

## Description

1. code_products contains the code for the Amazon Products experiments
- functions.py contains the main part of our implementation, in particular the functions for preparing the data splits, learning the logistic regression models, minimizing the bound, running the baselines, etc.
- main.py is a script for running the experiments described in the paper.
- get_results_products.ipynb is the script for creating the plots and tables from the paper (once the experiments have been run).

2. code_animals contains the code for the Animals with Attributes 2 experiments
- functions.py contains the main part of our implementation, in particular the functions for preparing the data splits, corrupting the data, learning the logistic regression models, minimizing the	bound, running the baselines, etc.
- main.py is a script for running the experiments described in the paper.
- get_results_animals.ipynb is the script for creating the plots and tables from the paper (once the experiments have been run).
- example_extract.py is an example script for extracing features from images using the [Tensornets package](https://github.com/taehoonlee/tensornets). Optionally, various types of curruptions (e.g. *blurring*) can be applied to the images before the feature representations are extracted.
-PCA_features.py is the code used for performing the PCA projection of the features (see below and our paper for more details).

3. robust_learning_from_untrusted_sources.pdf is the camera-ready version of our paper, together with the supplementary material.

## Dependencies

To run the code, you need:
- python 3
- Tensorflow 1.10
- The relevant dataset stored locally
- The [Tensornets package](https://github.com/taehoonlee/tensornets) and access to the [ImageNet dataset](http://www.image-net.org/), in case you want to extract the features yourself.
- A single experiment can easily be run on one machine. However, to reproduce the results, a large number of independent runs is required, so the experiments are best performed on a scientific cluster with multiple CPU cores, in parallel.

## Running the experiments
The main.py scripts can be used to run experiments like those in the paper.

For the Product Reviews, run:

```
specific=0  # Binary label, indicating the type of experiment (books and non-books, or the experiment on all products)
ind=0  # the index of the independent run or the product ID, depending on the type of experiment
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




<p align="center">
  <img width="756" height="411" src="http://pub.ist.ac.at/crypto/IST_Austria_Logo.jpg">
</p>
