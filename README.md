# wenda_gpu: fast domain adaptation for genomic data

## Overview

Domain adaptation allows for development of predictive models even in cases with limited or unlabeled sample data, by developing predictors for the data of interest (target data) using labeled data from a similar distribution (source data).
This repo is a fast implementation of one domain adaptation method, weighted elastic net domain adaptation, or wenda.
It leverages the complex interactions between biological features (such as genes) to optimize a model’s predictive power on both source and target datasets. 

## Installation

This package can be installed using pip:
```
pip install wenda_gpu
```

Alternatively, you can install the latest development version directly from this GitHub repository:
```
pip install git+https://github.com/greenelab/wenda_gpu
```

## Usage

The most basic usage of wenda is this:

```
from wenda_gpu import wenda_gpu as wg

source_data, target_data = wg.load_data(prefix="sample")
source_data_normed, target_data_normed = wg.normalize_data(source_data, target_data)
wg.train_feature_models(source_data_normed, target_data_normed, prefix="sample")
source_y = wg.load_labels(prefix="sample")
wg.train_elastic_net(source_data_normed, source_y, target_data_normed, prefix="sample")
```

For a step-by-step tutorial in running wenda_gpu, consult wenda_gpu_quick_usage.ipynb in the example folder.

## Directory structure

By default, wenda_gpu implements the following structure in your working directory:

```
working_directory
    ├── data
    │   └── prefix
    │       ├── source_data.tsv
    │       ├── source_y.tsv
    │       └── target_data.tsv
    ├── feature_models
    │   └── prefix
    │       ├── model_0.pth
    │       ├── model_1.pth
    │       └── ...
    ├── confidences
    │   └── prefix
    │       ├── confidences.tsv
    │       ├── model_0_confidence.txt
    │       ├── model_1_confidence.txt
    │       └── ...   
    └── output
        └── prefix
            ├── k_00
            │   ├── target_predictions.txt
            │   └── target_probabilities.txt
            ├── k_01
            │   ├── target_predictions.txt
            │   └── target_probabilities.txt
            └── ...
```
"prefix" is intended to be a unique identifier for your dataset, which allows you to run wenda_gpu on multiple datasets and have them nested within the same directory structure.

The user will need to create the files under the `data` directory, containing the feature information for both source and target datasets and the labels for the source data. Data can be loaded from a different source, for an example consult wenda_gpu_quick_usage.ipynb.
The files under the `feature_models`, `confidences`, and `output` directories will be automatically created by wenda_gpu. If you want intermediate files and output in a different location than inside your working directory, you can specify your own paths using the path arguments in the related functions, e.g.

```
wg.train_feature_models(source_data_normed, target_data_normed, prefix="sample", feature_model_path="~/wenda_gpu_run/feature_models", confidence_path="~/wenda_gpu_run/confidences")
```

## Helpful links

Example usage of this software and results can be found here: (https://github.com/greenelab/wenda_gpu_paper).
The original paper on wenda can be found here: (https://academic.oup.com/bioinformatics/article/35/14/i154/5529259).

## Citation
If you use this method, please cite the following:

`wenda_gpu: fast domain adaptation for genomic data
Ariel A. Hippen, Jake Crawford, Jacob R. Gardner, Casey S. Greene
bioRxiv 2022.04.09.487671; doi: https://doi.org/10.1101/2022.04.09.487671`
