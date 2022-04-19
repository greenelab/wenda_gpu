# wenda_gpu

Domain adaptation allows for development of predictive models even in cases with limited or unlabeled sample data.
This repo is a fast implementation of one domain adaptation method, weighted elastic net domain adaptation, or wenda.
The original paper on wenda can be found here: (https://academic.oup.com/bioinformatics/article/35/14/i154/5529259).

## Setup
We recommend running this repo using the conda environment specified in environment.yaml. 
To build and activate this environment run:
```
conda env create --file environment.yaml
conda activate wenda_gpu_paper
```

## Usage
The wenda_gpu software runs in two major steps, one GPU-dependent one not.

The first step, training feature models on the source data and generating confidence scores on the target data, is run in batches on the GPU.
This can be done manually using `train_feature_models.py` and specifying a batch size and starting location, but we recommend using `main.sh` for ease of use.

Once all batches have been run and confidence scores obtained, the final elastic net model is fitted. This step is performed by `train_elastic_net.py`, and can be called directly or through `main.sh`.

## Citation
If you use this method, please cite the following:

`wenda_gpu: fast domain adaptation for genomic data
Ariel A. Hippen, Jake Crawford, Jacob R. Gardner, Casey S. Greene
bioRxiv 2022.04.09.487671; doi: https://doi.org/10.1101/2022.04.09.487671`
