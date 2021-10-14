# wenda_gpu_MRE

This is a minimum reproducible example of the differences I'm seeing between wenda (https://academic.oup.com/bioinformatics/article/35/14/i154/5529259) implementations when using GPy to create the GP models vs GPyTorch. For example, when trained on the same data, one model's predicted variances on the target data can be different by as much as an order of magnitude. Hopefully there's some small difference in default parameter setting between the packages that can help us get comparable results across packages.
Included data is methylation data as used in the paper linked above.
