# CD-BPR
##### A simple yet effective interpretable Bayesian Personalized Ranking for the Assessment of Psychiatric Disorders, 2024

---

The current repository contains all the code and data necessary to reproduce the paper results. 

## Installation
You first need to download all the files at the following link to have the required datasets for computation : 

https://zenodo.org/records/11062298?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM2YWQ5Y2NlLTg3ZDQtNDEzNi04ZTgyLTlkZGVlMjVmYTFmNyIsImRhdGEiOnt9LCJyYW5kb20iOiJhNTg2NGI4NjUyZDFmZTA1ODk1MmQ2ZDRlMjFhN2I5YiJ9.ziZvmd7gdUrs2ff-wkayzmBXwg-Wp3VJMbwfiuYWgbnqlxWH0PoIOZaLGFzRch-nxKvBCZD3oSdjYt4cyQQ9dg

Place results.zip and data.zip at the root of the current folder.
Easily set up the environment (data and libraries) with `make` command :  

```
make
```

Prerequisite : [conda library]( https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)\
Attention: It is recommended to uninstall the pytorch library provided with the conda environment to download yourself the version most adapted to your machine from the pytorch [website](https://pytorch.org/get-started/locally/).

## Reproductibility

All the commands needed to reproduce the paper results are written in the `Experiments.ipynb` jupyter notebook at the root of the directory.
