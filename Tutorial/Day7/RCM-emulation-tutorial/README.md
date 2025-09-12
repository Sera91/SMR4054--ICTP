This folder contains a hands-on Python tutorial for training and evaluating a Deep Learning (DL) based Regional Climate Model (RCM) emulator using an early version of the CORDEX-ML-Benchmark. It serves as an illustration for users who are beginning to develop RCM emulators. Here, they will find notebooks that cover how to access the data, load the different experiments, and train DL models with PyTorch. The set of dependencies is kept minimal to ensure accessibility for most users.

In `./requirements`, there is an `environment.yaml` file to easily replicate the Conda environment required to run all the scripts in this folder. To create the environment, run the following command:

```bash
conda env create -f environment.yaml
```


Alternatively, the basic requirements to run these scripts are:

```
os
requests
zipfile
xarray
netcdf4
matplotlib
cartopy
numpy
torch
```

These packages can be installed using any package management tool.

The CORDEX-ML-Benchmark data is publicly available on [Zenodo](https://zenodo.org/records/15797226) as a `zip` file containing all the NetCDF files for the different training and evaluation experiments. The notebook `./data/data_download.ipynb` provides code for downloading this data (for any of the two geographical domains included so far). The data is around 5 GB per domain. After downloading, `./data/training_evaluation_experiments.ipynb` provides a walkthrough of the data, helping users understand which data to use for training, which to use for evaluation, and what each dataset represents. We encourage users to carefully review this notebook to become familiar with the benchmark.

Once the data is downloaded, `./dl-model/training_evaluation.ipynb` demonstrates the full process of building an RCM emulator using the benchmark dataset. This notebook uses pure PyTorch for the DL model and shows how to train and evaluate it across different frameworks (focusing on the Perfect Framework for training). It encompasses the main workflow for working with the benchmark, allowing users, once the benchmark is publicly available, to contribute by scoring their own RCM emulators against the baseline models (currently under construction) based on the various evaluation experiments.

If you would like to stay informed about the Benchmark and contribute when it becomes available, please contact Jose González-Abad (gonzabad@ifca.unican.es).