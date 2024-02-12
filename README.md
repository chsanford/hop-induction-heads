# README

Codebase accompanying the ICML submission entitled "Transformers, parallel computation, and logarithmic depth."
This repository contains the code necessary to train the models detailed in the paper, copies of the trained models used in the paper, and notebooks to generate the respective figures given locally trained models.

The code is written to be run locally on Apple Silicon. This can be adapted to other hardware by changing the `mps` device to your preferred type.

## Setup

First, create a conda environment with the proper dependencies.
To do so, run the following command from this directory:

```
conda env create --name [NAME] --file=environment.yml
```

## Training models

Models can be trained according to the experimental specs specified by the file `conf/runs/[CONFIG].yaml` by running the following command:

```
python src/train.py conf/runs/[CONFIG].yaml
```

## Plots

Run the jupyter notebooks `src/error_plots.ipynb` and `src/interpretability.ipynb` to reproduce the plots in Section 5 and Appendix G.