# MASD

## MASD (Metrics Analysis of Spectral Data) is an Unsupervised Machine Learning Pipeline for Noisy High-dimensional Spectral Data that can help quantify conformational changes of protein from its composite spectra in a end-to-end fashion
This repository contains two scripts in Python that implement a general unsupervised machine learning (ML) pipeline called MASD (Metrics Analysis of Spectral Data) developed under the supervision of professor Giancarlo Franzese (University of Barcelona, Spain). It also contains a dataset, named dataset_fibrinogen, of spectral data of the following type: UV Resonant Raman (UVRR), Circular Dichroism (CD) and UV Absorbance (Abs.) spectra, obtained from  Elettra Sincrotone Trieste facility, at Trieste (Italy). This ML methodology has been tested
with spectra of Fibrinogen (Fib) in solution, and also with Fib in presence of of either (Carbon) CNP nanoparticles or (Silica) SiNP nanoparticles. Details about the findings can be found in the following paper: 


## Table of contents
- [Python](#Python)
- [Usage](#usage)
  - [Input Data](#input-data)
  - [Data Standardization](#standardize-data)
  - [Running Scripts](#Running-script)
  - [Visualize](#visualize)
- [Documentation](#documentation)
- [Interactive](#interactive)
- [Contributing](#contributing)
- [Citation](#citation)

## Python Versions and Required Packages  
`MASD` has been tested  with Python 3.8.10. 3.8.16, 3.11.5

The following libraries are also required for running the scripts
 - NumPy, SciPy, and Matplotlib
 - [Scikit-learn](https://scikit-learn.org/stable/index.html)

## Usage
These steps introduce how to use `metrics.py` and  `manifold_learning.py`

### Input Data
The `fibrinogen_dataset`  ....

### Data Standardization 
The `fibrinogen_dataset`  ....


### Running Scripts
In order to launch the scripts:
 - Edit the script changing the variable  `path` to the place where  the folder `fibrinogen_dataset` is stored in your machine, e.g. path = '/Users/myfolder/dataset_fibrinogen/'
 - Open the terminal and write the line  ```$ python metrics.py``` or ```$ python manifold_learning.py```









```bash
$ conda install -c conda-forge kneed
```
