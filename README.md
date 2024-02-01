# MASD

## MASD (Metrics Analysis of Spectral Data) is an Unsupervised Machine Learning Pipeline for Noisy High-dimensional Spectral Data that can help quantify conformational changes of protein from its composite spectra in an end-to-end fashion
This repository contains two scripts in Python that implement a general unsupervised machine learning (ML) pipeline called MASD (Metrics Analysis of Spectral Data) developed under the supervision of professor Giancarlo Franzese (University of Barcelona, Spain). It also contains a dataset, named dataset_fibrinogen, of spectral data of the following type: UV Resonant Raman (UVRR), Circular Dichroism (CD) and UV Absorbance (UV Abs.) spectra, obtained from  Elettra Sincrotone Trieste facility, at Trieste (Italy). This ML methodology has been tested
with spectra of Fibrinogen (Fib) in solution, and also in presence of of either (Carbon) CNP nanoparticles or (Silica) SiNP nanoparticles. Details about the findings can be found in the following paper: (Authors List) A Machine Learning Tool to Analyse Spectroscopic Changes in High-Dimensional Data published in Journal


## Table of contents
- [Python](#Python)
- [Usage](#usage)
  - [Input Data](#input-data)
  - [Data Standardization](#standardize-data)
  - [Input Variables](#input-variables)
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
The folder `fibrinogen_dataset` contains an overall number of 254  two-column files with extension .txt corresponding to UVRR, CD UV spectra of Fibrinogen in water solution alone, and in combination with either CNP or SiNP nanoparticles. The are named in the following way. The files input_UVRR_i,  input_CD_i, input_UV_i where index i corresponds to a different temperature for bulk spectra. The files with same names above but with an additional label either CNP or SiO2NP, e.g.  input_UVRR_CNP_i, correspond to
spectral data of Fibrinogen in presence of either Carbon nanoparticles or Silica nanoparticles, respectively. 

### Data Standardization 
After the data are read by the scripts, the data are subjected to the following modifications: some spectral truncations, a process of combining UVRR, CD and UV specta into a single one, and finally a data standardization as routinely done in principal component analysis
[(PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis). For details we refer the reader to publication in (#citation).

### Input Variables



### Running Scripts
In order to launch the scripts:
 - Edit the scripts assigning to the variable  `path` the path where  the folder `fibrinogen_dataset` is stored in your machine, e.g. path = '/Users/myworkingfolder/dataset_fibrinogen/'
 - Open the terminal and write the line  ```$ python metrics.py``` or ```$ python manifold_learning.py```


### Visualize




### Citation

(Authors List) A Machine Learning Tool to Analyse Spectroscopic Changes in High-Dimensional Data published in Journal







