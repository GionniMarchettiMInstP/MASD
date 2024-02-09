# MASD

## MASD (Metrics Analysis of Spectral Data) is an Unsupervised Machine Learning Pipeline for Noisy High-dimensional Spectral Data that can help quantify conformational changes of protein from its composite spectra in an end-to-end fashion
This repository contains two scripts in Python that implement a general unsupervised machine learning (ML) pipeline called MASD (Metrics Analysis of Spectral Data) developed under the supervision of professor Giancarlo Franzese (University of Barcelona, Spain). It also contains a dataset, named dataset_fibrinogen, of spectral data of the following type: UV Resonant Raman (UVRR), Circular Dichroism (CD) and UV Absorbance (UV Abs.) spectra, obtained from  Elettra Sincrotone Trieste facility, at Trieste (Italy)(https://www.elettra.eu/it/index.html). This ML methodology has been tested
with spectra of Fibrinogen (Fib) in solution, and also in presence of of either (Carbon) CNP nanoparticles or (Silica) SiNP nanoparticles. Details about these findings can be found in the following paper [Citation](#citation).

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
 - [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Matplotlib](https://matplotlib.org/)
 - [Scikit-learn](https://scikit-learn.org/stable/index.html)

## Usage
These steps introduce how to use `metrics.py` and  `manifold_learning.py`

### Input Data
The folder `fibrinogen_dataset` contains an overall number of 254  two-column files with extension .txt corresponding to UVRR, CD UV spectra of Fibrinogen in water solution alone, and in combination with either CNP or SiNP nanoparticles. They are named in the following way. The files input_UVRR_i,  input_CD_i, input_UV_i where index i corresponds to a different temperature T for bulk spectra, i.e., Fibrinogen in solution alone. The files with same names above but with an additional label either CNP or SiO2NP, e.g.  input_UVRR_CNP_i, correspond to
spectral data of Fibrinogen in presence of either Carbon nanoparticles or Silica nanoparticles, respectively, at each temperature. 

### Data Standardization 
After the data are read by the scripts, the data are subjected to the following modifications: some spectral truncations, a process of combining each UVRR, CD and UV spectrum with same temperature T into a single one, and finally a data standardization as routinely done in principal component analysis [(PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis). For details we refer the reader to publication in [Citation](#citation)
.

### Input Variables
The variables used by the script are the following:
- nos: Number of UVRR spectra Fib, Fib + CNP, Fib + SIO2NP
- nou: Number of UV spectra  Fib + CNP
- noc: Number of CD spectra Fib + CNP
- noua: Number of UV spectra  Fib + SIO2NP
- noca: Number of CD spectra Fib + SIO2NP
- irs: Index for reference spectrum
### Running Scripts
In order to launch the scripts:
 - Edit the scripts assigning to the variable  `path` the path where  the folder `fibrinogen_dataset` is stored in your machine, e.g. path = '/Users/myworkingfolder/dataset_fibrinogen/'
 - Open the terminal and write the line  ```$ python metrics.py``` or ```$ python manifold_learning.py```


### Visualize




### Citation

(Authors List) A Machine Learning Tool to Analyse Spectroscopic Changes in High-Dimensional Data published in Journal







