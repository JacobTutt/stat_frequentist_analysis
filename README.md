# Signal Recovery in the Presence of Background: Multi-dimensional Likelihood vs. sWeight Reconstruction

## Author: Jacob Tutt, Department of Physics, University of Cambridge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/stat-frequentist-analysis/badge/?version=latest)](https://stat-frequentist-analysis.readthedocs.io/en/latest/?badge=latest)

## Description
This repository compares the statistical power and performance of a multidimensional Extended Maximum Likelihood Estimate (MLE) with an ‘sWeighted’ fit, which isolates the Signal distribution in the control variable using fits from the independent variables. It contains the package, its documentation, and implementation required for the analysis.


This repository forms part of the submission for the MPhil in Data Intensive Science's S1 Statistics Course at the University of Cambridge.


## Table of Contents
- [Pipeline Functionalities](#pipelines)
- [Notebooks](#notebooks)
- [Documentation](#documentation)
- [Installation](#installation-and-usage)
- [License](#license)
- [Support](#support)
- [Author](#author)


## Pipelines

This example is built upon **four fundamental probability distributions**, which are implemented as individual classes within the [Base_Dist](Stats_analysis/Base_Dist) module. These distributions serve as the building blocks which functions and properties can be inherited by classes which combine them to describe more.

### **[Base Probability Distributions](Stats_analysis/Base_Dist)**

The base probability distributions are as follows:

- **[Crystal Ball Distribution](Stats_analysis/Base_Dist/CrystalBall_Class.py)**
- **[Exponential Decay Distribution](Stats_analysis/Base_Dist/ExponentialDecay_Class.py)**
- **[Normal Distribution](Stats_analysis/Base_Dist/NormalDistribution_Class.py)**
- **[Uniform Distribution](Stats_analysis/Base_Dist/UniformDistribution_Class.py)**

Each of these distributions is encapsulated in its own class, providing methods for calculating probability density functions (PDFs), cumulative distribution functions (CDFs), and performing distribution fitting.

---

### **[Compound Distributions](Stats_analysis/Compound_Dist/)**

The compound distributions are **two-dimensional (2D) probability distributions** that combine properties of the base distributions. These are implemented as classes in the [Compound_Dist](Stats_analysis/Compound_Dist) module and inherit the behaviors of their constituent base distributions.

#### **[Signal Distribution](Stats_analysis/Compound_Dist/Signal_Class.py)**
Represents the signal region in 2D space.
- **Constituents**:
  - **Crystal Ball Distribution** in the X-dimension.
  - **Exponential Decay Distribution** in the Y-dimension.

#### **[Background Distribution](Stats_analysis/Compound_Dist/Background_Class.py)**
Represents the background noise in 2D space.
- **Constituents**:
  - **Uniform Distribution** in the X-dimension.
  - **Normal Distribution** in the Y-dimension.

---

### **[Overall Distribution](Stats_analysis/Compound_Dist/Signal_Background_Class.py)**

The **total distribution** is constructed from the Signal and Background distributions. This is implemented as a separate class in the `Compound_Dist` module and inherits the properties of the Signal and Background distributions, along with their respective base distributions.

- **Constituents**:
  - **Signal Distribution**
  - **Background Distribution**

By using inheritance, the total distribution can integrate all its constituent distributions in a modular way and easily adaptable for different base distributions in other senarios.

## Notebooks

The [notebooks](notebooks) in this repository serve as walkthroughs for the analysis performed. They include derivations of the mathematical implementations, explanations of key choices made, and present the main results. Five notebooks are provided:

| Notebook | Description |
|----------|-------------|
| [Notebook 1](notebooks/Notebook_Part_b.ipynb) | Introduces and implements the four base probability distributions and their combination into signal and background components. Verifies proper normalisation over the truncated domain. |
| [Notebook 2](notebooks/Notebook_Part_c.ipynb) | Demonstrates the calculation and visualisation of marginal probability distributions in both X and Y, including how to implement it in the pipeline. |
| [Notebook 3](notebooks/Notebook_Part_d.ipynb) | Overview of the sampler (accept/reject algorithm) with automatic scaling and recovery of model parameters using Extended Unbineed Maxmimium Likelihood Fitting with iminuit. |
| [Notebook 4](notebooks/Notebook_Part_e.ipynb) | Performs a full bootstrap simulation study, including generation of samples and analysing trends in bias and uncertainty as functions of sample size. |
| [Notebook 5](notebooks/Notebook_Part_f.ipynb) | This explores the use of Sweights, an algorithm in which fits the marginal distribution in a marginalised axis using an Extended Likelihood fit, assigns statistical weights to events, and reconstructs the signal distribution in an indenpendent axis, removing all consideration of background distribution for this dimension|

## Documentation

[Documentation on Read the Docs](https://stat-frequentist-analysis.readthedocs.io/en/latest/)

The pipeline uses a modular, inherited class-based structure, which is explained below, to make it adaptable to different probability distributions. As a result documentation has been created for easier understanding of each functions methods and implementation:

- **Class and Function References**: Includes detailed descriptions of all classes and functions used in the coursework.
- **Source Code Links**: Direct links to the source code for easy review.
- **Notebook Integration**: Hyperlinks throughout the notebooks provide direct access to relevant sections of the documentation.


## Installation and Usage

To run the notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository to your local machine.
Or your 
```bash
git clone https://github.com/JacobTutt/stat_frequentist_analysis.git
```

### 2. Create a Fresh Virtual Environment
Use a clean virtual environment to avoid dependency conflicts.
```bash
python -m venv env
source env/bin/activate   # For macOS/Linux
env\Scripts\activate      # For Windows
```

### 3. Install the Package and Dependencies
Navigate to the repository’s root directory and install the package along with its dependencies:
```bash
pip install -e .
```

### 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Statistical Analysis"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel (Statisical Analysis) to run the code.


## For Assessment
- The associated project report can be found under [Project Report](report/Report_jlt.pdf). 

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the [LICENSE](LICENSE) file for details.

## Support
If you have any questions, run into issues, or just want to discuss the project, feel free to:
- Open an issue on the [GitHub Issues](https://github.com/JacobTutt/stat_frequentist_analysis/issues) page.  
- Reach out to me directly via [email](mailto:jacobtutt@icloud.com).

## Author
This project is maintained by Jacob Tutt 


## Declaration of Use of Autogeneration Tools
This report made use of Large Language Models (LLMs) to assist in the development of the project.
These tools have been employed for:
- Formatting plots to enhance presentation quality.
- Performing iterative changes to already defined code.
- Debugging code and identifying issues in implementation.
- Helping with Latex formatting for the report.
- Identifying grammar and punctuation inconsistencies within the report.
- Helping to generate the repository's metadata files.
