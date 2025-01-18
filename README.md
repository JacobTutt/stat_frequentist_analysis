# Frequentist Statistical Analysis

This repository contains the package, its documentation, and implementation required for the analysis. It compares the statistical power and performance of a multidimensional Extended Maximum Likelihood Estimate (MLE) with an ‘sWeighted’ fit, which isolates the Signal distribution in the control variable using fits from the independent variables.

---

## Installation Instructions

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
cd jlt67
pip install -e .
```

### 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Statistical Analysis"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel (Statisical Analysis) to run the code.

## Report for the Pipeline

A report on the pipeline can be found under the **Report** directory of the repository

## Documentation for the Repository

[Documentation on Read the Docs](https://stat-frequentist-analysis.readthedocs.io/en/latest/)

The pipeline uses a modular, inherited class-based structure, which is explained below, to make it adaptable to different probability distributions. As a result documentation has been created for easier understanding of each functions methods and implementation

### Key Features of the Documentation

- **Class and Function References**: Includes detailed descriptions of all classes and functions used in the coursework.
- **Source Code Links**: Direct links to the source code for easy review.
- **Notebook Integration**: Hyperlinks throughout the notebooks provide direct access to relevant sections of the documentation.


## Structure of the Analysis

This example is built upon **four fundamental probability distributions**, which are implemented as individual classes within the `Base_Dist` module. These distributions serve as the building blocks which functions and properties can be inherited by classes which combine them to describe more.

### Base Probability Distributions

The base probability distributions are as follows:

- **Crystal Ball Distribution**
- **Exponential Decay Distribution**
- **Normal Distribution**
- **Uniform Distribution**

Each of these distributions is encapsulated in its own class, providing methods for calculating probability density functions (PDFs), cumulative distribution functions (CDFs), and performing distribution fitting.

---

### Compound Distributions

The compound distributions are **two-dimensional (2D) probability distributions** that combine properties of the base distributions. These are implemented as classes in the `Compound_Dist` module and inherit the behaviors of their constituent base distributions.

#### Signal Distribution
Represents the signal region in 2D space.
- **Constituents**:
  - **Crystal Ball Distribution** in the X-dimension.
  - **Exponential Decay Distribution** in the Y-dimension.

#### Background Distribution
Represents the background noise in 2D space.
- **Constituents**:
  - **Uniform Distribution** in the X-dimension.
  - **Normal Distribution** in the Y-dimension.

---

### Overall Distribution

The **total distribution** is constructed from the Signal and Background distributions. This is implemented as a separate class in the `Compound_Dist` module and inherits the properties of the Signal and Background distributions, along with their respective base distributions.

- **Constituents**:
  - **Signal Distribution**
  - **Background Distribution**

By using inheritance, the total distribution can integrate all its constituent distributions in a modular way and easily adaptable for different base distributions in other senarios.


