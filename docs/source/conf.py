# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Include your package directory

project = 'S1 Courework'
copyright = '2024, Jacob Tutt'
author = 'Jacob Tutt'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # Supports Google and NumPy style docstrings
    'sphinx.ext.viewcode',       # Adds links to source code
    'sphinx_autodoc_typehints',  # Includes type hints in the documentation
]

# Theme
html_theme = 'sphinx_rtd_theme'