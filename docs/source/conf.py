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
# Autodoc configuration
autodoc_default_options = {
    'members': True,             # Include all members (functions, classes, etc.)
    'undoc-members': True,       # Include members without docstrings
    'private-members': True,     # Include private members (_name)
    'special-members': '__init__',  # Include special methods (e.g., __init__)
    'show-inheritance': True,    # Show class inheritance
}
autodoc_mock_imports = []  # Mock imports if some dependencies are unavailable
html_theme = 'sphinx_rtd_theme'

import sys
print("Python Path:", sys.path)
try:
    import Stats_Analysis
    print("Stats_Analysis imported successfully")
except ImportError:
    print("Failed to import Stats_Analysis")