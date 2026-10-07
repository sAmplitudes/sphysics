# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import subprocess

project = 'sphysics'
copyright = '2022-2026, the sphysics authors'
author = 'Stefan Wallner'

def get_version():
    """Get the version from Git tags or the VERSION file."""
    try:
        # Get the latest Git tag
        version = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], stderr=subprocess.DEVNULL).decode().strip()
        return version
    except subprocess.CalledProcessError:
        return "0.0.0"

release = get_version()

master_doc = "index"

# -- General configuration --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.viewcode', 'sphinx_rtd_theme', 'sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'sphinx.ext.napoleon',]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autodoc_member_order = 'bysource'  # Keep function/class order as in source code

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members" : True,
    "inherited-members": True,
}

autodoc_mock_imports = [
    "lightning",
    "numba",
    "tensorflow",
    "tensorflow_probability",
    "tf_keras",
    "torch",
    "torchinfo",
    "torchmetrics",
    "triton",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_theme = "sphinx_book_theme"
#html_static_path = ['_static']
#html_js_files = ['custom.js']
