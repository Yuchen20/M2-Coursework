nbsphinx_kernel_name = 'python3'

import sphinx_rtd_theme
import os
import sys

# Add the project root to the path so that the notebooks can import the package
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Time Series Forecasting with LLMs'
copyright = '2025, Yuchen Mao'
author = 'Yuchen Mao'

# The full version, including alpha/beta/rc tags
release = 'beta'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'myst_parser',
    'sphinx.ext.autodoc',    # Autodoc extension for extracting docstrings
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',   # Add links to view source code
    'sphinx.ext.napoleon',   # Support for NumPy and Google style docstrings
    'sphinx_gallery.load_style',  # load CSS for gallery (needs SG >= 0.6)
]

# -- nbsphinx specific configuration -----------------------------------------

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = 'never'  # Set to 'never' to avoid long build times, change to 'auto' if desired

# Resolve notebook path issues by configuring the proper path
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

.. note::
   
   This page was generated from a Jupyter notebook. 
   `View the original notebook <https://github.com/yourusername/time_series_llm/blob/main/{{ docname }}>`_
"""

# Fix notebook path issues - specify the path to the notebooks
nbsphinx_notebooks = "../notebooks"

# Additional arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_allow_errors = True  # Continue building even if a notebook execution fails

# -- Options for HTML output -------------------------------------------------

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Fix autodoc issues by configuring sys.path for module imports
sys.path.append(os.path.abspath('../src'))
autodoc_mock_imports = ['torch', 'transformers', 'wandb', 'accelerate', 'tqdm', 'h5py', 'pandas', 'numpy', 'matplotlib']
autodoc_mock_imports = ["sklearn"]

master_doc = 'index'

highlight_language = 'python3'

# Disable section numbering
secnumber_suffix = ''  # No suffix means no section numbers

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'custom.css',
]

# Add custom CSS file if it doesn't exist
if not os.path.exists(os.path.join(os.path.dirname(__file__), '_static')):
    os.makedirs(os.path.join(os.path.dirname(__file__), '_static'))

if not os.path.exists(os.path.join(os.path.dirname(__file__), '_static', 'custom.css')):
    with open(os.path.join(os.path.dirname(__file__), '_static', 'custom.css'), 'w') as f:
        f.write("""
/* Add custom styling for nbsphinx notebooks */
div.nbinput.container div.input_area {
    background-color: #f5f5f5;
    border: 1px solid #ccc;
    border-radius: 4px;
}

div.nboutput.container div.output_area {
    background-color: #fcfcfc;
    border: 1px solid #eee;
    border-radius: 4px;
}

/* Better code highlighting */
pre {
    padding: 10px;
    background-color: #f8f8f8;
    border: 1px solid #e1e4e5;
    border-radius: 4px;
}
        """)

# Fix notebook paths in the toctree
def setup(app):
    app.connect('builder-inited', fix_notebook_paths)

def fix_notebook_paths(app):
    """Fix notebook paths to properly handle the notebooks directory."""
    # This ensures notebooks are found in the correct location
    app.config.nbsphinx_notebooks = os.path.join(app.srcdir, "..", "notebooks")