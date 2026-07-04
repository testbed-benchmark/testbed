"""Configuration for Sphinx."""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import pathlib
import sys
from datetime import date

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(PROJECT_DIR))

# -- Project information -----------------------------------------------------

project = "testbed"
copyright = f"{date.today().year}, The TestBED authors"
author = "The TestBED authors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "autoapi.extension",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# AutoAPI configuration
autoapi_dirs = ["../testbed"]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_options = ["show-module-summary", "members", "private-members"]
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/testbed-benchmark/testbed",
    "use_repository_button": True,
    "use_download_button": False,
    "show_toc_level": 2,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

autosummary_generate = True
