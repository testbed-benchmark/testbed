"""Configuration for Sphinx."""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import pathlib
import sys
from datetime import datetime, timezone

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(PROJECT_DIR))

# -- Project information -----------------------------------------------------

project = "testbed"
copyright = f"{datetime.now(timezone.utc).year}, The TestBED authors"
author = "The TestBED authors"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "autoapi.extension",
]

nb_execution_mode = "cache"

myst_enable_extensions = [
    "dollarmath",
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

# source_suffix = {
#     ".rst": "restructuredtext",
#     ".md": "markdown",
# }

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://anonymised-for-review",  # anonymised for double-blind review
    "use_repository_button": False,
    "use_download_button": False,
    "show_toc_level": 2,
    "show_prev_next": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

autosummary_generate = True
