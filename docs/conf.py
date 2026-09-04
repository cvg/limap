# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import limap
# version = limap.__version__
version = "2.1.0.dev"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LIMAP"
copyright = "CVG @ ETH Zurich"
author = "LIMAP Contributors"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_toolbox.more_autodoc.autonamedtuple",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ["_themes"]
html_css_files = [
    "css/fix-rtd-property.css"  # workaround readthedocs/sphinx_rtd_theme#1301
]


# -- Autodoc tweaks -------------------------------------------------------


# The overload lists pybind11 writes into the docstrings of the option classes
# contain `**kwargs`, which docutils reads as an unterminated strong emphasis.
# Escape the stars so those lines render literally.
def _escape_star_args(app, what, name, obj, options, lines):
    for i, line in enumerate(lines):
        if "*args" in line or "**kwargs" in line:
            lines[i] = line.replace("**kwargs", r"\*\*kwargs").replace(
                "*args", r"\*args"
            )


def setup(app):
    app.connect("autodoc-process-docstring", _escape_star_args)
