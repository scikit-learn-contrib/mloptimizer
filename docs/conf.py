# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import pathlib
from plotly.io._sg_scraper import plotly_sg_scraper

# Path to the VERSION file
version_file = pathlib.Path(__file__).parent.parent / "mloptimizer" / "VERSION"
version = version_file.read_text().strip()

sys.path.insert(0, os.path.abspath('..'))

project = 'mloptimizer'
copyright = '2024, Antonio Caparrini, Javier Arroyo'
author = 'Antonio Caparrini, Javier Arroyo'
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'sphinx_mdinclude',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    #'autoapi.extension',
    'sphinx_favicon',
    'sphinxcontrib.mermaid',
    'sphinx_github_changelog',
    'sphinx_contributors',
    'sphinx_copybutton',
]

sphinx_github_changelog_token = os.getenv("SPHINX_GITHUB_CHANGELOG_TOKEN")
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    # MLflow inventory temporarily disabled due to 403 errors
    # 'mlflow': ('https://www.mlflow.org/docs/latest/', None),
    'xgboost': ('https://xgboost.readthedocs.io/en/latest/', None),
    'lightgbm': ('https://lightgbm.readthedocs.io/en/latest/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),  # Updated: removed '/reference/' from URL
    'deap': ('https://deap.readthedocs.io/en/master/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    '*/test/*']

# autoapi_dirs = ['../mloptimizer']
# autoapi_type = "python"
# autoapi_options = [
#    "members",
#    "undoc-members",
#    "show-inheritance",
#    "show-module-summary",
#    "imported-members",
#]
#autodoc_typehints = "signature"
#autoapi_ignore = ['*test*']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

image_scrapers = ('matplotlib', plotly_sg_scraper,)

autodoc_default_flags = ['members']
autodoc_default_options = {
    'members': True,
    'undoc-members': True,  # Include undocumented members
    'show-inheritance': True,  # Display class inheritance
    'inherited-members': True  # Document inherited members
}
autodoc_typehints_format = 'fully-qualified' # has to be a one of ('fully-qualified', 'short')
autodoc_inherit_docstrings = True

autosummary_generate = False
autoclass_content = 'both'

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_show_sourcelink = True
set_type_checking_flag = True

sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        # The module you locally document uses None
        'mloptimizer': None,
    },
    'image_scrapers': image_scrapers,
}

html_theme_options = {
    "show_nav_level": 2,  # Control the navigation levels displayed
    "show_prev_next": False,  # Hide previous/next links
    "navigation_depth": 3,  # Set navigation depth to control sidebar depth
    "navbar_align": "left",  # Align the navbar to the left
    "icon_links": [
        {
            "name": "GitHub",  # The name that will appear on hover
            "url": "https://github.com/Caparrini/mloptimizer",  # Your repository URL
            "icon": "fab fa-github-square",  # FontAwesome icon for GitHub
            "type": "fontawesome",  # Use FontAwesome icons
        }
    ],
    "navbar_end": ["theme-switcher", "icon-links"],  # Add the icon to the end of the navbar
}

html_logo = 'http://raw.githubusercontent.com/Caparrini/mloptimizer-static/main/logos/mloptimizer_banner_readme.png'

favicons = [
    "https://raw.githubusercontent.com/Caparrini/mloptimizer-static/main/logos/favicon-16x16.png",
    "https://raw.githubusercontent.com/Caparrini/mloptimizer-static/main/logos/favicon-32x32.png",
    "https://raw.githubusercontent.com/Caparrini/mloptimizer-static/main/logos/favicon.ico",
]