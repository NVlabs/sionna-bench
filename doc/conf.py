# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Sionna Bench'
copyright = ' 2026 NVIDIA CORPORATION & AFFILIATES.'
author = 'Gian Marti'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


import os
import shutil
import sys
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOC_DIR.parent / "src"))

import sionna

LEADERBOARD_DIR = DOC_DIR.parent / "leaderboard" / "mimo_ofdm_chest"
LEADERBOARD_FILES = (
    "app.js",
    "color_palette.js",
    "data.json",
    "styles.css",
)

SIONNA_ROOT = Path(sionna.__file__).resolve().parents[2]
SIONNA_DOC_DIR = Path(
    os.environ.get("SIONNA_DOC_DIR", SIONNA_ROOT / "doc" / "source")
).resolve()
SIONNA_INPUT_OUTPUT_EXTENSION = (
    SIONNA_DOC_DIR / "_ext" / "input_output_fields.py"
)
if not SIONNA_INPUT_OUTPUT_EXTENSION.is_file():
    raise RuntimeError(
        "Building the documentation requires a Sionna source checkout "
        "containing doc/source/_ext/input_output_fields.py. Set "
        "SIONNA_DOC_DIR to its doc/source directory when Sionna is not "
        "installed editably."
    )

sys.path.insert(0, str(DOC_DIR))
sys.path.insert(0, str(SIONNA_DOC_DIR))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "_ext.input_output_fields",
    # "myst_parser",
    "sphinx_copybutton",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
bibtex_bibfiles = ["references.bib"]

add_module_names = False
toc_object_entries_show_parents = 'hide'   # Sphinx ≥ 5.2

root_doc = "index"

# -- Intersphinx configuration ---------------------------------------------------
sionna_objects_inv = SIONNA_DOC_DIR.parent / "build" / "html" / "objects.inv"
intersphinx_mapping = {
    "sionna": (
        "https://nvlabs.github.io/sionna/",
        str(sionna_objects_inv) if sionna_objects_inv.is_file() else None,
    ),
}

# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = "groupwise"
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "fully-qualified"  # Show full paths like np.random.Generator
# Prefer built-in type names for display; typing_links extension rewrites links to stdtypes
autodoc_type_aliases = {
    "List": "list",
    "Dict": "dict",
}
# Globally exclude inherited Keras/Block plumbing methods from rendered docs.
# `build` is a Keras lifecycle hook, not a user-facing API.
autodoc_default_options = {
    "exclude-members": "build",
}
autosummary_generate = True  # Generate autosummary pages for all modules
autosummary_generate_overwrite = True  # Overwrite existing autosummary pages

autosummary_context = {
    "shared_auxiliary_api_classes": {
        "sionna.bench.mimo_ofdm_chest.papers.koller2022.SpectralGMMChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.melgar2022.InterpolatedDNNChannelEstimator",
    },
    "nn_asset_api_classes": {
        "sionna.bench.mimo_ofdm_chest.papers.baur2022.VAEChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.melgar2022.DNNChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.melgar2022.InterpolatedDNNChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.li2020.ReEsNetChannelEstimator",
    },
    "statistical_asset_api_classes": {
        "sionna.bench.mimo_ofdm_chest.papers.koller2022.SpatialGMMChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.koller2022.SpectralGMMChannelEstimator",
    },
    "covariance_asset_api_classes": {
        "sionna.phy.ofdm.LMMSEChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.edfors1996.TruncatedChannelEstimator",
    },
    "asset_api_modules": {
        "sionna.phy.ofdm.LMMSEChannelEstimator":
            "sionna.bench.mimo_ofdm_chest.papers.builtin.asset",
    },
    "nn_asset_api_members": {
        "sionna.bench.mimo_ofdm_chest.papers.baur2022.VAEChannelEstimator": (
            "generate_training_data",
            "train_model",
        ),
        "sionna.bench.mimo_ofdm_chest.papers.li2020.ReEsNetChannelEstimator": (
            "generate_training_data",
            "train_model",
        ),
        "sionna.bench.mimo_ofdm_chest.papers.melgar2022.DNNChannelEstimator": (
            "generate_model2_training_data",
            "train_model2",
        ),
        "sionna.bench.mimo_ofdm_chest.papers.melgar2022.InterpolatedDNNChannelEstimator": (
            "generate_model3_training_data",
            "train_model3",
        ),
    },
    "tuning_api_classes": {
        "sionna.bench.mimo_ofdm_chest.papers.edfors1996.TruncatedChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.baur2022.VAEChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.koller2022.SpatialGMMChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.koller2022.SpectralGMMChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.li2020.ReEsNetChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.melgar2022.DNNChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.melgar2022.InterpolatedDNNChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.coleri2002.CombInterpolator",
        "sionna.bench.mimo_ofdm_chest.papers.delfeld2025.RKHSChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.lyu2026.SparseDiffuseChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.wei2017.SBLChannelEstimator",
        "sionna.bench.mimo_ofdm_chest.papers.balevi2020.DeepPriorChannelEstimator",
    }
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "auto"}
html_static_path = ['_static']
html_css_files = [
    "leaderboard/styles.css",
    "css/custom.css",
]
html_js_files = [
    "js/force-light.js",
    ("leaderboard/app.js", {"type": "module"}),
]


html_theme_options = {
    "logo": {
        "text": "Sionna Bench",
    },
    "disable_search": True,
    "header_links_before_dropdown": 5,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "navbar_center": ["navbar-nav"],
    "navbar_end": [],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "navigation_with_keys": False,
    "show_nav_level": 1,
    "navigation_depth": 5,
    "show_toc_level": 2,
    "navbar_align": "left",
    "show_prev_next": False,
}


html_sidebars = {
    "**": [
        "sidebar-nav-bs.html",
    ],
}


def _copy_leaderboard_assets(app, exception):
    """Copy the canonical standalone assets into a successful HTML build."""
    if exception is not None or app.builder.format != "html":
        return

    destination = Path(app.outdir) / "_static" / "leaderboard"
    for relative_name in LEADERBOARD_FILES:
        source = LEADERBOARD_DIR / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing leaderboard asset: {source}")
        target = destination / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _resolve_sionna_doc_images(app, what, name, obj, options, lines):
    """Resolve PHY docstring images against the Sionna source checkout."""
    if Path(app.srcdir).resolve() != DOC_DIR:
        return

    document_dir = DOC_DIR / Path(app.env.docname).parent
    for index, line in enumerate(lines):
        prefix = ".. image:: /phy/figures/"
        if line.lstrip().startswith(prefix):
            image_name = line.lstrip()[len(prefix):]
            image_path = SIONNA_DOC_DIR / "phy" / "figures" / image_name
            relative_path = Path(os.path.relpath(image_path, document_dir)).as_posix()
            lines[index] = line[:len(line) - len(line.lstrip())] + ".. image:: " + relative_path


def setup(app):
    """Register documentation-build hooks."""
    app.connect("build-finished", _copy_leaderboard_assets)
    app.connect("autodoc-process-docstring", _resolve_sionna_doc_images)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
