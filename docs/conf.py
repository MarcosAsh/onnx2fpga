import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "compiler"))

import onnx2fpga  # noqa: E402

project = "onnx2fpga"
copyright = "2026, Marcos Ashton Iglesias"
author = "Marcos Ashton Iglesias"
version = ".".join(onnx2fpga.__version__.split(".")[:3])
release = version

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "README.md"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "sphinx_book_theme"
html_title = "onnx2fpga " + version
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/MarcosAsh/onnx2fpga",
    "use_repository_button": True,
    "navigation_with_keys": False,
}
