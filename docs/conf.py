import pathlib  # noqa: INP001
import subprocess

##
# ABlog
##

blog_authors = {
    "Saul": ("Saul Shanabrook", "https://saul.shanabrook.com"),
}
blog_default_author = "Saul"
post_date_format = "%Y-%m-%d"
post_auto_image = 1


html_sidebars = {
    "**": [
        "sidebar-nav-bs",
        "ablog/postcard.html",
        "ablog/recentposts.html",
        "ablog/tagcloud.html",
        "ablog/categories.html",
    ]
}
##
# Myst
##

myst_enable_extensions = [
    # "attrs_inline",
    "smartquotes",
    "strikethrough",
    "html_image",
    "deflist",
]
myst_fence_as_directive = ["mermaid"]

##
# Built presentation in sphinx
##

cwd = pathlib.Path(__file__).parent
presentation_file = cwd / "explanation" / "pldi_2023_presentation.ipynb"
output_dir = cwd / "presentations"

subprocess.run(
    [  # noqa: S607
        "jupyter",
        "nbconvert",
        str(presentation_file),
        "--to",
        "slides",
        "--output-dir",
        str(output_dir),
        "--TagRemovePreprocessor.remove_input_tags",
        "remove-input",
    ],
    check=True,
)

html_extra_path = ["presentations/pldi_2023_presentation.slides.html"]

# load extensions
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.mermaid",
    "ablog",
    "sphinx.ext.intersphinx",
]


##
# Intersphinx
##

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

##
# Sphinx Gallery
# https://sphinx-gallery.github.io/stable/configuration.html#build-pattern
##

sphinx_gallery_conf = {
    # Run all scripts
    "filename_pattern": r".*",
    "examples_dirs": "../python/egglog/examples",
    "gallery_dirs": "auto_examples",
    "abort_on_example_error": True,
    "run_stale_examples": True,
}

##
# https://github.com/tox-dev/sphinx-autodoc-typehints#options
##
always_document_param_types = True
# typehints_defaults = "braces"
# typehints_use_signature = True
# typehints_use_signature_return = True

# specify project details
master_doc = "index"
project = "egglog Python"


# basic build settings
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "auto_examples/*.ipynb",
    # "auto_examples/*.md5",
]
nitpicky = True

html_theme = "pydata_sphinx_theme"

templates_path = ["_templates"]
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/header-links.html#fontawesome-icons
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/egraphs-good/egglog-python",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "navigation_with_keys": False,
    "article_footer_items": ["comments"],
}

html_context = {
    "github_user": "egraphs-good",
    "github_repo": "egglog-python",
    "github_version": "main",
    "doc_path": "docs",
}

# myst_nb default settings

# Custom formats for reading notebook; suffix -> reader
# nb_custom_formats = {}

# Notebook level metadata key for config overrides
# nb_metadata_key = 'mystnb'

# Cell level metadata key for config overrides
# nb_cell_metadata_key = 'mystnb'

# Mapping of kernel name regex to replacement kernel name(applied before execution)
# nb_kernel_rgx_aliases = {}

# Execution mode for notebooks
nb_execution_mode = "cache"

# Path to folder for caching notebooks (default: <outdir>)
# nb_execution_cache_path = ''

# Exclude (POSIX) glob patterns for notebooks
# nb_execution_excludepatterns = ()

# Execution timeout (seconds)
nb_execution_timeout = 60 * 10

# Use temporary folder for the execution current working directory
# nb_execution_in_temp = False

# Allow errors during execution
# nb_execution_allow_errors = False

# Raise an exception on failed execution, rather than emitting a warning
nb_execution_raise_on_error = False

# Print traceback to stderr on execution error
nb_execution_show_tb = True

# Merge stdout/stderr execution output streams
# nb_merge_streams = False

# The entry point for the execution output render class (in group `myst_nb.output_renderer`)
# nb_render_plugin = 'default'

# Remove code cell source
# nb_remove_code_source = False

# Remove code cell outputs
# nb_remove_code_outputs = False

# Prompt to expand hidden code cell {content|source|outputs}
# nb_code_prompt_show = 'Show code cell {type}'

# Prompt to collapse hidden code cell {content|source|outputs}
# nb_code_prompt_hide = 'Hide code cell {type}'

# Number code cell source lines
# nb_number_source_lines = False

# Overrides for the base render priority of mime types: list of (builder name, mime type, priority)
# nb_mime_priority_overrides = ()

# Behaviour for stderr output
# nb_output_stderr = 'show'

# Pygments lexer applied to stdout/stderr and text/plain outputs
# nb_render_text_lexer = 'myst-ansi'

# Pygments lexer applied to error/traceback outputs
# nb_render_error_lexer = 'ipythontb'

# Options for image outputs (class|alt|height|width|scale|align)
# nb_render_image_options = {}

# Options for figure outputs (classes|name|caption|caption_before)
# nb_render_figure_options = {}

# The format to use for text/markdown rendering
# nb_render_markdown_format = 'commonmark'

# Javascript to be loaded on pages containing ipywidgets
# nb_ipywidgets_js = {'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js': {'integrity': 'sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=', 'crossorigin': 'anonymous'}, 'https://unpkg.com/@jupyter-widgets/html-manager@^0.20.0/dist/embed-amd.js': {'data-jupyter-widgets-cdn': 'https://cdn.jsdelivr.net/npm/', 'crossorigin': 'anonymous'}}
