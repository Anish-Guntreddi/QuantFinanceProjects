"""Jupyter notebook viewer — renders full HTML via nbconvert in an iframe."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Root of the entire QuantFinanceProjects repo
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_analysis_notebook(notebook_dir: str) -> Path | None:
    """Find the *_analysis.ipynb file in the given notebook directory."""
    nb_path = REPO_ROOT / notebook_dir
    if not nb_path.is_dir():
        return None
    candidates = list(nb_path.glob("*_analysis.ipynb"))
    return candidates[0] if candidates else None


# Dark-theme CSS injected into the nbconvert HTML
_DARK_THEME_CSS = """
<style>
/* Dark background */
body, .jp-Notebook, html {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif !important;
}

/* Code cells */
.input_area, .jp-InputArea, div.highlight, .code_cell .input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

pre, code {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    font-size: 13px !important;
}

/* Syntax highlighting — GitHub dark style */
.highlight .k, .highlight .kn, .highlight .kd, .highlight .kp { color: #ff7b72 !important; }  /* keywords */
.highlight .n, .highlight .nn, .highlight .nx { color: #c9d1d9 !important; }  /* names */
.highlight .s, .highlight .s1, .highlight .s2, .highlight .sa, .highlight .sb { color: #a5d6ff !important; }  /* strings */
.highlight .mi, .highlight .mf, .highlight .mb, .highlight .mh, .highlight .mo { color: #79c0ff !important; }  /* numbers */
.highlight .nb, .highlight .bp { color: #79c0ff !important; }  /* builtins */
.highlight .nf, .highlight .fm { color: #d2a8ff !important; }  /* functions */
.highlight .nc { color: #f0883e !important; }  /* classes */
.highlight .c, .highlight .c1, .highlight .cm, .highlight .ch, .highlight .cs { color: #8b949e !important; font-style: italic; }  /* comments */
.highlight .o, .highlight .ow { color: #ff7b72 !important; }  /* operators */
.highlight .p { color: #c9d1d9 !important; }  /* punctuation */
.highlight .nd { color: #d2a8ff !important; }  /* decorators */
.highlight .si { color: #79c0ff !important; }  /* string interpolation */
.highlight .se { color: #79c0ff !important; }  /* string escape */

/* Output area */
.output_area, .jp-OutputArea, .output_subarea, .output_text, .output_stream {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    border: none !important;
}

.output_text pre, .output_stream pre, .output_subarea pre {
    color: #c9d1d9 !important;
    background-color: #0d1117 !important;
}

/* Markdown cells */
.text_cell, .text_cell_render, .jp-MarkdownOutput,
.rendered_html, .text_cell_render h1, .text_cell_render h2,
.text_cell_render h3, .text_cell_render h4, .text_cell_render p,
.text_cell_render li, .text_cell_render td, .text_cell_render th {
    color: #c9d1d9 !important;
    background-color: transparent !important;
}

.text_cell_render h1, .rendered_html h1 {
    color: #f0f6fc !important;
    font-size: 1.6em !important;
    border-bottom: 1px solid #30363d !important;
    padding-bottom: 0.3em !important;
}

.text_cell_render h2, .rendered_html h2 {
    color: #f0f6fc !important;
    font-size: 1.3em !important;
    border-bottom: 1px solid #30363d !important;
    padding-bottom: 0.2em !important;
}

.text_cell_render h3, .rendered_html h3 {
    color: #e6edf3 !important;
    font-size: 1.1em !important;
}

/* Bold / strong */
strong, b { color: #e6edf3 !important; }

/* Links */
a { color: #58a6ff !important; }

/* Tables */
table, .dataframe { border-collapse: collapse !important; margin: 0.5em 0 !important; }
th { background-color: #161b22 !important; color: #e6edf3 !important; border: 1px solid #30363d !important; padding: 6px 12px !important; }
td { background-color: #0d1117 !important; color: #c9d1d9 !important; border: 1px solid #30363d !important; padding: 6px 12px !important; }
tr:nth-child(even) td { background-color: #161b22 !important; }

/* Prompt labels In[N] / Out[N] */
.prompt, .input_prompt, .output_prompt,
.jp-InputPrompt, .jp-OutputPrompt {
    color: #484f58 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* Images — ensure they're visible on dark bg */
.output_png img, .jp-RenderedImage img {
    background-color: white;
    border-radius: 4px;
    padding: 4px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* Cell spacing */
.cell, .jp-Cell {
    margin-bottom: 12px !important;
    padding: 8px 0 !important;
}

/* Remove default white borders/backgrounds */
div.cell, .jp-Cell, .inner_cell, .jp-Cell-inputWrapper {
    background-color: transparent !important;
    border: none !important;
}

/* Error output styling */
.output_error pre, .ansi-red-fg { color: #f85149 !important; }
.ansi-green-fg { color: #3fb950 !important; }
.ansi-yellow-fg { color: #d29922 !important; }
.ansi-blue-fg { color: #58a6ff !important; }
.ansi-cyan-fg { color: #39c5cf !important; }
</style>
"""


@st.cache_data(show_spinner=False)
def _convert_notebook_to_html(nb_path_str: str, file_hash: str) -> str:
    """Convert a notebook file to dark-themed HTML. Cached by path + content hash."""
    import nbformat
    from nbconvert import HTMLExporter

    nb = nbformat.read(nb_path_str, as_version=4)

    exporter = HTMLExporter()
    exporter.template_name = "classic"
    exporter.exclude_input_prompt = False
    exporter.exclude_output_prompt = False

    html_body, _ = exporter.from_notebook_node(nb)

    # Inject dark theme CSS right before </head>
    html_body = html_body.replace("</head>", _DARK_THEME_CSS + "\n</head>")

    return html_body


def render_notebook(notebook_dir: str) -> None:
    """Render a Jupyter notebook as full HTML embedded in the Strategy page."""
    nb_file = _find_analysis_notebook(notebook_dir)
    if nb_file is None:
        st.info(f"No analysis notebook found in `{notebook_dir}`.")
        return

    with st.expander(f"View Full Notebook — {nb_file.name}", expanded=False):
        # Hash file content for cache invalidation
        file_hash = hashlib.md5(nb_file.read_bytes()).hexdigest()

        with st.spinner("Rendering notebook..."):
            html = _convert_notebook_to_html(str(nb_file), file_hash)

        # Embed as an iframe via st.components.v1.html
        # Height is generous to avoid excessive scrolling within the iframe
        components.html(html, height=800, scrolling=True)
