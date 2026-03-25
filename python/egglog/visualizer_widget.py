import base64
import json
import pathlib
import webbrowser

import anywidget
import traitlets
from IPython.display import display

# from ipywidgets.embed import embed_minimal_html
from .ipython_magic import IN_IPYTHON

CURRENT_DIR = pathlib.Path(__file__).parent


class VisualizerWidget(anywidget.AnyWidget):
    """
    Widget to render multiple graphs using the interactive visualizer.

    The index will choose the one that is currently displayed, defaulting to the last one.
    """

    _esm = CURRENT_DIR / "visualizer.js"
    _css = CURRENT_DIR / "visualizer.css"
    egraphs = traitlets.List[str]().tag(sync=True)

    def display_or_open(self) -> None:
        """
        Displays the widget if we are in a Jupyter environment, otherwise opens a standalone HTML page.
        """
        if IN_IPYTHON:
            display(self)
            return
        payload = json.dumps(self.egraphs).replace("</", "<\\/")
        html = HTML.replace("MAGIC_STRING", payload)
        data_uri = "data:text/html;base64," + base64.b64encode(html.encode()).decode()
        webbrowser.open(data_uri)


HTML = """
<div id="egraph-visualizer"></div>
<link rel="stylesheet" href="https://esm.sh/egraph-visualizer/dist/style.css" />
<script type="module">
  import { mount } from "https://esm.sh/egraph-visualizer";
  const egraphs = MAGIC_STRING;
  const mounted = mount(document.getElementById("egraph-visualizer"));
  mounted.render(egraphs);
</script>
"""
