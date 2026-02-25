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
        Displays the widget if we are in a Jupyter environment, otherwise saves it to a file and opens it.
        """
        if IN_IPYTHON:
            display(self)
            return
        # 1. Create a temporary html file that will stay open after close
        # 2. Write the widget to it with embed_minimal_html
        # 3. Open the file using the open function from graphviz
        file = pathlib.Path.cwd() / "tmp.html"
        file.write_text(HTML.replace("MAGIC_STRING", str(self.egraphs)))
        # https://github.com/manzt/anywidget/issues/339#issuecomment-1755654547
        # embed_minimal_html(file, views=[self], drop_defaults=False)
        # Instead of embedding widget, just save
        print("Visualizer widget saved to", file)
        webbrowser.open(file.as_uri())


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
