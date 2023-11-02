import pathlib

import anywidget
import ipywidgets
import traitlets


class GraphvizWidget(anywidget.AnyWidget):
    """
    Graphviz widget to render multiple graphviz graphs.

    The index will choose the one that is currently displayed, defaulting to the last one.
    If the index or the graphs change, there will be a re-rendering, with animation.
    """

    _esm = pathlib.Path(__file__).parent / "widget.js"
    _css = pathlib.Path(__file__).parent / "widget.css"
    dots = traitlets.List().tag(sync=True)
    index = traitlets.Int(None, allow_none=True).tag(sync=True)
    performance = traitlets.Bool(False).tag(sync=True)


def graphviz_widget_with_slider(dots: list[str], *, performance: bool = False) -> ipywidgets.VBox:
    n_dots = len(dots)
    graphviz_widget = GraphvizWidget()
    graphviz_widget.dots = dots
    graphviz_widget.performance = performance
    slider_widget = ipywidgets.IntSlider(max=n_dots - 1, value=0)
    ipywidgets.jslink((slider_widget, "value"), (graphviz_widget, "index"))
    # play_widget = ipywidgets.Play(max=n_dots - 1, repeat=True, interval=4000)
    # ipywidgets.jslink((slider_widget, "value"), (play_widget, "value"))
    # top = pywidgets.HBox([play_widget, slider_widget])
    top = slider_widget
    return ipywidgets.VBox([top, graphviz_widget])
