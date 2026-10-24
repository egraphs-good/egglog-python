from __future__ import annotations

import base64
import json
import pathlib
import struct
import webbrowser
from collections import defaultdict
from typing import NotRequired, TypedDict
from warnings import warn

import anywidget
import traitlets
from egglog_proto.egglog.v1 import egglog_pb as pb
from IPython.display import display

from ._presentation import CallKind, presentation
from ._program import Builder, Owner, _copy_record, _definition_key
from .ipython_magic import IN_IPYTHON
from .pretty import pretty_ref

CURRENT_DIR = pathlib.Path(__file__).parent


class _Node(TypedDict):
    op: str
    children: list[str]
    eclass: str
    subsumed: bool
    cost: NotRequired[str]


class _Graph(TypedDict):
    nodes: dict[str, _Node]
    class_data: dict[str, dict[str, str]]
    root_eclasses: list[str]


def _node_presentation(node: pb.Node, sort_label: str, labels: dict[str, str]) -> tuple[str, list[int]]:  # noqa: C901, PLR0912
    """Read explicit edges and format inert values, including opaque host costs."""
    children = []

    def collect(role: str, child: int) -> int:
        if role == "nodes":
            children.append(child)
        return child

    _copy_record(node, relocate=collect)
    if node.kind is None:
        msg = "Snapshot node has no kind"
        raise ValueError(msg)
    if node.kind.field in {"call", "get_cost"}:
        label = labels[node.kind.value.func]
        if node.kind.field == "get_cost":
            label = f"cost({label})"
    elif node.kind.field == "var":
        label = node.kind.value
    elif node.kind.field == "union":
        label = "Union"
    elif node.kind.field == "primitive_value":
        primitive = node.kind.value.value
        if primitive is None:
            msg = "Snapshot primitive has no value"
            raise ValueError(msg)
        if primitive.field == "custom":
            label = f"{sort_label}(<{len(primitive.value.payload)} opaque bytes>)"
        elif primitive.field == "f64_bits":
            label = repr(struct.unpack("!d", struct.pack("!Q", primitive.value))[0])
        elif primitive.field in {"i64", "bool", "string", "big_int"}:
            label = repr(primitive.value) if primitive.field != "big_int" else primitive.value
        elif primitive.field in {"rational", "big_rat"}:
            label = f"{primitive.value.numerator}/{primitive.value.denominator}"
        elif primitive.field == "partial_call":
            label = f"partial({labels[primitive.value.func]})"
        elif primitive.field == "maybe" and not primitive.value.has_field("value"):
            label = f"{sort_label}.none"
        else:
            label = sort_label
    else:
        msg = f"Unknown snapshot node kind: {node.kind.field}"
        raise ValueError(msg)
    return label, children


def _snapshot_to_data(  # noqa: C901, PLR0912
    snapshot: pb.EGraphSnapshot,
    *,
    max_functions: int | None = None,
    max_calls_per_function: int | None = None,
    n_inline_leaves: int = 0,
    split_primitive_outputs: bool = True,
    split_functions: set[str] | None = None,
    owner: Owner | None = None,
) -> _Graph:
    """
    Project the snapshot's stored row inventory, never evaluating its values.

    Children initially name display classes. Choose a representative only after
    filtering, so omitted rows and empty/cyclic classes never leave dangling edges.
    The protobuf, including saved code and opaque payloads, is never changed.
    """
    if snapshot.program is None:
        msg = "Snapshot has no reconstruction Program"
        raise ValueError(msg)
    for limit in (max_functions, max_calls_per_function, n_inline_leaves):
        if limit is not None and limit < 0:
            msg = "Visualization limits must be nonnegative"
            raise ValueError(msg)
    program = snapshot.program
    owner = Builder.from_program(program).publish() if owner is None else owner
    definitions = {
        _definition_key(reference.read()): reference
        for source in (*owner._environment, owner)
        for role in ("declarations", "primitive_sorts", "primitives")
        for index in range(len(source._slots[role]))
        for reference in (source.ref(role, index),)
    }
    sort_labels = {}
    labels = {}
    declared_costs = {}
    type_names = {}
    for key, reference in definitions.items():
        assert key is not None
        declaration = reference.read()
        if isinstance(declaration, pb.Declaration):
            assert declaration.kind is not None
            kind = declaration.kind.value
        else:
            kind = declaration
        if isinstance(kind, pb.EqSort | pb.PrimitiveSort):
            bindings = presentation(reference).type_view
            type_names[key] = bindings.path[-1] if bindings is not None else kind.name
    for declaration in program.declarations:
        if declaration.kind is None:
            continue
        kind = declaration.kind.value
        if not isinstance(kind, pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive):
            continue
        label = kind.name
        views = presentation(definitions["callable", kind.name]).views
        if views:
            for view in views[:1]:
                if view.path:
                    label = ".".join(view.path[-2:] if view.owner is not None else view.path[-1:])
                if view.owner is not None:
                    sort = view.owner.read().kind
                    assert sort is not None
                    if sort.field == "eq":
                        typ = type_names["eq_sort", sort.value]
                    elif sort.field == "application":
                        typ = type_names["primitive_sort", sort.value.name]
                    else:
                        msg = "A nominal Python callable owner requires an equality or family sort"
                        raise ValueError(msg)
                    label = typ if view.kind == CallKind.INITIALIZER else f"{typ}.{view.path[-1]}"
        labels[kind.name] = label
        if isinstance(kind, pb.Constructor) and kind.has_field("cost"):
            declared_costs[kind.name] = kind.cost

    class_ids = {i: f"value-{i}" for i in range(len(program.nodes))}
    inventory = set()
    empty_classes = set()
    actions = []
    pending = []
    terms = set()
    for command in program.commands:
        if command.kind is None or command.kind.field != "action" or command.kind.value.kind is None:
            msg = "Snapshot reconstruction must contain only Actions"
            raise ValueError(msg)
        action = command.kind.value.kind
        if action.field not in {"term", "set", "subsume", "set_cost"}:
            msg = f"Action is outside the snapshot row inventory: {action.field}"
            raise ValueError(msg)
        actions.append(action)
        if action.field == "term":
            terms.add(action.value)
            node = program.nodes[action.value]
            if node.kind is not None and node.kind.field == "union":
                cid = f"class-{action.value}"
                inventory.add(cid)
                if not node.kind.value.members:
                    empty_classes.add(cid)
                class_ids[action.value] = cid
                for member in node.kind.value.members:
                    class_ids[member] = cid
            pending.append(action.value)

    nodes: dict[str, _Node] = {}
    class_data: dict[str, dict[str, str]] = {}
    rows: dict[str, list[str]] = defaultdict(list)
    row_keys = {}
    costs = {}
    primitive_nodes = set()
    for index, action in enumerate(actions):
        if action.field == "set_cost":
            pending.append(action.value.cost)
        if action.field == "set":
            assignment = action.value
            if assignment.target is None or not assignment.has_field("value"):
                msg = "Snapshot Set has no target or value"
                raise ValueError(msg)
            target = assignment.target
            nid = f"row-{index}"
            cid = class_ids[assignment.value]
            nodes[nid] = _Node(
                op=labels[target.func], children=[class_ids[arg] for arg in target.args], eclass=cid, subsumed=False
            )
            rows[target.func].append(nid)
            pending.extend([assignment.value, *target.args])
    visited = set()
    while pending:
        index = pending.pop()
        if index in visited:
            continue
        visited.add(index)
        node = program.nodes[index]
        if node.kind is None:
            msg = "Snapshot node has no kind"
            raise ValueError(msg)
        cid = class_ids[index]
        if node.sort_id not in sort_labels:
            sort_labels[node.sort_id] = pretty_ref(owner.ref("sorts", node.sort_id), sort_names=type_names)
        class_data[cid] = {"type": sort_labels[node.sort_id]}
        if node.kind.field == "union" and cid in inventory:
            pending.extend(node.kind.value.members)
            continue
        label, children = _node_presentation(node, sort_labels[node.sort_id], labels)
        pending.extend(children)
        nid = f"node-{index}"
        if node.kind.field == "call" and (cid in inventory or index in terms):
            call = node.kind.value
            rows[call.func].append(nid)
            row_keys[call.func, tuple(class_ids[arg] for arg in call.args)] = nid
            if call.func in declared_costs:
                costs[nid] = declared_costs[call.func]
                pending.append(costs[nid])
        if node.kind.field == "primitive_value":
            primitive_nodes.add(nid)
        nodes[nid] = _Node(op=label, children=[class_ids[child] for child in children], eclass=cid, subsumed=False)

    for action in actions:
        if action.field in {"subsume", "set_cost"}:
            target = action.value if action.field == "subsume" else action.value.target
            if target is None:
                msg = "Snapshot annotation has no target"
                raise ValueError(msg)
            nid = row_keys[target.func, tuple(class_ids[arg] for arg in target.args)]
            if action.field == "subsume":
                nodes[nid]["subsumed"] = True
            else:
                costs[nid] = action.value.cost
    for nid, cost in costs.items():
        value = program.nodes[cost]
        label, children = _node_presentation(value, sort_labels[value.sort_id], labels)
        nodes[nid]["cost"] = (
            f"{label}({', '.join('@' + class_ids[child] for child in children)})" if children else label
        )

    removed: set[str] = set()
    tables = [name for name in labels if rows.get(name)]
    if max_functions is not None and len(tables) > max_functions:
        warn("Omitted: " + ", ".join(labels[name] for name in tables[max_functions:]), stacklevel=3)
        removed.update(nid for name in tables[max_functions:] for nid in rows[name])
        tables = tables[:max_functions]
    if max_calls_per_function is not None:
        truncated = [name for name in tables if len(rows[name]) > max_calls_per_function]
        if truncated:
            warn("Truncated: " + ", ".join(labels[name] for name in truncated), stacklevel=3)
            removed.update(nid for name in truncated for nid in rows[name][max_calls_per_function:])
    nodes = {nid: node for nid, node in nodes.items() if nid not in removed}
    groups: dict[str, list[str]] = defaultdict(list)
    for nid, node in nodes.items():
        groups[node["eclass"]].append(nid)
    needed = list(
        inventory
        | {nodes[nid]["eclass"] for name in tables for nid in rows[name] if nid in nodes}
        | {class_ids[cost] for nid, cost in costs.items() if nid in nodes}
    )
    visible = set()
    while needed:
        cid = needed.pop()
        if cid in visible:
            continue
        visible.add(cid)
        if not groups[cid]:
            nid = f"empty-{cid}"
            nodes[nid] = _Node(op="∅" if cid in empty_classes else "…", children=[], eclass=cid, subsumed=False)
            groups[cid].append(nid)
        for nid in groups[cid]:
            needed.extend(nodes[nid]["children"])
    nodes = {nid: node for nid, node in nodes.items() if node["eclass"] in visible}

    split = set(primitive_nodes if split_primitive_outputs else ())
    split.update(nid for name in split_functions or () for nid in rows[name])
    for cid, members in list(groups.items()):
        candidates = [nid for nid in members if nid in split and nid in nodes]
        if len(candidates) > 1:
            msg = "Cannot split multiple selected nodes in the same display class"
            raise ValueError(msg)
        if not candidates:
            continue
        selected = nodes[candidates[0]]
        for nid in members:
            if nid == candidates[0] or nid not in nodes:
                continue
            split_id = f"split-{nid}"
            nodes[nid]["eclass"] = split_id
            nodes[split_id] = _Node(**{**selected, "children": list(selected["children"]), "eclass": split_id})
            class_data[split_id] = dict(class_data.get(cid, {}))

    for _ in range(n_inline_leaves):
        groups = defaultdict(list)
        for nid, node in nodes.items():
            groups[node["eclass"]].append(nid)
        leaves = {
            cid: nodes[members[0]]["op"]
            for cid, members in groups.items()
            if len(members) == 1 and not nodes[members[0]]["children"] and not members[0].startswith("empty-")
        }
        inlined: set[str] = set()
        for node in nodes.values():
            if not any(child in leaves for child in node["children"]):
                continue
            args = [leaves.get(child, "·") for child in node["children"]]
            parts = node["op"].split("·")
            node["op"] = (
                "".join(part + (args[i] if i < len(args) else "") for i, part in enumerate(parts))
                if len(parts) == len(args) + 1
                else f"{node['op']}({', '.join(args)})"
            )
            inlined.update(child for child in node["children"] if child in leaves)
            node["children"] = [child for child in node["children"] if child not in leaves]
        nodes = {nid: node for nid, node in nodes.items() if node["eclass"] not in inlined}
    representatives = {node["eclass"]: nid for nid, node in nodes.items()}
    for node in nodes.values():
        node["children"] = [representatives[child] for child in node["children"]]
    return _Graph(
        nodes=nodes,
        class_data={cid: data for cid, data in class_data.items() if cid in representatives},
        root_eclasses=[],
    )


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
