# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

# Plots .cali output from t_blueprint_mesh_transform_benchmark, or compares
# two runs of it.
#
# Usage:
#     python3 plot_benchmark_output.py [path/to/file.cali]
#     python3 plot_benchmark_output.py --compare BASE.cali NEW.cali
#                                      [--label-base L] [--label-new L]
#                                      [--output DIR]
#     Either form also takes --rank-reduce {max,min,mean} (default: max).
#
#     With no arguments, the most recently modified .cali file in the current
#     working directory is plotted.
#
#     With --compare, two .cali files are matched case by case (same benchmark,
#     dim and mesh sizes) and the speedup of NEW over BASE is plotted. If BASE
#     holds a single series (for example a host-only run from before device
#     support), every series in NEW is compared against it. Otherwise each
#     series (backend + src/exec/out/sync configuration) is paired with the
#     same series in BASE, which compares two runs of the execution model,
#     such as before and after a change.
#
# Examples:
#     # Run the benchmark, which writes <YYYYmmdd_HHMMSS>.cali into its own
#     # working directory, then plot it from that directory.
#     ./t_blueprint_mesh_transform_benchmark
#     python3 plot_benchmark_output.py
#
#     # Plot a specific run, from anywhere.
#     python3 plot_benchmark_output.py /path/to/20260803_120000.cali
#
#     # Speedup of a device-support run over a develop-branch run.
#     python3 plot_benchmark_output.py --compare develop.cali branch.cali \
#         --label-base "develop" --label-new "execution model"
#
# Input:
#     A Caliper .cali file produced by t_blueprint_mesh_transform_benchmark.
#     When the benchmark ran under MPI, Caliper merges every rank's results
#     into one .cali file. Each rank converts its own domains without
#     communication.
#     --rank-reduce picks how the per-rank times are combined (default max).
#
# Output:
#     For a single file, a directory next to it named after it with the
#     extension stripped (e.g. 20260803_120000.cali -> 20260803_120000/),
#     containing:
#       - thicket_mesh_heatmap.png, thicket_generate_heatmap.png,
#         thicket_boxplot.png (skipped if seaborn is not installed; for an
#         MPI run the boxplot shows the spread across ranks)
#       - one subdirectory per execution backend found in the file (plus a
#         "combined" subdirectory when more than one backend is present),
#         each containing:
#           - mesh_scaling.png, mesh_scaling_per_element.png
#           - generate_scaling.png, generate_scaling_per_element.png
#             (one panel per benchmark, one line per src/exec/out/sync
#             configuration)
#           - mesh_conversion_heatmap_dim-<N>.png (and _per_element variant)
#           - generate_heatmap_dim-<N>.png (and _per_element variant)
#           - generate_growth_heatmap_dim-<N>.png
#           - the heatmap names above gain a _<src>-<exec>-<out>-<sync>
#             suffix when both the all-host and all-device configurations
#             are present; the combined subdirectory has no heatmaps since
#             several backends would share one cell
#           - individual/<figure name>/<panel>.png, one single-panel image
#             per panel of each multi-panel figure above
#
#     For --compare, a directory named <BASE stem>_vs_<NEW stem> next to BASE
#     (or --output), containing:
#       - comparison_slide.png: the geometric-mean speedup per series on the
#         left and the per-benchmark, per-size speedups on the right
#       - average_speedup.png: the summary panel on its own
#       - speedup_by_benchmark.png: the per-benchmark grid on its own
#       - matched_cases.csv, speedup_summary.csv, speedup_by_region.csv
#
# Requirements:
#     pip install thicket caliper-reader matplotlib numpy seaborn

import argparse
import csv
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import thicket as th
from caliperreader.readererror import ReaderError

NUMERIC_FIELDS = ("dim", "inverts", "inelems", "outverts", "outelems", "iter")
LOCATION_FIELDS = ("src", "exec", "out")
LOCATIONS = ("host", "device", "input")
SYNC_STRATEGIES = ("sync", "assume")

OPERATIONS = [
    "generate_centroids",
    "generate_points",
    "generate_faces",
    "generate_lines",
    "generate_corners",
    "generate_sides",
    "to_polytopal",
]
MESH_SRC_ORDER = ["structured", "rectilinear", "uniform"]
SHAPE_ORDER = ["quads", "hexs", "pyramids"]
# Panel order for the conversion benchmarks, read row by row in a two-column
# grid: conversions to structured on the left, to unstructured on the right.
CONVERSION_ORDER = [
    "rectilinear_to_structured", "structured_to_unstructured",
    "uniform_to_structured", "rectilinear_to_unstructured",
    "uniform_to_rectilinear", "uniform_to_unstructured",
]

BASELINE_CFG = ("host", "host", "host", "sync")
DEVICE_BASELINE_CFG = ("device", "device", "device", "sync")

CFG_ORDER = [
    ("host", "host", "host", "sync"),
    ("host", "host", "device", "sync"),
    ("host", "host", "device", "assume"),
    ("host", "device", "host", "sync"),
    ("host", "device", "host", "assume"),
    ("host", "device", "device", "sync"),
    ("host", "device", "input", "sync"),
    ("device", "host", "host", "sync"),
    ("device", "host", "device", "sync"),
    ("device", "host", "device", "assume"),
    ("device", "host", "input", "sync"),
    ("device", "device", "host", "sync"),
    ("device", "device", "host", "assume"),
    ("device", "device", "device", "sync"),
]
CFG_ABBREVIATIONS = {"device": "dev"}

COMBINED_LABEL = "combined"

INDIVIDUAL_FIGSIZE = (6.4, 3.6)
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def parse_scope_name(raw):
    tokens = raw.split("_")

    i = 0
    while i < len(tokens) and "-" not in tokens[i]:
        i += 1
    name = "_".join(tokens[:i])

    fields = {}
    for token in tokens[i:]:
        if "-" not in token:
            return None
        key, value = token.split("-", 1)
        fields[key] = value

    for key in NUMERIC_FIELDS:
        if not fields.get(key, "").isdigit():
            return None
    for key in LOCATION_FIELDS:
        if fields.get(key) not in LOCATIONS:
            return None
    sync = fields.get("sync", "sync")
    if sync not in SYNC_STRATEGIES:
        return None

    parsed = {
        "name": name,
        "cfg": (fields["src"], fields["exec"], fields["out"], sync),
        "backend": fields.get("backend"),
        "mem": fields.get("mem"),
    }
    for key in NUMERIC_FIELDS:
        parsed[key] = int(fields[key])
    parsed["threads"] = int(fields["threads"]) if "threads" in fields else None
    parsed["ranks"] = int(fields["ranks"]) if "ranks" in fields else None
    return parsed


def split_convert(name):
    tokens = name.split("_")
    if len(tokens) == 3 and tokens[1] == "to":
        return tokens[0], tokens[2]
    return None


def split_generate_name(name):
    for operation in OPERATIONS:
        if name.startswith(operation + "_"):
            return operation, name[len(operation) + 1:]
    return name, ""


def format_cfg_label(cfg):
    src, exec_, out, sync = cfg
    path = "→".join(CFG_ABBREVIATIONS.get(part, part) for part in (src, exec_, out))
    return path if exec_ == out else f"{path} [{sync}]"


def ordered(values, preferred_order):
    values = set(values)
    return [v for v in preferred_order if v in values] + \
        sorted(values - set(preferred_order))


def rank(value, order):
    return order.index(value) if value in order else len(order)


def find_cali_file():
    files = list(Path.cwd().glob("*.cali"))
    if not files:
        sys.exit("no .cali files found in current directory")
    return max(files, key=lambda path: path.stat().st_mtime)


def find_time_column(dataframe):
    for name in ("time", "sum#sum#time.duration", "sum#time.duration"):
        if name in dataframe.columns:
            return name
    for column in dataframe.columns:
        if str(column).endswith("time.duration"):
            return column
    sys.exit(f"no time column found; columns were {list(dataframe.columns)}")


INCLUSIVE_COLUMN = "inclusive#time.duration"

# How the per-rank times of an MPI run are combined into one time per region.
RANK_REDUCERS = {
    "max": max,
    "min": min,
    "mean": lambda values: sum(values) / len(values),
}


def add_inclusive_column(thicket, records):
    # thicket's own views read a dataframe column; fill it with the inclusive
    # times collect() already computed for the benchmark regions. An MPI run
    # has a rank index level, and each row gets its own rank's time so the
    # boxplot shows the spread across ranks.
    index = thicket.dataframe.index
    rank_level = index.names.index("rank") if "rank" in index.names else None
    by_node = {r["node"]: r for r in records}

    def inclusive(key):
        record = by_node.get(key[0])
        if record is None:
            return float("nan")
        if rank_level is None:
            return record["total_time"]
        return record["rank_times"].get(key[rank_level], float("nan"))

    thicket.dataframe[INCLUSIVE_COLUMN] = [inclusive(key) for key in index]
    return INCLUSIVE_COLUMN


def collect(thicket, rank_reduce):
    dataframe = thicket.dataframe
    time_column = find_time_column(dataframe)
    has_ranks = "rank" in dataframe.index.names

    # Own time per node for a serial run, or per (node, rank) for an MPI run
    # where Caliper merged every rank's rows into the one file.
    levels = ["node", "rank"] if has_ranks else ["node"]
    own_times = dataframe[time_column].groupby(level=levels).sum()
    if has_ranks:
        own_times = {node: group.droplevel("node").to_dict()
                     for node, group in own_times.groupby(level="node")}
    else:
        own_times = own_times.to_dict()

    def node_time(node):
        # A region's time is its own time plus every nested region beneath
        # it; Caliper stores only the own time. Kept per rank when present.
        if has_ranks:
            total = dict(own_times.get(node, {}))
            for child in node.children:
                for rank, time in node_time(child).items():
                    total[rank] = total.get(rank, 0.0) + time
            return total
        return own_times.get(node, 0.0) + sum(node_time(c) for c in node.children)

    def all_nodes(node):
        yield node
        for child in node.children:
            yield from all_nodes(child)

    records = []
    for root in thicket.graph.roots:
        for node in all_nodes(root):
            parsed = parse_scope_name(str(node.frame.get("name", "")))
            if parsed is None:
                continue
            parsed["node"] = node
            if has_ranks:
                parsed["rank_times"] = node_time(node)
                parsed["total_time"] = RANK_REDUCERS[rank_reduce](
                    list(parsed["rank_times"].values()) or [0.0])
            else:
                parsed["rank_times"] = None
                parsed["total_time"] = node_time(node)
            parsed["avg_time"] = parsed["total_time"] / parsed["iter"]
            records.append(parsed)
    return records


def load_records(cali_file, rank_reduce):
    if not cali_file.exists():
        sys.exit(f"no such file: {cali_file}")
    try:
        thicket = th.Thicket.from_caliperreader(str(cali_file))
    except ReaderError as error:
        sys.exit(f"failed to read {cali_file}: {error}")
    records = collect(thicket, rank_reduce)
    if not records:
        sys.exit(f"no benchmark regions found in {cali_file}")
    if records[0]["rank_times"] is not None:
        print(f"{cali_file}: MPI run, taking the {rank_reduce} across ranks")
    return thicket, records


def range_label(name, values):
    values = sorted(values)
    if not values:
        return ""
    if len(values) == 1:
        return f", {name}={values[0]}"
    return f", {name}={values[0]}-{values[-1]}"


def avg_time(record):
    return record["avg_time"]


def per_element_value(record):
    if record["outelems"] <= 0:
        return None
    return record["avg_time"] / record["outelems"]


# (file suffix, quantity, unit, scale from seconds) for the two time metrics
# every scaling plot and heatmap is made for.
METRICS = (
    ("", "iteration", "ms", 1e3),
    ("_per_element", "element", "ns", 1e9),
)
METRIC_VALUES = {"iteration": avg_time, "element": per_element_value}


def growth_value(record):
    if record["inelems"] <= 0:
        return None
    return record["outelems"] / record["inelems"]


def build_line_panels(records, value_fn):
    # {benchmark name: {configuration label: [(dim, value), ...]}}
    panels = {}
    for record in records:
        value = value_fn(record)
        if value is None:
            continue
        series = panels.setdefault(record["name"], {})
        series.setdefault(format_cfg_label(record["cfg"]), []).append((record["dim"], value))
    return panels


def plain_tick(value, _position):
    return f"{value:g}"


def render_series(ax, series, title, y_scale, series_order):
    labels = ordered(series, series_order)
    colors = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20").colors
    for idx, label in enumerate(labels):
        points = sorted(series[label])
        dims = [str(dim) for dim, _ in points]
        ys = [value * y_scale for _, value in points]
        ax.plot(dims, ys, marker=MARKERS[idx % len(MARKERS)], color=colors[idx % len(colors)],
                label=label)

    ax.set_title(pretty_region(title), fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    subs = (1.0,) if high / low > 1000 else (1.0, 2.0, 5.0)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(plain_tick))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="y", which="major", alpha=0.25)


def legend_height(n_series, columns):
    return 0.2 * math.ceil(n_series / columns) + 0.05


def legend_below(fig, axes, series_order, columns):
    # One legend under the panels. Its height and the axis label's are fixed
    # in inches, so convert them to the fraction tight_layout expects.
    handles = {}
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            handles[label] = handle
    labels = ordered(handles, series_order)
    inches = legend_height(len(labels), columns)
    height = fig.get_figheight()
    fig.legend([handles[label] for label in labels], labels, loc="lower center",
               ncol=min(len(labels), columns), frameon=False, fontsize=9)
    fig.supxlabel("dim (points per axis)", y=(inches + 0.1) / height)
    fig.tight_layout(rect=(0, (inches + 0.45) / height, 1, 0.95))


def plot_group(panels, path, suptitle, ylabel, y_scale, series_order):
    titles = sorted(panels, key=region_sort_key)
    n = len(titles)
    columns = min(2, n)
    rows = math.ceil(n / columns)
    all_series = set()
    for series in panels.values():
        all_series.update(series)

    height = 3.2 * rows + 0.9 + legend_height(len(all_series), 4)
    fig, axes = plt.subplots(rows, columns, figsize=(6.5 * columns, height), squeeze=False)
    axes = list(axes.flatten())
    for ax, title in zip(axes, titles):
        render_series(ax, panels[title], title, y_scale, series_order)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", wrap=True)
    fig.supylabel(ylabel)
    legend_below(fig, axes[:n], series_order, 4)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")

    individual_dir = path.parent / "individual" / path.stem
    individual_dir.mkdir(parents=True, exist_ok=True)
    height = INDIVIDUAL_FIGSIZE[1] + 0.9 + legend_height(len(all_series), 2)
    for title in titles:
        fig, ax = plt.subplots(figsize=(INDIVIDUAL_FIGSIZE[0], height))
        render_series(ax, panels[title], title, y_scale, series_order)
        ax.set_title(f"{suptitle}\n{pretty_region(title)}", fontsize=11)
        ax.set_ylabel(ylabel)
        legend_below(fig, [ax], series_order, 2)
        fig.savefig(individual_dir / f"{title}.png", dpi=150)
        plt.close(fig)
    print(f"saved {len(titles)} individual plot(s) to {individual_dir}/")


def format_plain(value, sig_figs=3):
    if value == 0:
        return f"{0:.{sig_figs - 1}f}"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, sig_figs - magnitude - 1)
    return f"{value:.{decimals}f}"


def format_growth_cell(value):
    return f"{format_plain(value, sig_figs=2)}x"


def render_heatmap(ax, matrix, rows, cols, title, cell_fmt):
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(title, fontsize=10)

    midpoint = (np.nanmin(matrix) + np.nanmax(matrix)) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = "white" if value > midpoint else "black"
            ax.text(j, i, cell_fmt(value), ha="center", va="center",
                    fontsize=7, color=color)
    return im


def plot_heatmaps(lookup, row_order, col_order, value_fn, scale, out_dir, filename_prefix,
                  suptitle, row_label, col_label, cbar_label, cell_fmt=format_plain):
    rows = ordered({key[0] for key in lookup}, row_order)
    cols = ordered({key[1] for key in lookup}, col_order)
    dims = sorted({key[2] for key in lookup})
    width = 4.0 + 0.9 * len(cols)
    height = 2.0 + 0.3 * len(rows)

    for dim in dims:
        matrix = np.full((len(rows), len(cols)), np.nan)
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                record = lookup.get((row, col, dim))
                value = value_fn(record) if record is not None else None
                if value is not None:
                    matrix[i, j] = value * scale

        fig, ax = plt.subplots(figsize=(width, height))
        im = render_heatmap(ax, matrix, rows, cols, f"{suptitle}\ndim={dim}", cell_fmt)
        cbar = fig.colorbar(im, ax=ax, label=cbar_label)
        cbar.ax.ticklabel_format(style="plain", useOffset=False)
        ax.set_xlabel(col_label)
        ax.set_ylabel(row_label)
        fig.tight_layout()
        path = out_dir / f"{filename_prefix}_dim-{dim}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")


def plot_thicket_views(thicket, records, time_column, out_dir):
    # time_column must be the inclusive (subtree-total) column so these
    # views agree with every other figure this script produces.
    if not hasattr(th.stats, "display_heatmap"):
        print("seaborn not installed; skipping thicket plots")
        return

    mean_column = th.stats.mean(thicket, columns=[time_column])[0]
    full_statsframe = thicket.statsframe.dataframe

    convert_nodes = [r["node"] for r in records if split_convert(r["name"])]
    generate_nodes = [r["node"] for r in records if not split_convert(r["name"])]
    for group, nodes in (("mesh", convert_nodes), ("generate", generate_nodes)):
        if not nodes:
            continue
        thicket.statsframe.dataframe = full_statsframe.loc[nodes]
        plt.figure(figsize=(8, 1.5 + 0.25 * len(nodes)))
        ax = th.stats.display_heatmap(thicket, columns=[mean_column], annot=True, fmt=".6f")
        ax.tick_params(axis="y", labelsize=7)
        if ax.collections and ax.collections[0].colorbar:
            ax.collections[0].colorbar.ax.ticklabel_format(style="plain", useOffset=False)
        path = out_dir / f"thicket_{group}_heatmap.png"
        ax.get_figure().savefig(path, dpi=150, bbox_inches="tight")
        plt.close(ax.get_figure())
        print(f"saved {path}")
    thicket.statsframe.dataframe = full_statsframe

    if convert_nodes:
        ax = th.stats.display_boxplot(thicket, nodes=convert_nodes, columns=[time_column])
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        path = out_dir / "thicket_boxplot.png"
        ax.get_figure().savefig(path, dpi=150, bbox_inches="tight")
        plt.close(ax.get_figure())
        print(f"saved {path}")


def generate_label(records):
    operations = {split_generate_name(r["name"])[0] for r in records}
    if len(operations) == 1:
        return operations.pop().replace("_", " ").capitalize()
    return "Generate-function"


def render_plot_set(records, out_dir, iters_label, heatmaps):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_order_labels = [format_cfg_label(c) for c in CFG_ORDER]

    cfgs_seen = {r["cfg"] for r in records}
    cfg_list = ", ".join(format_cfg_label(c) for c in ordered(cfgs_seen, CFG_ORDER))
    print(f"  {len(cfgs_seen)} execution config(s): {cfg_list}")

    baseline_cfg = BASELINE_CFG if BASELINE_CFG in cfgs_seen \
        else ordered(cfgs_seen, CFG_ORDER)[0]
    baseline_records = [r for r in records if r["cfg"] == baseline_cfg]
    mesh_records = [r for r in records if split_convert(r["name"])]
    generate_records = [r for r in records if not split_convert(r["name"])]
    generate_name = generate_label(generate_records)

    for suffix, quantity, unit, scale in METRICS:
        value_fn = METRIC_VALUES[quantity]
        ylabel = f"avg time / {quantity} ({unit})"
        for group, stem, benchmark in ((mesh_records, "mesh_scaling", "Mesh conversion"),
                                       (generate_records, "generate_scaling", generate_name)):
            panels = build_line_panels(group, value_fn)
            if panels:
                plot_group(panels, out_dir / f"{stem}{suffix}.png",
                           f"{benchmark} benchmark: avg time per {quantity} ({iters_label})",
                           ylabel, scale, cfg_order_labels)

    if not heatmaps:
        return baseline_records

    heatmap_cfgs = [c for c in (BASELINE_CFG, DEVICE_BASELINE_CFG) if c in cfgs_seen]
    if not heatmap_cfgs:
        heatmap_cfgs = ordered(cfgs_seen, CFG_ORDER)[:1]
    suffix_heatmaps = len(heatmap_cfgs) > 1

    for cfg in heatmap_cfgs:
        cfg_suffix = "_" + "-".join(cfg) if suffix_heatmaps else ""
        cfg_note = f" [{format_cfg_label(cfg)}]" if suffix_heatmaps else ""

        mesh_lookup = {}
        for r in mesh_records:
            if r["cfg"] == cfg:
                src, dst = split_convert(r["name"])
                mesh_lookup[(src, dst, r["dim"])] = r
        shape_lookup = {}
        for r in generate_records:
            operation, shape = split_generate_name(r["name"])
            if r["cfg"] == cfg and shape:
                shape_lookup[(operation, shape, r["dim"])] = r

        for suffix, quantity, unit, scale in METRICS:
            value_fn = METRIC_VALUES[quantity]
            cbar_label = f"avg time / {quantity} ({unit})"
            if mesh_lookup:
                plot_heatmaps(
                    mesh_lookup, MESH_SRC_ORDER, [], value_fn, scale, out_dir,
                    f"mesh_conversion_heatmap{suffix}{cfg_suffix}",
                    f"Mesh conversion avg time per {quantity}, {unit}{cfg_note} ({iters_label})",
                    "source representation", "target representation", cbar_label,
                )
            if shape_lookup:
                plot_heatmaps(
                    shape_lookup, OPERATIONS, SHAPE_ORDER, value_fn, scale, out_dir,
                    f"generate_heatmap{suffix}{cfg_suffix}",
                    f"{generate_name} avg time per {quantity}, {unit}{cfg_note} ({iters_label})",
                    "operation", "element shape", cbar_label,
                )
        if shape_lookup and cfg == heatmap_cfgs[0]:
            plot_heatmaps(
                shape_lookup, OPERATIONS, SHAPE_ORDER, growth_value, 1.0, out_dir,
                "generate_growth_heatmap",
                f"{generate_name} output/input element ratio ({iters_label})",
                "operation", "element shape", "output elements / input elements",
                cell_fmt=format_growth_cell,
            )

    return baseline_records


def backend_group(record):
    if record["backend"] is None:
        return None
    if record["mem"] == "unified":
        return record["backend"] + "-unified"
    return record["backend"]


def plot_single(cali_file, rank_reduce):
    thicket, records = load_records(cali_file, rank_reduce)

    iters = sorted({r["iter"] for r in records})
    iters_label = f"n={iters[0]}" if len(iters) == 1 else f"n={iters[0]}-{iters[-1]}"
    iters_label += range_label("threads", {r["threads"] for r in records
                                           if r["threads"] is not None})
    iters_label += range_label("ranks", {r["ranks"] for r in records
                                         if r["ranks"] is not None})

    out_dir = cali_file.with_suffix("")
    out_dir.mkdir(exist_ok=True)

    # One plot set per backend, plus everything together unless there is
    # exactly one backend. A file without backend fields gets combined only.
    groups = []
    for backend in sorted({backend_group(r) for r in records if backend_group(r)}):
        groups.append((backend, [r for r in records if backend_group(r) == backend]))
    if len(groups) != 1:
        groups.append((COMBINED_LABEL, records))

    for label, group_records in groups:
        print(f"{label}: {len(group_records)} records")
        baseline_records = render_plot_set(
            group_records, out_dir / label, f"{iters_label}, {label}",
            heatmaps=(len(groups) == 1 or label != COMBINED_LABEL),
        )

    plot_thicket_views(thicket, baseline_records, add_inclusive_column(thicket, records), out_dir)
    print(f"wrote plots to {out_dir}/")


# ---------------------------------------------------------------------------
# Two-run comparison (--compare)
# ---------------------------------------------------------------------------

MATCH_FIELDS = ("name", "dim", "inverts", "inelems", "outverts", "outelems")
BACKEND_ORDER = ["serial", "hip", "cuda", "sycl", "gpu"]
BASE_COLOR = "#1F77B4"
BACKEND_COLORS = {"serial": "#FF7F0E", "hip": "#2CA02C", "cuda": "#2CA02C",
                  "sycl": "#2CA02C", "gpu": "#2CA02C"}
EXTRA_COLORS = ["#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
SLIDE_DPI = 180
SLIDE_SIZE = (2400 / SLIDE_DPI, 1350 / SLIDE_DPI)
SLIDE_PANELS = 6
# Above this many series, grouped bars stop being readable: the summary
# becomes horizontal rows, the detail panels switch to lines and the series
# take matplotlib's tab10/tab20 palette.
MAX_BAR_SERIES = 4


def series_key(record):
    backend = record["backend"] or "serial"
    return (backend.lower(), record["cfg"])


def series_label(key, run_label=None):
    backend, cfg = key
    if backend in ("hip", "cuda", "sycl", "gpu"):
        text = backend.upper()
    else:
        text = backend.title()
    canonical = BASELINE_CFG if backend == "serial" else DEVICE_BASELINE_CFG
    if cfg != canonical:
        text += " " + format_cfg_label(cfg)
    if run_label:
        text += f" ({run_label})"
    return text


def series_sort_key(key):
    backend, cfg = key
    return (rank(backend, BACKEND_ORDER), rank(cfg, CFG_ORDER))


def index_series(records):
    indexed = {}
    for record in records:
        case = tuple(record[field] for field in MATCH_FIELDS)
        indexed.setdefault(series_key(record), {})[case] = record
    return indexed


def pair_runs(base_index, new_index, base_label, new_label):
    # A base run with a single series (a host-only run from before device
    # support) is the reference for every series in the new run. Otherwise
    # each series is paired with the same series in the base run.
    single_base = len(base_index) == 1
    if single_base:
        base_key = next(iter(base_index))
        pairs = [(base_key, key) for key in sorted(new_index, key=series_sort_key)]
        print(f"comparing every series in {new_label} against "
              f"{series_label(base_key)} from {base_label}")
    else:
        pairs = []
        for key in sorted(set(base_index) | set(new_index), key=series_sort_key):
            if key in base_index and key in new_index:
                pairs.append((key, key))
            else:
                side = base_label if key in base_index else new_label
                print(f"  skipped {series_label(key)}: only present in {side}")
        print(f"comparing each series in {new_label} against the same series in {base_label}")
    if not pairs:
        sys.exit("the two runs have no series in common")

    series = []
    if single_base:
        series.append({"label": series_label(base_key, base_label), "color": BASE_COLOR,
                       "base": base_index[base_key], "new": base_index[base_key]})
    used_backends = []
    extra = 0
    for base_key, new_key in pairs:
        backend = new_key[0]
        if backend in BACKEND_COLORS and backend not in used_backends:
            color = BACKEND_COLORS[backend]
        else:
            color = EXTRA_COLORS[extra % len(EXTRA_COLORS)]
            extra += 1
        used_backends.append(backend)
        label = series_label(new_key, new_label if single_base else None)
        series.append({"label": label, "color": color,
                       "base": base_index[base_key], "new": new_index[new_key]})
    if len(series) > MAX_BAR_SERIES:
        colors = plt.get_cmap("tab10" if len(series) <= 10 else "tab20").colors
        for index, s in enumerate(series):
            s["color"] = colors[index % len(colors)]
    return series


def region_sort_key(name):
    if split_convert(name):
        return (0, rank(name, CONVERSION_ORDER), name)
    operation, shape = split_generate_name(name)
    return (1, rank(operation, OPERATIONS), rank(shape, SHAPE_ORDER), name)


def row_sort_key(row):
    return (region_sort_key(row["name"]), row["dim"])


def match_cases(series):
    common = set(series[0]["base"])
    for s in series:
        common &= set(s["base"]) & set(s["new"])
    if not common:
        sys.exit("the two runs have no benchmark cases in common")

    rows = []
    for case in common:
        row = dict(zip(MATCH_FIELDS, case))
        row["speedups"] = []
        row["times"] = []
        for s in series:
            base_time = s["base"][case]["avg_time"]
            new_time = s["new"][case]["avg_time"]
            row["speedups"].append(base_time / new_time)
            row["times"].append((base_time, new_time))
        rows.append(row)
    rows.sort(key=row_sort_key)
    print(f"  {len(rows)} matched cases")
    return rows


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summarize(rows, series):
    summary = []
    for index, s in enumerate(series):
        values = [row["speedups"][index] for row in rows]
        item = dict(s)
        item["geomean"] = geometric_mean(values)
        item["min"] = min(values)
        item["max"] = max(values)
        summary.append(item)
    return summary


def pretty_region(name):
    convert = split_convert(name)
    if convert:
        return f"{convert[0].title()} → {convert[1].title()}"
    operation, shape = split_generate_name(name)
    if shape:
        return f"{operation.replace('_', ' ').title()}: {shape.title()}"
    return name.replace("_", " ").title()


def compact_count(value):
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= limit:
            return f"{value / limit:.0f}{suffix} elems"
    return f"{value} elems"


def speedup_tick(value, _position):
    return f"{value:,.0f}x" if value >= 10 else f"{value:.2f}x"


def wrap_label(label, width=18):
    name, _, detail = label.partition(" (")
    lines = textwrap.wrap(name, width)
    if detail:
        lines += textwrap.wrap("(" + detail, width)
    return "\n".join(lines)


def label_log_bar(ax, x, value, floor, fontsize, rotate):
    # White label inside a tall bar, dark label above a short one.
    if value <= floor:
        return
    decades = math.log10(value / floor)
    if decades < (0.55 if rotate else 0.28):
        ax.annotate(f"{value:.2f}x", xy=(x, value), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color="#222222", zorder=6)
    elif rotate:
        ax.text(x, math.sqrt(floor * value), f"{value:.2f}x", ha="center", va="center",
                rotation=90, fontsize=fontsize, fontweight="bold", color="white",
                clip_on=True, zorder=6)
    else:
        y = 10 ** (math.log10(value) - min(0.07, decades * 0.22))
        ax.text(x, y, f"{value:.2f}x", ha="center", va="top", fontsize=fontsize,
                fontweight="bold", color="white", clip_on=True, zorder=6)


def draw_range(ax, x, low, high):
    cap = 0.055
    ax.vlines(x, low, high, color="#333333", linewidth=1.2, zorder=4)
    ax.hlines([low, high], x - cap, x + cap, color="#333333", linewidth=1.2, zorder=4)
    if high / low >= 1.25:
        labels = [(high, f"{high:.2f}x"), (low, f"{low:.2f}x")]
    elif high == low:
        labels = [(low, f"{low:.2f}x")]
    else:
        labels = [(high, f"{low:.2f}x to\n{high:.2f}x")]
    for y, text in labels:
        ax.annotate(text, xy=(x + cap, y), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7.5, color="#555555", zorder=5)


def draw_summary(ax, summary, title):
    if len(summary) > MAX_BAR_SERIES:
        draw_summary_rows(ax, summary, title)
        return

    spacing, width = 1.15, 0.72
    positions = np.arange(len(summary)) * spacing
    floor = min(0.8, min(s["min"] for s in summary) * 0.8)
    ceiling = max(2.0, max(s["max"] for s in summary) * 2.0)
    bars = ax.bar(positions, [s["geomean"] - floor for s in summary], bottom=floor,
                  width=width, color=[s["color"] for s in summary], zorder=3)
    for bar, s in zip(bars, summary):
        center = bar.get_x() + bar.get_width() / 2
        label_log_bar(ax, center, s["geomean"], floor, fontsize=10, rotate=False)
        draw_range(ax, bar.get_x() + bar.get_width() + 0.07, s["min"], s["max"])

    ax.axhline(1.0, color="#777777", linewidth=0.7, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(floor, ceiling)
    ax.set_xlim(positions[0] - 0.65, positions[-1] + 0.90)
    ax.set_xticks(positions)
    ax.set_xticklabels([wrap_label(s["label"]) for s in summary], fontsize=9, linespacing=1.08)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(speedup_tick))
    ax.set_ylabel("Geometric-mean speedup, log scale")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_summary_rows(ax, summary, title):
    # One row per series: name, bar, geometric mean beside the bar end, and
    # the min-to-max range beneath the bar.
    positions = np.arange(len(summary))[::-1]
    floor = min(0.5, min(s["min"] for s in summary) * 0.7)
    ceiling = max(3.0, max(s["max"] for s in summary) * 4.0)
    ax.barh(positions, [s["geomean"] - floor for s in summary], left=floor, height=0.5,
            color=[s["color"] for s in summary], zorder=3)
    for position, s in zip(positions, summary):
        ax.annotate(f"{s['geomean']:.2f}x", xy=(s["geomean"], position), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9,
                    fontweight="bold", color="#222222", zorder=6)
        y = position - 0.36
        ax.hlines(y, s["min"], s["max"], color="#333333", linewidth=1.1, zorder=4)
        ax.vlines([s["min"], s["max"]], y - 0.06, y + 0.06, color="#333333", linewidth=1.1,
                  zorder=4)
        text = f"{s['min']:.2f}x" if s["min"] == s["max"] else f"{s['min']:.2f}x to {s['max']:.2f}x"
        ax.annotate(text, xy=(s["max"], y), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7.5, color="#555555", zorder=5)

    ax.axvline(1.0, color="#777777", linewidth=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(floor, ceiling)
    ax.set_ylim(-0.8, len(summary) - 0.3)
    ax.set_yticks(positions)
    ax.set_yticklabels([wrap_label(s["label"], 24) for s in summary], fontsize=9)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(speedup_tick))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Geometric-mean speedup, log scale")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", which="major", alpha=0.25, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_details(axes, regions, rows, series):
    values = []
    for row in rows:
        values += row["speedups"]
    floor = min(0.8, min(values) * 0.8)
    ceiling = max(2.0, max(values) * 1.55)
    width = min(0.8 / len(series), 0.25)
    use_lines = len(series) > MAX_BAR_SERIES

    for panel, (ax, region) in enumerate(zip(axes, regions)):
        region_rows = [row for row in rows if row["name"] == region]
        positions = np.arange(len(region_rows))
        for index, s in enumerate(series):
            heights = [row["speedups"][index] for row in region_rows]
            if use_lines:
                ax.plot(positions, heights, marker=MARKERS[index % len(MARKERS)],
                        markersize=4, linewidth=1.2, color=s["color"], zorder=3)
                continue
            offset = (index - (len(series) - 1) / 2) * width
            bars = ax.bar(positions + offset, [h - floor for h in heights], bottom=floor,
                          width=width, color=s["color"], zorder=3)
            for bar, value in zip(bars, heights):
                label_log_bar(ax, bar.get_x() + bar.get_width() / 2, value, floor,
                              fontsize=5.5, rotate=True)

        ax.set_yscale("log")
        ax.set_ylim(floor, ceiling)
        ax.axhline(1.0, color="#777777", linewidth=0.7, zorder=2)
        ax.grid(axis="y", which="major", alpha=0.25, zorder=1)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{row['dim']}\n{compact_count(row['inelems'])}" for row in region_rows],
                           fontsize=7)
        if use_lines:
            ax.set_xlim(-0.4, len(region_rows) - 0.6)
        ax.set_title(pretty_region(region), fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=7))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(speedup_tick))
        if panel % 2 == 0:
            ax.set_ylabel("Speedup", fontsize=8)
    for ax in axes[len(regions):]:
        ax.set_visible(False)


def save_comparison_plots(rows, summary, out_dir, base_label):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.axisbelow": True})
    title = f"Average Speedup Relative to {base_label}"

    if len(summary) > MAX_BAR_SERIES:
        figsize = (11.0, 1.6 + 0.7 * len(summary))
    else:
        figsize = (10.5, 5.5)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    draw_summary(ax, summary, title)
    fig.tight_layout()
    fig.savefig(out_dir / "average_speedup.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"saved {out_dir}/average_speedup.png")

    regions = sorted({row["name"] for row in rows}, key=region_sort_key)
    all_conversions = all(split_convert(region) for region in regions)
    subject = "Mesh Conversion" if all_conversions else "Benchmark"
    handles = [mpatches.Patch(color=s["color"], label=s["label"]) for s in summary]
    many = len(summary) > MAX_BAR_SERIES
    pages = math.ceil(len(regions) / SLIDE_PANELS)
    for page in range(pages):
        page_regions = regions[page * SLIDE_PANELS:(page + 1) * SLIDE_PANELS]
        page_suffix = "" if pages == 1 else f"_{page + 1:02d}"
        fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
        layout = fig.add_gridspec(1, 2, width_ratios=(1.20, 1.35), left=0.15 if many else 0.07,
                                  right=0.985, bottom=0.19 if many else 0.14, top=0.89,
                                  wspace=0.19)
        draw_summary(fig.add_subplot(layout[0]), summary, title)

        detail_rows = max(1, math.ceil(len(page_regions) / 2))
        detail = layout[1].subgridspec(detail_rows, 2, hspace=0.50, wspace=0.30)
        axes = []
        for r in range(detail_rows):
            for c in range(2):
                axes.append(fig.add_subplot(detail[r, c]))
        draw_details(axes, page_regions, rows, summary)

        fig.text(0.75, 0.95, f"Speedup by {subject} and Problem Size", ha="center", va="top",
                 fontsize=14, fontweight="bold")
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.75, 0.02 if many else 0.045),
                   ncol=min(len(summary), 3 if many else 4), frameon=False, fontsize=8)

        path = out_dir / f"comparison_slide{page_suffix}.png"
        fig.savefig(path, dpi=SLIDE_DPI, facecolor="white")
        plt.close(fig)
        print(f"saved {path}")

        # The right-hand grid again on its own, so it can be reviewed without
        # the summary panel.
        legend_inches = legend_height(len(summary), 3)
        height = 3.4 * detail_rows + 0.9 + legend_inches
        fig, axes = plt.subplots(detail_rows, 2, figsize=(11.0, height), squeeze=False,
                                 facecolor="white")
        draw_details(list(axes.flatten()), page_regions, rows, summary)
        fig.suptitle(f"Speedup by {subject} and Problem Size", fontsize=14, fontweight="bold")
        fig.legend(handles=handles, loc="lower center", ncol=min(len(summary), 3), frameon=False,
                   fontsize=9)
        fig.tight_layout(rect=(0, (legend_inches + 0.3) / height, 1, 0.95))
        path = out_dir / f"speedup_by_benchmark{page_suffix}.png"
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"saved {path}")


def write_comparison_csv(rows, summary, out_dir):
    with open(out_dir / "matched_cases.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        header = list(MATCH_FIELDS)
        for s in summary:
            header += [f"{s['label']} base_ms", f"{s['label']} new_ms", f"{s['label']} speedup"]
        writer.writerow(header)
        for row in rows:
            cells = [row[field] for field in MATCH_FIELDS]
            for (base_time, new_time), speedup in zip(row["times"], row["speedups"]):
                cells += [base_time * 1e3, new_time * 1e3, speedup]
            writer.writerow(cells)

    with open(out_dir / "speedup_summary.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["series", "geometric_mean_speedup", "minimum_speedup",
                         "maximum_speedup", "matched_cases"])
        for s in summary:
            writer.writerow([s["label"], s["geomean"], s["min"], s["max"], len(rows)])

    with open(out_dir / "speedup_by_region.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["region", "series", "geometric_mean_speedup", "minimum_speedup",
                         "maximum_speedup", "matched_cases"])
        for region in sorted({row["name"] for row in rows}, key=region_sort_key):
            for index, s in enumerate(summary):
                values = [row["speedups"][index] for row in rows if row["name"] == region]
                writer.writerow([region, s["label"], geometric_mean(values), min(values),
                                 max(values), len(values)])
    print(f"saved matched_cases.csv, speedup_summary.csv, speedup_by_region.csv to {out_dir}/")


def compare_runs(base_file, new_file, base_label, new_label, out_dir, rank_reduce):
    out_dir.mkdir(parents=True, exist_ok=True)

    _, base_records = load_records(base_file, rank_reduce)
    _, new_records = load_records(new_file, rank_reduce)
    print("timing: inclusive Caliper region time divided by the recorded iteration count")
    series = pair_runs(index_series(base_records), index_series(new_records),
                       base_label, new_label)
    rows = match_cases(series)
    summary = summarize(rows, series)

    print(f"\n{'series':48s} {'geo. mean':>10s} {'min':>10s} {'max':>10s}")
    for s in summary:
        print(f"{s['label'][:48]:48s} {s['geomean']:9.2f}x {s['min']:9.2f}x {s['max']:9.2f}x")
    write_comparison_csv(rows, summary, out_dir)
    save_comparison_plots(rows, summary, out_dir, base_label)
    print(f"compared {len(rows)} cases; wrote plots to {out_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot one benchmark run, or compare two runs with --compare.")
    parser.add_argument("cali_file", nargs="?", type=Path,
                        help="Caliper file to plot (default: newest .cali in the current directory)")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"), type=Path,
                        help="plot the speedup of run NEW over run BASE instead")
    parser.add_argument("--label-base", help="name for BASE in the comparison (default: file stem)")
    parser.add_argument("--label-new", help="name for NEW in the comparison (default: file stem)")
    parser.add_argument("--output", type=Path, help="comparison output directory")
    parser.add_argument("--rank-reduce", choices=sorted(RANK_REDUCERS), default="max",
                        help="how to combine the per-rank times of an MPI run (default: max)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.compare:
        base_file, new_file = args.compare
        out_dir = args.output or base_file.parent / f"{base_file.stem}_vs_{new_file.stem}"
        compare_runs(base_file, new_file, args.label_base or base_file.stem,
                     args.label_new or new_file.stem, out_dir, args.rank_reduce)
    else:
        plot_single(args.cali_file or find_cali_file(), args.rank_reduce)


if __name__ == "__main__":
    main()
