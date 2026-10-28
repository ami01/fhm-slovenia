"""
10h — Methodology flowchart for the flood susceptibility pipeline
=================================================================
Draws the manuscript's methodology figure: five bands (A-E) walking from
input data through collinearity screening, sampling and training, the
two-tier validation, and on to explainability and national map production.
Each band is a lettered header strip plus 2-3 small connected step-nodes,
each carrying a short bold label and one supporting fact/number — richer
than a single caption line per band, but each node is short enough to stay
legible after the manuscript's print-size shrink.

Design history:
  v1 (original): dense multi-sentence boxes on a 9.2in canvas, 7pt detail
      text. Printed at 0.88\\linewidth (~5.5in) this put real text under 5pt.
  v2 (2026-09-05): fixed legibility by cutting to one short clause per band,
      but read as too plain for a methods figure.
  v3 (this version, 2026-09-05): keeps v2's legible sizing but restores
      visual structure — per-band step-nodes with connecting arrows, a small
      pictogram per band, and a number badge per node — without returning to
      v1's paragraph-length text.

No data dependencies — this figure is drawn from constants, so it can be
regenerated on any machine without the raster stack.

Writes, into Manuscript/figures/main/:
  Fig02_methodology_flowchart.png   (300 dpi)
  Fig02_methodology_flowchart.pdf   (vector, for the LaTeX build)
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                 Polygon, Rectangle, Wedge)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Manuscript" / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM = "Fig02_methodology_flowchart"

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
BLUE_DARK = "#1f4e79"
BLUE_MID = "#3f7cac"
BLUE_FILL = "#eaf1f8"
ORNG_DARK = "#b1560f"
ORNG_MID = "#d97b2e"
ORNG_FILL = "#fdf1e7"
TXT_DARK = "#1a202c"
TXT_GREY = "#5a6472"
NODE_EDGE = "#c9ced6"
WHITE = "#ffffff"

FS_FIGTITLE = 13.5
FS_SUB = 10.0
FS_LETTER = 15
FS_BANDTITLE = 12.0
FS_REF = 9.0
FS_NODE_TITLE = 10.3
FS_NODE_STAT = 9.0

FIG_W = 7.6
MARGIN = 0.16
BX0 = MARGIN
BX1 = FIG_W - MARGIN

HDR_W = 1.78          # left header column: icon + letter + title + ref
NODE_GAP = 0.22        # horizontal gap between step-nodes (arrow lives here)
BAND_PAD_V = 0.16      # vertical padding above/below content inside a band
GAP = 0.16             # vertical gap between bands
H_TITLE = 0.92
H_FOOT = 0.02

N_BANDS = 5
IN = 1.0 / 72.0


def wrap_lines(s, width_in, fs):
    """Wrapped line list for text of font size fs inside width_in. Never
    breaks inside a word (a mid-word break, e.g. splitting a slash-joined
    term, is worse than a slightly ragged line)."""
    chars = max(8, int(width_in * 72 / (fs * 0.56)))
    return textwrap.wrap(s, chars, break_long_words=False, break_on_hyphens=False)


def wrap(s, width_in, fs):
    return "\n".join(wrap_lines(s, width_in, fs))


def text_block_h(lines, fs, leading=1.28):
    return len(lines) * fs * leading * IN


def node_height(title, stat, width_in):
    """Exact vertical space a node needs: top pad + title + gap + stat + bot pad."""
    t_lines = wrap_lines(title, width_in - 0.16, FS_NODE_TITLE)
    s_lines = wrap_lines(stat, width_in - 0.16, FS_NODE_STAT)
    return (0.14 + text_block_h(t_lines, FS_NODE_TITLE) + 0.06
            + text_block_h(s_lines, FS_NODE_STAT) + 0.13)


def header_height(title, ref_present=True):
    t_lines = wrap_lines(title, HDR_W - 0.22, FS_BANDTITLE)
    icon_zone = 0.62
    ref_zone = 0.30 if ref_present else 0.0
    return icon_zone + text_block_h(t_lines, FS_BANDTITLE) + 0.06 + ref_zone + 0.08


# ----------------------------------------------------------------------------
# Content — per band: letter, title, section ref, icon key, and 2-3 step
# nodes of (title, stat).
# ----------------------------------------------------------------------------
BANDS = [
    dict(letter="A", title="Input & alignment", ref="§2.1–2.3",
         dark=BLUE_DARK, mid=BLUE_MID, fill=BLUE_FILL, icon="layers",
         nodes=[
             ("4 input layers", "DEM, land use, climate/soil/lithology, flood inventory"),
             ("Grid alignment", "resample + reproject to one 12.5 m raster"),
             ("Analysis grid", "258.8 M cells, 129.7 M inside Slovenia"),
         ]),
    dict(letter="B", title="Factor screening", ref="§2.4",
         dark=BLUE_DARK, mid=BLUE_MID, fill=BLUE_FILL, icon="funnel",
         nodes=[
             ("24 candidates", "topography, hydrology, climate, substrate, cover"),
             ("Iterative VIF cut", "drop highest, recompute, repeat, stop below 10"),
             ("20 retained", "highest surviving VIF 3.98"),
         ]),
    dict(letter="C", title="Sampling & training", ref="§3.1–3.2",
         dark=ORNG_DARK, mid=ORNG_MID, fill=ORNG_FILL, icon="models",
         nodes=[
             ("5 sample sizes", "1x-10x flood points, stratified split"),
             ("4 learners + ensemble", "RF, XGBoost, CatBoost, DNN, mean ensemble"),
             ("DNN seeded x10", "reported as mean +/- s.d., not one run"),
         ]),
    dict(letter="D", title="Two-tier validation", ref="§3.3–3.4",
         dark=ORNG_DARK, mid=ORNG_MID, fill=ORNG_FILL, icon="split",
         nodes=[
             ("Random hold-out", "20% stratified test partition"),
             ("Spatial block CV", "5-fold, 50 km blocks, disjoint"),
             ("CSI-optimal threshold", "Youden's J reported alongside"),
         ]),
    dict(letter="E", title="Explain & map", ref="§3.5–3.6",
         dark=ORNG_DARK, mid=ORNG_MID, fill=ORNG_FILL, icon="chart",
         nodes=[
             ("SHAP + freq. ratio", "model-based and model-free, cross-checked"),
             ("National maps", "probability, binary, 5-class susceptibility"),
             ("Area & basin stats", "31 sub-basins, 5-model agreement layer"),
         ]),
]

NODE_AREA_W = (BX1 - BX0) - HDR_W - 0.12


def band_h(bd):
    n = len(bd["nodes"])
    w = (NODE_AREA_W - NODE_GAP * (n - 1)) / n
    node_need = max(node_height(t, s, w) for t, s in bd["nodes"])
    head_need = header_height(bd["title"])
    return max(node_need, head_need) + 2 * BAND_PAD_V - 0.10


FIG_H = H_TITLE + sum(band_h(b) for b in BANDS) + (N_BANDS - 1) * GAP + H_FOOT

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Title ------------------------------------------------------------------
y = FIG_H - 0.28
ax.text(FIG_W / 2, y, "Flood susceptibility modelling workflow, Slovenia, 12.5 m",
        fontsize=FS_FIGTITLE, fontweight="bold", color=TXT_DARK,
        ha="center", va="top")
ax.text(FIG_W / 2, y - 0.30,
        "20 factors  ·  4 learners + ensemble  ·  5 sample sizes  ·  two validation tiers",
        fontsize=FS_SUB, color=TXT_GREY, ha="center", va="top", style="italic")


# ----------------------------------------------------------------------------
# Small pictograms drawn from primitive shapes (no icon font needed)
# ----------------------------------------------------------------------------
def draw_icon(kind, cx, cy, color, r=0.145):
    lw = 1.4
    if kind == "layers":
        for i, dy in enumerate([-0.09, 0.0, 0.09]):
            ax.add_patch(Rectangle((cx - r, cy + dy - 0.032), 2 * r, 0.055,
                                    facecolor=color if i < 2 else WHITE,
                                    edgecolor=color, linewidth=lw, zorder=6,
                                    alpha=1.0 if i == 2 else 0.55 + 0.2 * i))
    elif kind == "funnel":
        pts = [(cx - r, cy + r * 0.75), (cx + r, cy + r * 0.75),
               (cx + r * 0.25, cy - 0.01), (cx + r * 0.25, cy - r * 0.85),
               (cx - r * 0.25, cy - r * 0.85), (cx - r * 0.25, cy - 0.01)]
        ax.add_patch(Polygon(pts, closed=True, facecolor="none",
                              edgecolor=color, linewidth=lw, zorder=6))
    elif kind == "models":
        for dx, dy in [(-0.08, -0.04), (0.08, -0.04), (0.0, 0.09)]:
            ax.add_patch(Circle((cx + dx, cy + dy), r * 0.52, facecolor="none",
                                 edgecolor=color, linewidth=lw, zorder=6))
    elif kind == "split":
        ax.add_patch(FancyArrowPatch((cx, cy + r * 0.9), (cx - r * 0.85, cy - r * 0.8),
                                      arrowstyle="-|>", mutation_scale=8, linewidth=lw,
                                      color=color, zorder=6, shrinkA=0, shrinkB=0))
        ax.add_patch(FancyArrowPatch((cx, cy + r * 0.9), (cx + r * 0.85, cy - r * 0.8),
                                      arrowstyle="-|>", mutation_scale=8, linewidth=lw,
                                      color=color, zorder=6, shrinkA=0, shrinkB=0))
    elif kind == "chart":
        base = cy - r * 0.7
        for dx, h in zip([-0.09, 0.0, 0.09], [0.10, 0.20, 0.15]):
            ax.add_patch(Rectangle((cx + dx - 0.028, base), 0.056, h,
                                    facecolor=color, edgecolor="none", zorder=6, alpha=0.85))


def draw_arrow(x, y_from, y_to, color, horiz=False):
    if horiz:
        ax.add_patch(FancyArrowPatch((y_from, x), (y_to, x), arrowstyle="-|>",
                                      mutation_scale=11, linewidth=1.4, color=color,
                                      zorder=7, shrinkA=0, shrinkB=0))
    else:
        ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>",
                                      mutation_scale=15, linewidth=1.8, color=color,
                                      zorder=7, shrinkA=0, shrinkB=0))


def draw_node(x0, x1, y0, y1, title, stat, dark, mid, fill):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.055",
        facecolor=WHITE, edgecolor=NODE_EDGE, linewidth=1.0, zorder=3))
    ax.add_patch(Rectangle((x0, y1 - 0.055), x1 - x0, 0.055,
                            facecolor=mid, edgecolor="none", zorder=4))
    cx = (x0 + x1) / 2
    w = (x1 - x0) - 0.16
    t_lines = wrap_lines(title, w, FS_NODE_TITLE)
    ax.text(cx, y1 - 0.14, "\n".join(t_lines), fontsize=FS_NODE_TITLE, fontweight="bold",
            color=TXT_DARK, ha="center", va="top", zorder=5, linespacing=1.28)
    # Stat starts a fixed gap below the *measured* bottom of the title block,
    # so it can never collide with a title that wrapped to more lines than
    # a fixed offset would have assumed.
    stat_top = y1 - 0.14 - text_block_h(t_lines, FS_NODE_TITLE) - 0.09
    s_lines = wrap_lines(stat, w, FS_NODE_STAT)
    ax.text(cx, stat_top, "\n".join(s_lines), fontsize=FS_NODE_STAT, color=dark,
            ha="center", va="top", zorder=5, linespacing=1.25)


def draw_band(bd, y_top, h):
    y_bot = y_top - h
    # Header column
    ax.add_patch(FancyBboxPatch(
        (BX0, y_bot), HDR_W, h,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=bd["dark"], edgecolor=bd["dark"], linewidth=1.0, zorder=2))
    ax.add_patch(Rectangle((BX0 + HDR_W - 0.08, y_bot), 0.08, h,
                            facecolor=bd["dark"], edgecolor="none", zorder=2))
    letter_cy = y_top - 0.30
    ax.add_patch(Circle((BX0 + 0.36, letter_cy), 0.155, facecolor=WHITE,
                         edgecolor="none", zorder=3, alpha=0.18))
    ax.text(BX0 + 0.36, letter_cy, bd["letter"], fontsize=FS_LETTER, fontweight="bold",
            color=WHITE, ha="center", va="center", zorder=4)
    draw_icon(bd["icon"], BX0 + HDR_W - 0.40, letter_cy, WHITE)
    t_lines = wrap_lines(bd["title"], HDR_W - 0.22, FS_BANDTITLE)
    ax.text(BX0 + HDR_W / 2, y_top - 0.58, "\n".join(t_lines), fontsize=FS_BANDTITLE,
            fontweight="bold", color=WHITE, ha="center", va="top", zorder=4,
            linespacing=1.24)
    ax.text(BX0 + HDR_W / 2, y_bot + 0.14, bd["ref"], fontsize=FS_REF, color=WHITE,
            style="italic", ha="center", va="bottom", zorder=4, alpha=0.9)

    # Step nodes
    n = len(bd["nodes"])
    x0 = BX0 + HDR_W + 0.12
    x1 = BX1
    w = ((x1 - x0) - NODE_GAP * (n - 1)) / n
    ny0, ny1 = y_bot + BAND_PAD_V, y_top - BAND_PAD_V
    for i, (t, s) in enumerate(bd["nodes"]):
        nx0 = x0 + i * (w + NODE_GAP)
        nx1 = nx0 + w
        draw_node(nx0, nx1, ny0, ny1, t, s, bd["dark"], bd["mid"], bd["fill"])
        if i < n - 1:
            draw_arrow((ny0 + ny1) / 2, nx1, nx1 + NODE_GAP, bd["mid"], horiz=True)
    return y_bot


y = FIG_H - H_TITLE
for i, bd in enumerate(BANDS):
    h = band_h(bd)
    y = draw_band(bd, y, h)
    if i < len(BANDS) - 1:
        draw_arrow(BX0 + HDR_W / 2, y - 0.02, y - GAP + 0.02, BANDS[i]["dark"])
        y -= GAP

fig.savefig(OUT_DIR / f"{STEM}.png", dpi=300, facecolor="white")
fig.savefig(OUT_DIR / f"{STEM}.pdf", facecolor="white")
plt.close(fig)

print(f"wrote {OUT_DIR / (STEM + '.png')}  ({FIG_W:.2f} x {FIG_H:.2f} in)")
print(f"wrote {OUT_DIR / (STEM + '.pdf')}")
