import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from pathlib import Path
import json
import re

"""
LOAD IN HEATMAPS
"""
with open("params.json", "r") as f:
    params = json.load(f)

L = params["simulation"]["L"]
Dt_set = params["ddm"]["Dt_set"]
ws = params["ddm"]["ws"][1:]

tile_sizes = L * np.array(ws) / ws[-1]
heatmap_dir = Path(".")

pattern = re.compile(r"heatmap_N(\d+)_R(\d+)_P(\d+)_v([\d.]+)\.npz")
heatmap_files = {}

for file in heatmap_dir.glob("heatmap_*.npz"):
    match = pattern.fullmatch(file.name)
    if match:
        N, R, P = map(int, match.groups()[:3])
        v = float(match.group(4))
        heatmap_files[(N, R, P, v)] = file

# Baseline when varying N, R or v
def get_baseline(vary):
    if vary == "P":
        return {"N": 2000, "R": 20, "P": 10, "v": 3}
    return {"N": 2000, "R": 20, "P": 10, "v": 1.5}


# flock heats available: 
all_measures = None
for file in heatmap_files.values():
    with np.load(file) as data:
        file_measures = set(data.files)
    if all_measures is None:
        all_measures = file_measures
    else:
        all_measures &= file_measures
measures = sorted(all_measures)

# vary from N,P,v, R. upon choosing what to vary, other available parameter options found.
def get_options(vary):
    fixed = get_baseline(vary)
    options = set()
    for key in heatmap_files:
        matches = all(
            key[i] == fixed[param]
            for i, param in enumerate(["N", "R", "P", "v"])
            if param != vary
        )
        if matches:
            options.add(key[["N", "R", "P", "v"].index(vary)])
    return sorted(options)


"""
PLOTTING
"""
# INITIAL VALUES:
varying = "R"
measure = "mean"
options = get_options(varying)

initial_value = get_baseline(varying)[varying]
initial_index = (
    options.index(initial_value)
    if initial_value in options
    else 0
)

# FIGURE:
fig = plt.figure(figsize=(13, 7.5))
ax = fig.add_axes([0.07, 0.20, 0.55, 0.72])
ax_vary = fig.add_axes([0.70, 0.72, 0.23, 0.20])
ax_measure = fig.add_axes([0.70, 0.25, 0.23, 0.42])
# Parameter slider
ax_slider = fig.add_axes([0.12, 0.09, 0.43, 0.035])
# Play/pause button
ax_play = fig.add_axes([0.70, 0.09, 0.12, 0.06])


def load_heatmap(vary, value, measure):
    fixed = get_baseline(vary)
    fixed[vary] = value
    key = (fixed["N"], fixed["R"], fixed["P"], fixed["v"])
    if key not in heatmap_files:
        raise FileNotFoundError(f"No heatmap available for {key}")
    with np.load(heatmap_files[key]) as data:
        heatmap = data[measure]
    return heatmap, key

# PLOTTING HEATMAPS:
initial_heatmap, initial_key = load_heatmap(varying, options[initial_index], measure)

im = ax.imshow(
    initial_heatmap,
    cmap="magma",
    aspect="auto",
    origin="lower"
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(measure)
ax.set_xticks(np.arange(len(ws)))
ax.set_xticklabels(tile_sizes, rotation=45)
ax.set_yticks(np.arange(len(Dt_set)))
ax.set_yticklabels(Dt_set)
ax.set_xlabel(r"$w$ / px")
ax.set_ylabel(r"$T_{\mathrm{eps}}$ / frames")


def update_title(key):
    N, R, P, v = key
    ax.set_title(
        f"Heatmap: N={N}, R={R} px, P={P} px, v={v}px/fr"
    )

update_title(initial_key)


# ------------------------------------------------------------
# SLIDER
# ------------------------------------------------------------

slider = Slider(
    ax=ax_slider,
    label=varying,
    valmin=0,
    valmax=max(len(options) - 1, 1),
    valinit=initial_index,
    valstep=1
)

slider.valtext.set_text(str(options[initial_index]))


# ------------------------------------------------------------
# RADIO BUTTONS
# ------------------------------------------------------------

vary_radio = RadioButtons(
    ax_vary,
    ("N", "R", "P", "v"),
    active=1
)

measure_radio = RadioButtons(
    ax_measure,
    measures,
    active=measures.index(measure)
)

ax_vary.set_title("Parameter to vary")
ax_measure.set_title("Heatmap measure")


# ------------------------------------------------------------
# UPDATE HEATMAP
# ------------------------------------------------------------

def redraw(value_index):
    value_index = int(value_index)
    value = options[value_index]

    heatmap, key = load_heatmap(varying, value, measure)

    im.set_data(heatmap)
    im.set_clim(np.nanmin(heatmap), np.nanmax(heatmap))

    cbar.update_normal(im)
    cbar.set_label(measure)

    slider.valtext.set_text(str(value))
    update_title(key)

    fig.canvas.draw_idle()

def slider_changed(value):
    redraw(int(value))

slider.on_changed(slider_changed)

# ------------------------------------------------------------
# CHANGE PARAMETER TO VARY
# ------------------------------------------------------------

def change_vary(label):
    global varying, options

    varying = label
    options = get_options(varying)

    if not options:
        ax.set_title(
            f"No heatmaps available for varying {varying}"
        )
        fig.canvas.draw_idle()
        return

    # Choose the baseline value if available
    fixed = get_baseline(varying)
    start_value = fixed[varying]

    start_index = (
        options.index(start_value)
        if start_value in options
        else 0
    )

    # Update slider range and label
    slider.label.set_text(varying)
    slider.valmin = 0
    slider.valmax = max(len(options) - 1, 1)
    slider.valstep = 1
    slider.ax.set_xlim(0, max(len(options) - 1, 1))

    slider.set_val(start_index)

    # Explicitly redraw in case the slider value did not change
    redraw(start_index)

vary_radio.on_clicked(change_vary)


# ------------------------------------------------------------
# CHANGE MEASURE
# ------------------------------------------------------------

def change_measure(label):
    global measure

    measure = label
    redraw(int(slider.val))


measure_radio.on_clicked(change_measure)


# ------------------------------------------------------------
# PLAY / PAUSE
# ------------------------------------------------------------

playing = False
timer = fig.canvas.new_timer(interval=700)

def advance():
    current = int(slider.val)
    next_index = (current + 1) % len(options)
    slider.set_val(next_index)
    return True

timer.add_callback(advance)

def toggle_play(event):
    global playing
    if playing:
        timer.stop()
        playing = False
        play_button.label.set_text("Play")
    else:
        timer.start()
        playing = True
        play_button.label.set_text("Pause")
    fig.canvas.draw_idle()

play_button = Button(ax_play, "Play")
play_button.on_clicked(toggle_play)

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

plt.show()