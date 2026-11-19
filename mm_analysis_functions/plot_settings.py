import matplotlib.pyplot as plt

# --------------------------------------------------
# GLOBAL PLOTTING SETTINGS
# --------------------------------------------------

plt.rcParams.update({

    # Font
    'font.family': 'serif',
    'font.size': 8,

    # Axes
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'axes.linewidth': 0.8,

    # Ticks
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,

    # Legend
    'legend.fontsize': 7.5,

    # Lines
    'lines.linewidth': 1.2,

    # PDF output
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# --------------------------------------------------
# STANDARD FIGURE SIZES (inches)
# --------------------------------------------------

FIG_WIDTH = 3.35
PANEL_HEIGHT = 1
FIG_WIDTH_FULL = 7.0


# --------------------------------------------------
# FIGURE CREATION
# --------------------------------------------------

def create_figure(num_rows=1, num_cols=1, full_width=False, **kwargs):

    width = FIG_WIDTH_FULL if full_width else FIG_WIDTH
    height = PANEL_HEIGHT * num_rows

    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(width, height),
        **kwargs
    )

    return fig, axs

"""
# One panel
fig, ax = create_figure()

# Three vertical panels
fig, axs = create_figure(3, 1)

# Three horizontal panels
fig, axs = create_figure(1, 3)

# 2 x 2 grid
fig, axs = create_figure(2, 2)

# Three vertical panels, full width
fig, axs = create_figure(3, 1, full_width=True)
"""