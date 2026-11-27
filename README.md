# GitHub for code accompanying two companion papers:
- An approach to characterise dynamics and extract space and time coherence without object tracking or identification, the example of flocking: m2DDM
- Extending Differential Dynamic Microscopy (DDM) to the continuum


## Overview
The code is organised into three main parts:
1) `simulation_functions` contains the forward simulations used to generate particle position data for passing through the m2DDM framework
2) `ddm_functions` contains the m2DDM analysis used to produce the anisotropy dataset
3) `mm_analysis_functions` contains the code used to interpret the anisotropy dataset, and recover the interaction radius using either the machine learning-based method or the inflection sigmoid method described in the text.

`heatmaps_N_r_P_v` contains a selection of example flock heat heatmaps, which can be explored via the `interactive.py` widget.


