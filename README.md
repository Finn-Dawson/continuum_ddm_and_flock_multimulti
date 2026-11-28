# GitHub for code accompanying two companion papers:
- An approach to characterise dynamics and extract space and time coherence without object tracking or identification, the example of flocking: m2DDM
- Extending Differential Dynamic Microscopy (DDM) to the continuum


## Overview
The code is organised into three main parts:
1) `simulation_functions` contains the forward simulations used to generate particle position data for passing through the m2DDM framework
2) `ddm_functions` contains the m2DDM analysis used to produce the anisotropy dataset
3) `mm_analysis_functions` contains the code used to interpret the anisotropy dataset, and recover the interaction radius using either the machine learning-based method or the inflection sigmoid method described in the text.

`heatmaps_N_r_P_v` contains a selection of example flock heat heatmaps, which can be explored via the `interactive.py` widget.

`example_flock_videos` contains a selection of example Vicsek flock videos, where particles are coloured by direction.

Parts 1 and 2 make use of the CuPy library and require a CUDA-enabled GPU with a compatible NVIDIA CUDA environment. Part 3 requires PyTorch, which can be run on either CPU or GPU.

## Installation

Clone the repository:
```bash
git clone https://github.com/Finn-Dawson/continuum_ddm_and_flock_multimulti.git
cd continuum_ddm_and_flock_multimulti
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```
To launch the interactive heatmap explorer:
```bash
cd heatmaps_N_r_P_v; python interactive.py
```

## Example usage

### Run forward simulation and generate anisotropy data

```python
import ddm_functions.ddm_functions_f as ddm
import simulation_functions.vicsek_sim as sim
import numpy as np

# simulation parameters
numberOfParticles = 2000
L = 1200
interactionRadius = 20
speed = 1.5
persistenceLength = 10
noise_strength = np.sqrt(2 * speed / persistenceLength) # for normal noise.
numberOfFrames = 2000 #Full simulation length in frames
T_ss = 1000 #Time after which m2DDM analysis is carried out 

# ddm parameters
Dt_set = [50, 100, 125, 250, 500]  # selection of episode lengths to be used in frames
ws = [2,3,4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120] # selection of tiles sizes in multiples of L/ws[-1]
# e.g. if ws = [2,3,4,12] and L = 1200px, tiles have widths 200, 300, 400, and 1200 px.
tauMax = 41  # maximum tau Fourier time
taus = np.arange(1, tauMax)
K = 30  # number of Fourier directions to measure at
anglesPerDirection = 1  # numer of angles used for each Fourier direction 
# then sorted into bins containing anglesPerDirection angles each, and an average is taken of values in each bin
prominence_threshold = 0.1 # required prominence for finding the first peak of the image structure function

# simulation run
Xs_full = sim.give_positions_unwrapped(speed, interactionRadius, noise_strength, numberOfParticles, L, numberOfFrames)
Xs = Xs_full[T_ss:]

# mm-DDM analysis Xs = unwrapped pos
qq_subs = ddm.give_outputpositions(0.4, K, anglesPerDirection)
all_splits = ddm.give_all_splits_f(Dt_set, Xs, len(Xs))
multiDDMs_q = ddm.give_AllmultiDDMs_q_f(Dt_set, all_splits, ws, qq_subs, ws[-1], L, taus)
fits_peak = ddm.give_Allfits_peak_method_f(Dt_set, multiDDMs_q, ws, prominence_threshold)
densities = ddm.box_densities(Dt_set, ws, Xs, L)
anisos = ddm.give_anisos(Dt_set, ws, fits_peak, num_a_steps = 100, tauMax = 41)

#Save densities and anisos
#densities[Dt_idx][episode index][ws index] has shape (xgrid_size, ygrid_size)
#anisos[Dt_idx][episode index][ws index] has shape (xgrid_size, ygrid_size)

```

### Make a flock heat heatmap

```python
from mm_analysis_functions.heatmap_funcs import aniso_distribution, load_anisos, make_heatmaps
import matplotlib.pyplot as plt
heatmap_figure = False

Afiles = [] #list of anisotropy path names
Dfiles = [] #list of density path names

#load in anisos with a choice of an occupancy cutoff (tiles with an average occupancy < occupancy_cutoff are ignored)
anisos = load_anisos(Afiles, Dfiles, Dt_set, ws, L, occupancy_cutoff = 0)
all_heatmaps = make_heatmaps(anisos, Dt_set, ws) #all_heatmaps[heat_choice e.g. "mean"][dt_idx][ws_idx] gives the flock heat H(w, T_eps)

# heatmap plot if desired
heat_choice = "mean"
if heatmap_figure:
    plt.clf()
    plt.imshow(all_heatmaps[heat_choice], cmap='magma', aspect='auto', origin = 'lower')
    plt.title(f"Ri = {Ri}" +"\n"+f"sigma_{heat_choice}", fontsize=11)
    box_numbers = (ws[-1] / np.array(ws)).astype(int)
    box_edges = (L / box_numbers).astype(int)
        
    plt.xticks(
        ticks=np.arange(len(ws))[::2],
        labels=box_edges[::2],
        rotation=45,
        ha='right',
        rotation_mode='anchor'
    )

    plt.yticks(
        ticks=np.arange(len(Dt_set)),
        labels=Dt_set
    )
    plt.xlabel("w / px")
    plt.ylabel("T_eps / s")
    plt.colorbar()
    plt.show()
```
### Recover `interactionRadius`

Inflection sigmoids method:
```python
import mm_analysis_functions.heatmap_funcs

heat_choice = "mean"
Dt_idx_slice = 0 #episode index to slice across
slice = all_heatmaps[heat_choice][Dt_idx_slice]
slice_peak, slice_inflect, popt = heatmap_funcs.find_peak_and_inflect(slice, tile_sizes)

```

Autoencoder method:
```python
import numpy as np
import torch
from mm_analysis_functions.autoencoder_funcs import split_files, run_experiment


#Heatmaps to be used, latent_dim, epochs, and heat choices
latent_dim = 4
data_split_mode = 1 #0 is a random split, 1 tests interpolation of Ri
epochs_ae = 50
epochs_reg = 100
# heat_choices = ["mean","S","std",]
heat_choices = ['mean', 'second_moment', 'third_moment', 'fourth_moment', 'binder', 'pearson_skew', 'S', 'positive_fraction', 'std', 'mod_ratio']

# FILTER INVALID HEATMAPS (data is a .npz with each heatmap and its labels (Ri, N, ...))
files = [] #list of heatmap filenames
files_original = files.copy()
invalid_files = []

for file in files_original:
    with np.load(file) as data:
        # Check all metrics being tested
        valid = all(np.all(np.isfinite(data[metric])) for metric in heat_choices)
    if not valid:
        invalid_files.append(file)

files = [file for file in files_original if file not in invalid_files]

# Common parameters for all heatmaps, load in from params.json
L = None
N = None
Dt_set = None
ws = None

# all the labels (in this case interaction radii)
radii = []
for file in files:
    with np.load(file) as data:
        radii.append(float(data["Ri"].item()))

r_tens = torch.tensor(radii, dtype=torch.float32).unsqueeze(1)


""" 
Train autoencoder and regressor. Results generated from test set.
"""
train_idx, val_idx, test_idx = split_files(files, data_split_mode, r_tens)

results = {}
for heat_choice in heat_choices:
    print(f"\n{'='*40}")
    print(f"Testing: {heat_choice}")
    print(f"{'='*40}")

    results[heat_choice] = run_experiment(
        files,
        heat_choice,
        r_tens,
        train_idx,
        val_idx,
        test_idx,
        latent_dim,
        epochs_ae,
        epochs_reg
    )

# SAVE results
```






































