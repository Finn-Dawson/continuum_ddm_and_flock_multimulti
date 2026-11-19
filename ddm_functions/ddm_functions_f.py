import ddm_functions.multiDDMProcessing as multiDDMProcessing
import cupy as cp
import numpy as np
from scipy.signal import find_peaks

"""
DDM functions for processing simulated particle data. Example useage:

# simulation parameters
N = 2000
L = 1200
speed = 1.5
persistence_length = 10
noise_strength = np.sqrt(2 * speed / persistence_length) # for normal noise.
Tf = 3000
T_ss = 1000

# ddm parameters
Dt_set = [50, 100, 125, 250, 500]  # selection of episode lengths to be used
ws = [2,3,4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
tauMax = 41  # maximum tau Fourier time
taus = np.arange(1, tauMax)
K = 30  # number of Fourier directions to measure at
anglesPerDirection = (
    1  # K * anglesPerDirection angles over the interval 0 to pi are measured at,
)
# then sorted into bins containing anglesPerDirection angles each, and an average is taken of values in each bin
prominence_threshold = 0.1 # required prominence for sec function fitting - can be changed!


# mm-DDM analysis Xs = unwrapped pos
qq_subs = ddm.give_outputpositions(0.4, K, anglesPerDirection)
all_splits = ddm.give_all_splits_f(Dt_set, Xs, len(Xs))
multiDDMs_q = ddm.give_AllmultiDDMs_q_f(
    Dt_set, all_splits, ws, qq_subs, ws[-1], L, taus
)
fits_peak = ddm.give_Allfits_peak_method_f(
    Dt_set, multiDDMs_q, ws, prominence_threshold
)
densities = ddm.box_densities(Dt_set, ws, Xs, L)
anisos = ddm.give_anisos(Dt_set, ws, fits_peak, num_a_steps = 100, tauMax = 41)

"""


# returns fourier-space wavevectors (q_x, q_y) for all sampled directions.
def give_outputpositions(waveNumber, numberOfDirections, anglesPerDirection):
    outputAngles = cp.linspace(
        0, cp.pi, numberOfDirections * anglesPerDirection, endpoint=False
    )
    outputAngles = outputAngles.reshape((numberOfDirections, anglesPerDirection))
    outputPositions = cp.zeros(
        (numberOfDirections, anglesPerDirection, 2), cp.float32
    )
    outputPositions[:, :, 0] = (waveNumber * cp.cos(outputAngles)).astype(
        cp.float32
    )
    outputPositions[:, :, 1] = (waveNumber * cp.sin(outputAngles)).astype(
        cp.float32
    )
    return outputPositions #shape is (K, angles_per_direction, 2)


# Returns list of episodes of length Dt, each containing time series of positions of particles
def split_frames_poss_f(Dt, poss, Tf):
    poss_split = []
    k = Tf // Dt
    for j in range(k):
        poss_split.append(poss[j * Dt : (j + 1) * Dt, :, :])
    return poss_split


# Returns list of lists of episodes of length Dt for each Dt in Dt_set
def give_all_splits_f(Dt_set, poss, Tf):
    all_splits = []
    for k in range(len(Dt_set)):
        all_splits.append(split_frames_poss_f(Dt_set[k], poss, Tf))
    return all_splits #nested, all_splits[Dt_idx][episode_number] has shape (episode_length, num_particles, 2)


# DDM is calculated chunk by chunk to reduce memory usage
def give_AllmultiDDMs_q_f(Dt_set, all_splits, ws, outposition_tot, base_gridSize, L, taus):

    all_DDMs_per_Dt = []

    for Dt_idx in range(len(Dt_set)):
        DDMs_for_this_Dt = []

        for episode_number in range(len(all_splits[Dt_idx])):
            base_transform_chunk = multiDDMProcessing.doBaseTransform(
                all_splits[Dt_idx][episode_number], outposition_tot, base_gridSize, L
            )

            DDMs_for_this_chunk = []
            for l in range(len(ws)):
                ddm_output = multiDDMProcessing.getDDMOutput(
                    base_transform_chunk, ws[l], taus
                ).get()
                DDMs_for_this_chunk.append(ddm_output)

            DDMs_for_this_Dt.append(DDMs_for_this_chunk)

            del base_transform_chunk
            cp.get_default_memory_pool().free_all_blocks()

        all_DDMs_per_Dt.append(DDMs_for_this_Dt)

    return all_DDMs_per_Dt #all_DDMs_per_Dt[Dt_idx][episode_number][ws_idx] has shape (xgrid_size, ygrid_size, K, num_taus)


def get_Allfits_peak_method(data, prominence_threshold=0.1):

    l_x, l_y, K, num_taus = data.shape #(xgrid_size, ygrid_size, K, num_taus)

    arg_maxes = np.full((l_x, l_y, K), num_taus, dtype=np.float32) #default to largest delay  
    for ilx in range(l_x):
        for ily in range(l_y):
            for i in range(K):
                isf_curve = data[ilx, ily, i, :]

                if not np.any(isf_curve):
                    arg_maxes[ilx, ily, i] = 0
                    continue

                peak_range = np.max(isf_curve) - np.min(isf_curve)

                if peak_range == 0:
                    continue

                required_prominence = peak_range * prominence_threshold

                peaks, irr = find_peaks(isf_curve, prominence=required_prominence)

                if peaks.size > 0:
                    arg_maxes[ilx, ily, i] = peaks[0] + 1
    return cp.asarray(arg_maxes) #shape: (xgrid_size, ygrid_size, K)


def give_Allfits_peak_method_f(Dt_set, multiDs, ws, prominence_threshold=0.1):
    fit_params = []
    for Dt_idx in range(len(Dt_set)):
        fit_params.append([])
        for episode_number in range(len(multiDs[Dt_idx])):
            fit_params[Dt_idx].append([])
            for k in range(len(ws)):
                fit_params[Dt_idx][episode_number].append(
                    get_Allfits_peak_method(multiDs[Dt_idx][episode_number][k], prominence_threshold)
                )

    return fit_params #fit_params[Dt_idx][episode_number][ws_idx] has shape (xgrid_size, ygrid_size, K)

 
def recip_cos(x, a, b, c, maxim=41):
    return np.where(
        a / np.abs(np.cos(c * (x - b))) < maxim, a / np.abs(np.cos(c * (x - b))), maxim
    )

def give_anisos(Dt_set, ws, fits, num_a_steps = 100, num_b_steps = 100, tauMax = 41):
    qualities = []
    a_cutoff = tauMax - 1
    tauMax_minus = tauMax - 1 #max observable tau
    a_candidates = cp.linspace(0,a_cutoff,num_a_steps, dtype=cp.float32 )
    b_candidates = cp.linspace(0, cp.pi, num_b_steps, endpoint=False)
    for Dt_idx in range(len(Dt_set)):
        qualities.append([])
        for episode_idx in range(len(fits[Dt_idx])):
            qualities[Dt_idx].append([])
            for ws_idx in range(len(ws)):
                # P(lx,ly,K)
                P = fits[Dt_idx][episode_idx][ws_idx][:, :, :]
                Nx, Ny, K = cp.shape(P)
                fourier_angles = cp.linspace(0, np.pi, K, endpoint=False, dtype=cp.float32)
                angle_diffs = b_candidates[:, None] - fourier_angles[None, :] #anglediffs[i,j] b_i - thetas_j

                g = 1 / cp.maximum(cp.abs(cp.cos(angle_diffs)), 1e-12) #shape(num_bs, num_fourierangles)
                y_fits = cp.minimum(g[:,:, None] * a_candidates[None, None, :], tauMax_minus) 
                #y_fits[i,j,k] is the fit value for b option i theta option j for a-optionk
                #P broadcast shape(Nx, Ny, 1,           num_fourierangles, 1           )
                #y broadcast shape(1 , 1 , num_bangles, num_fourierangles, num_aoptions)
                # discard data points with no maxima found 
                errors = cp.abs(P[:,:,None, :, None]-y_fits[None,None,:,:,:])
                all_residuals = cp.sum(errors, axis = 3) #error for (Nx, Ny, b_option, a_option)
                # best_residual_a_index = cp.argmin(all_residuals, axis=-1) # findthe best a for x,y,b
                best_residuals = cp.min(all_residuals, axis=-1) #best for (Nx, Ny, b_option)
                # best_a = a_candidates[best_residual_a_index] #(Nx,Ny,b_option)
                # best_b_idx = cp.argmin(best_residuals, axis=-1) #(Nx,Ny)
                # best_a_overall = cp.take_along_axis(best_a, best_b_idx[..., None], axis=-1)[..., 0]
                # above does:
                # for x in range(Nx):
                #     for y in range(Ny):
                #         best_a_overall[x,y] = best_a[x,y,best_b_idx[x,y]]

                # compare with iso fit
                med = cp.median(P, axis=-1)
                iso_residual = cp.sum(cp.abs(P - med[...,None]), axis=-1)
            
                # denom = cp.sum(best_residuals, axis=-1) #sum for all b options (Nx,Ny)
                min_res = cp.min(best_residuals, axis=-1) #min res for each cell (Nx,Ny)
                aniso_residual = min_res
                quality = (iso_residual - aniso_residual) / (iso_residual + aniso_residual)
                # quality = cp.where(denom==0,0,1 - (K*min_res)/denom) #quality for (Nx,Ny)
                # quality = cp.where(best_a_overall > a_cutoff, 0, quality)
                # nan out empty pixels or pixels where no peak is found.
                mask = cp.all(P==0, axis=-1)
                quality = cp.where(mask, cp.nan, quality)
                qualities[Dt_idx][episode_idx].append(cp.asnumpy(quality))

                
    return qualities
                
# time averaged desnity = counts / area calculation

def box_densities(Dt_set, ws, timeseries, L):
    Dt_set = cp.asarray(Dt_set)
    ws = cp.asarray(ws)
    num_frames = timeseries.shape[0]
    num_particles = timeseries.shape[1]
    num_ws = len(ws)

    box_count = [ws[-1] / w for w in ws]
    densities = []

    all_box_positions = cp.empty(
        (num_ws, num_frames, num_particles , 2), dtype=cp.int32
    )

    for k in range(num_ws):
        grid_size = int(box_count[k])
        all_box_positions[k] = multiDDMProcessing.assignBoxNumbers(
            timeseries, L, grid_size
        ) #shape (numframes, numparticles, 2)

    for dt in Dt_set:

        episode_count = int(num_frames // dt)
        results_for_this_dt = []

        for j in range(episode_count):
            start_frame = j * dt
            end_frame = (j + 1) * dt
            results_for_this_episode = []

            for k in range(num_ws):
                grid_size = int(box_count[k])
                grid_shape = (grid_size, grid_size)
                current_positions = all_box_positions[k, start_frame:end_frame]

                flat_indices = cp.ravel_multi_index(
                    (current_positions[:,:, 0], current_positions[:,:, 1]), grid_shape
                )
                total_counts_flat = cp.bincount(
                    flat_indices.ravel(), minlength=grid_shape[0] * grid_shape[1]
                )
                mean_counts_grid = (total_counts_flat / dt).reshape(grid_shape)
                # can impose a cutoff where the density is set to zero
                # mean_counts_grid[mean_counts_grid < 1] = 0
                box_measure = (L * ws[k] / ws[-1]) ** 2
                density_grid = mean_counts_grid / box_measure

                results_for_this_episode.append(density_grid)
            results_for_this_dt.append(results_for_this_episode)
        densities.append(results_for_this_dt)

    return densities #densities[Dt_idx][episode index][ws index] has shape (xgrid_size, ygrid_size)

# nested element-wise product
def parameter_product(qualities, densities, L):
    if isinstance(qualities, list):
        return [
            parameter_product(quality, density, L)
            for quality, density in zip(qualities, densities)
        ]
    else:
        return qualities * densities * L**2

# np/ cp arrays --> python lists
def structure_consistency(nested):
    if isinstance(nested, cp.ndarray):
        return nested.tolist()
    if isinstance(nested, np.ndarray):
        return nested.tolist()
    if isinstance(nested, list):
        return [structure_consistency(item) for item in nested]
    return nested


# collapse tiles and episodes for each Dt and tile size
def collapse_osc_f(Dt_set, aniso_fits, ws):
    a_tot = []
    for j in range(len(Dt_set)):
        w_results = []
        for w in range(len(ws)):
            flattened_values = []
            for q in range(len(aniso_fits[j])):
                list_of_lists = aniso_fits[j][q][w]
                for sublist in list_of_lists:
                    flattened_values.extend(sublist)
            w_results.append(np.array(flattened_values))
        a_tot.append(w_results)
    return a_tot