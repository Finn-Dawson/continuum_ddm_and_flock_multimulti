# simulation functions for circular flocks (ballistic + some noise) with a background gas of diffusive particles (random walk with a diffusion coefficient D)
# e.g. Xs = give_poss_hmg(N, L, v0, D, Tf, Thetas, x0, y0, num=number_of_flocks, N_bg=N_bg) generates the simulation data

# Generate a circular flock of randomly located particles with the specified number of particles, radius, and starting position
import numpy as np
import cupy as cp
import copy

def circle(N, ri, x0, y0):
    r0 = cp.array([x0, y0], dtype=cp.float32)
    r = ri * cp.sqrt(cp.random.uniform(0, 1, size=(N)))
    phi = cp.random.uniform(0, 2 * cp.pi, size=(N), dtype=cp.float32)

    pos = cp.zeros((N, 2), dtype=cp.float32)
    pos[:, 0] = r * cp.cos(phi)
    pos[:, 1] = r * cp.sin(phi)

    return r0 + pos

# sets up random initial particle positions
def ini_hmg(N, L, ri, x0, y0, num=1, N_bg=0): 
    pos = cp.zeros((N * num + N_bg, 2), dtype=cp.float32)

    for j in range(num):
        pos[j * N : (j + 1) * N] = circle(N, ri, x0=x0[j], y0=y0[j])

    flock_centres = cp.stack([x0, y0], axis=1)
    bg = []

    bg_left = N_bg
    while bg_left > 0:
        candidates = cp.random.uniform(0, L, size=(bg_left, 2)).astype(cp.float32)
        # delta[i,j,k] is the k-th component of the vector from the j-th flock centre to the i-th background particle
        # shape = (N_bg, num, 2)
        delta = candidates[:, None, :] - flock_centres[None, :, :]
        delta = delta - L * cp.round(delta / L)  
        distances2 = cp.sum(delta**2, axis=2) 
        valid = cp.all(distances2 > ri**2, axis = 1)
        accepted = candidates[valid]
        bg.append(accepted)
        bg_left -= accepted.shape[0]
    background_particles = cp.concatenate(bg, axis=0)[:N_bg]
    pos[N * num:] = background_particles
    return pos
    

def step_fwd_hmg(
    pos,
    v0,
    Theta,
    N,
    L,
    flock_x,
    flock_y,
    num=1,
    v_bgs=cp.asarray([]),
    Theta_bgs=cp.asarray([]),
    N_bg=0,
):
    # equal flock and bg density
    density = (N * num + N_bg) / L**2
    radius_hmg = cp.min(cp.array([(N / (cp.pi * density)) ** (1 / 2), L / 2]))

    pos_new = cp.zeros(cp.shape(pos))
    Theta = cp.array(Theta, dtype=cp.float32)

    # update the flocks
    for j in range(num):
        pos_new[j * N : (j + 1) * N, 0] = (
            pos[j * N : (j + 1) * N, 0]
            + v0 * cp.cos(Theta[j])   
        )
        pos_new[j * N : (j + 1) * N, 1] = (
            pos[j * N : (j + 1) * N, 1]
            + v0 * cp.sin(Theta[j])     
        )
    
    # particles in the background gas advance at a constant velocity (that is different for each particle),
    pos_new[num * N:, 0] = pos[num * N:, 0] + v_bgs * cp.cos(
        Theta_bgs
    )
    pos_new[num * N:, 1] = pos[num * N:, 1] + v_bgs * cp.sin(
        Theta_bgs
    )
    bg = pos_new[num*N:]

    # check bg overlaps with flocks
    # if multiple flock overlaps, correct the first one and re check validity. 
    # if more than 100 attempts to correct, give up and accept the position
    flock_centres = cp.stack([flock_x, flock_y], axis=1).astype(cp.float32)
    check_counter = 0
    
    # split into overlapping and non-overlapping particles
    delta = bg[:, None, :] - flock_centres[None, :, :]
    delta = delta - L * cp.round(delta / L)
    distances2 = cp.sum(delta**2, axis=2)
    active = cp.any(distances2 < radius_hmg**2, axis=1)
    active_idx = cp.where(active)[0]

    while check_counter < 1000:
        if active_idx.size == 0:
            break
        delta = bg[active_idx, None, :] - flock_centres[None, :, :]
        delta = delta - L * cp.round(delta / L)
        distances2 = cp.sum(delta**2, axis=2)
        overlaps = distances2 < radius_hmg**2

        if not cp.any(overlaps):
            break

        # invalid[i] is True if the i-th active particle overlaps with any flock, and False otherwise
        # the ith actice particle is the active_idx[i]th particle in the original bg array 
        invalid = cp.any(overlaps, axis=1)
        idx_local = cp.where(invalid)[0]
        idx_global = active_idx[idx_local]
        flock_idx = cp.argmax(overlaps[idx_local], axis=1)

        # vectors flock centre to particle for overlaps
        vector = -delta[idx_local, flock_idx]
        norm = cp.sqrt(cp.sum(vector**2, axis=1))
        reciprocal_norm = cp.where(norm == 0, 0, 1/norm)

        # flips all overlapping particles across the flock centre and adds a component in the direction of the flip:
        # bg[idx_global] += 2 * vector + v0 * vector * reciprocal_norm[:, None]
        bg[idx_global] += 2*vector + v0 * vector * reciprocal_norm[:, None]

        # recheck which particles are still overlapping after the correction
        delta = bg[active_idx, None, :] - flock_centres[None, :, :]
        delta = delta - L * cp.round(delta / L)
        distances2 = cp.sum(delta**2, axis=2)
        active_mask = cp.any(distances2 < radius_hmg**2, axis=1)
        active_idx = active_idx[active_mask]

        check_counter += 1
        
    pos_new[N*num:] = bg
    return pos_new


def give_poss_hmg(N, L, v0, Tf, Theta, x0, y0, num=1, N_bg=1):

    density = (N * num + N_bg) / L**2
    radius_hmg = cp.min(cp.array([(N / (cp.pi * density)) ** (1 / 2), L / 2]))

    flocks_x = copy.deepcopy(x0)
    flocks_y = copy.deepcopy(y0)

    poss = cp.zeros((Tf, N * num + N_bg, 2), dtype=cp.float32)
    poss_initialised = ini_hmg(N, L, radius_hmg, x0=x0, y0=y0, num=num, N_bg=N_bg)
    v_bgs = cp.zeros(N_bg)
    Theta_bgs = cp.zeros(N_bg)
    for i in range(N_bg):
        v_bgs[i] = cp.random.uniform(0, 2 * v0)
        Theta_bgs[i] = cp.random.uniform(0, 2 * cp.pi)
    poss[0] = step_fwd_hmg(
        poss_initialised,
        v0,
        Theta,
        N,
        L,
        x0,
        y0,
        num=num,
        v_bgs=v_bgs,
        Theta_bgs=Theta_bgs,
        N_bg=N_bg,
    )
    for frame in range(1, Tf):
        flocks_x = (flocks_x + v0 * cp.cos(Theta))
        flocks_y = (flocks_y + v0 * cp.sin(Theta))
        poss[frame] = step_fwd_hmg(
            poss[frame - 1],
            v0,
            Theta,
            N,
            L,
            flocks_x,
            flocks_y,
            num=num,
            v_bgs=v_bgs,
            Theta_bgs=Theta_bgs,
            N_bg=N_bg,
        ) 
    N_tot = N*num + N_bg
    all_thetas = cp.zeros((N_tot), dtype=cp.float32)  
    for j in range(num):
        all_thetas[j*N:(j+1)*N] = Theta[j]
    all_thetas[N*num:] = Theta_bgs   
    return poss, all_thetas