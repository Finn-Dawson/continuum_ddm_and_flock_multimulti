# N particles Vicsek simulation 
import numpy as np
import cupy as cp

# sets up random initial particle positions and speeds and directions
def ini(N, L, speed):

    theta = cp.random.uniform(0, 2*np.pi, N, dtype=cp.float32)
    pos = cp.random.uniform(0, L, (N,2), dtype=cp.float32)

    vel = cp.zeros((N, 2), dtype=cp.float32)
    vel[:,0] = speed * cp.cos(theta)
    vel[:,1] = speed * cp.sin(theta)
    
    return pos, vel
    

def step_fwd_unwrapped_position(
    pos,
    vel,
    speed,
    interaction_radius,
    noise_strength,
    N,
    L
):
    vel_new = cp.zeros((N ,2), dtype=cp.float32)

    # neighbour distances
    # delta[i,j,k] gives the kth component of the vector from particle j to particle i
    delta = pos[:, None, :] - pos[None, :, :]
    delta = delta - L * cp.round(delta / L)
    distances2 = cp.sum(delta**2, axis=2)
    if interaction_radius == 0:
        interaction_strength = cp.identity(N, dtype=cp.float32)
    else:
        interaction_strength = distances2 <= interaction_radius**2

    interacting_float = interaction_strength.astype(cp.float32)

    # gives the sums of neighbour velocities for each particle.
    sum_x_vels = cp.matmul(interacting_float, vel[:,0])
    sum_y_vels = cp.matmul(interacting_float, vel[:,1])

    average_direction = cp.arctan2(sum_y_vels, sum_x_vels)
    noise = cp.random.normal(0, noise_strength, size = N, dtype=cp.float32)
    # can also do softer allignment, e.g. angle moves 3/4 to the average of the neighbours
    # old_theta = cp.arctan2(vel[:, 1], vel[:, 0])
    new_theta = (average_direction) + noise
    # new_theta = old_theta + noise
    vel_new[:, 0] = speed*cp.cos(new_theta)
    vel_new[:, 1] = speed*cp.sin(new_theta)
    pos_new = (pos + vel_new)

    return pos_new, vel_new
    


def give_positions_unwrapped(speed, interaction_radius, noise_strength, N, L, Tf):
    poss = cp.zeros((Tf, N, 2), dtype=cp.float32)
    vels = cp.zeros((Tf, N, 2), dtype=cp.float32)

    poss_initialised, vel_initialised = ini(N, L, speed)

    poss[0], vels[0] = step_fwd_unwrapped_position(
        poss_initialised,
        vel_initialised,
        speed,
        interaction_radius,
        noise_strength,
        N,
        L
    )
    
    for frame in range(1, Tf):
        poss[frame], vels[frame] = step_fwd_unwrapped_position(
            poss[frame - 1],
            vels[frame - 1],
            speed,
            interaction_radius,
            noise_strength,
            N,
            L
        )  
    return poss, vels
