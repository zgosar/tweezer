"""
Tracks particles.
Opens file, tracks particles, links particles, gets trap data and saves
everything into .dat file.
"""

import unittest
import queue
import numpy as np
try:
    import pims
    from tweezer import TWV_Reader
    have_pims = True
except ImportError:
    have_pims = False
except Exception:
    raise
try:
    import trackpy as tp
    have_trackpy = True
except ImportError:
    have_trackpy = False
except Exception:
    raise

def save_tracked_data_pandas(filename, frames, trajectories, times, laser_powers, traps):
    """
    Converts all data to a (.dat) file.
    Important NOTE:
    The format will change in the future. Particle data and trap data will go into separate files,
    and additional trap metadata will be saved.
    
    Inputs:
     - filename: Output filename.
     - frames: TWV_Reader or other pims object. __len__ needs to be defined.
     - trajectories: Pandas DataFrame with all the particles. Same structure as returned by TrackPy.
     - times: List of frame times.
     - laser powers: List of laser powers for every frame.
     - traps: List of traps data for every frame. traps[trap number][frame number][position_x/position_y/power]
     
    Output format, tab separated:
     - time, laser power, trap_1_power, trap_1_x, trap_1, y,
         the same for traps 2-4, particle_1_x, particle_1_y,
         the same for all particles
    If a particle is missing from a frame, empty string ('') is placed
    instead of coordinates.
    """
    max_particles = int(round(trajectories.max()['particle']))
    with open(filename, 'w') as f:
        for i in range(len(frames)):
            tmp = ''
            tmp += str(times[i]) + '\t'
            tmp += str(laser_powers[i]) + '\t'
            for j in range(4): # for j in traps
                for k in range(3): # for k in x/y/power of a trap
                    tmp += str(traps[j][i][k]) + '\t'
            for j in range(max_particles+1):
                tmp_particle = trajectories.loc[
                    trajectories['particle'] == j].loc[
                        trajectories['frame'] == i]
                # find the particle j on frame i
                if tmp_particle.empty:
                    tmp += '\t\t'
                    # if no such particle exists, write two tabs
                else:
                    tmp += str(tmp_particle.iloc[0]['x']) + '\t'
                    tmp += str(tmp_particle.iloc[0]['y']) + '\t'
                    # else write the particles position
            tmp += '\n'
            f.write(tmp)

def save_tracked_data(filename, Nframes, trajectories, times, laser_powers, traps):
    """
    Converts all data to a (.dat) file.
    Important NOTE:
    The format will change in the future. Particle data and trap data will go into separate files,
    and additional trap metadata will be saved.
    
    Inputs:
     - filename: Output filename.
     - Nframes: Number of frames
     - trajectories: trajectories[x, y, other data][particle][frame]
     - times: List of frame times.
     - laser powers: List of laser powers for every frame.
     - traps: List of traps data for every frame. traps[trap number][frame number][position_x/position_y/power]
     
    Output format, tab separated:
     - time, laser power, trap_1_power, trap_1_x, trap_1, y,
         the same for traps 2-4, particle_1_x, particle_1_y,
         the same for all particles
    If a particle is missing from a frame, empty string ('') is placed
    instead of coordinates.
    """
    max_particles = len(trajectories[0])
    with open(filename, 'w') as f:
        for i in range(Nframes):
            tmp = ''
            tmp += str(times[i]) + '\t'
            tmp += str(laser_powers[i]) + '\t'
            for j in range(4): # for j in traps
                for k in range(3): # for k in x/y/power of a trap
                    tmp += str(traps[j][i][k]) + '\t'
            for j in range(max_particles):
                #print(1, j, i, '|', len(trajectories), len(trajectories[0]), len(trajectories[0][0]))
                tmp += str(trajectories[0][j][i]) + '\t'
                tmp += str(trajectories[1][j][i]) + '\t'
            tmp += '\n'
            f.write(tmp)

def flood_fill(frame, flood_frame, start_x, start_y, particle_number, treshold=100, invert=False, return_area=False):
    """
    An implementation of a common flood fill algorithm.
    Pixels with brightness above treshold are considered to be inside (below treshold if invert is True).
    flood_frame and particle_number are used to remember where particles were.
    Start position is start_x, start_y, where it is assumed to be above treshold and not checked.

    It also calculates on the fly the center of the pixel.

    Args:
     - frame: image array
     - flood_frame: Same shape as frame. 0 where there were no particles found yet.
     - prev_x, prev_y: Previous positions of particle.
     - particle_number: Used to mark this particle in flood_frame
     - treshold, invert, min_size, max_size, max_distance: See simple_tracking.
     - return_area: 

    Returns:
     - flood_frame: Updated flood_frame.
     - particle: Array with data about particle. Position x, y, size in pixels, average brightness and normalization weight.
       the last element is optional list of coordinates where particle was found
    """
    q = queue.Queue()
    visited = set()
    q.put((start_x, start_y))
    if return_area:
        particle = [0, 0, 0, 0, 0, []]
    else:
        particle = [0, 0, 0, 0, 0]
    while not q.empty():
        cx, cy = q.get()
        if (cx, cy) in visited:
            continue
        if (cx >= len(frame) or cy >= len(frame[0])
            or cx < 0 or cy < 0):
            continue
        visited.add((cx, cy))
        if ((not invert and frame[cx][cy] > treshold) or
            (invert and frame[cx][cy] < treshold)):
            flood_frame[cx][cy] = particle_number
            particle[0] += cx*(frame[cx][cy] - treshold)
            particle[1] += cy*(frame[cx][cy] - treshold)
            particle[2] += 1
            particle[3] += frame[cx][cy]
            particle[4] += frame[cx][cy] - treshold
            if return_area:
                particle[5].append([cx, cy])
            q.put((cx+1, cy))
            q.put((cx-1, cy))
            q.put((cx, cy+1))
            q.put((cx, cy-1))
    particle[0] /= particle[4]
    particle[1] /= particle[4]
    particle[3] /= particle[2]
    return flood_frame, particle
            

def find_particles_first_frame(frame, treshold=100, invert=False, min_size=16, max_size=2500, return_area=False):
    """
    Finds particles on the first frame.

    Every pixel with brightness larger than treshold starts a flood fill around its position.
    Particles that are too small or too big (min_size and max_size are in total number of pixels)
    are filtered out.

    Args:
     - frame: image array
     - treshold, invert, min_size, max_size: See simple_tracking.

    Returns:
     - flood_frame: Array with the same shape as frame. With 0 where there were no particles found yet and
                    markers of particles where particles were found.
     - particle: Array with data about particle. Position x, y, size in pixels, average brightness and normalization weight.    
    """
    flood_frame = np.zeros_like(frame)
    particle_number = 1
    particles = []
    for cx in range(len(frame)):
        for cy in range(len(frame[0])):
            if (flood_frame[cx][cy] == 0 and
                ((not invert and frame[cx][cy] > treshold) or
                 (invert and frame[cx][cy] < treshold))):
                #print('start', cx, cy, particle_number)
                flood_frame, particle = flood_fill(frame, flood_frame,
                                         cx, cy, particle_number, treshold=treshold,
                                         invert=invert, return_area=return_area)
                if min_size < particle[2] < max_size:
                    particles.append(particle[:])
                else:
                    pass
                    #print("filtering", particle)
                particle_number += 1
    return flood_frame, particles

def spiral(R):
    """
    Generates a square spiral walk around (0, 0).
    Adapted from https://stackoverflow.com/a/398302/
    """
    x = y = 0
    dx = 0
    dy = -1
    for i in range((2*R)**2):
        if (-R < x <= R) and (-R < y <= R):
            yield (x, y)
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy

def find_particle_around_position(frame, flood_frame, prev_x, prev_y, particle_number,
                                  treshold=100, invert=False, min_size=16, max_size=2500, max_distance=50,
                                  return_area=False):
    """
    Finds a particle around prev_x, prev_y up to max_distance away (square with a 2*max_distance side).

    Args:
     - frame: image array
     - flood_frame: Same shape as frame. 0 where there were no particles found yet.
     - prev_x, prev_y: Previous positions of particle.
     - particle_number: Used to mark this particle in flood_frame
     - treshold, invert, min_size, max_size, max_distance: See simple_tracking.

    Returns:
     - flood_frame: Updated flood_frame.
     - particle: Array with data about particle. Position x, y, size in pixels, average brightness and normalization weight.
    """
    for dx, dy in spiral(max_distance):
        cx = prev_x + dx
        cy = prev_y + dy
        if (cx < len(flood_frame) and cy < len(flood_frame[cx]) and
            cx >= 0 and cy >= 0 and
            flood_frame[cx][cy] == 0 and
            frame[cx][cy] > treshold):
            flood_frame, particle = flood_fill(frame, flood_frame,
                                         cx, cy, particle_number, treshold=treshold,
                                         invert=invert, return_area=return_area)
            if min_size < particle[2] < max_size:
                return flood_frame, particle[:]
    return flood_frame, [0, 0, 0, 0, 0, []]
    
def simple_tracking(frames, treshold=100, invert=False, min_size=16, max_size=2500, max_distance=50, return_area=False):
    """
    Tracks particles on all frames using a simple flood-fill of pixels above treshold.
    Args:
     - frames: an array of frames. Each frame must be like 2D array with single numbers as values.
               works with pims, but it is not needed.
     - treshold: Pixels above treshold are considered particle.
     - invert: If True, it looks for dark patches instead of bright.
     - min_size, max_size: If particle is below/above this pixel count, it is discarded.
     - max_distance: The distance in pixels, when we stop looking for a particle on the nex frame
                     (distance from the previous frame position). Distance is calculated as the
                     supremum distance (max of distances in x and y directions).

    Returns:
     - positions: Array. positions[frame number][particle number][x, y, other particle data]

    Note:
     - The new particle is searched for from the previous particles position in a spiral. That may
       introduce some bias when multiple particles are near eachother and jumps from one to the other.

    """
    positions = [] # positions[frame][particle][x, y, ...]
    flood_frame, particles = find_particles_first_frame(frames[0],
        treshold=treshold, invert=invert, min_size=min_size, max_size=max_size,
        return_area=return_area)
    positions.append(particles[:])
    for i in range(1, len(frames)):
        flood_frame = np.zeros_like(frames[i])
        cparticles = []
        for particle_number in range(len(positions[0])):
            flood_frame, particle = find_particle_around_position(frames[i], flood_frame,
                int(round(positions[-1][particle_number][0])),
                int(round(positions[-1][particle_number][1])),
                particle_number + 1, treshold=treshold, invert=invert, # particle_number + 1 to avoid using 0.
                min_size=min_size, max_size=max_size,
                max_distance=max_distance, return_area=return_area)
            cparticles.append(particle)
        positions.append(cparticles)
    return positions


if 0:
    import matplotlib.pyplot as plt
    frames = pims.open("Q:\\Dropbox\\Work\\PkP Pinceta\\19.02 - Trajectory Detection\\passiveInTrapP1.twv")
    frames = pims.open("Q:\\Dropbox\\Work\\PkP Pinceta\\Forked Repo\\tweezer\\tweezer\\generated_test\\*.png", as_grey=True) 

    flood_frame, particles = find_particles_first_frame(frames[0], 0.9, invert=False)
    #print(particles)
    #plt.imshow(frames[0])
    #plt.show()
    plt.imshow(flood_frame)
    particles = np.array(particles)
    plt.plot(particles.T[1], particles.T[0], 'rx')
    plt.show()

    positions = simple_tracking(frames, 0.9)
    positions = np.array(positions).T
    #positions[column x, y, size, avg brightness][particle][frame]

def simulation_example(show=True):
    import os
    import generate_gaussian_particles
    from calibration_generate_data import generate as generate_particle_in_trap
    from calibration_generate_data import generate_time
    from PIL import Image
    import matplotlib.pyplot as plt
    
    if not os.path.isdir("generated_test1"):
        os.mkdir("generated_test1")
    Nframes = 100
    positions1 = np.array(generate_particle_in_trap((1, 2), phi=0.1, center=(25, 25), number_of_points=Nframes))
    positions2 = np.array(generate_particle_in_trap((1, 2), phi=1.1, center=(25, 75), number_of_points=Nframes))

    if 0: # generate images. Takes some time - do it once and then skip it.
        for i in range(len(positions1)):
            img = generate_gaussian_particles.make_frame((100, 50),
                [[*positions1[i], 220, 5],
                 [*positions2[i], 220, 5]], noise_lvl=5)
            # 220 in brightness of particle (max=255)
            # 5 is particle radius.
            # noise_lvl=5 is additional noise on image (abs of gaussian with STD noise_lvl)
            img.save('generated_test1\\example{:}.png'.format(i))

    frames = [np.array(Image.open("generated_test1\\example{:}.png".format(i))) for i in range(len(positions1))]

    # Optional open with pims.
    # Treshold needs to change units.
    #frames = pims.open("generated_test\\*.png", as_grey=True) 

    flood_frame, particles = find_particles_first_frame(
        frames[0], 180, invert=False)

    # Display found particles - in image or flood fill background.
    if 1:
        plt.imshow(frames[0])
    else:
        plt.imshow(flood_frame)
    particles = np.array(particles)
    plt.plot(particles.T[1], particles.T[0], 'rx', label='FF')
    if have_trackpy:
        features = tp.locate(frames[0], 11, minmass=1000)
        plt.scatter(features.x, features.y, s=80, facecolors='none', edgecolors='b', label='Trackpy')
    plt.scatter([positions1[0][1], positions2[0][1]], [positions1[0][0], positions2[0][0]],
                s=20, facecolors='c', edgecolors='c', label='Original')
    plt.legend()
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()

    # track all particles.
    positions_st = simple_tracking(frames, 180)
    positions_st = np.array(positions_st).T #positions_st[x, y, other data][particle][frame]
    plt.plot(positions1.T[1], label='original1')
    plt.plot(positions2.T[1], label='original2')
    plt.plot(positions_st[1][0], label='FF1')
    plt.plot(positions_st[1][1], label='FF2')

    if have_trackpy:
        features = tp.batch(frames, 15, minmass=1000, invert=False)
        positions_tp = tp.link_df(features, 15, memory=10)

        particles = list(set(positions_tp.particle))
        for i in particles:
            plt.plot(positions_tp[positions_tp.particle==i].frame, positions_tp[positions_tp.particle==i].x, label='TrackPy{:}'.format(i))

    plt.legend()    
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()

    # plot differences to the original trajectory. The order of particles may need to be adjusted.
    plt.plot(positions_st[1][0] - positions1.T[1], label='FF1')
    plt.plot(positions_st[1][1] - positions2.T[1], label='FF2')

    if have_trackpy:
        plt.plot(positions_tp[positions_tp.particle==0].frame, positions_tp[positions_tp.particle==0].x - positions1.T[1], label='TrackPy1')
        plt.plot(positions_tp[positions_tp.particle==1].frame, positions_tp[positions_tp.particle==1].x - positions2.T[1], label='TrackPy2')
    
    plt.legend()    
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()

    # generate other data for saving into file.
    stationary_traps = [[25, 25, 1], [25, 75, 1], [0, 0, -1], [0, 0, -1]]
    traps = [[stationary_traps[i][:] for j in range(Nframes)]for i in range(4)]
    laser_powers = [1 for i in range(Nframes)]
    

    save_tracked_data("generated_test1\\example_FF.dat", Nframes, positions_st, generate_time(Nframes, 10**-3), laser_powers, traps)
    save_tracked_data_pandas("generated_test1\\example_TP.dat", frames, positions_tp, generate_time(Nframes, 10**-3), laser_powers, traps)

class Test(unittest.TestCase):
    """
    Also tests TWV_Reader module.
    """

    def setUp(self):
        pass

    def test_everything(self):
        """
        End to end test.
        Test the whole tracking pipeline from input file to particle and trap
        positions in the output file.
        """
        filename = "../examples/data/test_example.twv"
        frames = pims.open(filename)
        times, laser_powers, traps = frames.get_all_tweezer_positions()
        features = tp.batch(frames, 25, minmass=1000, invert=False)
        tracks = tp.link_df(features, 15, memory=10)
        save_tracked_data_pandas(filename[:-4] + '_out.dat', frames, tracks, times, laser_powers, traps)
        with open(filename[:-4] + '_out.dat', 'r') as calculated_file:
            with open(filename[:-4] + '_expected.dat', 'r') as expected_file:
                for calculated, expected in zip(calculated_file, expected_file):
                    self.assertEqual(calculated, expected)

def example_with_trackpy_and_twv(filename):
    """
    Example usecase from input file to particle and trap positions in .dat file.
    """
    frames = pims.open(filename)
    # Open file with pims. Works with many file extensions.
    # This example assumes .twv file.

    # metadata = frames.get_all_metadata()
    # Optional access to additional metadata.

    times, laser_powers, traps = frames.get_all_tweezer_positions()
    # Obtain frame times, laser power at each frame time and
    # traps powers and positions at each frame.

    features = tp.batch(frames, 25, minmass=1000, invert=False)
    # Obtain features (particle positions) using trackpy's batch function.
    # It is verbose.
    # The 25 in arguments is diameter. It be odd number.
    # It is recommended to obtain parameters using GUI.
    
    tracks = tp.link_df(features, 15, memory=10)
    # Joins particles positions to tracks (connects them in time).
    # See trackpy documentation for parameters.

    save_tracked_data_pandas(filename[:-4] + '_out.dat', frames, tracks, times, laser_powers, traps)
    # Save data in a format readable by other scripts.

if __name__ == "__main__" and 1:
    example_with_trackpy_and_twv("../examples/data/test_example.twv")
    simulation_example(True)
					
if __name__ == "__main__" and 0:
    unittest.main()
