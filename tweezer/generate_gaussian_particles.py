"""Module for drawing gaussian particles on images.

make_frame is used in other modules, other functions are not used."""

from PIL import Image, ImageDraw
from scipy.stats import norm
import numpy as np

RENORM = np.sqrt(2*np.pi)

def make_frame(size, particles, noise_lvl=5, invert=False):
    """
    Generates a frame with particles.
    Args:
     - size: touple of width, height dimensions
     - particles: y, x positions, amplitude (0-255), diameter (in px) for eaxh particle.
    Returns:
     - PIL(LOW) object of the image.

    TODO speedup improvement:
    Generate array and convert to int and then image at the end. Also improves accuracy with conversion to int.
    """
    
    img = Image.new('P', size)
    for particle in particles:
        for j in range(img.height):
            for k in range(img.width):
                dist = np.sqrt( (j - particle[0])**2 + (k - particle[1])**2)
                cval = img.getpixel((k, j))
                img.putpixel((k, j), int(cval + RENORM*particle[2]*norm.pdf(dist/particle[3])))
    if noise_lvl > 0:                
        for j in range(img.height):
            for k in range(img.width):
                cval = img.getpixel((k, j))
                img.putpixel((k, j), int(cval + abs(np.random.normal()*noise_lvl)))
    return img

def make_frame_array(size, particles, noise_lvl=5, invert=False):
    """
    The same as make_frame, but only generates an array, not image object. Probably faster.
    Not used.
    """
    
    img = [[0 for i in range(size[1])] for j in range(size[0])]
    for particle in particles:
        for j in range(size[1]):
            for k in range(size[0]):
                dist = np.sqrt( (j - particle[0])**2 + (k - particle[1])**2)
                img[k][j] += RENORM*particle[2]*norm.pdf(dist/particle[3])
    if noise_lvl > 0:                
        for j in range(size[1]):
            for k in range(size[0]):
                img[k][j] +=abs(np.random.normal()*noise_lvl)
    for j in range(size[1]):
        for k in range(size[0]):
            img[k][j] = min(255, int(img[k][j]))
    return img

def example():
    import matplotlib.pyplot as plt
    img = make_frame((100, 10), [[5, 5, 255, 5], [5, 50, 255, 20]])
    plt.imshow(img)
    plt.show()
    img.save('example.png')

def two_particles_approaching():
    NMax = 33
    for i in range(NMax):
        img = make_frame((100, 50), [[25, 10+i, 255, 5], [25, 90-i, 255, 5]])
        img.save('generated_test/example{:}.png'.format(i))
    for i in range(NMax):
        img = make_frame((100, 50), [[25, 10+NMax-i, 255, 5], [25, 90-NMax+i, 255, 5]])
        img.save('generated_test/example{:}.png'.format(NMax + i))

def two_particles_yield(dimensions, start1, start2, d1, d2, b1, b2, dr1, dr2, noise_lvl, min_dist):
    i = 0
    j = 0
    while True:
        img = make_frame(dimensions,
                         [[start1[0] + dr1[0]*i, start1[1] + dr1[1]*i, b1, d1],
                          [start2[0] + dr2[0]*i, start2[1] + dr2[1]*i, b2, d2]], noise_lvl=noise_lvl)
        img.save('generated_test/example{:}.png'.format(j))
        yield img
        if np.sqrt( (start1[0] + dr1[0]*i - start2[0] - dr2[0]*i)**2 +
                    (start1[1] + dr1[1]*i - start2[1] - dr2[1]*i)**2 ) < min_dist:
            break
        i += 1
        j += 1
        print(i)
    i -= 1
    while True:
        img = make_frame(dimensions,
                         [[start1[0] + dr1[0]*i, start1[1] + dr1[1]*i, b1, d1],
                          [start2[0] + dr2[0]*i, start2[1] + dr2[1]*i, b2, d2]], noise_lvl=noise_lvl)
        img.save('generated_test/example{:}.png'.format(j))
        yield img
        if i == 0:
            break
        i -= 1
        j += 1

if __name__ == '__main__':
    two_particles_approaching()
