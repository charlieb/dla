from math import sqrt, sin, cos, radians, pi, isnan
from random import choice, random, randrange
import numpy as np
from numba import jit, int64, float64, boolean
import svgwrite
import time

@jit(boolean(int64, float64[:,:], float64[:]))
def check_collision(nparts, parts, new_part):
    for i in range(nparts):
        d = new_part - parts[i]
        d2 = d*d
        r2 = (new_part[2]+parts[i][2])**2
        if d2[0] + d2[1] <= r2:
            return True
    return False

@jit((int64, float64[:,:], float64, float64))
def add_particle(nparts, parts, particle_radius, start_radius):
    angle = 2*pi * random()
    pos = np.array([start_radius * cos(angle),
                    start_radius * sin(angle),
                    particle_radius])
    d = -pos
    d /= sqrt(d[0]**2 + d[1]**2)
    d[2] = 0.
    for _ in range(int(start_radius)):
        if check_collision(nparts, parts, pos):
            break
        pos += d
    parts[nparts] = pos

@jit(int64(int64, float64[:,:]))
def generate_dla(array_len, parts):
    nparts = 1
    particle_radius = .5
    parts[0][0] = parts[0][1] = 0.
    parts[0][2] = particle_radius
    radius = particle_radius*2

    for _ in range(array_len-1):
        particle_radius /= 0.9995
        add_particle(nparts, parts, particle_radius, radius)
        r = sqrt(parts[nparts][0]*parts[nparts][0] +
                 parts[nparts][1]*parts[nparts][1])
        if r + 2*particle_radius > radius:
            radius = r+ 2*particle_radius
        nparts += 1
    return nparts

def draw(nparts, parts, frame=0):
    dwg = svgwrite.Drawing('test%05i.svg'%frame, profile='tiny')
    minx = miny =  9999999
    maxx = maxy = -9999999
    line_width = 1
    for i in range(nparts):
        minx = parts[i][0] - parts[i][2] if parts[i][0] - parts[i][2] < minx else minx
        maxx = parts[i][0] + parts[i][2] if parts[i][0] + parts[i][2] > maxx else maxx
        miny = parts[i][1] - parts[i][2] if parts[i][1] - parts[i][2] < miny else miny
        maxy = parts[i][1] + parts[i][2] if parts[i][1] + parts[i][2] > maxy else maxy

        c = svgwrite.shapes.Circle((parts[i][0], parts[i][1]), parts[i][2],
                                    fill='none', stroke='black', stroke_width=1)
        dwg.add(c)
    dwg.viewbox(minx=minx-line_width, miny=miny-line_width, 
                width=maxx-minx+2*line_width, height=maxy-miny+2*line_width)
    dwg.save()

def main():
    array_len = 5000
    parts = np.zeros((array_len, 3), dtype=np.float64)
    nparts = generate_dla(array_len, parts)
    draw(nparts, parts)

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()

    print("Elapsed:", t1-t0)

