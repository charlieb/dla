from math import sqrt, sin, cos, radians, pi, isnan
from random import choice, random, randrange
import numpy as np
from numba import jit, int64, float64, boolean
import svgwrite
import time

@jit(int64(int64, float64[:,:], float64[:]))
def check_collision(nparts, parts, new_part):
    hit = -1
    err = -1
    for i in range(nparts):
        d = new_part - parts[i]
        d2 = d*d
        r2 = (new_part[2]+parts[i][2])**2
        if d2[0] + d2[1] < r2:
            # find the one that we're furthest into
            if err < r2 - (d2[0] + d2[1]):
                err = r2 - (d2[0] + d2[1])
                hit = i
    return hit

@jit((int64, float64[:,:], float64, float64))
def add_particle(nparts, parts, particle_radius, start_radius):
    angle = 2*pi * random()
    pos = np.array([start_radius * cos(angle),
                    start_radius * sin(angle),
                    particle_radius, 0.])
    d = -pos
    d /= sqrt(d[0]**2 + d[1]**2) / particle_radius
    d[2] = 0.
    hit = False
    want_hit = True
    err = 1.
    hit_particle = -1
    # Find a collision particle then walk back and forth along our path
    # until the hit position is within 0.1
    while abs(err) > 0.1:
        while hit != want_hit:
            pos += d
            if hit_particle == -1: # we're looking for a particle
                hit_particle = check_collision(nparts, parts, pos)
                hit = hit_particle != -1
            else: # we've found one now we just want to see how far away it is
                v2 = (pos - parts[hit_particle])**2
                r2 = (pos[2]+parts[hit_particle][2])**2
                hit = v2[0] + v2[1] < r2
        want_hit = not want_hit
        d *= -0.5
        err = sqrt((parts[hit_particle][0] - pos[0])**2 +
                (parts[hit_particle][1] - pos[1])**2) - \
                (parts[hit_particle][2] + particle_radius)

    parts[nparts] = pos
    parts[nparts][2] = particle_radius
    parts[nparts][3] = hit_particle # NB the last particle we detect a hit on is the link

@jit(int64(int64, float64[:,:], float64, float64))
def generate_dla(array_len, parts, particle_radius, radius_multiplier):
    nparts = 1
    parts[0][0] = parts[0][1] = 0.
    parts[0][2] = particle_radius
    radius = 4*particle_radius # two particle widths away

    for _ in range(array_len-1):
        particle_radius *= radius_multiplier
        add_particle(nparts, parts, particle_radius, radius)
        r = sqrt(parts[nparts][0]*parts[nparts][0] +
                 parts[nparts][1]*parts[nparts][1])
        if r + 4*particle_radius > radius:
            radius = r + 4*particle_radius
        nparts += 1
        print(nparts)
    return nparts

def draw(nparts, parts, frame=0, circles=True, links=True):
    dwg = svgwrite.Drawing('test%05i.svg'%frame, profile='tiny')
    minx = miny =  9999999
    maxx = maxy = -9999999
    line_width = 0.1
    for i in range(nparts):
        minx = parts[i][0] - parts[i][2] if parts[i][0] - parts[i][2] < minx else minx
        maxx = parts[i][0] + parts[i][2] if parts[i][0] + parts[i][2] > maxx else maxx
        miny = parts[i][1] - parts[i][2] if parts[i][1] - parts[i][2] < miny else miny
        maxy = parts[i][1] + parts[i][2] if parts[i][1] + parts[i][2] > maxy else maxy

        if circles:
            c = svgwrite.shapes.Circle((parts[i][0], parts[i][1]), parts[i][2],
                                        fill='none', stroke='black', stroke_width=line_width)
            dwg.add(c)

        if links:
            link = svgwrite.shapes.Line((parts[i][0], parts[i][1]),
                                        (parts[int(parts[i][3])][0], parts[int(parts[i][3])][1]),
                                        stroke='red', stroke_width=line_width)
            dwg.add(link)

    dwg.viewbox(minx=minx-line_width, miny=miny-line_width, 
                width=maxx-minx+2*line_width, height=maxy-miny+2*line_width)
    dwg.save()

def generate_particles(nparticles, particle_radius, radius_multiplier):
    parts = np.zeros((nparticles, 4), dtype=np.float64)
    nparts = generate_dla(nparticles, parts, particle_radius, radius_multiplier)
    return parts

def unpack(particle):
    return {
            'x':particle[0],
            'y':particle[1],
            'r':particle[2],
            'link':particle[3],
            }

def main():
    nparts = 5000
    parts = generate_particles(nparts, 2., 0.9995)
    draw(nparts, parts, circles=False, links=True)

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()

    print("Elapsed:", t1-t0)

