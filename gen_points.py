from math import sqrt, sin, cos, radians, pi, isnan
from random import choice, random, randrange
import numpy as np
from numba import jit, int64, float64, boolean
import svgwrite as svg
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

@jit((int64, float64[:,:], float64, float64, float64, float64))
def add_particle(nparts, parts, particle_radius, start_radius, start_angle_range, start_angle_range_center):
    angle = start_angle_range_center + (start_angle_range / 2)  - start_angle_range * random()
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

@jit(int64(int64, float64[:,:], float64, float64, float64, float64))
def generate_dla(array_len, parts, particle_radius, radius_multiplier,
                start_angle_range, start_angle_range_center):
    nparts = 1
    parts[0][0] = parts[0][1] = 0.
    parts[0][2] = particle_radius
    radius = 4*particle_radius # two particle widths away

    for _ in range(array_len-1):
        particle_radius *= radius_multiplier
        #start_angle_range_center += pi / array_len
        add_particle(nparts, parts, particle_radius, radius,
                    start_angle_range, start_angle_range_center)
        r = sqrt(parts[nparts][0]*parts[nparts][0] +
                 parts[nparts][1]*parts[nparts][1])
        if r + 4*particle_radius > radius:
            radius = r + 4*particle_radius
        nparts += 1
        print(nparts)
    return nparts

def draw(parts, drawing, circles=True, links=True, circle_color='black', link_color='red', line_width=1., prune=[]):
    minx = miny =  9999999
    maxx = maxy = -9999999
    for i, p in enumerate(parts):
        minx = p[0] - p[2] if p[0] - p[2] < minx else minx
        maxx = p[0] + p[2] if p[0] + p[2] > maxx else maxx
        miny = p[1] - p[2] if p[1] - p[2] < miny else miny
        maxy = p[1] + p[2] if p[1] + p[2] > maxy else maxy

        if circles and i not in prune:
            c = svg.shapes.Circle((p[0], p[1]), p[2],
                                        fill='none', 
                                        stroke=circle_color,
                                        stroke_width=line_width)
            drawing.add(c)

        if links:
            link = svg.shapes.Line((p[0], p[1]),
                                        (parts[int(p[3])][0], parts[int(p[3])][1]),
                                        stroke=link_color,
                                        stroke_width=line_width)
            drawing.add(link)

    drawing.viewbox(minx=minx-line_width, miny=miny-line_width, 
                width=maxx-minx+2*line_width, height=maxy-miny+2*line_width)

def generate_particles(nparticles, particle_radius, radius_multiplier, start_angle_range=2*pi, start_angle_range_center=0.):
    parts = np.zeros((nparticles, 4), dtype=np.float64)
    nparts = generate_dla(nparticles, parts, particle_radius, radius_multiplier,
                        start_angle_range, start_angle_range_center)
    return parts

def find_leaves(particles):
    leaves = list(range(len(particles)))
    for p in particles:
        try:
            leaves.remove(p[3])
        except ValueError:
            pass
    return leaves

def unpack(particle):
    return {
            'x':particle[0],
            'y':particle[1],
            'r':particle[2],
            'link':particle[3],
            }

def main():
    nparts = 1000
    #parts = generate_particles(nparts, 2., 1.-1/(nparts*0.7), start_angle_range=pi/4)
    parts = generate_particles(nparts, 1., 0.999, start_angle_range_center=3*pi/2, start_angle_range=pi/3)

    dwg = svg.Drawing('test.svg')
    draw(parts, dwg, circles=True, links=True, line_width=0.1)
    dwg.save()

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()

    print("Elapsed:", t1-t0)

