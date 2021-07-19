import math
import random
from multiprocessing import Pool
import pygame
import numpy as np
from numba import jit


max_iteration = 1000
size = (1080, 720)

@jit(nopython=True, fastmath=True)
def scale(value, oldmin, oldmax, newmin, newmax):
    normalized = (value - oldmin) / (oldmax - oldmin)
    return normalized * (newmax - newmin) + newmin 

random.seed(1)
random_values = [math.floor(scale(random.random(), 0, 1, 0, 360)) for _ in range(500)]


def generate_colour(iteration, max_iteration):
    col = pygame.Color(0, 0, 0)
    h = scale(iteration, 0, max_iteration, 45, 360)
    col.hsla = (h, 100, 50, 1)
    return (col.r, col.g, col.b)

@jit(nopython=False, fastmath=True)
def generate_random_colour(iteration, max_iteration, random_values):
    col = pygame.Color(0, 0, 0)
    
    val = scale(iteration, 0, max_iteration, 0, len(random_values))
    h = random_values[math.floor(val)-1]

    val2 =  math.sin(scale(val % 1, 0, 1, -2 * math.pi, 2 * math.pi))
    l = scale(val2, -1,1 , 0, 100)

    col.hsla = (h, 100, l, 1)
    return (col.r, col.g, col.b)

    
@jit(nopython=True, fastmath=True)
def find_iteration(px, py, max_iteration, size):
    def scale(value, oldmin, oldmax, newmin, newmax):
        normalized = (value - oldmin) / (oldmax - oldmin)
        return normalized * (newmax - newmin) + newmin    

    x0 = scale(px, 0, size[0], -2.5, 1)
    y0 = scale(py, 0, size[1], -1, 1)
    x = 0
    y = 0
    iteration = 0
    x2 = 0
    y2 = 0
    while (x2 + y2 <= 4 and iteration < max_iteration):
        y = 2 * x * y + y0
        x = x2 - y2 + x0
        x2 = x * x
        y2 = y * y
        iteration += 1
        
    if iteration < max_iteration:
        log_zn = math.log(x * x + y * y)  / 2
        nu = math.log(log_zn / math.log(2)) / math.log(2)
        iteration += 1 - nu
    return iteration


@jit(nopython=True, fastmath=True)
def all_pixels(size, shift, zoom):
    for x in range(size[0]):
        for y in range(size[1]):
            yield shift[0] + x / zoom, shift[1] + y / zoom


def find_colour(pos):
    iteration = find_iteration(*pos, max_iteration, size)
    return generate_random_colour(iteration, max_iteration, random_values)


@jit(nopython=False, fastmath=True)
def main():
    display = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    p = Pool(23)
    shift = [0, 0]
    zoom = 1
    
    dx, dy = 0, 0
    render =  True
    
    while 1: 
        if render:
            colours = p.map(find_colour, all_pixels(size, shift, zoom))

            colours_2d = np.array(colours).reshape(size[0], size[1], 3)
            pygame.surfarray.blit_array(display, colours_2d)
            render = False
            
            clock.tick()
            print(clock.get_time())
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                p.close()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dx = event.pos[0]
                    dy = event.pos[1]
                    zoom *= 2
                    shift[0] += dx / zoom
                    shift[1] += dy / zoom
                    render = True

        pygame.display.update()
        clock.tick(30)

if __name__ == "__main__":
    main()