import dearpygui.dearpygui as dpg
from numba import cuda
import numpy as np, random

VP_WIDTH = 640
VP_HEIGHT = 640

@cuda.jit
def update_board(an_array):
  x, y = cuda.grid(2)

  active_neighbors = an_array[x-1, y-1] + an_array[x-1, y] + an_array[x-1, y+1] + an_array[x, y-1] + an_array[x, y+1] + an_array[x+1, y-1] + an_array[x+1, y] + an_array[x+1, y+1]
  
  if an_array[x, y] == 1:
    if active_neighbors < 2 or active_neighbors > 3:
      an_array[x , y] = 0
  else:
    if active_neighbors == 3:
      an_array[x, y] = 1

def create_board():
    pass
