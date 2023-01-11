import dearpygui.dearpygui as dpg
from numba import cuda
import numpy as np, random, time
import argparse

def update_board(an_array):
  a_new_array = np.copy(an_array)

  for i in range(an_array.shape[0]):
    for j in range(an_array.shape[1]):

      active_neighbors = an_array[i-1, j-1] + an_array[i-1, j] + an_array[i-1, (j+1)%(an_array.shape[1])] + \
         an_array[i, j-1] + an_array[i, (j+1)%(an_array.shape[1])] + an_array[(i+1)%(an_array.shape[0]), j-1] + \
         an_array[(i+1)%(an_array.shape[0]), j] + an_array[(i+1)%(an_array.shape[0]), (j+1)%(an_array.shape[1])]

      if an_array[i, j] == 1:
        if active_neighbors < 2 or active_neighbors > 3:
          a_new_array[i , j] = 0
      else:
        if active_neighbors == 3:
          a_new_array[i, j] = 1
  
  return a_new_array

def conway(HEIGHT, WIDTH, ITERATIONS, an_array):
  if (not RUNTIME_FLAG):
    time.sleep(0.5)
    dpg.delete_item("viewport_back", children_only=True)

  for _ in range(ITERATIONS):
    an_array = update_board(an_array)

    if (not RUNTIME_FLAG):
      for i in range(WIDTH):
        for j in range(HEIGHT):
          value = an_array[i, j]
          if value == 0:
            dpg.draw_rectangle((i * PIXEL_DIM[0], j * PIXEL_DIM[1]), ((i * PIXEL_DIM[0]) + PIXEL_DIM[0], (j * PIXEL_DIM[1]) + PIXEL_DIM[1]), fill = (255, 255, 255, 255), parent = "viewport_back")
          else:
            dpg.draw_rectangle((i * PIXEL_DIM[0], j * PIXEL_DIM[1]), ((i * PIXEL_DIM[0]) + PIXEL_DIM[0], (j * PIXEL_DIM[1]) + PIXEL_DIM[1]), fill = (0, 0, 0, 0), parent = "viewport_back")

      dpg.render_dearpygui_frame()
      time.sleep(0.5)
      dpg.delete_item("viewport_back", children_only=True)

  if (RUNTIME_FLAG):
    END_TIME = time.time()
    print('Execution time: ', END_TIME - START_TIME)

def create_board(HEIGHT, WIDTH, ITERATIONS):
  an_array = np.asmatrix([[random.randint(0, 1) for _ in range(HEIGHT)] for _ in range(WIDTH)])

  if (not RUNTIME_FLAG):
    dpg.setup_dearpygui()
    dpg.add_viewport_drawlist(front=False, tag="viewport_back")
    dpg.show_viewport()

    for i in range(WIDTH):
      for j in range(HEIGHT):
        value = an_array[i, j]
        if value == 0:
          dpg.draw_rectangle((i * PIXEL_DIM[0], j * PIXEL_DIM[1]), ((i * PIXEL_DIM[0]) + PIXEL_DIM[0], (j * PIXEL_DIM[1]) + PIXEL_DIM[1]), fill = (255, 255, 255, 255), parent = "viewport_back")
        else:
          dpg.draw_rectangle((i * PIXEL_DIM[0], j * PIXEL_DIM[1]), ((i * PIXEL_DIM[0]) + PIXEL_DIM[0], (j * PIXEL_DIM[1]) + PIXEL_DIM[1]), fill = (0, 0, 0, 0), parent = "viewport_back")
    
    dpg.render_dearpygui_frame()
  conway(HEIGHT, WIDTH, ITERATIONS, an_array)

parser = argparse.ArgumentParser(description = "Initialize board dimensions")
parser.add_argument('--height', type = int, help = "specify height of the board, must be a multiple of 16", required = True)
parser.add_argument('--width', type = int, help = "specify width of the board, must be a multiple of 16", required = True)
parser.add_argument('--iterations', type = int, help = "number of iteration; must be a positive integer", required = True)
parser.add_argument('--runtime', help = "measure runtime instead of presenting simulation", action='store_true')

args = vars(parser.parse_args())
WIDTH = args["width"]
HEIGHT = args["height"]
ITERATIONS = args["iterations"]
RUNTIME_FLAG = args["runtime"]
PIXEL_DIM = (16, 16)

if (not RUNTIME_FLAG):
  dpg.create_context()
  dpg.create_viewport(title='Conway\'s Game of Life: CPU', width=WIDTH*PIXEL_DIM[0], height=HEIGHT*PIXEL_DIM[1]+50, resizable = False)

if (RUNTIME_FLAG):
  START_TIME = time.time()

create_board(HEIGHT, WIDTH, ITERATIONS)
