import dearpygui.dearpygui as dpg
from numba import cuda
import numpy as np, random, time, math
import argparse

@cuda.jit
def update_board(an_array, a_new_array):
  x, y = cuda.grid(2)

  active_neighbors = an_array[x-1, y-1] + an_array[x-1, y] + an_array[x-1, (y+1)%(an_array.shape[1])] + \
    an_array[x, y-1] + an_array[x, (y+1)%(an_array.shape[1])] + an_array[(x+1)%(an_array.shape[0]), y-1] + \
    an_array[(x+1)%(an_array.shape[0]), y] + an_array[(x+1)%(an_array.shape[0]), (y+1)%(an_array.shape[1])]
  
  if an_array[x, y] == 1:
    if active_neighbors < 2 or active_neighbors > 3:
      a_new_array[x , y] = 0
  else:
    if active_neighbors == 3:
      a_new_array[x, y] = 1

def conway(HEIGHT, WIDTH, ITERATIONS, an_array):
  time.sleep(0.5)
  dpg.delete_item("viewport_back", children_only=True)

  while dpg.is_dearpygui_running():
    for _ in range(ITERATIONS):
      gpu_old_array = cuda.to_device(an_array)
      gpu_new_array = cuda.to_device(np.copy(an_array))

      threadsperblock = (32, 32) #fully utilize 1024 threads per block
      blockspergrid_x = math.ceil(WIDTH / threadsperblock[0])
      blockspergrid_y = math.ceil(HEIGHT / threadsperblock[1])
      blockspergrid = (blockspergrid_x, blockspergrid_y)

      update_board[blockspergrid, threadsperblock](gpu_old_array, gpu_new_array)
      an_array = gpu_new_array.copy_to_host()

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

    break
  dpg.destroy_context()

def create_board(HEIGHT, WIDTH, ITERATIONS):
  dpg.setup_dearpygui()
  dpg.add_viewport_drawlist(front=False, tag="viewport_back")
  dpg.show_viewport()

  an_array = np.asmatrix([[random.randint(0, 1) for _ in range(HEIGHT)] for _ in range(WIDTH)])

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

dpg.create_context()
dpg.create_viewport(title='Conway\'s Game of Life: GPU Naive', width=WIDTH*PIXEL_DIM[0], height=HEIGHT*PIXEL_DIM[1]+50, resizable = False)
create_board(HEIGHT, WIDTH, ITERATIONS)
