import dearpygui.dearpygui as dpg
from numba import cuda
import numpy as np, random, time, math
import argparse

@cuda.jit(device=True)
def update_value(x, y, an_array, an_array_dim_x, an_array_dim_y):
  cell = an_array[x, y]
  active_neighbors = an_array[x-1, y-1] + \
                     an_array[x-1, y] + \
                     an_array[x-1, (y+1) % an_array_dim_y] + \
                     an_array[x, y-1] + \
                     an_array[x, (y+1) % an_array_dim_y] + \
                     an_array[(x+1) % an_array_dim_x, y-1] + \
                     an_array[(x+1) % an_array_dim_x, y] + \
                     an_array[(x+1) % an_array_dim_x, (y+1) % an_array_dim_y]
  
  if cell == 1:
    if active_neighbors < 2 or active_neighbors > 3:
      cell = 0
  else:
    if active_neighbors == 3:
      cell = 1

  return cell

@cuda.jit
def update_value_kernel(an_array, an_array_new, an_array_dim_x, an_array_dim_y):
  startX, startY = cuda.grid(2)
  gridX = cuda.gridDim.x * cuda.blockDim.x
  gridY = cuda.gridDim.y * cuda.blockDim.y

  for x in range(startX, gridX):
    for y in range(startY, gridY):
      if y >= an_array_dim_y:
        break
      an_array_new[x, y] = update_value(x, y, an_array, an_array_dim_x, an_array_dim_y)
    if x >= an_array_dim_x:
      break

def conway(VP_HEIGHT, VP_WIDTH, ITERATIONS, an_array, an_array_dim):
  an_array_dim_x, an_array_dim_y = an_array_dim

  block_dim_x, block_dim_y = (32, 32)
  
  grid_dim_x = math.ceil(an_array_dim_x / block_dim_x)
  grid_dim_y = math.ceil(an_array_dim_y / block_dim_y)

  block_dim = (block_dim_x, block_dim_y)
  grid_dim = (grid_dim_x, grid_dim_y)

  time.sleep(0.5)
  dpg.delete_item("viewport_back", children_only=True)

  while dpg.is_dearpygui_running():
    for _ in range(ITERATIONS):
      d_an_array = cuda.to_device(an_array)
      d_an_array_new = cuda.to_device(an_array)
      update_value_kernel[block_dim, grid_dim](d_an_array, d_an_array_new, an_array_dim_x, an_array_dim_y)
      an_array = d_an_array_new.copy_to_host()
      for i in range(0, VP_WIDTH, 16):
        for j in range(0, VP_HEIGHT, 16):
          value = an_array[i//16, j//16]
          if value == 0:
            dpg.draw_rectangle((i, j), (i+16, j+16), fill = (255, 255, 255, 255), parent = "viewport_back")
          else:
            dpg.draw_rectangle((i, j), (i+16, j+16), fill = (0, 0, 0, 0), parent = "viewport_back")
      
      dpg.render_dearpygui_frame()
      time.sleep(0.5)
      dpg.delete_item("viewport_back", children_only=True)
    break

  dpg.destroy_context()

def create_board(VP_HEIGHT, VP_WIDTH, ITERATIONS):
  dpg.setup_dearpygui()
  dpg.add_viewport_drawlist(front=False, tag="viewport_back")
  dpg.show_viewport()

  an_array = np.asmatrix([[random.randint(0, 1) for _ in range(VP_HEIGHT//16)] for _ in range(VP_WIDTH//16)])
  an_array_dim = an_array.shape

  for i in range(0, VP_WIDTH, 16):
    for j in range(0, VP_HEIGHT, 16):
      value = an_array[i//16, j//16]
      if value == 0:
        dpg.draw_rectangle((i, j), (i+16, j+16), fill = (255, 255, 255, 255), parent = "viewport_back")
      else:
        dpg.draw_rectangle((i, j), (i+16, j+16), fill = (0, 0, 0, 0), parent = "viewport_back")
  
  dpg.render_dearpygui_frame()
  conway(VP_HEIGHT, VP_WIDTH, ITERATIONS, an_array, an_array_dim)

parser = argparse.ArgumentParser(description = "Initialize board dimensions")
parser.add_argument('--height', type = int, help = "specify height of the board, must be a multiple of 16", required = True)
parser.add_argument('--width', type = int, help = "specify width of the board, must be a multiple of 16", required = True)
parser.add_argument('--iterations', type = int, help = "number of iteration; must be a positive integer", required = True)

args = vars(parser.parse_args())
VP_WIDTH = args["width"]
VP_HEIGHT = args["height"]
ITERATIONS = args["iterations"]

dpg.create_context()
dpg.create_viewport(title='Conway\'s Game of Life Simulation: GPU', width=VP_WIDTH, height=VP_HEIGHT, resizable = False)
create_board(VP_HEIGHT, VP_WIDTH, ITERATIONS)
