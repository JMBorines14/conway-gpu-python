import dearpygui.dearpygui as dpg
from numba import cuda
import numpy as np, random, time
import argparse

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

def conway(VP_HEIGHT, VP_WIDTH, ITERATIONS, an_array):
  while dpg.is_dearpygui_running():
    for k in range(ITERATIONS):
      pass
      time.sleep(0.5)
      dpg.delete_item("viewport_back", children_only=True)

  dpg.destroy_context()

def create_board(VP_HEIGHT, VP_WIDTH, ITERATIONS):
  dpg.setup_dearpygui()
  dpg.add_viewport_drawlist(front=False, tag="viewport_back")
  dpg.show_viewport()

  an_array = np.asmatrix([[random.randint(0, 1) for i in range(VP_WIDTH//16)] for i in range(VP_HEIGHT//16)])

  for i in range(0, VP_HEIGHT - 16, 16):
    for j in range(0, VP_WIDTH - 16, 16):
      value = an_array[i//16, j//16]
      if value == 1:
        dpg.draw_rectangle((i, j), (i+16, j+16), fill = (255, 255, 255, 255), parent = "viewport_back")
      else:
        dpg.draw_rectangle((i, j), (i+16, j+16), fill = (0, 0, 0, 0), parent = "viewport_back")
  
  dpg.render_dearpygui_frame()

parser = argparse.ArgumentParser(description = "Initialize board dimensions")
parser.add_argument('--height', type = int, help = "specify height of the board, must be a multiple of 16", required = True)
parser.add_argument('--width', type = int, help = "specify width of the board, must be a multiple of 16", required = True)
parser.add_argument('--iterations', type = int, help = "number of iteration; must be a positive integer", required = True)

args = vars(parser.parse_args())
VP_WIDTH = args["width"]
VP_HEIGHT = args["height"]
ITERATIONS = args["iterations"]

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=VP_WIDTH, height=VP_HEIGHT, resizable = False)
create_board(VP_HEIGHT, VP_WIDTH, ITERATIONS)