# Conway's Game of Life Python GPU Implementation

This is the project repository for the implementation of *Conway's Game of Life* using a parallel algorithm through Python 3. The project repository contains two Python files that focus on the GPU implementation of the Game of Life (`app.py`), and the naive CPU implementation (`app_cpu.py`). To run the Python scripts, type the command shown below:

```
python3 app.py --width 40 --height 40 --iterations 50
```

1. `--width` pertains to the width of the board. It can be any positive integer.
2. `--height` pertains to the height of the board. It can be any positive integer.
3. `--iterations` pertains to the number of iterations to be simulated
4. `--runtime` is an optional flag; if included in the command, the Python script will not deploy a GUI but rather would print the running time of the game based on the specified grid size and iterations

Based on the example command above, the script will generate a 40 by 40 board and will simulate the *Game of Life* for 50 iterations with randomly generated cells. For this mini-project, the game board is randomly created instead of getting user input. 

## Required Libraries

The following Python 3 libraries are required to be installed on your machine before running the scripts: `numba` and `dearpygui`. Both libraries can be installed using `pip` command:

```
pip install numba
pip install dearpygui
```