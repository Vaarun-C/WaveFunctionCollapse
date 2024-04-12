import cv2 as cv
import math
import json
import random
import numpy as np

canvas_height, canvas_width = 10,10
tile_size = 64

with open('rules.json') as f:
    rules = json.load(f)

entropy_grid = [[0.0 for _ in range(canvas_width)] for _ in range(canvas_height)]
grid = [[[] for _ in range(canvas_width)] for _ in range(canvas_height)]

def calculate_entropy(num_states: int) -> float:
    p = 1 / num_states
    entropy = -sum(p * math.log2(p) for _ in range(num_states))
    return entropy

def get_min_entropy_cell() -> tuple[int,int]:
    min_ent_cell = (0,0)
    min_ent = float('inf')

    for i in range(canvas_height):
        for j in range(canvas_width):
            if (entropy_grid[i][j] < min_ent) and (abs(entropy_grid[i][j])//1 > 0.0): # Skip the already collapsed cells
                min_ent_cell = (i,j)
                min_ent = entropy_grid[i][j]
    return min_ent_cell

def init() -> None:
    global entropy_grid, grid

    all_possible_states = [
        "wall_25.png",
        "spike_0.png",
        "pillar_0.png",
        "ledge_1.png",
        "box_0.png",
        "background_dark_0.png"
    ]

    for i in range(canvas_height):
        for j in range(canvas_width):
            entropy_grid[i][j] = calculate_entropy(len(all_possible_states))
            grid[i][j] = all_possible_states
            
def display_grid() -> None:
    for i in range(canvas_height):
        for j in range(canvas_width):
            print(grid[i][j], end="\t")
        print()
    print("*"*50)

def display_ent_grid() -> None:
    for i in range(canvas_height):
        for j in range(canvas_width):
            print(entropy_grid[i][j], end="\t")
        print()
    print("-"*50)

def render_grid() -> np.ndarray:
    canvas_height = len(grid) * tile_size
    canvas_width = len(grid[0]) * tile_size

    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            image_name ,= grid[i][j]
            if image_name:
                image_path = f"tileset/{image_name}"
                image = cv.imread(image_path)
                if image is not None:
                    # Resize image to tile size
                    image = cv.resize(image, (tile_size, tile_size))
                    # Calculate coordinates for rendering
                    start_x = j * tile_size
                    start_y = i * tile_size
                    end_x = start_x + tile_size
                    end_y = start_y + tile_size
                    # Render image onto canvas
                    canvas[start_y:end_y, start_x:end_x] = image

    return canvas

def direction(dx: int, dy: int) -> str:
    if dx == -1 and dy == 0:
        return "top"
    elif dx == 1 and dy == 0:
        return "down"
    elif dx == 0 and dy == -1:
        return "left"
    elif dx == 0 and dy == 1:
        return "right"
    else:
        print(dx,dy)
        return "unknown"

def get_possible_states(coords: tuple[int,int], dir: str) -> list[str]:
    x,y = coords
    possible_states = []

    for state in grid[x][y]:
        for poss_neigh in rules[state][dir]:
            if poss_neigh not in possible_states:
                possible_states.append(poss_neigh)
    return possible_states


def valid_direcs(coords: tuple[int,int]) -> list[tuple[int,int]]:
    x,y = coords
    possible_dirs = []

    if (x-1)>=0:
        possible_dirs.append((-1,0))
    if (x+1)<canvas_height:
        possible_dirs.append((+1,0))
    if (y-1)>=0:
        possible_dirs.append((0,-1))
    if (y+1)<canvas_width:
        possible_dirs.append((0,+1))

    return possible_dirs

def constrain(coords: tuple[int, int], impossible_state) -> None:

    global grid, entropy_grid

    x,y = coords
    current_possibilities = grid[x][y][:]
    current_possibilities.remove(impossible_state)
    grid[x][y] = current_possibilities
    entropy_grid[x][y] = calculate_entropy(max(len(current_possibilities),1))

def propogate_constraints(coords: tuple[int,int]) -> None:
    stack = []
    stack.append(coords)

    while len(stack)>0:
        curr_coords = stack.pop(-1)
        x,y = curr_coords
        for dx,dy in valid_direcs(curr_coords):
            nx,ny = x+dx, y+dy
            neigh_states = grid[nx][ny][:]
            possible_states = get_possible_states(curr_coords, direction(dx,dy))

            for possibile_neigh_state in neigh_states:
                if possibile_neigh_state not in possible_states:
                    constrain((nx,ny), possibile_neigh_state)
                    if (nx,ny) not in stack:
                        stack.append((nx,ny))

def collapse_at(coords: tuple[int,int]) -> None:

    global grid, entropy_grid

    x,y = coords
    chosen_state = random.choice(grid[x][y])
    grid[x][y] = [chosen_state]
    entropy_grid[x][y] = 0.0

def is_wave_collapsed() -> bool:

    global entropy_grid

    for row in entropy_grid:
        for col in row:
            if abs(col) != 0.0:
                return False
    return True

def reached_valid_state() -> bool:
    global grid

    for row in grid:
        for col in row:
            if len(col) == 0:
                return False
    return True

def wave_function_collapse() -> None:

    global grid, entropy_grid
    while not is_wave_collapsed():
        coords = get_min_entropy_cell()
        collapse_at(coords)
        propogate_constraints(coords)

MAX_COUNTER = 100

while True:
    init()
    wave_function_collapse()

    if reached_valid_state():
        break

    if MAX_COUNTER <= 0:
            break

    MAX_COUNTER -= 1

    print("REACHED INVALID:", MAX_COUNTER)

if MAX_COUNTER == 0:
    quit()

canvas = render_grid()
cv.imshow("Grid", canvas)
cv.waitKey(0)
cv.destroyAllWindows()