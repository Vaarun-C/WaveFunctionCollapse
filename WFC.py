import numpy as np
import cv2 as cv
import math
import json

canvas_height, canvas_width = 4, 4
tile_size = 16
new_tile_size = tile_size*5

with open('rules.json') as f:
    rules = json.load(f)

entropy_grid = [[0.0 for _ in range(canvas_width)] for _ in range(canvas_height)]
grid = [[[] for _ in range(canvas_width)] for _ in range(canvas_height)]

canvas = np.ones((canvas_width*new_tile_size, canvas_height*new_tile_size, 3), np.uint8) * 255

def calculate_entropy(num_states: int) -> float:
    p = 1 / num_states
    entropy = -sum(p * math.log2(p) for _ in range(num_states))
    return entropy

def get_min_entropy_cell() -> tuple[int,int]:
    min_ent_cell = (0,0)
    min_ent = float('inf')

    for i in range(canvas_height):
        for j in range(canvas_width):
            if entropy_grid[i][j] < min_ent:
                min_ent_cell = (i,j)

    return min_ent_cell

def create_tile_canvas(tiles: list[str]) -> np.ndarray:
    row_size = int(round(len(tiles)**0.5))
    try:
        shrunk_tile_size = new_tile_size // max(row_size, len(tiles))
    except ZeroDivisionError:
        shrunk_tile_size = new_tile_size // 1

    tile_canvas = np.ones((new_tile_size, new_tile_size, 3), np.uint8) * 255
    tile_count = 0

    while tile_count < len(tiles):
        for i in range(row_size):

            if tile_count >= len(tiles):  # Ensure to exit loop when all tiles are processed
                break

            image = cv.imread(f"tileset/{tiles[tile_count]}")
            start_point = (i*shrunk_tile_size, (tile_count//row_size)*shrunk_tile_size)
            end_point = ((i+1)*shrunk_tile_size, ((tile_count//row_size) + 1)*shrunk_tile_size)

            new_width, new_height = end_point[0]-start_point[0], end_point[1]-start_point[1]
            image = cv.resize(image, (new_width, new_height))
            tile_canvas[start_point[0]:end_point[0],start_point[1]:end_point[1]] = image
            tile_count += 1

    return tile_canvas

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

    fixed_tiles = {
        (2,2): "wall_25.png",
    }

    for i in range(canvas_height):
        for j in range(canvas_width):
            possible_states = []
            coords = (j,i)

            if(coords in fixed_tiles):
                possible_states.append(fixed_tiles[coords])
            else:
                possible_states = all_possible_states

            entropy_grid[i][j] = calculate_entropy(len(possible_states))
            grid[i][j] = possible_states
    
    display_grid()
            
def display_grid() -> None:
    for i in range(canvas_height):
        for j in range(canvas_width):
            start_point = (j*new_tile_size, i*new_tile_size)
            end_point = ((j+1)*new_tile_size, (i+1)*new_tile_size)
            canvas[start_point[0]:end_point[0],start_point[1]:end_point[1]] = create_tile_canvas(grid[i][j])

def propogate_contraints(collapsed_cell: tuple[int, int]) -> None:
    visited = [[False for _ in range(canvas_width)] for _ in range(canvas_height)]
    queue = [(collapsed_cell[0],collapsed_cell[1])]

    while len(queue) > 0:
        i,j = queue.pop(0)
        if not visited[i][j]:
            visited[i][j] = True
            update_constraints((i,j))
            if (i-1)>=0:
                queue.append((i-1,j))
            if (i+1)<canvas_height:
                queue.append((i+1,j))
            if (j-1)>=0:
                queue.append((i,j-1))
            if (j+1)<canvas_width:
                queue.append((i,j+1))

def direction(dx: int, dy: int) -> str:
    if dx == -1 and dy == 0:
        return "left"
    elif dx == 1 and dy == 0:
        return "right"
    elif dx == 0 and dy == -1:
        return "top"
    elif dx == 0 and dy == 1:
        return "down"
    else:
        print(dx,dy)
        return "unknown"

def update_constraints(collapsed_cell: tuple[int, int]) -> None:
    i, j = collapsed_cell
    states = grid[i][j]

    for di in range(-1, 2):
        for dj in range(-1, 2):
            if abs(di) == abs(dj): # Skip the cell itself and diagonals
                continue

            ni, nj = i + di, j + dj
            if 0 <= ni < canvas_height and 0 <= nj < canvas_width:
                possible_states = set()
                
                for state in states:
                    states_from_rules = set(rules[state][direction(di,dj)])

                    if possible_states:
                        possible_states.intersection_update(states_from_rules)
                    else:
                        possible_states = states_from_rules
                        
                updated_states = [state for state in grid[ni][nj] if state in possible_states]
                grid[ni][nj] = updated_states


def wave_function_collapse() -> None:
    propogate_contraints((2,2))
    display_grid()

init()
wave_function_collapse()
cv.imshow("WFC", canvas)
cv.waitKey(0)
cv.destroyAllWindows()