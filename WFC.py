import numpy as np
import cv2 as cv
import math

canvas_height, canvas_width = 16, 16
tile_size = 16
new_tile_size = tile_size*5

grid = [[0.0 for _ in range(canvas_width)] for _ in range(canvas_height)]

canvas = np.ones((canvas_width*new_tile_size, canvas_height*new_tile_size, 3), np.uint8) * 255

def calculate_entropy(num_states: int) -> float:
    p = 1 / num_states
    entropy = -sum(p * math.log2(p) for _ in range(num_states))
    return entropy

def create_tile_canvas(tiles: list[str]) -> np.ndarray:
    row_size = int(round(len(tiles)**0.5))
    shrunk_tile_size = new_tile_size // max(row_size, len(tiles))

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

    global grid

    all_possible_states = [
        "wall_25.png",
        "spike_0.png",
        "pillar_0.png",
        "ledge_1.png",
        "box_0.png",
        "background_dark_0.png"
    ]

    fixed_tiles = {
        (0,4): "wall_25.png",
        (3,8): "wall_25.png",
        (2,8): "spike_0.png"
    }

    for i in range(canvas_height):
        for j in range(canvas_width):
            start_point = (j*new_tile_size, i*new_tile_size)
            end_point = ((j+1)*new_tile_size, (i+1)*new_tile_size)

            possible_states = []
            coords = (j,i)

            if(coords in fixed_tiles):
                possible_states.append(fixed_tiles[coords])
            else:
                possible_states = all_possible_states
            canvas[start_point[0]:end_point[0],start_point[1]:end_point[1]] = create_tile_canvas(possible_states)
            grid[i][j] = calculate_entropy(len(possible_states))
            cv.putText(canvas, f"{grid[i][j]:.2f}", (start_point[1] + 5, start_point[0] + 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)


# def wave_function_collapse() -> None:

init()
cv.imshow("WFC", canvas)
cv.waitKey(0)
cv.destroyAllWindows()