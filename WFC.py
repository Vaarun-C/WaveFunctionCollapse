import cv2 as cv
import math
import json
import random
import numpy as np
import os

canvas_height, canvas_width = 13,13
tile_size = 64
MAX_COUNTER = 10000

with open('rules.json') as f:
    rules = json.load(f)

entropy_grid = [[0.0 for _ in range(canvas_width)] for _ in range(canvas_height)]
grid = [[[] for _ in range(canvas_width)] for _ in range(canvas_height)]

def calculate_entropy(num_states: int) -> float:
    """
    Calculate the entropy of a cell using the Shannon entropy formula [ HSh = -Σp*log(p) ]

    Parameters:
    - num_states (int): The number of states in superposition.

    Returns:
    (float): The entropy of a cell.
    """

    if num_states <= 1:
        return 0

    p = 1 / num_states
    entropy = -sum(p * math.log2(p) for _ in range(num_states))
    return entropy

def get_min_entropy_cell() -> tuple[int,int]:
    """
    Find the cell coordinates with the lowest entropy to collapse next.
    This cell is chosen because it represents least uncertainty.

    Parameters:
    None

    Returns:
    tuple[int, int]: The coordinates of the cell with least entropy.
    """

    min_ent_cell = (0,0)
    min_ent = float('inf')

    for i in range(canvas_height):
        for j in range(canvas_width):
            if (entropy_grid[i][j] < min_ent) and (abs(entropy_grid[i][j]) // 1 > 0.0): # Skip the already collapsed cells
                min_ent_cell = (i,j)
                min_ent = entropy_grid[i][j]
    return min_ent_cell

def init() -> None:
    """
    Initialize the grid with all cells set to a superposition of all possible states ( Highest entropy ). Initially all states are equally probable.
    This function sets up the initial state of the grid for the Wave Function Collapse algorithm.

    Parameters:
    None

    Returns:
    None
    """

    global entropy_grid, grid

    all_possible_states = os.listdir("tileset")
    initial_entropy = calculate_entropy(len(all_possible_states))

    for i in range(canvas_height):
        for j in range(canvas_width):
            entropy_grid[i][j] = initial_entropy
            grid[i][j] = all_possible_states
            
def display_grid() -> None:
    """
    This function prints the current state of the grid to the console in a tabular format.

    Parameters:
    None

    Returns:
    None
    """

    for i in range(canvas_height):
        for j in range(canvas_width):
            print(grid[i][j], end="\t")
        print()
    print("*" * 50)

def display_ent_grid() -> None:
    """
    This function prints the current entropies of the cells in the grid to the console in a tabular format.

    Parameters:
    None

    Returns:
    None
    """
    
    for i in range(canvas_height):
        for j in range(canvas_width):
            print(entropy_grid[i][j], end="\t")
        print()
    print("-" * 50)

def render_grid() -> np.ndarray:
    """
    Render the rendering tile images onto a canvas array.
    Each cell in the grid corresponds to a tile image, which is retrieved from the "tileset" directory.
    The canvas array is then filled with these tile images, appropriately resized and positioned based on the grid.

    Parameters:
    None

    Returns:
    np.ndarray: A 3-dimensional NumPy array representing the rendered canvas.
    """

    canvas_height_pixels = canvas_height * tile_size
    canvas_width_pixels = canvas_width * tile_size

    canvas = 255 * np.ones((canvas_height_pixels, canvas_width_pixels, 3), dtype=np.uint8)

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
    """
    Map the change in row and column to a cardinal direction,
    which is used to check possible states in the rules.json file.

    Parameters:
    - dx (int): The change in row.
    - dy (int): The change in column.

    Returns:
    str: The direction to corresponding to change in coordinates.

    Raises:
    ValueError: If the delta values do not correspond to a valid direction.
    """
        
    directions = {
        (-1, 0): "top",
        (1, 0): "down",
        (0, -1): "left",
        (0, 1): "right",
    }

    result = directions.get((dx, dy)) # type: ignore ( Here we are ignoring the type check as it raises a value error anyway. )
    if result is None:
        raise ValueError(f"Unknown direction: delta_row={dx}, delta_column={dy}")
    return result

def get_possible_states(coords: tuple[int,int], direction: str) -> list[str]:
    """
    Retrieve the list of possible states for a cell in a given direction, based on the rules.
    The function returns a list of all possible states that the cell can collapse to,
    considering the neighboring cells and their current states.

    Parameters:
    - coords (tuple[int, int]): The coordinates of the cell in the grid.
    - direction (str): The direction to check for possible states.

    Returns:
    list[str]: The possible states that for the given cell in the given direction.
    """
        
    x,y = coords
    possible_states = []

    for state in grid[x][y]:
        for poss_neigh in rules[state][direction]:
            if poss_neigh not in possible_states:
                possible_states.append(poss_neigh)
    return possible_states


def valid_direcs(coords: tuple[int,int]) -> list[tuple[int,int]]:
    """
    Return the list of neighboring cells surrounding a given cell.
    The neighboring cells are those immediately adjacent to the specified cell,
    excluding diagonal neighbors and any cells outside the grid boundaries.

    Parameters:
    - coords (tuple[int, int]): The coordinates of the cell.

    Returns:
    list[tuple[int, int]]: The list containing the coordinates of neighboring cells.
    """
        
    x, y = coords
    neighboring_cells = []

    if (x - 1) >= 0:
        neighboring_cells.append((-1, 0))
    if (x + 1) < canvas_height:
        neighboring_cells.append((+1, 0))
    if (y - 1) >= 0:
        neighboring_cells.append((0, -1))
    if (y + 1) < canvas_width:
        neighboring_cells.append((0, +1))

    return neighboring_cells

def constrain(coords: tuple[int, int], state_to_remove: str) -> None:
    """
    This function updates the superposition of a cell in the grid by removing a state
    that is determined to be impossible based on the rules and constraints of the system.

    Parameters:
    - coords (tuple[int, int]): The index of the cell.
    - state_to_remove (str): The state to be removed from the cell's superposition.

    Returns:
    None
    """
        
    global grid, entropy_grid

    x,y = coords
    current_possibilities = grid[x][y][:]
    current_possibilities.remove(state_to_remove)
    grid[x][y] = current_possibilities
    entropy_grid[x][y] = calculate_entropy(len(current_possibilities))

def propogate_constraints(coords: tuple[int,int]) -> None:
    """
    Propogate the constraints to the other cells in the grid until there are no changes to be propogated.
    Constraints are propagated using a DFS algorithm and causes the neighbouring cells to update their superpositions
    based on the rules.

    Parameters:
    - coords (tuple[int, int]): The coordinates of the cell that was recently changed.

    Returns:
    None
    """

    stack = []
    stack.append(coords)

    while len(stack)>0:
        curr_coords = stack.pop(-1)
        x,y = curr_coords

        # Iterate over valid neighboring directions
        for dx,dy in valid_direcs(curr_coords):
            nx,ny = x + dx, y + dy
            neigh_states = grid[nx][ny][:]
            possible_states = get_possible_states(curr_coords, direction(dx,dy))

            # Constrain neighbor states based on possible states
            for possibile_neigh_state in neigh_states:
                if possibile_neigh_state not in possible_states:
                    constrain((nx,ny), possibile_neigh_state)
                    if (nx,ny) not in stack:
                        stack.append((nx,ny))

def collapse_at(coords: tuple[int,int]) -> None:
    """
    Collapse the superposition of a cell to a single state randomly chosen from the possible states.

    Parameters:
    - coords (tuple[int, int]): The coordinates of the cell to be collapsed.

    Returns:
    None
    """

    global grid, entropy_grid

    x,y = coords
    collapsed_state = random.choice(grid[x][y])
    grid[x][y] = [collapsed_state]

    # Entropy now becomes 0 as there is only 1 state and no uncertainty
    entropy_grid[x][y] = 0.0

def is_wave_collapsed() -> bool:
    """
    Check if all states have collapsed ( Entropy of system is 0 )

    Parameters:
    None

    Returns:
    bool: True if the wave function is fully collapsed, False otherwise.
    """
    global entropy_grid

    for row in entropy_grid:
        for col in row:
            if abs(col) != 0.0:
                return False
    return True

def reached_valid_state() -> bool:
    """
    Check if we have reached a valid state ( All cells have 1 possibility )
    If any cell has no remaining possible states (empty superposition), 
    the function returns False, indicating that an invalid state has been reached.

    Parameters:
    None

    Returns:
    bool: True if a valid state has been reached, False otherwise.
    """
    global grid

    for row in grid:
        for col in row:
            if len(col) == 0:
                return False
    return True

def wave_function_collapse() -> None:
    """
    Apply the Wave Function Collapse (WFC) algorithm to fully collapse the wave function.

    The function operates on the global `grid` and `entropy_grid` arrays and
    continues collapsing the wave function until all states have collapsed,
    as determined by the `is_wave_collapsed` function.

    Parameters:
    None

    Returns:
    None
    """

    global grid, entropy_grid
    while not is_wave_collapsed():
        coords = get_min_entropy_cell()
        collapse_at(coords)
        propogate_constraints(coords)

if __name__ == "__main__":
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
        raise RuntimeError("Wave Function Collapse failed to converge to a valid state")

    canvas = render_grid()
    cv.imshow("Grid", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()