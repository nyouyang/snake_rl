# Created by Nicholas Ouyang
# July 2022

'''
Simple end goals for this file:
    Write up basic logic for the game "Snake"
        Snake main object
            Head, tail, direction
            Grid the game is played on
        How the game updates
            Control the snake
            Add length to the snake
            Win/lose conditions
    Ensure the game can be output as an image
        For later training of our deep CNN/RL model
'''

import numpy as np
from collections import deque
from random import choice


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'(x: {self.x}, y: {self.y})'

    def __eq__(self, compare):
        assert isinstance(compare, Coordinate), 'Cannot compare equality of a Coordinate object with a non-Coordinate object'
        return self.x == compare.x and self.y == compare.y

    def __add__(self, other):
        assert isinstance(other, Coordinate), 'Cannot add a Coordinate object with a non-Coordinate object'
        return Coordinate(self.x + other.x, self.y + other.y)

    def __hash__(self):
        x, y = self.x, self.y
        return int((x + y) * (x + y + 1) / 2 + y + 1) # interesting NxN hash function, courtesy of Surya Subbarao

DIRECTION_MAP = {
    'up': Coordinate(0, -1),
    'down': Coordinate(0, 1),
    'right': Coordinate(1, 0),
    'left': Coordinate(-1, 0)
}

class Snake:
    def __init__(self):
        self.head = Coordinate(2, 0) # make it start at (2, 0)
        self.tail = deque([Coordinate(0, 0), Coordinate(1, 0)]) # [first in, ..., last in]
        self.snake_size = 3
        self.direction = Coordinate(1, 0)

    def __repr__(self):
        return f'''
        Head: {self.head}
        Tail: {self.tail}
        Size: {self.snake_size}
        '''

    def apply_direction(self, new_direction):
        assert new_direction in DIRECTION_MAP, f'Direction {new_direction} is not a valid direction'
        self.direction = DIRECTION_MAP[new_direction]

    def update(self): # update and return the last (removed) cell of tail
        new_head = self.head + self.direction # check if in bounds later
        if self.tail:
            last = self.tail.popleft()
        self.tail.append(self.head)
        self.head = new_head
        return last

class Grid:
    def __init__(self, grid_size = 10):
        assert grid_size >= 3, 'This is only designed for grids larger than 2x2'
        self.snake = Snake()
        self.all_coords = {Coordinate(x, y) for x in range(grid_size) for y in range(grid_size)}
        self.grid_size = grid_size
        self.setCoin()

    def getCoin(self):
        assert self.coin_location is not None, 'No coins have been set yet'
        return self.coin_location
    
    def setCoin(self):
        snake_locations = set([self.snake.head] + list(self.snake.tail))
        possible_setting = self.all_coords.difference(snake_locations)
        self.coin_location = choice(list(possible_setting))

    def _check_bounds(self, coord):
        assert isinstance(coord, Coordinate), 'Cannot check bounds of object that is not of type Coordinate'
        return 0 <= coord.x < self.grid_size and 0 <= coord.y < self.grid_size

    def nn_image(self):
        ret = np.zeros((self.grid_size, self.grid_size, 3), 'uint8') + 100 # [0, 255] format
        ret[self.coin_location.y, self.coin_location.x, -1] = 255
        for cell in [self.snake.head] + list(self.snake.tail):
            ret[cell.y, cell.x] = 255
        return ret

    def apply_move(self, direction):
        self.snake.apply_direction(direction)
        last = self.snake.update()
        if self.snake.head == self.coin_location:
            self.setCoin() # set a new coin
            self.snake.tail.appendleft(last)
            self.snake.snake_size += 1

    def check_state(self):
        if not self._check_bounds(self.snake.head) or self.snake.head in self.snake.tail:
            return 'LOSE'
        elif self.snake.snake_size == self.grid_size ** 2:
            return 'WIN'
        else:
            return 'FINE'

if __name__ == '__main__':
    import cv2
    grid = Grid(5)
    while True:
        cv2.imwrite('snake_rl/tmp_outputs/test.png', cv2.resize(grid.nn_image(), (160, 160), interpolation=cv2.INTER_NEAREST))
        move = input()
        grid.apply_move(move)
        state = grid.check_state()
        if state == 'FINE':
            continue
        elif state == 'LOSE':
            print('You lost.')
        elif state == 'WIN':
            print('You won!')
        break



    # from PIL import Image
    # image = Image.fromarray(grid.nn_image(snake), 'RGB')
    # image = image.resize((640, 640)) # x16 from grid size, fix the quality
    # image.save(fp='snake_rl/tmp_outputs/test.png', mode='PNG', quality=90)
    # image.close() # needed?

    