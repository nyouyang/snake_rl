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


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'(x: {self.x}, y: {self.y})'

    def __eq__(self, compare):
        assert isinstance(compare, Vector), 'Cannot compare equality of a Vector object with a non-Vector object'
        return self.x == compare.x and self.y == compare.y

    def __add__(self, other):
        assert isinstance(other, Vector), 'Cannot add a Vector object with a non-Vector object'
        return Vector(self.x + other.x, self.y + other.y)

    def __hash__(self):
        x, y = self.x, self.y
        return int((x + y) * (x + y + 1) / 2 + y + 1) # interesting NxN hash function, courtesy of Surya Subbarao

DIRECTION_MAP = {
    'up': Vector(0, -1),
    'down': Vector(0, 1),
    'right': Vector(1, 0),
    'left': Vector(-1, 0)
}

class Snake:
    def __init__(self):
        self.head = Vector(2, 0) # make it start at (2, 0)
        self.tail = deque([Vector(0, 0), Vector(1, 0)]) # [first in, ..., last in]
        self.snake_size = 3
        self.direction = Vector(1, 0)
        self.snake_locations = set([self.head] + list(self.tail))

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
        self.snake_locations.add(new_head)
        self.snake_locations.remove(last)
        return last

class Grid:
    def __init__(self, grid_size = 10):
        assert grid_size >= 3, 'This is only designed for grids larger than 2x2'
        self.snake = Snake()
        self.all_coords = {Vector(x, y) for x in range(grid_size) for y in range(grid_size)}
        self.grid_size = grid_size
        self.setCoin()
    
    def setCoin(self):
        possible_setting = self.all_coords.difference(self.snake.snake_locations)
        self.coin_location = choice(list(possible_setting))

    def _check_bounds(self, coord):
        assert isinstance(coord, Vector), 'Cannot check bounds of object that is not of type Vector'
        return 0 <= coord.x < self.grid_size and 0 <= coord.y < self.grid_size

    def nn_image(self): # outputs the image in a [0, 255] RGB tensor
        ret = np.zeros((self.grid_size, self.grid_size, 3), 'uint8') + 100 # [0, 255] format
        ret[self.coin_location.y, self.coin_location.x, -1] = 255
        for cell in self.snake.snake_locations:
            ret[cell.y, cell.x] = 255
        # print(f'shape: {ret.shape}')
        return ret

    def apply_move(self, direction):
        self.snake.apply_direction(direction)
        last = self.snake.update()
        if self.snake.head == self.coin_location:
            self.snake.tail.appendleft(last)
            self.snake.snake_locations.add(last)
            self.snake.snake_size += 1
            self.setCoin() # set a new coin

    def check_state(self):
        if not self._check_bounds(self.snake.head) or self.snake.head in self.snake.tail:
            return 'LOSE'
        elif self.snake.snake_size == self.grid_size ** 2:
            return 'WIN'
        else:
            return 'FINE'

if __name__ == '__main__':
    import cv2
    grid = Grid(4)
    while True:
        state = grid.check_state()
        if state == 'FINE':
            cv2.imwrite('snake_rl/tmp_outputs/test.png', cv2.resize(grid.nn_image(), (160, 160), interpolation=cv2.INTER_NEAREST))
            move = input()
            grid.apply_move(move)
            continue
        elif state == 'LOSE':
            print('You lost.')
        elif state == 'WIN':
            print('You won!')
        break
        
    