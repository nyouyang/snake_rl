from snakegame import Grid
import numpy as np
import torch

class Game:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = Grid(grid_size)
        self.state = 'FINE'
    
    def image_with_permuted_channels(self):
        image = self.grid.nn_image()
        image_ = torch.from_numpy(image)
        image_.unsqueeze_(0)
        image_ = image_.permute(0, 3, 1, 2)
        return image_ / 255

    def reset(self):
        self.grid = Grid(self.grid_size)
        self.state = 'FINE'
        return self.image_with_permuted_channels()

    def step(self, action): # return observation_, reward, done
        if self.grid.apply_move(action): # action is [right, left, up, down]
            return None, 1, True
        elif self.grid.check_state() == 'LOSE':
            return None, -1, True

        image = self.image_with_permuted_channels()

        if self.grid.ate_this_turn:
            return image / 255, 1, False
        else:
            return image / 255, -0.001, False
        

if __name__ == '__main__':
    image = Game(4).image_with_permuted_channels()
    print(image.shape)