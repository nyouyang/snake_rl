from snakegame import Grid

class Game:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = Grid(grid_size)
        self.state = 'FINE'
    
    def reset(self):
        self.grid = Grid(self.grid_size)
        self.state = 'FINE'

    def step(self, action): # return observation_, reward, done
        pass
