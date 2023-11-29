import pygame
import sys
from pygame.locals import *
import numpy as np

class FrozenLake():
    def __init__(self):
        # Constants
        self.GRID_SIZE = 4
        self.SQUARE_SIZE = 100
        self.WINDOW_SIZE = self.GRID_SIZE * self.SQUARE_SIZE
        self.PLAYER_COLOR = (0, 255, 0)  # Green

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Frozen Lake")
        self.clock = pygame.time.Clock()

        # Player position
        self.player_x, self.player_y = 0, 0

       
    def reset(self):
        self.player_x = 0
        self.player_y = 0
        return [self.player_x , self. player_y]
    
    def draw(self,env_map):
     
        while True:
            for event in pygame.event.get():
                if event.type == QUIT or event.type == KEYUP and event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_RIGHT and self.player_x < self.GRID_SIZE - 1:
                        self.player_x += 1
                    elif event.key == K_LEFT and self.player_x > 0:
                        self.player_x -= 1
                    elif event.key == K_DOWN and self.player_y < self.GRID_SIZE - 1:
                        self.player_y += 1
                    # elif event.key == K_UP and self.player_y > 0:
                    #     self.player_y -= 1

            self.screen.fill((255, 255, 255))

            # Draw the grid
            for x in range(len(env_map)):
                for y in range(len(env_map)):
                    rect = pygame.Rect(y * self.SQUARE_SIZE, x * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE)
                    if env_map[x,y]== 2:
                        pygame.draw.rect(self.screen, (79,55, 39), rect)
                        if self.player_x == y and self.player_y == x:
                            reward = 0
                            next_state = [y,x]
                            done = False
                        
                        
                    elif env_map[x,y]== 1:
                        pygame.draw.rect(self.screen, ('black'), rect,1)
                        if self.player_x == y and self.player_y == x:
                            reward = 0
                            next_state = [y,x]
                            if env_map[2,1] or env_map[2,2] or env_map[3,1] or env_map[3,1]:
                                reward = 40
                      
                    elif env_map[x,y] == 3:    
                        pygame.draw.rect(self.screen, ('red'), rect)
                        if self.player_x == y and self.player_y == x:
                            reward = 200
                            next_state = [y,x]
                            done = True

                    else:
                        pygame.draw.rect(self.screen, ('black'), rect)
                        if self.player_x == y and self.player_y == x:
                            reward =  -120
                            next_state = [y,x]
                            done = True
                         
            # Draw the player
            player_rect = pygame.Rect(self.player_x * self.SQUARE_SIZE, self.player_y * self.SQUARE_SIZE,
                                      self.SQUARE_SIZE, self.SQUARE_SIZE)
            pygame.draw.rect(self.screen, self.PLAYER_COLOR, player_rect)
            if done:
                self.reset()
            pygame.display.update()
            self.clock.tick(30)

       
# Create an instance of the GridGame class to run the game
env = FrozenLake()
env_map = np.array([
                    [2,1,1,1],
                    [1,0,1,0],
                    [1,1,1,0],
                    [0,1,1,3]
                  ])
env.draw(env_map)