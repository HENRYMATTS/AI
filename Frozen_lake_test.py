import pygame,random,sys
from pygame.locals import*
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# Environment class
class FrozenLake():
    def __init__(self):
        # Constants
        self.GRID_SIZE = 4
        self.SQUARE_SIZE = 100
        self.WINDOW_SIZE = self.GRID_SIZE * self.SQUARE_SIZE
        self.PLAYER_COLOR = (0, 0, 255)  # Green
       
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Frozen Lake")
        self.clock = pygame.time.Clock()

        # Player position
        self.player_x, self.player_y = 0,0
    
    def reset(self):
        self.player_x = 0
        self.player_y = 0
        return [self.player_x , self. player_y]
        
    def draw(self,action,env_map):
        
        done = False
        if action == 2 and self.player_x < self.GRID_SIZE - 1:
            self.player_x += 1
        elif action == 1 and self.player_x > 0:
            self.player_x -= 1
        elif action == 0 and self.player_y < self.GRID_SIZE - 1:
            self.player_y += 1
       
     
      
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
        
        pygame.display.flip()
        self.clock.tick(2)
        return action,reward,next_state, done


# Deep Q-Neural Network
class PolicyNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(PolicyNet,self).__init__()
        self.l1 = nn.Linear(input_size,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256, output_size)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x



#hyperparameters
input_size = 2  
output_size = 3 

#initialise all the classes
#  PolicyNet
policy = PolicyNet(input_size,output_size)
policy.load_state_dict(torch.load('model_10000.pth'))
policy.eval()  # Set the model to evaluation mode

env = FrozenLake()



# frozen lake environment map
env_map = np.array([
                    [2,1,1,1],
                    [1,0,1,0],
                    [1,1,1,0],
                    [0,1,1,3]
                  ])

state = torch.tensor( env.reset(), dtype=torch.float)
state = state.view(1,-1)
done = False
while True:
   
    for event in pygame.event.get():
        if event.type == QUIT or event.type == KEYUP and event.key == K_ESCAPE:
            pygame.quit()
            sys.exit()
    
       
    with torch.no_grad():
        q_values = policy(state)
      
    action = q_values.argmax().item()
    action,reward,next_state, done = env.draw(action,env_map)
    next_state = torch.tensor(next_state,dtype=torch.float) 
    if done == True:
        state = torch.tensor( env.reset(), dtype=torch.float)
        state = state.view(1,-1)
        
    state = next_state
       
