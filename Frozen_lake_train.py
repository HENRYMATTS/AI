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
                        print('goal')
                       
                   
              
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
        self.clock.tick(60)
        return action,reward,next_state, done


# Neural Network
class PolicyNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(PolicyNet,self).__init__()
        self.l1 = nn.Linear(input_size,256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, output_size)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# Experience Replay
class Experience():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


#hyperparameters
input_size = 2  
output_size = 3
gamma = 0.9
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.001
replay_capacity = 10000
batch_size = 100

#initialise all the classes
#  PolicyNet and target DQNs
policy = PolicyNet(input_size,output_size)
target = PolicyNet(input_size,output_size)
target.load_state_dict(policy.state_dict())  # Copy policy weights to target network


#  environment and experience replay memory
env = FrozenLake()
replay = Experience(replay_capacity)

# loss $ optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# frozen lake environment map
env_map = np.array([
                    [2,1,1,1],
                    [1,0,1,0],
                    [1,1,1,0],
                    [0,1,1,3]
                  ])

# Training loop
for episodes in range(15000):
    state = torch.tensor( env.reset(), dtype=torch.float)
    state = state.view(1,-1)
    done = False
    for episode in range(200):
        for event in pygame.event.get():
            if event.type == QUIT or event.type == KEYUP and event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
      
        if random.random() < epsilon:
           
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                q_values = policy(state)
                action = torch.argmax(q_values).item()

               

        action,reward,next_state, done = env.draw(action,env_map)
        next_state = torch.tensor( next_state, dtype=torch.float)
        next_state = next_state.view(1,-1)
        
        replay.push(state,torch.tensor([action]),torch.tensor([reward]), next_state)
        
        if len(replay.memory) >= batch_size:
            samples = random.choice(replay.memory)
            
            q_values = policy(samples[0]).squeeze(0)
            q_value = q_values[samples[1]]
            
            
            max_q_value = target(samples[3]).max(1)[0].detach()
            target_q_value = samples[2] + gamma * max_q_value
           
            loss = criterion(q_value, target_q_value)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        
        state = next_state
        if done:
            break
    epsilon = max(epsilon * epsilon_decay, epsilon_min)      
    if episodes % 10 == 0:
        target.load_state_dict(policy.state_dict())
        
torch.save(policy.state_dict(), 'model_15000.pth')    
pygame.quit()
