import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np 

import random

from Game import SnakeGame
from Agent import Agent
from utils import Direction

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(DEVICE)

### the model predicts the q-value for each action given the state
class LinearModel(nn.Module):
    def __init__(self,input_size=6, hidden_size=128):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 4) ### output_size = 4 because there are 4 actions
    
    def forward(self, x):
        x = torch.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, r, hidden_size=16):
        super(CNNModel, self).__init__()
        ### kernel size need to be tuned
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels =1, kernel_size = 5, padding = 2) ### in & out channels are 1, stride = 1
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels =1, kernel_size = 3)
        ### size after conv1: (2*r+1 - 5 + 2*2）// 1 + 1 = 2*r+1
        ### size after conv2: (2*r+1 - 3) // 1 + 1
        self.fc1 = nn.Linear((2*r+1-3 + 1) * (2*r+1-3 + 1), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 4) ### output_size = 4 because there are 4 actions
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





class DeepQLearningAgent(Agent):
    def __init__(self, game, model_type, pretrained_model=None):
        super(DeepQLearningAgent, self).__init__(game)

        self.model_type = model_type ### set the model type

        if model_type == "linear":
            # self.model = LinearModel().to(DEVICE)
            self.model = LinearModel(input_size=121).to(DEVICE)

        elif model_type == "cnn":
            self.model = CNNModel(5).to(DEVICE) ### r = 5
        
        if pretrained_model:
            self._loadModel(pretrained_model)

    def __cur_state(self):
        if self.model_type == "linear":
            # return self.__naive_state().to(DEVICE)
            return self.__surrounding_state().to(DEVICE)

        elif self.model_type == "cnn":
            return self.__surrounding_state().to(DEVICE)
    
    def __naive_state(self):
        '''
        The same state as Q learning agent used.
        Naive Q state : 
        (surrounding obstacle, food direction)
        I used 4 bits to indicate whether there are obstacles surrounding the snake head
        (0/1, 0/1, 0/1, 0/1): (up, down, left, right)
        I used 2 bits to indicate the food direction:
        (-1, 1) ( 0, 1) ( 1, 1)
        (-1, 0)   head  ( 1, 0)
        (-1,-1) (0, -1) ( 1,-1)
        There are 2*2*2*2*8 = 128 states in total.
        There are 4 possible actions UP, DOWN, LEFT, RIGHT in each state.
        Therefore, the Q table size will be 128 * 4 = 512.
        '''

        head_pos = self.game.head_pos
        food_pos = self.game.food_pos
        whitespace = self.game.whitespace
        assert food_pos is not None ### if there is no food on the board, the snake must have filled the board

        ### 6 bits state (up, down, left, right, food_dirX, food_dirY)
        state = [None]*6
        
        up_pos = (head_pos[0], head_pos[1]-1)
        if up_pos in whitespace or up_pos == food_pos: ### empty space at up pos
            state[0] = 0
        else: ### there is obstacle at up pos
            state[0] = 1

        down_pos = (head_pos[0], head_pos[1]+1)
        if down_pos in whitespace or down_pos == food_pos: ### empty space at down pos
            state[1] = 0
        else: ### there is obstacle at down pos
            state[1] = 1
        
        left_pos = (head_pos[0]-1, head_pos[1])
        if left_pos in whitespace or left_pos == food_pos: ### empty space at left pos
            state[2] = 0
        else: ### there is obstacle at left pos
            state[2] = 1

        right_pos = (head_pos[0]+1, head_pos[1])
        if right_pos in whitespace or right_pos == food_pos: ### empty space at right pos
            state[3] = 0
        else: ### there is obstacle at right pos
            state[3] = 1
        
        ### food direction x
        if food_pos[0] > head_pos[0]:
            state[4] = 1
        elif food_pos[0] < head_pos[0]:
            state[4] = -1
        else:
            state[4] = 0
        ### food direction y
        if food_pos[1] > head_pos[1]:
            state[5] = 1
        elif food_pos[1] < head_pos[1]:
            state[5] = -1
        else:
            state[5] = 0
        
        return torch.tensor(state, dtype=torch.float).to(DEVICE)


    ### Theoretically, CNN may not work well because of its strong prior assumption:
    ### the image pixels have strong relationship with its neighbours
    ### local features may not be useful in snake game
    def __surrounding_state(self, r=5): 
        '''
        the surrounding state is defined as the area of which the snake head is the center
                  |
                  r
                  |
        <-- r -- head -- r -->
                  |
                  r
                  |        
        '''
        WALL = 0
        EMPTY = 1
        SNAKE_BODY = 0
        SNAKE_HEAD = 0
        SNAKE_TAIL = 0
        FOOD = 5

        W = self.game.W
        H = self.game.H
        food_pos = self.game.food_pos
        head_pos = self.game.snake[-1]
        tail_pos = self.game.snake[0] 
        body = list(self.game.snake)[1: len(self.game.snake)-1]
        wall = list(self.game.wall)
        
        state = np.zeros((2*r+1, 2*r+1))
        ### (offset_x, offset_y) is the coordinate of the left-up corner of the surrounding area
        offset_x, offset_y = head_pos[0] - r, head_pos[1] - r 

        ### set the position outside the map as WALL
        for x in range(2*r+1):
            for y in range(2*r+1):
                if x + offset_x < 0 or y + offset_y < 0 or x + offset_x >= W or y + offset_y >= H:
                    state[y][x] = WALL

        ### set wall
        for px, py in wall:
            x, y = px - offset_x, py - offset_y
            if x < 0 or y < 0 or x >= 2*r+1 or y >= 2*r+1:
                continue
            else:
                state[y][x] = WALL     
        
        ### set snake body
        for px, py in body:
            x, y = px - offset_x, py - offset_y
            if x < 0 or y < 0 or x >= 2*r+1 or y >= 2*r+1:
                continue
            else:
                state[y][x] = SNAKE_BODY
        
        ### set snake tail
        px, py = tail_pos
        x, y = px - offset_x, py - offset_y
        if x < 0 or y < 0 or x >= 2*r+1 or y >= 2*r+1:
            pass
        else:
            state[y][x] = SNAKE_TAIL

        ### set snake head
        px, py = head_pos
        x, y = px - offset_x, py - offset_y
        if x < 0 or y < 0 or x >= 2*r+1 or y >= 2*r+1:
            pass
        else:
            state[y][x] = SNAKE_HEAD

        ### set food
        px, py = food_pos
        x, y = px - offset_x, py - offset_y
        if x < 0 or y < 0 or x >= 2*r+1 or y >= 2*r+1:
            pass
        else:
            state[y][x] = FOOD
        
        state = torch.tensor(state, dtype=torch.float).to(DEVICE)
        return state.reshape(1,1, 2*r+1, 2*r+1)

        
    def _move(self):
        directions = [d.value for d in Direction]
        state = self.__cur_state()
        prediction = self.model(state)
        move = directions[torch.argmax(prediction).item()] ### choose the move with the highest score
        return move

    def _saveModel(self, filename="model/linear.pth"):
        torch.save(self.model.state_dict(), filename)

    def _loadModel(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=torch.device(DEVICE)))

    def __train_step(self, optimizer, criterion, discount, state, action, reward, new_state):
        directions = [d.value for d in Direction]
        action = directions.index(action) ### 0 1 2 3
        pred = self.model(state)
        target = pred.clone()

        sample = 0
        if new_state == None: # terminal state
            sample = reward
        else:
            ### only the current best action's Q-value will be updated according Bellman-Equation
            sample = reward + discount * torch.max(self.model(new_state))
        target[action] = sample

        optimizer.zero_grad()
        loss = criterion(target, pred)
        loss.backward()
        optimizer.step()


    def train(self, lr=0.01, discount=0.8, epsilon=1.0, ed=0.01, n_epoch=200, filename="model/new_model.pth"):
        '''
        lr: learning rate
        discount: discount to make the Q-value converge
        epsilon: possibility to make the random move to explore the state
        ed: epsilon decay rate of epsilon after each round training
        n_epoch: total training round
        '''
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        ### the action is based on the Q-values predicted by the model
        criterion = nn.MSELoss()

        for epoch in range(n_epoch):
            t = 0 ### to avoid infinite loop like the snake keep chasing its tail
            while t < 200:
                action = self._move()
                if random.uniform(0,1) < epsilon: ### force to make random move to explore state space
                    action = random.choice(list(Direction)).value

                state = self.__cur_state()
                reward, dead = self.game._play(move=action) ### reward and whether the game has ended
                new_state = self.__cur_state() if not dead else None ### s'

                self.__train_step(optimizer, criterion, discount, state, action, reward, new_state)

                if dead:
                    epsilon -= ed ### decay the epsilon
                    break

                t += 1
            if epoch % 10 == 0:
                print(f"Current epoch: {epoch} Highest Score: {max(self.game.record)}")
        
        self._saveModel(filename=filename)


if __name__ == "__main__":
    '''
    It is highly recommended that train on the small-sized map first and then go to larger map to train the snake
    When train in the small-sized map, remember to set the t in train() to a small number (e.g. 100)
    During the further training round, remember to set the epsilon in train() to 0 
    '''
    game = SnakeGame(W=5, H=5, SPEED=50)
    # agent = DeepQLearningAgent(game, "linear")
    # agent = DeepQLearningAgent(game, "linear", pretrained_model='model/linear_with_surroungding_input-cpu-2.pth')

    agent = DeepQLearningAgent(game, "cnn")
    # agent = DeepQLearningAgent(game, "cnn", pretrained_model='model/linear_with_surroungding_input-cpu-1.pth')

    agent.train(filename='model/cnn-cpu.pth')
   

    # agent = DeepQLearningAgent(game, "linear", pretrained_model="model/linear.pth")

    # while True:
    #     game._play()