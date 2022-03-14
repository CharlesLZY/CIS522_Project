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

class LinearModel(nn.Module):
    def __init__(self,input_size=6, hidden_size=128, output_size=4):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class DeepQLearningAgent(Agent):
    def __init__(self, game, model_type, pretrained_model=None):
        super(DeepQLearningAgent, self).__init__(game)

        self.model_type = model_type ### set the model type

        if model_type == "linear":
            self.model = LinearModel().to(DEVICE)
        
        if pretrained_model:
            self._loadModel(pretrained_model)

    def __cur_state(self):
        if self.model_type == "linear":
            return self.__naive_state().to(DEVICE)

    
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
        
        return torch.tensor(state, dtype=torch.float).cuda()
    
    def _move(self):
        directions = [d.value for d in Direction]
        state = self.__cur_state()
        prediction = self.model(state)
        move= directions[torch.argmax(prediction).item()] ### choose the move with the highest score
        return move

    def _saveModel(self, filename="model/linear.pth"):
        torch.save(self.model.state_dict(), filename)

    def _loadModel(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def __train_step(self, optimizer, criterion, discount, state, action, reward, new_state):
        directions = [d.value for d in Direction]
        action = directions.index(action) ### 0 1 2 3
        pred = self.model(state)
        target = pred.clone()

        sample = 0
        if new_state == None: # terminal state
            sample = reward
        else:
            sample = reward + discount * torch.max(self.model(new_state))
        target[action] = sample

        optimizer.zero_grad()
        loss = criterion(target,pred)
        loss.backward()
        optimizer.step()


    def train(self, lr=0.01, discount=0.8, epsilon=1.0, ed=0.01, n_epoch=200):
        '''
        lr: learning rate
        discount: discount to make the Q-value converge
        epsilon: possibility to make the random move to explore the state
        ed: epsilon decay rate of epsilon after each round training
        n_epoch: total training round
        '''
        optimizer = optim.Adam(self.model.parameters(), lr=lr)    
        criterion = nn.MSELoss()

        for epoch in range(n_epoch):
            t = 0 ### to avoid infinite loop like the snake keep chasing its tail
            while t < 1000:
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
        
        self._saveModel()


if __name__ == "__main__":
    game = SnakeGame(W=7, H=7)
    agent = DeepQLearningAgent(game, "linear")
    agent.train()
    # agent = DeepQLearningAgent(game, "linear", pretrained_model="model/linear.pth")
    # agent._saveModel()
    # while True:
    #     game._play()