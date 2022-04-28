import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

from collections import deque
import random
import pickle

from utils import Direction, Value

from Game import SnakeGame
from Agent import Agent
from DeepQLearningAgent import DeepQLearningAgent

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

'''
IMPORTANT!!!
For auto-encoder, we need to use fixed map size 
and values which represent different type of block.
The Value is defined in utils.py
If you modify the map-size or Value, you have to 
re-train an auto-encoder.
'''

'''
Currently, we set the map size as W = 16 and H = 16.
Please check the Value in utils.py

Encoder.encoder should have the same structure as AutoEncoder.encoder
Encoder will use the trained parameter from AutoEncoder
'''
class Encoder(nn.Module):
    def __init__(self, W=16, H=16, feature_size=8):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            torch.nn.Linear(W * H, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, feature_size),
        )


    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class AutoEncoder(nn.Module):
    def __init__(self, W=16, H=16, feature_size=8):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            torch.nn.Linear(W * H, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, feature_size),
        )

        self.decoder = nn.Sequential(
            torch.nn.Linear(feature_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, W*H),
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


### Simplified snake game without pygame stuff, only considering the map state
class GameState:
    def __init__(self, W=16, H=16):
        self.W = W
        self.H = H
        self._setWall()
        self._reset()
    
    # def _setWall(self):
    #     self.wall = set()
    def _setWall(self):
        y = int(self.H / 2-2)
        self.wall = set()
        for i in range(self.W//2, self.W):
            self.wall.add((y, i))
    
    def _reset(self):
        self.whitespace = set([(x,y) for x in range(self.W) for y in range(self.H)]) ### for fast generating new food 
        ### remove wall
        for wall in self.wall:
            self.whitespace.remove(wall)
        ### initialize food
        self.food_pos = random.choice(list(self.whitespace))
        self.whitespace.remove(self.food_pos)
        ### initialize head
        head_pos = random.choice(list(self.whitespace))
        self.snake = deque([head_pos]) ### snake[-1] is head and snake[0] is tail
        self.whitespace.remove(self.snake[0])

    '''
    The state will be 2D array which looks like:
    1 1 1 1 1
    1 5 1 0 1 
    1 1 0 0 1
    1 0 0 1 1
    1 0 1 1 1 
    where food is located in (1,1) which is five and the snake is represented by 0, 1 means EMPTY
    '''
    def cur_state(self): ### same as the map_state in SnakeGame class
        ### Value is defined in utils.py
        WALL = Value.WALL.value 
        EMPTY = Value.EMPTY.value 
        SNAKE_BODY = Value.SNAKE_BODY.value 
        SNAKE_HEAD = Value.SNAKE_HEAD.value 
        SNAKE_TAIL = Value.SNAKE_TAIL.value 
        FOOD = Value.FOOD.value 

        W = self.W
        H = self.H
        empty_space = list(self.whitespace)
        food_pos = self.food_pos
        head_pos = self.snake[-1]
        tail_pos = self.snake[0] 
        body = list(self.snake)[1: len(self.snake)-1]
        wall = list(self.wall)
        
        state = np.zeros((H, W))

        ### set wall
        for x, y in wall:
            state[y][x] = WALL     

        ### set empty space
        for x, y in empty_space:
            state[y][x] = EMPTY   
        
        ### set snake body
        for x, y in body:
            state[y][x] = SNAKE_BODY
        
        ### set snake tail
        x, y = tail_pos
        state[y][x] = SNAKE_TAIL

        ### set snake head
        x, y = head_pos
        state[y][x] = SNAKE_HEAD

        ### set food
        x, y = food_pos
        state[y][x] = FOOD
        
        # print(state)
        return state

        # state = torch.tensor(state, dtype=torch.float).to(DEVICE)
        # return state.reshape(1,1, H, W)

    ### extend the snake body randomly until it grows to the length
    def grow(self, length):
        directions = set([d.value for d in Direction])
        for i in range(length-1):
            valid_pos = []
            cur_tail = self.snake[0]
            for d in directions:
                pos = (cur_tail[0] + d[0], cur_tail[1] + d[1])
                if pos in self.whitespace:
                    valid_pos.append(pos)
            if len(valid_pos) == 0: ### the snake can not grow any more
                break
            
            new_body = random.choice(valid_pos)
            self.snake.appendleft(new_body)
            self.whitespace.remove(new_body)

    def naive_state(self):
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

        head_pos = self.snake[-1]
        food_pos = self.food_pos
        whitespace = self.whitespace
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

### refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class MapStateDataset(Dataset):
    def __init__(self, filename, flatten=True):
        self.flatten = flatten
        f = open(filename, 'rb')
        self.map_states = pickle.loads(f.read())
        f.close()
    
    def __len__(self):
        return len(self.map_states)
    
    def __getitem__(self, idx):
        if self.flatten: ### for fully connected encoder
            return torch.from_numpy(self.map_states[idx]).flatten().float()
        else: ### return 2d array (for CNN encoder (not implemented))
            return torch.from_numpy(np.expand_dims(self.map_states[idx], axis=0)).float() ### shape: (batch_size, 1, W, H)


### refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class MapStateActionDataset(Dataset):
    def __init__(self, filename):
        f = open(filename, 'rb')
        self.map_states = pickle.loads(f.read())
        # from ipdb import set_trace
        # set_trace()
        f.close()
    
    def __len__(self):
        return len(self.map_states)
    
    def __getitem__(self, idx):
        return self.map_states[idx][0].flatten(),self.map_states[idx][1].flatten()

### generate map states with different snakes 
def generateMapState(W=16, H=16, N=10000, filename="data/empty_map.pkl"):
    game = GameState(W=W, H=H)
    states = []
    for i in range(N): ### generate N scenes
        game._reset()
        length = random.randint(1, W*H-2)
        game.grow(length)
        states.append(game.cur_state())
        # print(game.cur_state())
        # from ipdb import set_trace
        # set_trace()

    f = open(filename, 'wb')
    f.write(pickle.dumps(states))
    f.close()

def generateMapStateWithActions(agent, W=16, H=16, N=10000, filename="data/empty_map.pkl"):
    game = GameState(W=W, H=H)
    states = []
    for i in range(N): ### generate N scenes
        game._reset()
        length = random.randint(1, (W*H-2)/2)
        game.grow(length)
        action = agent.model(game.naive_state())
        states.append((game.cur_state(), action.detach()))
        # from ipdb import set_trace
        # set_trace()
    f = open(filename, 'wb')
    f.write(pickle.dumps(states))
    f.close()

def train(n_epoch=10, batch_size=4, W=16, H=16, feature_size=8, dataset="data/empty_map.pkl", save_path="model/autoencoder-cpu.pth"):
    dataset = MapStateDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(W=W, H=H, feature_size=feature_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)
    for epoch in range(n_epoch):
        for data in dataloader:
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if epoch % 10 == 0:
        print(f"epoch: {epoch} loss: {loss.item()}")

    param = model.encoder.state_dict()
    torch.save(param, save_path)


def train_agent_loss(agent, n_epoch=10, batch_size=4, W=16, H=16, feature_size=6, dataset="data/empty_map.pkl", save_path="model/autoencoder-cpu.pth"):
    dataset = MapStateActionDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(W=W, H=H, feature_size=feature_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)
    for epoch in range(n_epoch):
        for data, action in dataloader:
            latent = model.encoder(data.float())
            action_reconstructed = agent.model(latent)
            
            loss = criterion(action_reconstructed, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if epoch % 10 == 0:
        print(f"epoch: {epoch} loss: {loss.item()}")

    param = model.encoder.state_dict()
    torch.save(param, save_path)

if __name__ == "__main__":
    # model = AutoEncoder().to(DEVICE)
    # encoder = Encoder().to(DEVICE)

    # param = model.encoder.state_dict()
    # torch.save(param, "test.pth")
    # print(param)

    # encoder.encoder.load_state_dict(torch.load("test.pth"))
    # print(encoder.state_dict())

    generateMapState()
    
    # dataset = MapStateDataset("data/empty_map.pkl")
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # for sample in dataloader:
    #     print(sample.size())
    #     break

    # generateMapState(N=1000000)

    outf = "model/autoencoder-cpu.pth"
    train(n_epoch=2, save_path=outf, feature_size=32)
    
    # data_path = "data/empty_map.pkl"
    # game = SnakeGame(W=10, H=10, SPEED=1000)
    # agent = DeepQLearningAgent(game, "linear", pretrained_model='model/linear-cpu.pth')
    # generateMapStateWithActions(agent,filename=data_path, N=1000000)

    # outf = "model/autoencoder-cpu-agent.pth"
    # train_agent_loss(agent, n_epoch=2, dataset=data_path, save_path=outf)