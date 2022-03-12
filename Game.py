import pygame

from enum import Enum
from collections import deque
import random

from utils import Direction, Reward, ManhattanDistance


class Color(Enum):
    WHITE = (255,255,255)
    GREY1 = (100,100,100)
    GREY2 = (200,200,200)
    BLACK = (0,0,0)
    RED   = (255,0,0)
    GREEN = (0,255,0)

class SnakeGame:
    def __init__(self, W=16, H=16, BLOCK_SIZE=20, SPEED=50, VERBOSE=False, SEED=None):
        ### set random seed
        random.seed(SEED)

        ### Set Game Parameter
        self.W = W
        self.H = H
        self.Width = W * BLOCK_SIZE ### window width
        self.Height = H * BLOCK_SIZE ### window height
        self.BLOCK_SIZE = BLOCK_SIZE if BLOCK_SIZE > 20 else 20 ### block size for display
        self.SPEED = SPEED
        self.VERBOSE = VERBOSE ### whether to print game information

        ### The game agent
        self.agent = None
        self.round = 0 ### current round
        self.record = []

        ### Initial GUI
        pygame.init()
        self.display = pygame.display.set_mode((self.Width, self.Height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        ### Initial Game State
        self._restart()
        self._renderGUI()
    
    ### reset the game state to initial state
    def _restart(self):
        self.round += 1 ### a new round start

        self.head_pos = (self.W // 2, self.H // 2) ### the initial postion of head
        self.snake = deque([(self.head_pos[0], self.head_pos[1] + 1), self.head_pos]) ### for fast updating snake
        self.whitespace = set([(x,y) for x in range(self.W) for y in range(self.H)]) ### for fast generating new food 
        self.whitespace.remove(self.snake[0])
        self.whitespace.remove(self.snake[1])
        self.score = 0 ### current score
        self.current_step = 0 ### how many steps the snake has moved
        
        self._placeFood() ### place food


    def _setAgent(self, agent):
        self.agent = agent ### set player

    def _placeFood(self):
        if len(self.whitespace) > 0:
            # self.food_pos = self.whitespace.pop() ### set food position by randomly choosing a whitespace
            self.food_pos = random.choice(list(self.whitespace))
            self.whitespace.remove(self.food_pos)

        else: ### there is no space to place food
            self.food_pos = None

    def _nextMove(self):
        self.current_step += 1

        if self.agent:
            for event in pygame.event.get():
                if(event.type == pygame.QUIT):
                    pygame.quit()
                    quit()
            
            return self.agent._move()

        else: ### default: manipulate the snake by keyboard
            for event in pygame.event.get():
                if(event.type == pygame.QUIT):
                    pygame.quit()
                    quit()
                if(event.type == pygame.KEYDOWN):
                    if(event.key == pygame.K_LEFT):
                        return Direction.LEFT.value
                    elif(event.key == pygame.K_RIGHT):
                        return Direction.RIGHT.value
                    elif(event.key == pygame.K_UP):
                        return Direction.UP.value
                    elif(event.key == pygame.K_DOWN):
                        return Direction.DOWN.value
            
            return self.head_dir() ### default: move forward

    ### play the game
    def _play(self, move=None): ### we can specify the move instead of let the agent decide the next move
        reward = Reward.LIVE.value
        dead = False

        if self.food_pos:
            prev_distance = ManhattanDistance(self.head_pos, self.food_pos) ### previous distance from snake head to food

            ### if the next move has been specified, we will ignore the agent's decision
            dx, dy = move if move else self._nextMove() ### direction
            new_head_pos = (self.head_pos[0] + dx, self.head_pos[1] + dy)
            
            self.head_pos = new_head_pos ### update head pos
            self.snake.append(new_head_pos) ### add new head to the snake

            if new_head_pos != self.food_pos:
                self.whitespace.add(self.snake.popleft()) ### add the tail to the whitespace and remove the tail from the snake

                if self._isCollision(): ### snake hit the wall or ate itself, it dead
                    reward = Reward.DEATH.value 
                    dead = True

                    if self.VERBOSE:
                        print(f"Round: {self.round} Score: {self.score} Steps: {self.current_step}")
                    self.record.append((self.score, self.current_step))
                    self._restart()
                else:
                    cur_distance = ManhattanDistance(self.head_pos, self.food_pos) ### current distance from snake head to food
                    if cur_distance > prev_distance: ### get further from the food
                        reward = Reward.FURTHER.value
                    elif cur_distance < prev_distance: ### get closer to the food
                        reward = Reward.CLOSER.value

            else: ### the snake ate the food
                ### in this case, there must not be collision
                self.score += 1
                self._placeFood()
                
                reward = Reward.FOOD.value 
                dead = True
            
            self.whitespace.discard(new_head_pos) ### remove the new head from the whitespace

        else: ### the snake filled the whole board
            reward = Reward.WIN.value 
            dead = True

            if self.VERBOSE:
                print(f"Round: {self.round} Score: {self.score} Steps: {self.current_step}")
            self.record.append((self.score, self.current_step))
            self._restart()

        self._renderGUI() ### render current state
        return reward, dead


    def _isCollision(self):
        if self.head_pos in self.whitespace:
            return False
        else:
            return True

    ### render the GUI
    def _renderGUI(self):
        self.display.fill(Color.WHITE.value)
        BLOCK_SIZE = self.BLOCK_SIZE
        BORDER = 4

        ### draw the snake
        for x,y in self.snake:
            pygame.draw.rect(self.display, Color.GREY1.value, pygame.Rect(x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, Color.GREY2.value, pygame.Rect(x*BLOCK_SIZE + BORDER, y*BLOCK_SIZE + BORDER, BLOCK_SIZE-2*BORDER, BLOCK_SIZE-2*BORDER))
        
        ### color the head
        pygame.draw.rect(self.display, Color.BLACK.value, pygame.Rect(self.snake[-1][0]*BLOCK_SIZE, self.snake[-1][1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        ### color the tail
        pygame.draw.rect(self.display, Color.RED.value, pygame.Rect(self.snake[0][0]*BLOCK_SIZE, self.snake[0][1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        if self.food_pos: ### if there is food, draw the food
            pygame.draw.rect(self.display, Color.GREEN.value, pygame.Rect(self.food_pos[0]*BLOCK_SIZE, self.food_pos[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.display.update()
        self.clock.tick(self.SPEED)

    ### forward direction
    def head_dir(self): 
        ### body -> head = head - body : BH = OH - OB
        return (self.snake[-1][0] - self.snake[-2][0], self.snake[-1][1] - self.snake[-2][1])
    
    def tail_dir(self):
        ### tail -> body = body - tail : TB = OB - OT
        return (self.snake[1][0] - self.snake[0][0], self.snake[1][1] - self.snake[0][1])

    ### current valid moves
    def valid_move(self):
        forward = self.head_dir()
        back = (-forward[0], -forward[1])

        directions = set([d.value for d in Direction])
        directions.remove(back)

        valid = []
        for d in directions:
            pos = (self.head_pos[0] + d[0], self.head_pos[1] + d[1])
            if pos == self.snake[0] or pos == self.food_pos: ### tail or food
                valid.append(d)
            elif pos in self.whitespace:
                valid.append(d)
        
        return valid
        


