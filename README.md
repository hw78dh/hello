# 뱀 게임을 하는 인공지능 만들기 
Genetic algorithm을 이용한 뱀게임을 하는 인공지능 프로그램  

![111](https://user-images.githubusercontent.com/54670559/69914917-c6b5c680-148c-11ea-84b1-f19a21dee061.png)  
21600614 장현우,  21600265 박성환 , 모두를 위한 인공지능 활용, HGU  


## 프로젝트를 하게 된 계기
파이썬에 pygame이라는 게임 개발 도구가 있다는 것을 보고, 뱀게임을 만들어 보았습니다. 뱀게임을 사람이 하는 것이 아니라, 인공지능이 스스로 학습하고 훈련하면 어떤 결과가 나올지 궁금해서 프로젝트를 시작하게 되었습니다.

## 다루게 될 기술
1. pygame
2. nympy
3. Genetic algorithm
4. os, random
 
## 프로젝트 개요
1. 인공지능에 앞쪽, 왼쪽, 오른쪽 방향을 가르친다.
2. 인공지능에 장애물을 감지하는 센서 3개를 달아준다.
3. 인공지능에 목표물을 찾는 센서 3개를 달아준다.
4. 원래 유전자에 변이를 일으켜서 강화학습 시킨다.
 
## 기대효과
1. 뱀게임에 대한 강화 학습을 오래 시킬수록 더 높은 점수를 획득하는 모델이 나온다.
2. 인공지능이 발전하는 과정을 볼 수 있다.  


## 실행코드
### Snake
~~~
import pygame
import os, random
import numpy as np 

FPS = 60
SCREEN_SIZE = 30
PIXEL_SIZE = 20
LINE_WIDTH = 1 

DIRECTIONS = np.array([
    (0, -1), # UP
    (1, 0), # RIGHT
    (0, 1), # DOWN
    (-1, 0) # LEFT
]) 

class Snake():
    snake, fruit = None, None

    def __init__(self, s, genome):
        self.genome = genome
        
        self.s = s
        self.score = 0
        self.snake = np.array([[15, 26], [15, 27], [15, 28], [15, 29]])
        self.direction = 0 # UP
        self.place_fruit()
        self.timer = 0
        self.last_fruit_time = 0
        
        # fitness
        self.fitness = 0.
        self.last_dist = np.inf

        
    # make fruit
    def place_fruit(self, coord=None):
        if coord:
            self.fruit = np.array(coord)
            return
        
        while True:
            x = random.randint(0, SCREEN_SIZE-1)
            y = random.randint(0, SCREEN_SIZE-1)
            if list([x, y]) not in self.snake.tolist():
                break
            self.fruit = np.array([x, y])

    def step(self, direction):
        old_head = self.snake[0]
        movement = DIRECTIONS[direction]
        new_head = old_head + movement
        
        if (
            new_head[0] < 0 or
            new_head[0] >= SCREEN_SIZE or
            new_head[1] < 0 or
            new_head[1] >= SCREEN_SIZE or
            new_head.tolist() in self.snake.tolist()
        ):
            # self.fitness -= FPS/2
            return False
        # eat fruit
        if all(new_head == self.fruit):
            self.last_fruit_time = self.timer
            self.score += 1
            self.fitness += 10
            self.place_fruit()
        else:
            tail = self.snake[-1]
            self.snake = self.snake[:-1, :]
        
        self.snake = np.concatenate([[new_head], self.snake], axis=0)
        return True

    def get_inputs(self):
        head = self.snake[0]
        result = [1., 1., 1., 0., 0., 0.]
        
        # check forward, left, right
        possible_dirs = [
            DIRECTIONS[self.direction], # straight forward
            DIRECTIONS[(self.direction + 3) % 4], # left
            DIRECTIONS[(self.direction + 1) % 4] # right
        ]
        
        # 0 - 1 ... danger - safe
        
        for i, p_dir in enumerate(possible_dirs):
            # sensor range = 5
            for j in range(5):
                guess_head = head + p_dir * (j + 1)
                
                if (
                    guess_head[0] < 0 or
                    guess_head[0] >= SCREEN_SIZE or
                    guess_head[1] < 0 or
                    guess_head[1] >= SCREEN_SIZE or
                    guess_head.tolist() in self.snake.tolist()
                ):
                    result[i] = j * 0.2
                    break
                    
        # finding fruit
        # heading straight forward to fruit
        if np.any(head == self.fruit) and np.sum(head * possible_dirs[0]) <= np.sum(self.fruit * possible_dirs[0]):
            result[3] = 1
        # fruit is on the left side
        if np.sum(head * possible_dirs[1]) < np.sum(self.fruit * possible_dirs[1]):
            result[4] = 1
        # fruit is on the right side
        # if np.sum(head * possible_dirs[2]) < np.sum(self.fruit * possible_dirs[2]):
        else:
            result[5] = 1
            
        return np.array(result)
    
    def run(self):
        self.fitness = 0
        
        prev_key = pygame.K_UP
        
        font = pygame.font.Font('/Users/David/Library/Fonts/3270Medium.otf', 20)
        font.set_bold(True)
        appleimage = pygame.Surface((PIXEL_SIZE, PIXEL_SIZE))
        appleimage.fill((0, 255, 0))
        img = pygame.Surface((PIXEL_SIZE, PIXEL_SIZE))
        img.fill((255, 0, 0))
        clock = pygame.time.Clock()
        
        while True:
            self.timer += 0.1
            if self.fitness < -FPS/2 or self.timer - self.last_fruit_time > 0.1 * FPS * 5:
                # self.fitness -= FPS/2
                print('Terminate!')
                break
                
                clock.tick(FPS)
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                    elif e.type == pygame.KEYDOWN:
                        # QUIT
                        if e.key == pygame.K_ESCAPE:
                            pygame.quit()
                            exit()
                        # PAUSE
                        if e.key == pygame.K_SPACE:
                            pause = True
                            while pause:
                                for ee in pygame.event.get():
                                    if ee.type == pygame.QUIT:
                                        pygame.quit()
                                    elif ee.type == pygame.KEYDOWN:
                                        if ee.key == pygame.K_SPACE:
                                            pause = False
                        if __name__ == '__main__':
                            # CONTROLLER
                            if prev_key != pygame.K_DOWN and e.key == pygame.K_UP:
                                self.direction = 0
                                prev_key = e.key
                            elif prev_key != pygame.K_LEFT and e.key == pygame.K_RIGHT:
                                self.direction = 1
                                prev_key = e.key
                            elif prev_key != pygame.K_UP and e.key == pygame.K_DOWN:
                                self.direction = 2
                                prev_key = e.key
                            elif prev_key != pygame.K_RIGHT and e.key == pygame.K_LEFT:
                                self.direction = 3
                                prev_key = e.key
                # action
                if __name__ != '__main__':
                    inputs = self.get_inputs()
                    outputs = self.genome.forward(inputs)
                    outputs = np.argmax(outputs)
                    
                    if outputs == 0: # straight
                        pass
                    elif outputs == 1: # left
                        self.direction = (self.direction + 3) % 4
                    elif outputs == 2: # right
                        self.direction = (self.direction + 1) % 4
                        
                if not self.step(self.direction):
                    break
                
                # compute fitness
                current_dist = np.linalg.norm(self.snake[0] - self.fruit)
                if self.last_dist > current_dist:
                    self.fitness += 1.
                else:
                    self.fitness -= 1.5
                self.last_dist = current_dist
                
                self.s.fill((0, 0, 0))
                pygame.draw.rect(self.s, (255,255,255), [0,0,SCREEN_SIZE*PIXEL_SIZE,LINE_WIDTH])
                pygame.draw.rect(self.s, (255,255,255), [0,SCREEN_SIZE*PIXEL_SIZE-LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE,LINE_WIDTH])
                pygame.draw.rect(self.s, (255,255,255), [0,0,LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE])
                pygame.draw.rect(self.s, (255,255,255), [SCREEN_SIZE*PIXEL_SIZE-LINE_WIDTH,0,LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE+LINE_WIDTH])
                for bit in self.snake:
                    self.s.blit(img, (bit[0] * PIXEL_SIZE, bit[1] * PIXEL_SIZE))
                self.s.blit(appleimage, (self.fruit[0] * PIXEL_SIZE, self.fruit[1] * PIXEL_SIZE))
                score_ts = font.render(str(self.score), False, (255, 255, 255))
                self.s.blit(score_ts, (5, 5))
                pygame.display.update()
                
            return self.fitness, self.score

if __name__ == '__main__':
    pygame.init()
    pygame.font.init()
    s = pygame.display.set_mode((SCREEN_SIZE * PIXEL_SIZE, SCREEN_SIZE * PIXEL_SIZE))
    pygame.display.set_caption('Snake')
    
    while True:
        snake = Snake(s, genome=None)
        fitness, score = snake.run()
        
        print('Fitness: %s, Score: %s' % (fitness, score)) 
~~~  
  
### Genome
~~~
import numpy as np

class Genome():
    def __init__(self):
        self.fitness = 0
        
        hidden_layer = 10
        self.w1 = np.random.randn(6, hidden_layer)
        self.w2 = np.random.randn(hidden_layer, 20)
        self.w3 = np.random.randn(20, hidden_layer)
        self.w4 = np.random.randn(hidden_layer, 3)
        
    def forward(self, inputs):
        net = np.matmul(inputs, self.w1)
        net = self.relu(net)
        net = np.matmul(net, self.w2)
        net = self.relu(net)
        net = np.matmul(net, self.w3)
        net = self.relu(net)
        net = np.matmul(net, self.w4)
        net = self.softmax(net)
        return net
    
    def relu(self, x):
        return x * (x >= 0)
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01) 
~~~  


### Evolution
~~~
import pygame, random 
import numpy as np 
from copy import deepcopy 
from snake import Snake, SCREEN_SIZE, PIXEL_SIZE 
from genome import Genome 
 
N_POPULATION = 50 
N_BEST = 5 
N_CHILDREN = 5 
PROB_MUTATION = 0.4

pygame.init() 
pygame.font.init() 
s = pygame.display.set_mode((SCREEN_SIZE * PIXEL_SIZE, SCREEN_SIZE * PIXEL_SIZE)) 
pygame.display.set_caption('Snake') 

 
# generate 1st population 
genomes = [Genome() for _ in range(N_POPULATION)] 
best_genomes = None 
 
n_gen = 0 
while True:
    n_gen += 1 

for i, genome in enumerate(genomes):
    snake = Snake(s, genome=genome) 
    fitness, score = snake.run() 

genome.fitness = fitness  
 
# print('Generation #%s, Genome #%s, Fitness: %s, Score: %s' % (n_gen, i, fitness, score)) 
 
if best_genomes is not None:
    genomes.extend(best_genomes)
genomes.sort(key=lambda x: x.fitness, reverse=True) 
 
print('===== Generaton #%s\tBest Fitness %s =====' % (n_gen, genomes[0].fitness)) 
# print(genomes[0].w1, genomes[0].w2) 
 
best_genomes = deepcopy(genomes[:N_BEST]) 

# crossover 
for i in range(N_CHILDREN):
    new_genome = deepcopy(best_genomes[0])
    a_genome = random.choice(best_genomes)
    b_genome = random.choice(best_genomes)
    
    cut = random.randint(0, new_genome.w1.shape[1])
    new_genome.w1[i, :cut] = a_genome.w1[i, :cut]
    new_genome.w1[i, cut:] = b_genome.w1[i, cut:] 
    
    cut = random.randint(0, new_genome.w2.shape[1])
    new_genome.w2[i, :cut] = a_genome.w2[i, :cut]
    new_genome.w2[i, cut:] = b_genome.w2[i, cut:] 
    
    cut = random.randint(0, new_genome.w3.shape[1])
    new_genome.w3[i, :cut] = a_genome.w3[i, :cut]
    new_genome.w3[i, cut:] = b_genome.w3[i, cut:] 

    cut = random.randint(0, new_genome.w4.shape[1]) 
    new_genome.w4[i, :cut] = a_genome.w4[i, :cut]
    new_genome.w4[i, cut:] = b_genome.w4[i, cut:] 

    best_genomes.append(new_genome) 
    
# mutation 
genomes = [] 
for i in range(int(N_POPULATION / (N_BEST + N_CHILDREN))):
    for bg in best_genomes:
        new_genome = deepcopy(bg) 
        
        mean = 20 
        stddev = 10
        
        if random.uniform(0, 1) < PROB_MUTATION:
            new_genome.w1 += new_genome.w1 * np.random.normal(mean, stddev, size=(6, 10)) / 100 * np.random.randint(-1, 2, (6, 10)) 
        if random.uniform(0, 1) < PROB_MUTATION:
            new_genome.w2 += new_genome.w2 * np.random.normal(mean, stddev, size=(10, 20)) / 100 * np.random.randint(-1, 2, (10, 20))
        if random.uniform(0, 1) < PROB_MUTATION:
            new_genome.w3 += new_genome.w3 * np.random.normal(mean, stddev, size=(20, 10)) / 100 * np.random.randint(-1, 2, (20, 10))
        if random.uniform(0, 1) < PROB_MUTATION:
            new_genome.w4 += new_genome.w4 * np.random.normal(mean, stddev, size=(10, 3)) / 100 * np.random.randint(-1, 2, (10, 3))
        genomes.append(new_genome)
        
  ~~~

## 한계
1. 뱀이 먹이를 향한 의지와 살아남기 위한 의지가 충돌하면서 뱀이 길어졌을 때 몸통에 충돌하게 됨
2. 시간이 지날수록 학습시간이 오래걸림
