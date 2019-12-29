import retro
import pickle
import neat
import cv2
import numpy as np

envv = retro.make(game='SuperMarioBros-Nes', state='Level1-1', record=False)
ob = envv.reset()
random_action = envv.action_space.sample()
oned_image = []

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')



genome = pickle.load(open('winner.pkl', 'rb'))


inx,iny,inc = envv.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

# 20 networks

net = neat.nn.FeedForwardNetwork.create(genome, config)
current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0

done = False

while not done:
    envv.render()
    frame += 1
    ob = cv2.resize(ob,(inx,iny)) # Ob is the current frame
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # convert to grayscale
    ob = np.reshape(ob,(inx,iny))
    
    imgarray = np.ndarray.flatten(ob)
    imgarray = np.interp(imgarray, (0, 254), (-1, +1))

    # cv2.imshow('image', ob)
    neuralnet_output = net.activate(imgarray) # Give an output for current frame from neural network
    ob, rew, done, info = envv.step(neuralnet_output) # Try given output from network in the game

    fitness_current += rew

    if fitness_current>current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    else:
        counter+=1

    if current_max_fitness >= 3106.0:
        current_max_fitness += 100000
        done = True
    # Train mario for max 250 frames
    elif done or counter == 250:
        done = True 
    
envv.close()


