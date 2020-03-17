import retro        
import numpy as np  
import cv2          
import neat         
import pickle       


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
    def work(self):
        
        self.env = retro.make(game='SuperMarioBros-Nes', state='Level2-1')
        
        ob = self.env.reset()
        inx,iny,inc = self.env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        fitness_current = 0
        frame = 0
        counter = 0
        imgarray= []
        done = False
        total_rew = 0
        while not done:
            # self.env.render()
            frame += 1
            ob = cv2.resize(ob,(inx,iny)) # Ob is the current frame
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # convert to grayscale
            ob = np.reshape(ob,(inx,iny))
            
            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))

            # cv2.imshow('image', ob)
            neuralnet_output = net.activate(imgarray) # Give an output for current frame from neural network
            ob, rew, done, info = self.env.step(neuralnet_output) # Try given output from network in the game
            total_rew += rew
            if rew >= 1.0:
                fitness_current = fitness_current + rew 
                counter = 0
            else:
                counter += 1
            
            if total_rew >= 3427.0:
                print('solution found')
                fitness_current += 1000000
                done = True
            # Train mario for max 250 frames
            elif done or counter == 250:
                done = True 

        print(int(fitness_current), "| distance :", total_rew)
        return int(fitness_current)

def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)
# p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-102')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(50))

pe = neat.ParallelEvaluator(10, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

