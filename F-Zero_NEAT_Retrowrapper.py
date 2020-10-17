## Runs, but is weird. DOES NOT WORK CORRECTLY


import retro
import numpy as np
import cv2 
import neat
import pickle
import random 
import retrowrapper 


imgarray = []

xpos_end = 0





states = ['go', 'l45', 'r45', 'l90', 'r90']
actions = [[1,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,1,0,0,0,0],
            [1,0,0,0,0,0,1,0,0,0,0,0]]
                        
class Worker(object):
    def __init__(self, genome, config):
        self.name = "Worker"
        self.config = config
        self.genome = genome 
        
    def work(self):
        best_fitness = 0
        randstate = random.choice(states)
        fitness_current = 0        
        current_max_fitness = 0
        
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
        
        for st in states:
            frame = 0
            counter = 0
            xpos = 0
            xpos_max = 0
            health = 2048
            health_current = 2048
            done = False
            

            self.env = retrowrapper.RetroWrapper('FZero-Snes', state=st)
            ob = self.env.reset()
            ac = self.env.action_space.sample()
            inx, iny, inc = self.env.observation_space.shape
            inx = int(inx/8)
            iny = int(iny/8)
            ob, rew, done, info = self.env.step(actions[0])

            while not done:
                frame += 1
                self.env.render()
                ob = cv2.resize(ob, (inx, iny))
                ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)                             
                ob = np.reshape(ob, (inx,iny))
                imgarray = np.ndarray.flatten(ob)
                #imgarray = [info['x'], info['y'], info['pos'], info['speed'], info['health']]
                
                nnOutput = net.activate(imgarray)
                ob, rew, done, info = self.env.step(actions[np.argmax(nnOutput)])
                health = info['health']
                xpos = info['pos']
                if health < health_current:
                    health_current = health
                if xpos > xpos_max:
                    fitness_current += 10
                    xpos_max = xpos
                    
                if info['speed'] > 0 and health == health_current and info['reverse'] == 0:
                    fitness_current += 1
                else:
                    fitness_current -= 1
                
                
                if fitness_current > 9030:
                    fitness_current += 100000
                    done = True
                
                #fitness_current += rew
                
                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    counter = 0
                else:
                    counter += 1
                    
                if done or counter == 250:
            
                    done = True
            #self.env.close()
            if fitness_current > best_fitness: 
                best_fitness = fitness_current
                print("Best Fitness So Far!")
                with open('best_yet.pkl', 'wb') as output:
                    pickle.dump(self.genome, output, 1)
        
        print("Current Fitness: ", fitness_current, "Best Ever: ", best_fitness)
        if fitness_current < 0:
            fitness_current = -1
        
        return fitness_current


def eval_genomes(genome, config):
  

    
    worky = Worker(genome, config)
    return worky.work()
    
            
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-572')

pe = neat.ThreadedEvaluator(5, eval_genomes)
pe.stop()
winner = p.run(pe.evaluate)


with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
    

