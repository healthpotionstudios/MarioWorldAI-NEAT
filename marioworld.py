import retro
from gym.envs.classic_control import rendering
import numpy as np
import cv2 
import neat
import pickle
import sys
import win32api

viewer = rendering.SimpleImageViewer()
env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2Yellow')

imgarray = []

xpos_end = 0

def kill():
    mx, my = win32api.GetCursorPos()
    if mx==0 and my==0:
        print("over")
        cv2.destroyAllWindows()
        env.reset()
        env.viewer = None
        env.close()
        env.render(close=True)
        sys.exit()

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print ("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/6)
        iny = int(iny/6)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xMax = 0
        
        done = False
        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("main", (256*2,240*2))
        cv2.namedWindow("ob", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ob", (256*2,240*2))

        while not done:
            kill()
            rgb = env.render('rgb_array')
            upscaled=repeat_upsample(rgb,3, 3)
            viewer.imshow(upscaled)
            
            frame += 1
            scaledimg = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            scaledimg = cv2.resize(scaledimg, (inx, iny)) 
            #ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(scaledimg, cv2.COLOR_RGB2GRAY)
            #ob = np.reshape(ob, (inx,iny))
            
            cv2.imshow('main', scaledimg)
            cv2.imshow('ob', ob)
            #cv2.waitKey(1) 

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            #print(nnOutput)
            ob, rew, done, info = env.step(nnOutput)
            
            #print(info['lives'])
            
            #xpos = info['x']
            #xpos_end = info['screen_x_end']
            
            #make sure he goes right
            if xMax < info['x']:
                xMax = info['x']
            fitness_current += rew
            
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            if done or counter == 350 or info['lives'] == 3:
                done = True
                fitness_current += info['score']
                print(genome_id, fitness_current)
                
            genome.fitness = fitness_current
                
            
            
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-mw')



p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-260")

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))


winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
    

