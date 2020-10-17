import retro

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

env.reset()

done = False

while not done:
    env.render()
    
    #action = env.action_space.sample()
    action = [0,0,1,0,0,0,0, 1,1,1,0,0]

    
    ob, rew, done, info = env.step(action)
    print("Action ", action, "Reward ", rew)
