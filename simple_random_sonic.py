# First, import retro. Make sure it's installed.
import retro 

# next start an environment to interact with Sonic
env = retro.make('SonicTheHedgehog-Genesis', state='GreenHillZone.Act1') 

# reset the env to the first frame so we can start interacting
env.reset() 

# loop forever 
while true: 
# env.step takes a random set of button presses and moves the emulator forward one frame 
    obs, rew, info, done = env.step(env.action_space.sample()) 
# env.render shows you in a little screen what's happening.
    env.render()