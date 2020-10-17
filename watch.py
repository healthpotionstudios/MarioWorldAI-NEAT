# to run this: python watch.py --vid the_name_of_some_bk2_file.bk2

import retro
import argparse

parser = argparse.ArgumentParser(description='Play a vid')
parser.add_argument('--vid', type=str, help='bk2 file name')

args = parser.parse_args()

print(args.vid)

movie = retro.Movie(args.vid)
movie.step()

env = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL)
env.initial_state = movie.get_state()
env.reset()

while movie.step():
    keys = []
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    _obs, _rew, _done, _info = env.step(keys)
    env.render()
