from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
keys = {"w": 0,}

done = True
action = 0
while True:
    for step in range(10):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(action)
        env.render()
    action = int(input())

env.close()