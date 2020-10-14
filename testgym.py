import gym
import time

game = "IceHockey-v0"
env = gym.make(game)

print(env.unwrapped.get_action_meanings())

state = env.reset()
done = False

start = time.time()
while not done:
    env.render() #current state
    state, rewards, done, info = env.step(env.action_space.sample())
    # input()

env.close()
print(time.time() - start)