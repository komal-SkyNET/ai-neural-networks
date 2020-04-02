import tensorflow as tf
from gym import wrappers
import numpy as np
import gym
from wrapper import ChannelsFirstImageShape, FrameStack, ProcessFrame84
fn = 'breakout-1585731598.model'

model = tf.keras.models.load_model('{}'.format(fn))
def get_qs(state):
    state = np.array(state)
    state = state/255.0
    return model.predict(np.array(state).reshape(-1, *state.shape))[0]

env = gym.make('BreakoutDeterministic-v4')
env = ProcessFrame84(env)
env = ChannelsFirstImageShape(env)
env = FrameStack(env, 4)
env = wrappers.Monitor(env, "./gym-results", force=True)
obs = env.reset()
print(env.unwrapped.get_action_meanings())
done = False
total_rew = 0
step = 0 
epsilon = 0.02
lives = 5
print("started..")
while not done:
    if np.random.random() > epsilon:
        # Get action from Q table
        action = np.argmax(get_qs(obs))
    else:
        # Get random action
        action = np.random.randint(0, env.action_space.n)
    obs, rew, done, info = env.step(action)
    if not done and info['ale.lives'] < lives:
        #fire to drop the ball
        lives = info['ale.lives']
        obs, rew, done, info = env.step(1)
    if rew != 0:
        print("reward: {}, done: {}, info: {}".format(total_rew, done, info))
    total_rew += rew
env.close()

