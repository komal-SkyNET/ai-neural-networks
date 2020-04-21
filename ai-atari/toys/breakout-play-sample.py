import gym, time, cv2
import numpy as np
env_name='Breakout-v0'

class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(2080,), dtype=np.uint8)
    def observation(self, obs):
        inx,iny,inc = self.env.observation_space.shape
        inx = int(inx/6)
        iny = int(iny/6)
        obs = cv2.resize(obs,(inx,iny)) # Ob is the current frame
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # convert to grayscale
        obs = np.reshape(obs,(inx,iny))
        obs = np.ndarray.flatten(obs)
        return obs

env = gym.make(env_name)
env = PreProcessFrame(env)
done = False
while True:
    env.reset() 
    while not done:
        env.render()
        ob, reward, done, info = env.step(env.action_space.sample())
        print('####\n',ob, info)