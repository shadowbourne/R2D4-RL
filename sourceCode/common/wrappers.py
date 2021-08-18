import numpy as np
import gym
from gym import spaces
import collections
import cv2


class ForceDeath(gym.Wrapper):
    def __init__(self, env, is_monitor_enabled=False):
        gym.Wrapper.__init__(self, env)
        self.upper_x = 0
        self.is_monitor_enabled = is_monitor_enabled

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info["x_pos"] > self.upper_x:
            self.upper_x = info["x_pos"]
            self.counter = 0
        else:
            self.counter += 1
        if self.counter > 150:
            if self.is_monitor_enabled:
                self.env.stats_recorder.save_complete()
                self.env.stats_recorder.done = True
            return obs, float(reward), True, info
        return obs, float(reward), done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.upper_x = 0
        return obs
class CustomReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.status = None
        self.score = None
        self.y = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.status == None:
            self.score = info["score"]
            self.status = info["status"]
            self.y = info["y_pos"]
        if info["status"] != self.status:
            if not done:
                if self.status == "small" and info["status"] == "tall":
                    reward += 10
                elif self.status == "small" and info["status"] == "fireball":
                    reward += 20
                elif self.status == "tall" and info["status"] == "fireball":
                    reward += 10
                elif self.status == "tall" and info["status"] == "small":
                    reward -= 10
                elif self.status == "fireball" and info["status"] == "tall":
                    reward -= 10
                elif self.status == "fireball" and info["status"] == "small":
                    reward -= 20
                else:
                    raise Exception
            self.status = info["status"]
        if not done:
            reward += (info["score"] - self.score) / 200
            reward += (info["y_pos"] - self.y) / 10
        self.score = info["score"]
        self.y = info["y_pos"]

        return obs, float(reward), done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.score = None
        self.status = None
        return obs


class RepeatAction(gym.Wrapper):
    def __init__(self, env, n):
        """
        Repeat each action for n frames. 
        """
        gym.Wrapper.__init__(self, env)
        self.repeatFor = n
    
    def step(self, action):
        reward = 0
        for _ in range(self.repeatFor):
            obs, tempR, done, info = self.env.step(action)
            reward += tempR
            if done == True:
                break
        return obs, reward, done, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class AutoReset(gym.Wrapper):
    def __init__(self, env):
        """
        Reset env
        """
        gym.Wrapper.__init__(self, env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            _ = self.env.reset()
        return obs, reward, done, info

# Taken from RL Adventure's [10] and OpenAI's Baselines [11] Repos.
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

# Taken from RL Adventure's [10] and OpenAI's Baselines [11] Repos.
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        return np.array(frame[:, :, None])

# Edited from RL Adventure's [10] and OpenAI's Baselines [11] Repos.
# class FrameStack(gym.Wrapper):
#     def __init__(self, env, k):
#         """Stack k last frames.
#         Returns lazy array, which is much more memory efficient.
#         See Also
#         --------
#         baselines.common.atari_wrappers.LazyFrames
#         """
#         gym.Wrapper.__init__(self, env)
#         self.k = k
#         self.frames = collections.deque([], maxlen=k)
#         shp = env.observation_space.shape
#         self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k * shp[2]), dtype=np.uint8)
# 
#     def reset(self):
#         ob = self.env.reset()#.transpose(2,0,1)
#         for _ in range(self.k):
#             self.frames.append(ob)
#         return self._get_ob()
# 
#     def step(self, action):
#         ob, reward, done, info = self.env.step(action)
#         self.frames.append(ob)#.transpose(2,0,1))
#         return self._get_ob(), reward, done, info
# 
#     def _get_ob(self):
#         assert len(self.frames) == self.k
        # return LazyFrames(list(self.frames))
        
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = collections.deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(k * shp[2], shp[0], shp[1]), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset().transpose(2,0,1)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob.transpose(2,0,1))
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

# Taken from RL Adventure's [10] and OpenAI's Baselines [11] Repos.
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]