import torch.multiprocessing as multiprocessing
import sys
import gym
import gym_super_mario_bros # DO NOT REMOVE THIS IMPORT IF PLAYING SUPER MARIO BROS
from nes_py.wrappers import JoypadSpace

from common.multiprocessingEnv import SubprocVecEnv, VecFrameStack
from common.wrappers import CustomReward, ForceDeath, MaxAndSkipEnv, WarpFrame, FrameStack, RepeatAction

def build_multiprocessing_env(env_ids, nenvs, seed, MOVEMENT, max_pool=True, frame_skip=4, custom_rewards=True, force_death=True):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    ncpu //= 2
    print(ncpu)
    assert nenvs % ncpu == 0
    in_series = nenvs // ncpu
    frame_stack_size = 4
    env = make_vec_env(env_ids, nenvs, seed, in_series, MOVEMENT, max_pool=max_pool, frame_skip=frame_skip, custom_rewards=custom_rewards, force_death=force_death)
    env = VecFrameStack(env, frame_stack_size)
    return env

# def make_vec_env(env_id, num_env, env_seed, in_series, MOVEMENT, max_pool=True, frame_skip=4, custom_rewards=True):
#     def make_thunk():
#         return lambda: make_env(env_id, env_seed, MOVEMENT, max_pool=max_pool, frame_skip=frame_skip, custom_rewards=custom_rewards)
#     # return SubprocVecEnv([make_thunk() for _ in range(num_env)])
#     return ShmemVecEnv([make_thunk() for _ in range(num_env)])
def make_vec_env(env_ids, num_env, env_seed, in_series, MOVEMENT, max_pool=True, frame_skip=4, custom_rewards=True, force_death=True):
    def make_thunk(env_id):
        return lambda: make_env(env_id, env_seed, MOVEMENT, max_pool=max_pool, frame_skip=frame_skip, custom_rewards=custom_rewards, force_death=force_death)
    # return SubprocVecEnv([make_thunk() for _ in range(num_env)])
    return SubprocVecEnv([make_thunk(env_id) for env_id in env_ids], in_series=in_series)

def make_env(env_id, env_seed, MOVEMENT, max_pool=True, frame_skip=4, custom_rewards=True, force_death=True):
    env = gym.make(env_id)
    env = JoypadSpace(env, MOVEMENT)
    if custom_rewards:
        env = CustomReward(env)
    # Repeat action in multienv setup is done before framestacking unfortunately due to the extreme inefficiency of the alternative.
    if max_pool:
        env = MaxAndSkipEnv(env, frame_skip)
    else:
        env = RepeatAction(env, frame_skip)
    if force_death:
        env = ForceDeath(env)
    
    env = WarpFrame(env)
    # env = AutoReset(env)
    env.seed(env_seed)
    env.action_space.seed(env_seed)
    return env

def build_singlecore_env(env_id, seed, MOVEMENT, video_every, tag, max_pool=True, frame_skip=4, custom_rewards=True, force_death=True):
    env = gym.make(env_id)
    env = JoypadSpace(env, MOVEMENT)

    # Enables video recording.
    if video_every:
        env = gym.wrappers.Monitor(env, "./video/{}/{}".format(tag, env_id), video_callable=lambda episode_id: (episode_id%video_every)==0, force=True, write_upon_reset=False)

    # Unlike R2D2's reported results [1], using episodic life (with roll-over LSTM states) resulted in worse performance, so this wrapper is disabled for our model (NOTE: requires tweaking in the acting loop to enable roll-over)
    # env = EpisodicLifeEnv(env)
    if custom_rewards:
        env = CustomReward(env)
    # Repeat each action for frame_skip frames.
    if max_pool:
        env = MaxAndSkipEnv(env, frame_skip)
    else:
        env = RepeatAction(env, frame_skip)

    if force_death:
        env = ForceDeath(env, is_monitor_enabled= True if video_every else False)
        
    # Resize frame to 84x84 and make grayscale.
    env = WarpFrame(env)

    # Makes env observation into last 4 frames instead.
    env = FrameStack(env, 4)

    

    # Setup Reproducible Environment and Action Spaces.
    env.seed(seed)
    env.action_space.seed(seed)
    return env
