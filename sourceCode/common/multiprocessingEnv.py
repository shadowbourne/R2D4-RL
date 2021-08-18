import torch.multiprocessing as multiprocessing
import os
import numpy as np
from gym import spaces
# from collections import OrderedDict
# import ctypes

# # All of this section is adapted from the OpenAI Baselines Repo.

# _NP_TO_CT = {np.float32: ctypes.c_float,
#              np.int32: ctypes.c_int32,
#              np.int8: ctypes.c_int8,
#              np.uint8: ctypes.c_char,
#              np.bool: ctypes.c_bool}
             
def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = multiprocessing.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            # with clear_mpi_env_vars():
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews).astype(np.float32), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    # @abstractmethod
    def reset(self):
        pass

    # @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)

class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

# class VecRepeatAction(VecEnvWrapper):
#     def __init__(self, venv, n):
#         self.venv = venv
#         self.repeatFor = n
#         wos = venv.observation_space  # wrapped ob space
#         low = np.repeat(wos.low, self.nstack, axis=-1)
#         high = np.repeat(wos.high, self.nstack, axis=-1)
#         self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
#         observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
#         for (i, new) in enumerate(news):
#             if new:
#                 self.stackedobs[i] = 0
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs, rews, news, infos

#     def reset(self):
#         obs = self.venv.reset()
#         self.stackedobs[...] = 0
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]

def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

# class ShmemVecEnv(VecEnv):
#     """
#     Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
#     """

#     def __init__(self, env_fns, spaces=None, context='spawn'):
#         """
#         If you don't specify observation_space, we'll have to create a dummy
#         environment to get it.
#         """
#         ctx = multiprocessing.get_context(context)
#         if spaces:
#             observation_space, action_space = spaces
#         else:
#             print('Creating dummy env object to get spaces')
#             dummy = env_fns[0]()
#             observation_space, action_space = dummy.observation_space, dummy.action_space
#             dummy.close()
#             del dummy
#         VecEnv.__init__(self, len(env_fns), observation_space, action_space)
#         self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
#         self.obs_bufs = [
#             {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
#             for _ in env_fns]
#         self.parent_pipes = []
#         self.procs = []
#         for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
#             wrapped_fn = CloudpickleWrapper(env_fn)
#             parent_pipe, child_pipe = ctx.Pipe()
#             proc = ctx.Process(target=_subproc_worker,
#                         args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys))
#             proc.daemon = True
#             self.procs.append(proc)
#             self.parent_pipes.append(parent_pipe)
#             proc.start()
#             child_pipe.close()
#         self.waiting_step = False
#         self.viewer = None

#     def reset(self):
#         if self.waiting_step:
#             print('Called reset() while waiting for the step to complete')
#             self.step_wait()
#         for pipe in self.parent_pipes:
#             pipe.send(('reset', None))
#         return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])

#     def step_async(self, actions):
#         assert len(actions) == len(self.parent_pipes)
#         for pipe, act in zip(self.parent_pipes, actions):
#             pipe.send(('step', act))
#         self.waiting_step = True

#     def step_wait(self):
#         outs = [pipe.recv() for pipe in self.parent_pipes]
#         self.waiting_step = False
#         obs, rews, dones, infos = zip(*outs)
#         return self._decode_obses(obs), np.array(rews), np.array(dones), infos

#     def close_extras(self):
#         if self.waiting_step:
#             self.step_wait()
#         for pipe in self.parent_pipes:
#             pipe.send(('close', None))
#         for pipe in self.parent_pipes:
#             pipe.recv()
#             pipe.close()
#         for proc in self.procs:
#             proc.join()

#     def get_images(self, mode='human'):
#         for pipe in self.parent_pipes:
#             pipe.send(('render', None))
#         return [pipe.recv() for pipe in self.parent_pipes]

#     def _decode_obses(self, obs):
#         result = {}
#         for k in self.obs_keys:

#             bufs = [b[k] for b in self.obs_bufs]
#             o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
#             result[k] = np.array(o)
#         return dict_to_obs(result)


# def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys):
#     """
#     Control a single environment instance using IPC and
#     shared memory.
#     """
#     def _write_obs(maybe_dict_obs):
#         flatdict = obs_to_dict(maybe_dict_obs)
#         for k in keys:
#             dst = obs_bufs[k].get_obj()
#             dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
#             np.copyto(dst_np, flatdict[k])

#     env = env_fn_wrapper.x()
#     parent_pipe.close()
#     try:
#         while True:
#             cmd, data = pipe.recv()
#             if cmd == 'reset':
#                 pipe.send(_write_obs(env.reset()))
#             elif cmd == 'step':
#                 obs, reward, done, info = env.step(data)
#                 if done:
#                     obs = env.reset()
#                 pipe.send((_write_obs(obs), reward, done, info))
#             elif cmd == 'render':
#                 pipe.send(env.render(mode='rgb_array'))
#             elif cmd == 'close':
#                 pipe.send(None)
#                 break
#             else:
#                 raise RuntimeError('Got unrecognized cmd %s' % cmd)
#     except KeyboardInterrupt:
#         print('ShmemVecEnv worker: got KeyboardInterrupt')
#     finally:
#         env.close()

# """
# Helpers for dealing with vectorized environments.
# """


# def copy_obs_dict(obs):
#     """
#     Deep-copy an observation dict.
#     """
#     return {k: np.copy(v) for k, v in obs.items()}


# def dict_to_obs(obs_dict):
#     """
#     Convert an observation dict into a raw array if the
#     original observation space was not a Dict space.
#     """
#     if set(obs_dict.keys()) == {None}:
#         return obs_dict[None]
#     return obs_dict


# def obs_space_info(obs_space):
#     """
#     Get dict-structured information about a gym.Space.
#     Returns:
#       A tuple (keys, shapes, dtypes):
#         keys: a list of dict keys.
#         shapes: a dict mapping keys to shapes.
#         dtypes: a dict mapping keys to dtypes.
#     """
#     if isinstance(obs_space, spaces.Dict):
#         assert isinstance(obs_space.spaces, OrderedDict)
#         subspaces = obs_space.spaces
#     elif isinstance(obs_space, spaces.Tuple):
#         assert isinstance(obs_space.spaces, tuple)
#         subspaces = {i: obs_space.spaces[i] for i in range(len(obs_space.spaces))}
#     else:
#         subspaces = {None: obs_space}
#     keys = []
#     shapes = {}
#     dtypes = {}
#     for key, box in subspaces.items():
#         keys.append(key)
#         shapes[key] = box.shape
#         dtypes[key] = box.dtype
#     return keys, shapes, dtypes


# def obs_to_dict(obs):
#     """
#     Convert an observation into a dict.
#     """
#     if isinstance(obs, dict):
#         return obs
#     return {None: obs}
