import torch
import numpy as np
import random

from common.replayBufferPrerequisites import SumSegmentTree, MinSegmentTree

# Edited from RL Adventure's [10] and OpenAI's Baselines [11] Repos.
class ReplayBuffer(object):
    def __init__(self, size, seq_len_with_burn_in, device):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.seq_len_with_burn_in = seq_len_with_burn_in
        self.device = device

    def __len__(self):
        return len(self._storage)

    def push(self, seq):
        if self._next_idx >= len(self._storage):
            self._storage.append(seq)
        else:
            self._storage[self._next_idx] = seq
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        s_lst = torch.from_numpy(np.array([[self._storage[idx][0][time] for idx in idxes] for time in range(self.seq_len_with_burn_in+1)])).to(self.device).float()/255.0 # .transpose(0,1,4,2,3)
        a_lst = torch.tensor([[[self._storage[idx][1][time]] for idx in idxes] for time in range(self.seq_len_with_burn_in+1)]).to(self.device)
        r_lst = torch.tensor([[[self._storage[idx][2][time]] for idx in idxes] for time in range(self.seq_len_with_burn_in+1)]).to(self.device)
        done_mask_lst = torch.tensor([[[self._storage[idx][3][time]] for idx in idxes] for time in range(self.seq_len_with_burn_in+1)]).to(self.device)
        h0, c0 = torch.cat([self._storage[idx][4]["h0"] for idx in idxes]).squeeze(1).unsqueeze(0).to(self.device), torch.cat([self._storage[idx][4]["c0"] for idx in idxes]).squeeze(1).unsqueeze(0).to(self.device)
        return s_lst, a_lst, r_lst, done_mask_lst, {"h0": h0, "c0": c0}

    def sample(self, batch_size, _):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes), (None, None)

# Edited from RL Adventure's [10] and OpenAI's Baselines [11] Repos.
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, importance_sampling_exponent, seq_len_with_burn_in, add_progression_bias, device):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, seq_len_with_burn_in, device)
        assert alpha > 0
        self.importance_sampling_exponent = importance_sampling_exponent
        self._alpha = alpha
        self.add_progression_bias = add_progression_bias
        self.m = 0
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super(PrioritizedReplayBuffer, self).push(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=None):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        if beta == None:
            beta = self.importance_sampling_exponent
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, device=self.device)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, [weights, idxes]

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            if self.add_progression_bias:
                progression_bias = 1 + self._storage[idx][5] / 4000
                self.m = max(self.m, progression_bias)
                priority = priority * progression_bias
            
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)