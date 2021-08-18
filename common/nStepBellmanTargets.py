import torch

# Edited from Google's SEED RL [8] Repo.
def n_step_bellman_target(rewards, done_mask, q_target, gamma, n_steps):
    r"""Computes n-step Bellman targets.
    See section 2.3 of R2D2 [1] paper (which does not mention the logic around end of
    episode).
    Args:
        rewards: <float32>[time, batch_size] tensor. This is r_t in the equations
        below.
        done_mask: <bool>[time, batch_size] tensor. This is done_mask_t in the equations
        below. done_mask_t should be false if the episode is done just after
        experimenting reward r_t.
        q_target: <float32>[time, batch_size] tensor. This is Q_target(s_{t+1}, a*)
        (where a* is an action chosen by the caller).
        gamma: Exponential RL discounting.
        n_steps: The number of steps to look ahead for computing the Bellman
        targets.
    Returns:
        y_t targets as <float32>[time, batch_size] tensor.
        When n_steps=1, this is just:
        $$r_t + gamma * (1 - done_t) * Q_{target}(s_{t+1}, a^*)$$
        In the general case, this is:
        $$(\sum_{i=0}^{n-1} \gamma ^ {i} * notdone_{t, i-1} * r_{t + i}) +
        \gamma ^ n * notdone_{t, n-1} * Q_{target}(s_{t + n}, a^*) $$
        where notdone_{t,i} is defined as:
        $$notdone_{t,i} = \prod_{k=h0}^{k=i}(1 - done_mask_{t+k})$$
        The last n_step-1 targets cannot be computed with n_step returns, since we
        run out of Q_{target}(s_{t+n}). Instead, they will use n_steps-1, .., 1 step
        returns. For those last targets, the last Q_{target}(s_{t}, a^*) is re-used
        multiple times.
        However, in this implementation the last n_steps-1 are truncated in the
        training function and are not used.
        
    """
    # We append n_steps - 1 times the last q_target. They are divided by gamma **
    # k to correct for the fact that they are at a 'fake' indice, and will
    # therefore end up being multiplied back by gamma ** k in the loop below.
    # We prepend 0s that will be discarded at the first iteration below.
    bellman_target = torch.cat(
        [torch.zeros_like(q_target[0:1]), q_target] +
        [q_target[-1:] / gamma ** k
        for k in range(1, n_steps)],
        axis=0)
    # Pad with n_steps 0s. They will be used to compute the last n_steps-1
    # targets (having 0 values is important).
    done_mask = torch.cat([done_mask] + [torch.ones_like(done_mask[0:1])] * n_steps, axis=0)
    rewards = torch.cat([rewards] + [torch.zeros_like(rewards[0:1])] * n_steps,
                        axis=0)
    # Iteratively build the n_steps targets. After the i-th iteration (1-based),
    # bellman_target is effectively the i-step returns.
    for _ in range(n_steps):
        rewards = rewards[:-1]
        done_mask = done_mask[:-1]
        bellman_target = (
            rewards + gamma * done_mask * bellman_target[1:])

    return bellman_target