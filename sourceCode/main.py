from R2D4 import R2D4
from model.model import DistributionalIQNDuelingLSTMNet
from model.replayBuffer import PrioritizedReplayBuffer, ReplayBuffer
from common.buildEnv import build_multiprocessing_env, build_singlecore_env
import os
import numpy as np
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def main():
    # Set the tag for saving and TensorBoard.
    tag                                     = "R2D4+"   
    print(tag)
    # Whether to restart training from a checkpoint located at training/{tag}.
    restart                                 = False
    train                                   = True
    eval_only                               = False
    interactive                             = False

    if not os.path.exists("training"):
        os.makedirs("training")

    # Setup TensorBoard to write to runs/{tag}.
    writer                                  = SummaryWriter("runstrue//{}".format(tag))

    use_cuda                                = torch.cuda.is_available()
    print(use_cuda)
    device                                  = torch.device("cuda" if use_cuda else "cpu")

    # Whether to setup multiprocessing environments or just a single environment.
    multi                                   = True
    # Must be divisible by (cpu count // 2)
    nenvs                                   = 256

    # Whether to pool over the skipped frames (repeated actions).
    max_pool                                = True
    frame_skip                              = 3
    # Whether to use the custom rewards wrapper
    custom_rewards                          = True
    force_death                             = True
    
    # What type of epsilon exploration to use.
    eps_mode                                = "observe"
    eps_param                               = 0.10
    
    noisy                                   = True
    noisy_std                               = 0.4
    
    dueling                                 = True
    
    distributional                          = False
    num_tau                                 = 32
    num_tau_dash                            = 32
    
    double_q                                = True
    
    # Epsilon value used in value function scaling and rescaling (taken from R2D2).
    vfEpsilon                               = 0.001
    value_function_rescaling                = True

    # Number of steps used for n-step Bellman rewards and targets.
    n_steps                                 = 5
    n_step_returns                          = True

    # Discount factor for future rewards.
    gamma                                   = 0.997
    use_lstm                                = True
    
    PER                                     = True
    add_progression_bias                    = True
    
    gradient_clipping                       = False
    actions_and_rewards                     = True
    
    dropout                                 = False
    conv_dropout                            = 0.1
    linear_dropout                          = 0.4
    
    deeper                                  = False
    skip                                    = True

    # How many epochs to train for (including buffer warmup).
    n_epochs                                = int(1e32)

    # Adam optimizer hyperparameters.
    learning_rate                           = 0.0001
    adam_eps                                = 0.001

    # Sync rate between target and online networks for Double Q-Learning.
    sync_target_every                       = 300

    # Batch size of length seq_len_with_burn_in sequences of transitions to use for the networks and Replay Buffer.
    batch_size                              = 64

    # Hyperparameters used by the Prioritized Replay Buffer
    eta                                     = 0.9  # Weighting of max (eta) and mean (1-eta) of TD errors when calculating the priority of a batch post sampling to update priorities within buffer.
    priority_exponent                       = 0.9  # alpha
    importance_sampling_exponent_0          = 0.4  # beta
    buffer_limit                            = 18000     # Size of buffer.
    minimum_buffer_training_size            = 3000


    # Lengths used for storing sequences of experience into the Replay Buffer (See R2D2 for details).
    seq_len                                 = 80 + (n_steps - 1 if n_step_returns else 0)
    l_burnin                                = 40
    seq_len_with_burn_in                    = seq_len + l_burnin
    overlap                                 = int((seq_len - (n_steps - 1 if n_step_returns else 0))/ 2) # 40
    seq_len_with_burn_in_minus_overlap      = seq_len_with_burn_in - overlap

    # Minimum length of sequence to submit to the buffer (otherwise will be discarded). This is a personal contribution.
    minimumLen                              = 10

    # For handling statistics and videos.
    video_every                             = 10
    print_every                             = 500

    # Random seed.
    seed                                    = 101

    # Setup Reproducible Environment and Action Spaces.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Make environment
    env_name = 'SuperMarioBros'
    # worlds_plus_stages = ["1-1"]
    # worlds_plus_stages = ["8-4"]
    # worlds_plus_stages = ["5-1","5-2","5-3","5-4","2-1","2-3","2-4","3-1"]
    # worlds_plus_stages = ["1-1","1-2","1-3","1-4","2-1","2-2","2-3","2-4","3-1", "3-2","3-3","3-4", "4-1","4-2","4-3","4-4", "5-1","5-2","5-3","5-4", "6-1","6-2","6-3","6-4", "7-1","7-2","7-3","7-4", "8-1","8-2","8-3","8-4"]
    # worlds_plus_stages = ["1-2","1-3","1-4","2-2","2-3","2-4","3-1","3-3","3-4", "4-1","4-2","4-3","4-4", "5-1","5-2","5-3","5-4", "6-1","6-2","6-3","7-1","7-2","7-3", "8-1","8-2","8-3"]
    # worlds_plus_stages = ["7-1","1-2","1-3","1-4","2-1","2-3","2-4","3-1", "3-2","3-3","3-4", "4-1","4-2","4-3","4-4", "5-1"]
    # worlds_plus_stages = ["3-1","3-2","3-3","3-4"]
    worlds_plus_stages = ["1-1","1-2","1-3","1-4"]
    # worlds_plus_stages = ["5-1","5-2","5-3","5-4"]
    # assert nenvs % len(worlds_plus_stages) == 0
    env_gfx = '-v1'
    repeat_envs = nenvs // len(worlds_plus_stages)
    env_ids = [env_name + "-" + w_plus_s + env_gfx for w_plus_s in worlds_plus_stages for _ in range(repeat_envs)]
    # Change += args if no of distinct envs doesnt divide nenvs and also change the equivalent 0s in R2D4.py
    if nenvs % len(worlds_plus_stages) != 0:
        env_ids += [env_name + "-" + w_plus_s + env_gfx for w_plus_s in ["1-3","1-4","2-2","2-3","2-4","3-1","3-3","3-4", "4-1","4-2","4-3","4-4", "5-1","5-2","5-3","5-4", "6-1","6-2","6-3","7-1","7-2","7-3"]]
    assert len(env_ids) == nenvs
    MOVEMENT = SIMPLE_MOVEMENT

    if multi:
        env = build_multiprocessing_env(env_ids, nenvs, seed, MOVEMENT, max_pool=max_pool, frame_skip=frame_skip, custom_rewards=custom_rewards, force_death=force_death)
    else:
        env = build_singlecore_env(env_name+env_gfx, seed, MOVEMENT, video_every, tag, max_pool=max_pool, frame_skip=frame_skip, custom_rewards=custom_rewards, force_death=force_death)

    num_actions = env.action_space.n

    # Pack up hyper parameters:
    hyperParams = {
        "multi"                               : multi,
        "device"                              : device,
        "eps_mode"                            : eps_mode,
        "eps_param"                           : eps_param,
        "max_pool"                            : max_pool,
        "frame_skip"                          : frame_skip,
        "custom_rewards"                      : custom_rewards,
        "distributional"                      : distributional,
        "num_tau"                             : num_tau,
        "num_tau_dash"                        : num_tau_dash,
        "dueling"                             : dueling,
        "noisy"                               : noisy,
        "PER"                                 : PER,
        "double_q"                            : double_q,
        "use_lstm"                            : use_lstm,
        "value_function_rescaling"            : value_function_rescaling,
        "n_steps"                             : n_steps,
        "n_step_returns"                      : n_step_returns,
        "gradient_clipping"                   : gradient_clipping,
        "actions_and_rewards"                 : actions_and_rewards,
        "nenvs"                               : nenvs,
        "repeat_envs"                         : repeat_envs,
        "n_epochs"                            : n_epochs,
        "learning_rate"                       : learning_rate,
        "adam_eps"                            : adam_eps,
        "sync_target_every"                   : sync_target_every,
        "batch_size"                          : batch_size,
        "eta"                                 : eta,
        "priority_exponent"                   : priority_exponent,
        "importance_sampling_exponent_0"      : importance_sampling_exponent_0,
        "buffer_limit"                        : buffer_limit,
        "minimum_buffer_training_size"        : minimum_buffer_training_size,
        "vfEpsilon"                           : vfEpsilon,
        "gamma"                               : gamma,
        "seq_len"                             : seq_len,
        "l_burnin"                            : l_burnin,
        "seq_len_with_burn_in"                : seq_len_with_burn_in,
        "overlap"                             : overlap,
        "seq_len_with_burn_in_minus_overlap"  : seq_len_with_burn_in_minus_overlap,
        "minimumLen"                          : minimumLen,
        "video_every"                         : video_every,
        "print_every"                         : print_every,
        "seed"                                : seed,
        "num_actions"                         : num_actions,
        "movement"                            : MOVEMENT,
        "tag"                                 : tag
    } 

    # Setup Prioritized Replay Buffer.
    if PER:
        replayBuffer = PrioritizedReplayBuffer(buffer_limit, priority_exponent, importance_sampling_exponent_0, seq_len_with_burn_in, add_progression_bias, device)
    else:
        replayBuffer = ReplayBuffer(buffer_limit, seq_len_with_burn_in, device)

    # Q-Networks Setup for Double Q Learning.
    q = DistributionalIQNDuelingLSTMNet(device=device, num_action=num_actions, nenvs=nenvs, multi=multi, noisy=noisy, noisy_std=noisy_std, dueling=dueling, distributional=distributional, num_tau=num_tau, dropout=dropout, conv_dropout=conv_dropout, linear_dropout=linear_dropout, actions_and_rewards=actions_and_rewards, deeper=deeper, skip=skip, use_lstm=use_lstm)
    q_target = DistributionalIQNDuelingLSTMNet(device=device, num_action=num_actions, nenvs=nenvs, multi=multi, noisy=noisy, noisy_std=noisy_std, dueling=dueling, distributional=distributional, num_tau=num_tau_dash, dropout=False, conv_dropout=conv_dropout, linear_dropout=linear_dropout, actions_and_rewards=actions_and_rewards, deeper=deeper, skip=skip, use_lstm=use_lstm)

    # Setup Optimizer for Online Network Backpropagation.
    optimizer = optim.Adam(q.parameters(), lr=learning_rate, eps=adam_eps)

    # Initialize R2D4 (effectively the trainer and actor conductor).
    trainer = R2D4(q, q_target, optimizer, replayBuffer, env, writer, hyperParams)

    # Load model and optimizer from checkpoint.
    if restart:
        trainer.loadModel()

    # Sync Online and Target Q-Networks.
    trainer.syncNetworks()

    # If only wanting videos.
    if eval_only:
        # trainer.eval()
        trainer.eval("SuperMarioBros-1-1-v1", n_epochs=2, current_episode=0, tag=hyperParams["tag"], print_out=True, interactive_mode=interactive)
        trainer.eval("SuperMarioBros-1-2-v1", n_epochs=2, current_episode=0, tag=hyperParams["tag"], print_out=True, interactive_mode=interactive)
        trainer.eval("SuperMarioBros-1-3-v1", n_epochs=2, current_episode=0, tag=hyperParams["tag"], print_out=True, interactive_mode=interactive)
        trainer.eval("SuperMarioBros-2-1-v1", n_epochs=2, current_episode=0, tag=hyperParams["tag"], print_out=True, interactive_mode=interactive)
        trainer.eval("SuperMarioBros-3-1-v1", n_epochs=2, current_episode=0, tag=hyperParams["tag"], print_out=True, interactive_mode=interactive)
        trainer.eval("SuperMarioBros-4-1-v1", n_epochs=2, current_episode=0, tag=hyperParams["tag"], print_out=True, interactive_mode=interactive)

    # (Re)commence learning (acting and training).
    if train:
        trainer.run(n_epochs=n_epochs)
    
    # Save a final checkpoint.
    trainer.saveModel()

if __name__ == "__main__":
    with open(__file__) as f: print(f.read())
    main()
