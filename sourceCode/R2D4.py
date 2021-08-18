from common.loss import calculate_huber_loss
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import one_hot
import numpy as np
import time
import random

from common.valueFunctionScalers import valueFunctionScaler
from common.buildEnv import build_singlecore_env
from common.nStepBellmanTargets import n_step_bellman_target

class R2D4():

    def __init__(self, q, q_target, optimizer, replayBuffer, env, writer, hyperParams) -> None:
        
        self.q = q
        self.q_target = q_target
        self.optimizer = optimizer
        self.replayBuffer = replayBuffer
        self.env = env
        self.writer = writer
        
        self.batch_size = hyperParams["batch_size"]
        self.minimumLen = hyperParams["minimumLen"]
        self.overlap = hyperParams["overlap"]
        self.minimum_buffer_training_size = hyperParams["minimum_buffer_training_size"]
        
        self.sync_target_every = hyperParams["sync_target_every"]
        self.print_every = hyperParams["print_every"]
        self.seq_len_with_burn_in_minus_overlap = hyperParams["seq_len_with_burn_in_minus_overlap"]
        self.seq_len_with_burn_in = hyperParams["seq_len_with_burn_in"]
        self.l_burnin = hyperParams["l_burnin"]
        self.eta = hyperParams["eta"]
        self.n_steps = hyperParams["n_steps"]
        self.gamma = hyperParams["gamma"]
        self.vfScaler = valueFunctionScaler(vfEpsilon=hyperParams["vfEpsilon"])
        self.device = hyperParams["device"]
        self.num_actions = hyperParams["num_actions"]
        self.actions_and_rewards = hyperParams["actions_and_rewards"]
        self.nenvs = hyperParams["nenvs"]
        self.repeat_envs = hyperParams["repeat_envs"]
        self.hyperParams = hyperParams
        self.eps_mode = hyperParams["eps_mode"]
        self.noisy = hyperParams["noisy"]
        self.distributional = hyperParams["distributional"]
        self.num_tau = hyperParams["num_tau"]
        self.num_tau_dash = hyperParams["num_tau_dash"]
        self.gradient_clip = hyperParams["gradient_clipping"]
        self.importance_sampling_exponent_0 = hyperParams["importance_sampling_exponent_0"]
        self.PER = hyperParams["PER"]
        self.double_q = hyperParams["double_q"]
        self.value_function_rescaling = hyperParams["value_function_rescaling"]
        self.n_step_returns = hyperParams["n_step_returns"]

        temp_dictionary = {"A": "JUMP", "B": "RUN", "right": "RIGHT", "left": "LEFT"}
        temp_movement = [[hyperParams["movement"][i][j] if hyperParams["movement"][i][j] not in temp_dictionary else temp_dictionary[hyperParams["movement"][i][j]] for j in range(len(hyperParams["movement"][i]))] for i in range(len(hyperParams["movement"]))]
        self.action_dictionary = {i : " + ".join(temp_movement[i]) for i in range(len(temp_movement))}

        self.zeroFloatList = [0.0]
        self.zeroIntList   = [0]
        self.oneFloatList  = [1.0]
        
        self.eval_envs = {}
        self.eval_envs_s = {}
        self.firstRunSetup()

    def train(self, n_iter=1):
        # Switch both networks to training mode (enable dropout etc.).
        # self.q.train()
        # self.q_target.train()
        for _ in range(n_iter):
            (s,a,r,done_mask,hids), (weights, idxes)  = self.replayBuffer.sample(self.batch_size, self.importance_sampling_exponent)
            
            # WARNING: 
            # s also has last s_prime attached!
            # done_mask has 1.0 prepended once!

            # one_hot_a and r_scaled are only used if passing actions and rewards too (instead) like R2D2 does.
            one_hot_a = one_hot(a, num_classes=self.num_actions).squeeze()
            r_scaled = r/15.0

            # Handle LSTM burnin.
            if self.l_burnin:
                with torch.no_grad():
                    _, online_hid, _ = self.q(s[:self.l_burnin], one_hot_a[:self.l_burnin], r_scaled[:self.l_burnin], hids)
                    _, target_hid, _ = self.q_target(s[:self.l_burnin], one_hot_a[:self.l_burnin], r_scaled[:self.l_burnin], hids)
            else:
                online_hid = hids 
                target_hid = hids
            # s, r, done_mask, a = s[self.l_burnin:], r[self.l_burnin:], done_mask[self.l_burnin:], a[self.l_burnin:]
            s, one_hot_a, r, r_scaled, done_mask, a = s[self.l_burnin:], one_hot_a[self.l_burnin:], r[self.l_burnin:], r_scaled[self.l_burnin:],  done_mask[self.l_burnin:], a[self.l_burnin:]

            # Retrieve online network's Q-values.
            q_out, _, taus = self.q(s, one_hot_a, r_scaled, online_hid)

            # Get Q-value for maximum action.
            if self.distributional:
                q_a = q_out[:-1].gather(-1,
                                        a[1:].unsqueeze(-1).expand(-1, -1, self.num_tau, 1)
                                        )#* done_mask[:-1].unsqueeze(-1).expand(-1, -1, self.num_tau, 1)  # replay_q
            else:
                q_a = q_out[:-1].gather(-1, a[1:]) #* done_mask[:-1] # replay_q
            
            if self.n_step_returns:
                # Truncate last n_steps-1 Q-values where the n-step Bellman targets are "incomplete".
                q_a = q_a[:-self.n_steps+1]

            # Compute n-step Bellman targets.
            with torch.no_grad():
                if self.double_q:
                    # Double Q-Learning {
                    # This means using argmax from online Q-network, but corresponding Q-values from target network to increase stability
                    greedy_action = (
                                    q_out[1:] if not self.distributional else q_out[1:].mean(-2) # Take mean of quantiles
                                    ).max(-1)[1].unsqueeze(-1)
    
                    target_q_prime, _, _ = self.q_target(s, one_hot_a, r_scaled, target_hid)
                    target_q_prime = target_q_prime[1:]
                    if self.distributional:
                        max_q_prime = target_q_prime.gather(-1,
                                                            greedy_action.unsqueeze(-1).expand(-1, -1, self.num_tau_dash, 1)
                                                            ) * done_mask[1:].unsqueeze(-1).expand(-1, -1, self.num_tau_dash, 1)
                    else:
                        max_q_prime = target_q_prime.gather(-1, greedy_action) * done_mask[1:]
                    # }
                else:
                    greedy_action = None
                    target_q_prime, _, _ = self.q_target(s, one_hot_a, r_scaled, target_hid)
                    target_q_prime = target_q_prime[1:]
                    if self.distributional:
                        max_q_prime = target_q_prime.gather.max(-1) * done_mask[1:].unsqueeze(-1).expand(-1, -1, self.num_tau_dash, 1)
                    else:
                        max_q_prime = target_q_prime.max(-1, keepdim=True)[0] * done_mask[1:]

                if self.value_function_rescaling:
                    max_q_prime = self.vfScaler.inverse_value_function_rescaling(max_q_prime)
                else:
                    r = r.clip(-1, 1)
                
                if self.n_step_returns:
                    target = n_step_bellman_target(
                        r[1:] if not self.distributional else r[1:].unsqueeze(-1),
                        done_mask[1:] if not self.distributional else done_mask[1:].unsqueeze(-1),
                        max_q_prime,
                        self.gamma,
                        self.n_steps
                    )[:-self.n_steps+1]
                else:
                    if not self.distributional:
                        target = r[1:] + self.gamma * max_q_prime * done_mask[1:]
                    else:
                        target = r[1:].unsqueeze(-1) + self.gamma * max_q_prime * done_mask[1:].unsqueeze(-1)
                
                if self.value_function_rescaling:
                    target = self.vfScaler.value_function_rescaling(target)

            if self.distributional:
                # target = target.transpose()
                # [seq, batch, tau, tau]
                td_errors = (target.transpose(-2, -1).detach() - q_a).float()
                huber_l = calculate_huber_loss(td_errors, 1.0)
                # The minus 1 below is to account for the final s prime appended to the s array.
                quantil_l = abs(taus[:(-self.n_steps+1) - 1] -(td_errors.detach() < 0).float()) * huber_l / 1.0
                loss =  0.5 *  (quantil_l
                                .sum(dim=-2) # Sum over N_tau
                                .mean(dim=-1) # Mean over N'_tau
                                .sum(dim=0) # Sum over time (sequence)
                                )
                # abs_td_errors = torch.abs(td_errors.sum(dim=-2).mean(dim=-1, keepdim=True))
                abs_td_errors = torch.abs(td_errors).sum(dim=-2).mean(dim=-1, keepdim=True)
            else:
                # Calculate TD errors and batch-wise weighted loss.
                abs_td_errors = torch.abs(q_a - target.detach()).float()
                loss = 0.5 * abs_td_errors.square().sum(0)

            if self.PER:
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()

            # Perform an optimizer step using clipped gradients, as per Dueling Network Architectures for Deep Reinforcement Learning.
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip:
                clip_grad_norm_(self.q.parameters(), 10)
            self.optimizer.step()
            
            self.loss.append(loss.item())

            if self.PER:
                # Update Prioritized Replay Buffer Priorities.
                with torch.no_grad():
                    priorities = torch.max(abs_td_errors, axis=0).values * self.eta + torch.mean(abs_td_errors, axis=0) * (1 - self.eta)
                    self.replayBuffer.update_priorities(idxes, priorities)
            else:
                priorities = None
            
            if self.noisy:
                self.q.reset_noise()
                self.q_target.reset_noise()

            # Clean up memory
            del s,a,r,done_mask,hids, q_out, q_a, greedy_action, max_q_prime, target, abs_td_errors, priorities, loss, weights, idxes
        
        # self.q.eval()
        
    def run(self, n_epochs=int(1e32)):
        s = self.current_s
        a = self.current_a
        r = self.current_r
        new_buffer_pushes = 0
        nettime, steptime, proctime, modeltime, evaltime = 0, 0, 0, 0, 0
        if self.eps_mode == "no_eps":
            epsilon = 0
        else:
            if self.start_ep == 0:
                epsilon = 1
            else:
                # Change from 0 if number of distinct env types doesnt divide nenvs.
                if self.eps_mode == "apex":
                    epsilon = torch.Tensor([self.hyperParams["eps_param"] ** (1 + 7*(i/(self.repeat_envs - 1))) for _ in range((self.nenvs-0)//self.repeat_envs) for i in range(self.repeat_envs)]+[0.02]*0).to(self.device)
                elif self.eps_mode == "observe":
                    epsilon = torch.Tensor([1.5 * self.hyperParams["eps_param"] ** (((i) / (self.repeat_envs - 1)) + 3*(1 - ((i) / (self.repeat_envs - 1)))) for _ in range((self.nenvs-0)//self.repeat_envs) for i in range(self.repeat_envs)]+[0.02]*0).to(self.device)
                elif self.eps_mode == "linear":
                    epsilon = max(0.02, 0.50 - 0.01*(self.start_ep/2000), 0)             
        
        self.q.train()
        self.q_target.train()
        # s, _, _, _ = self.env.step(0)
        for n_episode in range(self.start_ep, self.start_ep + n_epochs):
            self.n_episode = n_episode
            self.importance_sampling_exponent = min(1, self.importance_sampling_exponent_0 + (1 - self.importance_sampling_exponent_0) / (25000 - 750) * (n_episode - 750))
            for step in range(1):
                t = time.time()
                with torch.no_grad():
                    # Retrieve action to take from online Q-network.
                    a, self.hid = self.q.act(torch.Tensor(s).to(self.device).contiguous().float()/255.0, one_hot(torch.tensor(a).view(1,self.nenvs), num_classes=self.num_actions).to(self.device), torch.tensor(r).view(1,self.nenvs,1).to(self.device)/15.0, self.hid, epsilon)
                nettime += time.time() - t
                
                t = time.time()
                # Step all environments.
                s_prime, r, done, info = self.env.step(a)
                s_prime = s_prime.transpose(0, 3, 1, 2)
                steptime += time.time() - t
                
                t = time.time()
                # Handle submitting finished episodes to buffer etc.. for each env number.
                for nenv in range(self.nenvs):
                    self.current_seq1_len[nenv] += 1
                    self.length_counter[nenv] += 1
                    done_mask = 0.0 if done[nenv] else 1.0
                    self.done_mask_list[nenv].append(done_mask)
                    self.s_list[nenv].append(s[nenv])
                    self.a_list[nenv].append(a[nenv])
                    self.r_list[nenv].append(r[nenv])
                    self.score[nenv] += r[nenv]
                    
                    # If done:
                    if not done_mask:
                        deltaEndSeq0 = self.seq_len_with_burn_in_minus_overlap - self.current_seq1_len[nenv]
                        if self.notFirstSeq[nenv]:
                            # Pad end of episode with zeros and dones and build seq0.
                            s0_list = self.s_list[nenv] + [self.s_list[nenv][-1]] * (deltaEndSeq0+1)
                            a0_list = self.a_list[nenv] + self.zeroIntList * deltaEndSeq0
                            r0_list = self.r_list[nenv] + self.zeroFloatList * deltaEndSeq0
                            done0_mask_list = self.done_mask_list[nenv] + self.zeroFloatList * deltaEndSeq0
                            
                            # NOTE: seq0 is always long enough if this is not the first sequence (chunk) of the episode.
                            # Submit longer sequence seq0 into buffer.
                            self.replayBuffer.push((s0_list,a0_list,r0_list, done0_mask_list, self.seq0_initial_episode_hid[nenv], self.initial_x_queue[nenv].pop(0)))
                            new_buffer_pushes += 1
                            
                            # If seq1 is long enough.
                            if self.current_seq1_len[nenv] > self.l_burnin + self.minimumLen:
                                # Pad end of episode with zeros and dones and build seq1.
                                s1_list = s0_list[self.overlap:] + [s0_list[-1]] * self.overlap
                                a1_list = a0_list[self.overlap:] + self.zeroIntList * self.overlap
                                r1_list = r0_list[self.overlap:] + self.zeroFloatList * self.overlap
                                done1_mask_list = done0_mask_list[self.overlap:] + self.zeroFloatList * self.overlap

                                # Submit shorter sequence seq1 into buffer.
                                self.replayBuffer.push((s1_list,a1_list,r1_list, done1_mask_list, self.seq1_initial_episode_hid[nenv], self.initial_x_queue[nenv].pop(0)))
                                new_buffer_pushes += 1
                        
                        # If this is the first sequence (chunk) of the episode.
                        elif self.current_seq1_len[nenv] > self.l_burnin + self.minimumLen:
                            deltaEndSeq1 = deltaEndSeq0 + self.overlap
                            # Pad end of episode with zeros and dones and build seq1.
                            s1_list = self.s_list[nenv] + [self.s_list[nenv][-1]] * (deltaEndSeq1+1)
                            a1_list = self.a_list[nenv] + self.zeroIntList * deltaEndSeq1
                            r1_list = self.r_list[nenv] + self.zeroFloatList * deltaEndSeq1
                            done1_mask_list = self.done_mask_list[nenv] + self.zeroFloatList * deltaEndSeq1

                            # Submit the only sequence into buffer.
                            self.replayBuffer.push((s1_list,a1_list,r1_list, done1_mask_list, self.seq1_initial_episode_hid[nenv], self.initial_x_queue[nenv].pop(0)))
                            new_buffer_pushes += 1
                        
                        # Handle end of episode resetting: {

                        # Get LSTM initial hidden state.
                        temp_hid = self.q.get_h0(1)
                        self.seq1_initial_episode_hid[nenv] = {
                            "h0" :
                            temp_hid["h0"].squeeze(0).to("cpu"),
                            "c0":
                            temp_hid["c0"].squeeze(0).to("cpu")
                        }
                        self.hid["h0"][:, nenv], self.hid["c0"][:, nenv] = self.seq1_initial_episode_hid[nenv]["h0"].to(self.device), self.seq1_initial_episode_hid[nenv]["c0"].to(self.device)
                        # Reset variables.
                        self.marking.append(self.score[nenv])
                        self.final_x.append(info[nenv]["x_pos"])
                        self.finishing_rate.append(1 if info[nenv]["flag_get"] else 0)
                        
                        self.score[nenv] = 0.
                        self.length_counter[nenv] = self.overlap - 1
                        self.initial_x_queue[nenv] = [0]
                        self.initial_hid_queue[nenv] = []
                        self.current_seq1_len[nenv] = self.overlap - 1
                        self.notFirstSeq[nenv] = False
                        # Set previous action to NOOP etc...
                        self.s_list[nenv] ,self.a_list[nenv] ,self.r_list[nenv] , self.done_mask_list[nenv] = [s_prime[nenv]]*(self.overlap-1), [0]*(self.overlap), [0.]*(self.overlap), [1.]*(self.overlap)
                        # }
                        continue

                    if self.length_counter[nenv] == self.overlap:
                        self.length_counter[nenv] = 0
                        self.initial_x_queue[nenv].append(info[nenv]["x_pos"])
                        self.initial_hid_queue[nenv].append({
                            "h0" :
                            self.hid["h0"].view(self.nenvs, -1)[nenv].unsqueeze(0).to("cpu"),
                            "c0":
                            self.hid["c0"].view(self.nenvs, -1)[nenv].unsqueeze(0).to("cpu")
                        })
                    
                    # When longer of 2 sequences (seq0) reaches correct length, submit to buffer and handle sequence switching.
                    if self.seq_len_with_burn_in_minus_overlap == self.current_seq1_len[nenv]: # If seq0len = seq_len_with_burn_in : handle switch over
                        # Unless seq0 does not yet exist.
                        if self.notFirstSeq[nenv]:
                            self.replayBuffer.push((self.s_list[nenv][:self.seq_len_with_burn_in] + [s_prime[nenv]],self.a_list[nenv][:self.seq_len_with_burn_in+1],self.r_list[nenv][:self.seq_len_with_burn_in+1], self.done_mask_list[nenv][:self.seq_len_with_burn_in+1], self.seq0_initial_episode_hid[nenv], self.initial_x_queue[nenv].pop(0)))
                            new_buffer_pushes += 1
                            self.s_list[nenv],self.a_list[nenv],self.r_list[nenv], self.done_mask_list[nenv] = self.s_list[nenv][self.overlap:], self.a_list[nenv][self.overlap:], self.r_list[nenv][self.overlap:],  self.done_mask_list[nenv][self.overlap:]

                        self.current_seq1_len[nenv] -= self.overlap
                        self.notFirstSeq[nenv] = True
                        self.seq0_initial_episode_hid[nenv] = self.seq1_initial_episode_hid[nenv]
                        # seq1_initial_episode_hid[nenv] = hid[nenv]
                        self.seq1_initial_episode_hid[nenv] = self.initial_hid_queue[nenv].pop(0)
                    
                
                proctime += time.time() - t
                # Once there is nothing left to do set current state as the previous next state ready for next iteration.
                s = s_prime

            # If the buffer is big enough, begin training at the end of each new episode.
            if len(self.replayBuffer) > self.minimum_buffer_training_size:  
                # Switch from fully random exploration once the buffer is big enough to train
                if isinstance(epsilon, int):
                    # Change from 0 if number of distinct env types doesnt divide nenvs.
                    if self.eps_mode == "apex":
                        epsilon = torch.Tensor([self.hyperParams["eps_param"] ** (1 + 7*(i/(self.repeat_envs - 1))) for _ in range((self.nenvs-0)//self.repeat_envs) for i in range(self.repeat_envs)]+[0.02]*0).to(self.device)
                    elif self.eps_mode == "observe":
                        epsilon = torch.Tensor([1.5 * self.hyperParams["eps_param"] ** (((i) / (self.repeat_envs - 1)) + 3*(1 - ((i) / (self.repeat_envs - 1)))) for _ in range((self.nenvs-0)//self.repeat_envs) for i in range(self.repeat_envs)]+[0.02]*0).to(self.device)
                elif self.eps_mode == "linear":
                    # 3 stage linear epsilon annealing:
                    if n_episode < 200000:
                        # Linear annealing from 50% to 2%, then constant from episode 100k -> 200k.
                        epsilon = max(0.02, 0.50 - 0.01*(n_episode/2000), 0) 
                    else:
                        # Linear annealing from 2% to 0% from episode 200k
                        epsilon = max(0,  0.02 - 0.02*(n_episode-200000)/200000)
                # else:
                #     # epsilon1 = max(0.002, 0.50 - 0.01*(n_episode/2000)) # linear annealing from 50% to 0.002%
                #     # epsilon2 = max(0.04, 0.50 - 0.01*(n_episode/200)) # linear annealing from 50% to 4%
                #     # epsilon3 = max(0.02, 0.50 - 0.01*(n_episode/400)) # linear annealing from 50% to 4%
                #     # epsilon4 = max(0.005, 0.30 - 0.01*(n_episode/2000)) # linear annealing from 50% to 4%
                #     # epsilon5 = max(0.06, 0.20 - 0.01*(n_episode/3000)) # linear annealing from 50% to 4%
                #     # epsilonGrid = [epsilon1, epsilon2, epsilon3, 0.4, epsilon4, 0.25, epsilon5, 0.15, 0.1, 0.08, 0.05, 0.03, 0.01]
                #     # epsilon = epsilonGrid[random.randrange(0, len(epsilonGrid))]
                t = time.time()
                self.train(1)
                modeltime += time.time() - t

            # Sync target and online Q-networks periodically.
            if n_episode%self.sync_target_every==0 and n_episode!=0:
                self.syncNetworks()

            # Save model checkpoint periodically.
            if n_episode%2000 == 0:
                self.saveModel(n_episode, overwrite=True)

            if n_episode%50 == 0 and len(self.replayBuffer) > self.minimum_buffer_training_size:
                # eval_score = 0
                self.eval_score.append(self.eval("SuperMarioBros-1-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"]))
                self.eval_score.append(self.eval("SuperMarioBros-2-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"]))
                self.eval_score.append(self.eval("SuperMarioBros-3-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"]))
                self.eval_score.append(self.eval("SuperMarioBros-6-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"]))
                
            if n_episode%2000 == 0 and len(self.replayBuffer) > self.minimum_buffer_training_size:
                t = time.time()
                # eval_score = 0
                # eval_score += self.eval("SuperMarioBros-1-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # eval_score += self.eval("SuperMarioBros-2-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # eval_score += self.eval("SuperMarioBros-3-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # eval_score += self.eval("SuperMarioBros-6-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-2-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-2-1-v2", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-2-1-v3", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-1-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-1-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-1-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-2-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-2-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-2-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-3-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-3-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-3-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-4-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-4-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-4-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-4-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-5-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-5-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-5-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-5-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-6-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-6-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-6-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-6-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-7-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-7-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-7-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-7-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-8-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-8-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                self.eval("SuperMarioBros-8-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-8-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])

                # self.eval("SuperMarioBros-8-4-v0", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-1-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-1-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-1-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-1-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-3-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-3-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-3-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-3-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-5-1-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-2-5-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-2-3-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-2-4-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # self.eval("SuperMarioBros-3-5-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                # if n_episode%3000 == 0:
                #     self.eval("SuperMarioBros-3-2-v1", n_epochs=1, current_episode=n_episode, tag=self.hyperParams["tag"])
                evaltime += time.time() - t

            
            # Handle statistics printing and empty cache.
            if n_episode%500 == 0 and n_episode!=0:
                self.writer.add_scalar("new episodes per training epoch", len(self.marking)/500 , n_episode)
                self.writer.add_scalar("new sequences per training epoch", new_buffer_pushes/500 , n_episode)
                self.writer.add_scalar("multienvironment efficiency", (steptime + nettime + proctime)/500/1/self.nenvs , n_episode)
                
                # TensorBoard integration.
                self.writer.add_scalar('self.score',
                                np.array(self.marking).mean(),
                                    n_episode)
                self.writer.add_scalar('self.eval_score',
                                np.array(self.eval_score).mean(),
                                    n_episode)
                self.writer.add_scalar('self.x',
                                np.array(self.final_x).mean(),
                                    n_episode)
                self.writer.add_scalar('self.finishing_rate',
                                np.array(self.finishing_rate).mean(),
                                    n_episode)
                self.writer.add_scalar('self.loss',
                                np.array(self.loss).mean(),
                                    n_episode)
                try:
                    if isinstance(epsilon, float):
                        self.writer.add_scalar("eps", epsilon, n_episode)
                    else:
                        self.writer.add_scalar("eps", epsilon.mean(), n_episode)
                except:
                    pass
                
                
                # print(nettime, steptime, proctime, modeltime, evaltime)
                # print(len(self.replayBuffer))
                
                self.loss = []
                self.marking = []
                self.eval_score = []
                self.final_x = []
                self.finishing_rate = []

                new_buffer_pushes = 0
                nettime, steptime, proctime, modeltime, evaltime = 0, 0, 0, 0, 0
                
                torch.cuda.empty_cache()

            # if n_episode%self.print_every==0 and n_episode!=0:
                # print("episode: {}, self.score: {:.1f}, epsilon: {:.2f}".format(n_episode, self.score, epsilon))

        # Once we have trained for an additional n_epochs (including buffer warmup epochs), update new start_ep.
        self.start_ep = self.start_ep + n_epochs
        self.current_s = s_prime
        self.current_a = a
        self.current_r = r

    def eval(self, env_id, n_epochs, current_episode, tag, print_out=False, interactive_mode=False):
        self.q.eval()
        if interactive_mode:
            n_epochs = 1
        if env_id not in self.eval_envs:
            self.eval_envs[env_id] = build_singlecore_env(env_id, self.hyperParams["seed"], self.hyperParams["movement"], video_every=1, tag=tag, max_pool=self.hyperParams["max_pool"], frame_skip=self.hyperParams["frame_skip"], custom_rewards=self.hyperParams["custom_rewards"])
            self.eval_envs_s[env_id] = self.eval_envs[env_id].reset()
        eval_env = self.eval_envs[env_id]
        avg_x = 0
        avg_finishing_rate = 0
        eval_score = 0.
        finishing_time = []
        for n_epoch in range(n_epochs):
            max_x = 0
            a = 0
            r = 0
            s = self.eval_envs_s[env_id]
            done = False
            hid = self.q.get_h0(1)
            if not interactive_mode:
                while not done:
                    with torch.no_grad():
                        # Retrieve action to take from online Q-network.
                        a, hid = self.q.act(torch.from_numpy(np.array(s)).to(self.device).contiguous().unsqueeze(0).float()/255.0, one_hot(torch.tensor(a).view(1,1), num_classes=self.num_actions).to(self.device), torch.tensor(r).view(1,1,1).to(self.device)/15.0, hid, multi=False)
                    
                    # Step environment.
                    s, r, done, info = eval_env.step(a)
                    eval_score += r
            else:
                while not done:
                    if eval_score < 50:
                        print("Calculating Q-values for the current state to determine best action...")
                    with torch.no_grad():
                        # Retrieve action to take from online Q-network.
                        a, hid = self.q.act(torch.from_numpy(np.array(s)).to(self.device).contiguous().unsqueeze(0).float()/255.0, one_hot(torch.tensor(a).view(1,1), num_classes=self.num_actions).to(self.device), torch.tensor(r).view(1,1,1).to(self.device)/15.0, hid, multi=False)
                    input("Best action to take is: {}".format(self.action_dictionary[a]))
                    # Step environment.
                    s, r, done, info = eval_env.step(a)
                    if eval_score < 50:
                        print("Rendering environment...\n")
                    eval_env.render()
                    eval_score += r
            avg_x += info["x_pos"]
            avg_finishing_rate += 1 if info["flag_get"] else 0
            if info["flag_get"]:
                finishing_time.append(info["time"])
            self.eval_envs_s[env_id] = eval_env.reset()
        avg_x /= n_epochs
        avg_finishing_rate /= n_epochs
        eval_score /= n_epochs
        if interactive_mode:
            print("\nEpisode finished with Mario {}".format({1:"alive :) !", 0:"dead :( ."}[avg_finishing_rate]))
            if avg_finishing_rate == 1:
                print("Mario manage to finish the level with {} seconds remaining!".format(info["time"]))
            else:
                print("Therefore Mario did not manage to finish the level.")
            print("\nScore: {}".format(eval_score))
            print("Final distance reached (x-coord): {}\n".format(avg_x))
        if print_out:
            print("eval-{}".format(env_id), eval_score)
            print("eval-lives-{}".format(env_id), avg_finishing_rate)
            print("eval-x-{}".format(env_id), avg_x, "\n")
        else:
            if len(finishing_time) > 0:
                self.writer.add_scalar("eval-time-{}".format(env_id), np.array(finishing_time).mean(), current_episode)
            self.writer.add_scalar("eval-{}".format(env_id), eval_score, current_episode)
            self.writer.add_scalar("eval-lives-{}".format(env_id), avg_finishing_rate, current_episode)
            self.writer.add_scalar("eval-x-{}".format(env_id), avg_x, current_episode)
        
        self.q.train()

        return eval_score

    def firstRunSetup(self):
        self.start_ep = 0

        _ = self.env.reset()
        # Perform 1 random action at beginning of episode.
        a = torch.randint(low=0, high=self.num_actions, size=(self.nenvs,)).tolist()
        s, r, done, _ = self.env.step(a)
        s = s.transpose(0, 3, 1, 2)
        self.current_s = s
        self.current_a = a
        self.current_r = r
        
        self.s_list,self.a_list,self.r_list, self.done_mask_list = [[s0]*self.overlap for s0 in s], [[a0]*self.overlap for a0 in a], [[r0]*self.overlap for r0 in r], [[done0]*self.overlap for done0 in done]
        self.initial_x_queue = [[0] for _ in range(self.nenvs)]
        self.length_counter = [self.overlap-1 for _ in range(self.nenvs)]

        # Get LSTM initial hidden state.
        self.hid = self.q.get_h0(self.nenvs)
        self.seq1_initial_episode_hid = [
                                    { 
                        "h0" :
                        self.hid["h0"].view(self.nenvs, -1)[nenv].unsqueeze(0).to("cpu"),
                        "c0":
                        self.hid["c0"].view(self.nenvs, -1)[nenv].unsqueeze(0).to("cpu")
                                    }                       
        for nenv in range(self.nenvs)]

        self.seq0_initial_episode_hid = [
                                    None                      
        for nenv in range(self.nenvs)]
        self.initial_hid_queue = [[] for _ in range(self.nenvs)]

        # Reset variables.
        self.loss = []
        self.score = [0. for _ in range(self.nenvs)]
        self.current_seq1_len = [self.overlap-1 for _ in range(self.nenvs)]
        self.notFirstSeq = [False for _ in range(self.nenvs)]
        self.marking = []
        self.eval_score = []
        self.final_x = []
        self.finishing_rate = []

    def syncNetworks(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def saveModel(self, n_episode, overwrite=True):
        if overwrite:
            torch.save({'q':self.q.state_dict(), 'optimizer':self.optimizer.state_dict(), 'n_episode':n_episode}, 'training/save{0}.chkpt'.format(self.hyperParams["tag"]))
        else:
            torch.save({'q':self.q.state_dict(), 'optimizer':self.optimizer.state_dict(), 'n_episode':n_episode}, 'training/save{0}-{1}.chkpt'.format(self.hyperParams["tag"], n_episode))

    def loadModel(self, fileName=None):
        if fileName == None:
            fileName = 'training/save{}.chkpt'.format(self.hyperParams["tag"])
        params = torch.load(fileName)
        self.q.load_state_dict(params['q'])
        self.optimizer.load_state_dict(params["optimizer"])
        self.start_ep = params["n_episode"]
