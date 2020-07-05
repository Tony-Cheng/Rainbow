# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

from model import DQN, DQN_ENS


class Agent():
    def __init__(self, args, env):
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(
            device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                state_dict = torch.load(args.model, map_location='cpu')
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        # Re-map state dict for old pretrained models
                        state_dict[new_key] = state_dict[old_key]
                        # Delete old keys for strict load_state_dict
                        del state_dict[old_key]
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    # High ε can reduce evaluation scores drastically
    def act_e_greedy(self, state, epsilon=0.001):
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(
            self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        # Log probabilities log p(s_t, ·; θonline)
        log_ps = self.online_net(states, log=True)
        # log p(s_t, a_t; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            # Calculate nth next state probabilities
            # Probabilities p(s_t+n, ·; θonline)
            pns = self.online_net(next_states)
            # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            dns = self.support.expand_as(pns) * pns
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.target_net.reset_noise()  # Sample new target net noise
            # Probabilities p(s_t+n, ·; θtarget)
            pns = self.target_net(next_states)
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = returns.unsqueeze(
                1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            # Clamp between supported values
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(
                1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a *
                                                             (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a *
                                                             (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m * log_ps_a, 1)
        self.online_net.zero_grad()
        # Backpropagate importance-weighted minibatch loss
        (weights * loss).mean().backward()
        # Clip gradients by L2 norm
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
        self.optimiser.step()

        # Update priorities of sampled transitions
        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()


def ens_entropy(qs):
    prob = 0
    for i in range(len(qs)):
        q = qs[i]
        prob += F.softmax(q, dim=1)
    prob /= len(qs)
    entropy = - torch.sum(prob * torch.log(prob + 1e-7), dim=1)
    return entropy


def ens_cond_entropy(qs):
    cond_entropy = 0
    for i in range(len(qs)):
        q = qs[i]
        prob = F.softmax(q, dim=1)
        cond_entropy += torch.sum(prob * torch.log(prob + 1e-7), dim=1)
    cond_entropy /= len(qs)
    return cond_entropy


def ens_BALD(qs):
    return ens_entropy(qs) + ens_cond_entropy(qs)


class EnsembleAgent(Agent):
    def __init__(self, args, env):
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(
            device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip

        self.online_net = DQN_ENS(
            args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                state_dict = torch.load(args.model, map_location='cpu')
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        # Re-map state dict for old pretrained models
                        state_dict[new_key] = state_dict[old_key]
                        # Delete old keys for strict load_state_dict
                        del state_dict[old_key]
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQN_ENS(
            args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

        self.use_BALD = args.use_BALD

    def learn(self, mem):
        samples = mem.sample(self.batch_size)
        idxs, states, actions, returns, next_states, nonterminals, weights = samples

        loss = 0

        for i in range(self.online_net.get_ens_size()):
            loss += self.loss_member(samples, self.online_net.get_model(i),
                                     self.target_net.get_model(i))

        self.online_net.zero_grad()
        # Backpropagate importance-weighted minibatch loss
        (weights * loss).mean().backward()
        # Clip gradients by L2 norm
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
        self.optimiser.step()

        # Update priorities of sampled transitions
        if self.use_BALD:
            with torch.no_grad():
                qs = [self.q_value_batch(states, self.online_net.get_model(
                    i)) for i in range(self.online_net.get_ens_size())]
            BALD_value = torch.abs(ens_BALD(qs) + 1)
            mem.update_priorities(idxs, BALD_value.detach().cpu().numpy())
        else:
            mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def loss_member(self, samples, online_net, target_net):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = samples

        # Calculate current state probabilities (online network noise already sampled)
        # Log probabilities log p(s_t, ·; θonline)
        log_ps = online_net(states, log=True)
        # log p(s_t, a_t; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            # Calculate nth next state probabilities
            # Probabilities p(s_t+n, ·; θonline)
            pns = online_net(next_states)
            # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            dns = self.support.expand_as(pns) * pns
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)
            target_net.reset_noise()  # Sample new target net noise
            # Probabilities p(s_t+n, ·; θtarget)
            pns = target_net(next_states)
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = returns.unsqueeze(
                1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            # Clamp between supported values
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(
                1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a *
                                                             (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a *
                                                             (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m * log_ps_a, 1)

        return loss

    def q_value_batch(self, states, online_net):
        with torch.no_grad():
            qs = online_net(states)
            return (qs * self.support.expand_as(qs)).sum(2)

    def update_target_net(self):
        for i in range(self.target_net.get_ens_size()):
            self.target_net.get_model(i).load_state_dict(
                self.online_net.get_model(i).state_dict())
