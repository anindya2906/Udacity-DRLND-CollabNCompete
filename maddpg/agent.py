import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .model import Actor, Critic
from .storage import ReplayBuffer
from .utils import OUNoise, encode, decode, soft_update

# BUFFER_SIZE = int(1e6)  # replay buffer size
# BATCH_SIZE = 128        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 6e-2              # for soft update of target parameters
# LR_ACTOR = 1e-3         # learning rate of the actor
# LR_CRITIC = 1e-3        # learning rate of the critic
# WEIGHT_DECAY = 0        # L2 weight decay

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed,
                 device, lr_actor, lr_critic, weight_decay):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.device = device
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(self.device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size*num_agents, action_size*num_agents, random_seed).to(self.device)
        self.critic_target = Critic(
            state_size*num_agents, action_size*num_agents, random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(
        ), lr=self.lr_critic, weight_decay=self.weight_decay)

        # make both local and target network parameters same, tau=1.0 will do hard update
        # soft_update(self.actor_local, self.actor_target, tau=1.0)
        # soft_update(self.critic_local, self.critic_target, tau=1.0)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    # def soft_update(self, local_model, target_model, tau):
    #     """Soft update model parameters.
    #     θ_target = τ*θ_local + (1 - τ)*θ_target
    #     Params
    #     ======
    #         local_model: PyTorch model (weights will be copied from)
    #         target_model: PyTorch model (weights will be copied to)
    #         tau (float): interpolation parameter 
    #     """
    #     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #         target_param.data.copy_(
    #             tau*local_param.data + (1.0-tau)*target_param.data)

    def reset(self):
        self.noise.reset()

class MADDPGAgent():
    def __init__(self, state_size, action_size, num_agents, random_seed,
                    device, buffer_size, batch_size, gamma, tau, lr_actor, lr_critic, weight_decay, update_every):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agnets = num_agents
        self.seed = random.seed(random_seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.update_every = update_every
        self.time_step = 0

        self.agents = [Agent(state_size, action_size, num_agents, random_seed, device, lr_actor, lr_critic, weight_decay) for i in range(num_agents)]

        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed, self.device)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        actions = [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.time_step += 1
        self.memory.add(encode(states), encode(actions), rewards, encode(next_states), dones)
        if self.time_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                for i in range(3):
                    experiences = self.memory.sample()
                    self.learn(experiences, index=0, other_index=1, gamma=self.gamma)
                    experiences = self.memory.sample()
                    self.learn(experiences, index=1, other_index=0, gamma=self.gamma)

    def learn(self, experiences, index, other_index, gamma):
        states, actions, rewards, next_states, dones = experiences

        own_states = decode(states, self.state_size, index)
        own_actions = decode(actions, self.action_size, index)
        own_next_states = decode(next_states, self.state_size, index)

        other_states = decode(states, self.state_size, other_index)
        other_actions = decode(actions, self.action_size, other_index)
        other_next_states = decode(states, self.state_size, other_index)

        all_states = torch.cat((own_states, other_states), dim=1).to(self.device)
        all_actions = torch.cat((own_actions, other_actions), dim=1).to(self.device)
        all_next_states = torch.cat((own_next_states, other_next_states), dim=1).to(self.device)

        agent = self.agents[index]

        # Update Critic
        all_next_actions = torch.cat((agent.actor_target(own_states), agent.actor_target(other_states)), dim=1).to(self.device)
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Update Actor
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()), dim=1).to(self.device)
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # soft update
        soft_update(agent.critic_local, agent.critic_target, self.tau)
        soft_update(agent.actor_local, agent.actor_target, self.tau)
