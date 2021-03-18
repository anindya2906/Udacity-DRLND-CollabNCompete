import random
import config
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .model import Actor, Critic
from .utils import OUNoise, concat, split, soft_update
from .storage import ReplayBuffer


class Agent():
    """DDPG Agent : Interacts with the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents):
        """Initialize a DDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int) : Number of agents
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents

        # Actor Networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(config.DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(config.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR)
        
        # Copy the weights of the local network to the target network to make them identical at the beginning.
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)

        # Critic Networks
        self.critic_local = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(config.DEVICE)
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(config.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC, weight_decay=config.WEIGHT_DECAY)
        
        # Copy the weights of the local network to the target network to make them identical at the beginning.
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)        

    def act(self, state, noise=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state: the state in which the agent is now
            noise (float): factor to scale the noise value
        """
        state = torch.from_numpy(state).float().to(config.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if config.ADD_OU_NOISE:
            action += self.noise.sample() * noise 
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()


class MaddpgAgent():
    """MADDPG Agent : Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize a MADDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        
        # Instantiate Multiple  Agent
        self.agents = [ Agent(state_size,action_size, random_seed, num_agents) 
                       for i in range(num_agents) ]
        
        # Instantiate Memory replay Buffer (shared between agents)
        self.memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, random_seed)
          
    def reset(self):
        """Reset all the agents"""
        for agent in self.agents:
            agent.reset()

    def act(self, states, noise):
        """Return action to perform for each agents (per policy)"""        
        return [ agent.act(state, noise) for agent, state in zip(self.agents, states) ]
                
    
    def step(self, states, actions, rewards, next_states, dones, current_episode):
        """ # Save experience in replay memory, and use random sample from buffer to learn
        
        Param
        =====
            states: states each agent is in
            actions: actions taken by each agents
            rewards: rewards received by each agent
            next_state: next state of each agent
            dones: if any of the agent has completed the env or lost the game
            current_episode: current episode number
        """
 
        self.memory.add(concat(states), 
                        concat(actions), 
                        rewards,
                        concat(next_states),
                        dones)

        if (len(self.memory) > config.BATCH_SIZE) and (current_episode % config.UPDATE_EVERY==0) :
            # Learn multiple times in each update
            for i in range(config.MULTIPLE_LEARN_PER_UPDATE):
                # Update Agent #0
                experiences = self.memory.sample()   
                self.learn(experiences, own_idx=0, other_idx=1)
                # Update Agent #1
                experiences = self.memory.sample()   
                self.learn(experiences, own_idx=1, other_idx=0)
                
    
    def learn(self, experiences, own_idx, other_idx, gamma=config.GAMMA):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(states) -> action
            critic_target(all_states, all_actions) -> Q-value

        This is done for only the agent with own index

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            own_idx (int) : index of the own agent to update in self.agents
            other_idx (int) : index of the other agent to update in self.agents
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
               
        # Filter out the agent OWN states, actions and next_states batch
        own_states =  split(self.state_size, own_idx, states)
        own_actions = split(self.action_size, own_idx, actions)
        own_next_states = split(self.state_size, own_idx, next_states) 
                
        # Filter out the OTHER agent states, actions and next_states batch
        other_states =  split(self.state_size, other_idx, states)
        other_actions = split(self.action_size, other_idx, actions)
        other_next_states = split(self.state_size, other_idx, next_states)
        
        # Concatenate both agent information (own agent first, other agent in second position)
        all_states=torch.cat((own_states, other_states), dim=1).to(config.DEVICE)
        all_actions=torch.cat((own_actions, other_actions), dim=1).to(config.DEVICE)
        all_next_states=torch.cat((own_next_states, other_next_states), dim=1).to(config.DEVICE)
   
        agent = self.agents[own_idx]
        
            
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models        
        all_next_actions = torch.cat((agent.actor_target(own_states), agent.actor_target(other_states)),
                                     dim =1).to(config.DEVICE) 
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)
        
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()),
                                     dim = 1).to(config.DEVICE)      
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
        
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()        
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        soft_update(agent.critic_local, agent.critic_target, config.TAU)
        soft_update(agent.actor_local, agent.actor_target, config.TAU)                   
    
    def save(self, model_path="."):
        """Save checkpoints for all Agents"""
        for idx, agent in enumerate(self.agents):
            actor_local_filename = model_path+'/actor_local_' + str(idx) + '.pt'
            actor_target_filename = model_path+'/actor_target_' + str(idx) + '.pt'
            critic_local_filename = model_path+'/critic_local_' + str(idx) + '.pt'           
            critic_target_filename = model_path+'/critic_target_' + str(idx) + '.pt'            
            torch.save(agent.actor_local.state_dict(), actor_local_filename) 
            torch.save(agent.critic_local.state_dict(), critic_local_filename)             
            torch.save(agent.actor_target.state_dict(), actor_target_filename) 
            torch.save(agent.critic_target.state_dict(), critic_target_filename)

    def load(self, device, model_path="."):
        """Load the trained models"""
        for idx, agent in enumerate(self.agents):
            actor_local_filename = model_path+'/actor_local_' + str(idx) + '.pt'
            actor_target_filename = model_path+'/actor_target_' + str(idx) + '.pt'
            critic_local_filename = model_path+'/critic_local_' + str(idx) + '.pt'           
            critic_target_filename = model_path+'/critic_target_' + str(idx) + '.pt'
            agent.actor_local.load_state_dict(torch.load(actor_local_filename, map_location=device))
            agent.actor_target.load_state_dict(torch.load(actor_target_filename, map_location=device))
            agent.critic_local.load_state_dict(torch.load(critic_local_filename, map_location=device))
            agent.critic_target.load_state_dict(torch.load(critic_target_filename, map_location=device))