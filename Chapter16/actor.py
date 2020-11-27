import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import DQNetworkImageSensor

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 50       # how often to update the network
ACTION_SIZE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor():
    def __init__(self):
        
        # Q-Network
        self.qnetwork_local = DQNetworkImageSensor().to(device)
        self.qnetwork_target = DQNetworkImageSensor().to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, 10)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        images, lidars, sensors = state['image'], state['lidar'], state['sensor']
        images = torch.from_numpy(images).float().unsqueeze(0).to(device)
        lidars = torch.from_numpy(lidars).float().unsqueeze(0).to(device)
        sensors = torch.from_numpy(sensors).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(images, lidar=lidars, sensor=sensors)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(9))
        # return action_values

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        images, lidars, sensors = states
        next_images, next_lidars, next_sensors = next_states
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_images, lidar=next_lidars, sensor=next_sensors).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # import pdb; pdb.set_trace()
        Q_expected = self.qnetwork_local(images, lidar=lidars, sensor=sensors).gather(1, actions.long())
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        images = torch.from_numpy(np.vstack([e.state['image'][None] for e in experiences if e is not None])).float().to(device)
        lidars = torch.from_numpy(np.vstack([e.state['lidar'][None] for e in experiences if e is not None])).float().to(device)
        sensors = torch.from_numpy(np.vstack([e.state['sensor'] for e in experiences if e is not None])).float().to(device)
        states = [images, lidars, sensors]
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_images = torch.from_numpy(np.vstack([e.next_state['image'][None] for e in experiences if e is not None])).float().to(device)
        next_lidars = torch.from_numpy(np.vstack([e.next_state['lidar'][None] for e in experiences if e is not None])).float().to(device)
        next_sensors = torch.from_numpy(np.vstack([e.next_state['sensor'] for e in experiences if e is not None])).float().to(device)
        next_states = [next_images, next_lidars, next_sensors]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
