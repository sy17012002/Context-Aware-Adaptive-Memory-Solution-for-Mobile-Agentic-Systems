import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. THE RL AGENT BRAIN (Deep Q-Network)
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # State: [RAM_Usage, Oracle_Pred_1, Oracle_Pred_2, Oracle_Pred_3]
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 2. THE OS ENVIRONMENT (The "Game")
# ==========================================
class RLOsEnvironment:
    def __init__(self, max_ram=2560):
        self.max_ram = max_ram
        self.current_ram = 0
        self.cache = {}
        
        # App weights (Simulated MB)
        self.app_weights = {
            0: 400, 1: 100, 2: 150, 3: 350, 4: 300, 5: 200
        }
        self.num_apps = len(self.app_weights)
        
        # RL Tracking
        self.state_size = 4 # [RAM_ratio, Pred1, Pred2, Pred3]
        self.action_size = 3 # 0: Keep, 1: Compress, 2: Kill

    def reset(self):
        self.current_ram = 0
        self.cache = {}
        return self._get_state([0, 0, 0])

    def _get_state(self, oracle_preds):
        ram_ratio = self.current_ram / self.max_ram
        state = [ram_ratio] + oracle_preds
        return np.array(state, dtype=np.float32)

    def step(self, action, target_app, oracle_preds):
        reward = 0
        done = False
        
        # 1. Process Agent's Action on the Oldest App
        oldest_app = list(self.cache.keys())[0] if self.cache else None
        
        if oldest_app is not None:
            if action == 1: # COMPRESS
                if not self.cache[oldest_app].get('compressed', False):
                    freed_ram = self.app_weights[oldest_app] / 2
                    self.current_ram -= freed_ram
                    self.cache[oldest_app]['compressed'] = True
            elif action == 2: # KILL
                freed_ram = self.cache[oldest_app]['weight']
                self.current_ram -= freed_ram
                del self.cache[oldest_app]
            # If action == 0 (KEEP), do nothing.

        # 2. User Requests New App
        app_weight = self.app_weights.get(target_app, 200)

        if target_app in self.cache:
            # CACHE HIT! Massive Reward as per project requirements
            reward += 1
            # Uncompress if it was compressed when opened
            if self.cache[target_app].get('compressed', False):
                self.current_ram += (app_weight / 2)
                self.cache[target_app]['compressed'] = False
            
            # Move to back (most recently used)
            val = self.cache.pop(target_app)
            self.cache[target_app] = val
        else:
            # CACHE MISS
            self.current_ram += app_weight
            self.cache[target_app] = {'weight': app_weight, 'compressed': False}

        # 3. Check for Thrashing (RAM Overflow)
        if self.current_ram > self.max_ram:
            reward -= 10 # THRASHING PENALTY
            done = True # Episode fails
            
        next_state = self._get_state(oracle_preds)
        return next_state, reward, done

# ==========================================
# 3. RL TRAINING LOOP
# ==========================================
def train_rl_agent():
    env = RLOsEnvironment()
    agent = DQN(env.state_size, env.action_size)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    memory = deque(maxlen=2000)
    gamma = 0.95    # Discount rate
    epsilon = 1.0   # Exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32

    episodes = 500
    print("🎮 Starting Deep Q-Network Training (Phase 2, Step 4)...")
    
    for e in range(episodes):
        # Simulate a fake "Oracle" prediction for the sake of RL training
        fake_oracle_preds = [random.randint(0,5), random.randint(0,5), random.randint(0,5)]
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        
        total_reward = 0
        
        # Simulate a user opening 50 apps in a row
        for time in range(50):
            # Agent decides: Keep (0), Compress (1), or Kill (2)
            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_size)
            else:
                with torch.no_grad():
                    q_values = agent(state)
                    action = np.argmax(q_values[0].numpy())
            
            # Simulate the user requesting an app
            target_app = random.randint(0, 5)
            next_oracle_preds = [random.randint(0,5), random.randint(0,5), random.randint(0,5)]
            
            # Step the environment
            next_state_np, reward, done = env.step(action, target_app, next_oracle_preds)
            next_state = torch.FloatTensor(next_state_np).unsqueeze(0)
            total_reward += reward
            
            # Save to replay buffer
            memory.append((state, action, reward, next_state, done))
            state = next_state
            
            if done:
                break
                
        # Experience Replay (Learning from past mistakes)
        # Experience Replay (Learning from past mistakes)
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
                
                # 1. Calculate target in pure PyTorch (using .item() to get a clean float)
                target = float(m_reward)
                if not m_done:
                    next_q_values = agent(m_next_state).detach()
                    target = float(m_reward + gamma * torch.max(next_q_values).item())
                
                # 2. Get current predictions
                current_q_values = agent(m_state)
                
                # 3. Clone and detach to create a safe target tensor
                target_f = current_q_values.clone().detach()
                target_f[0][m_action] = target
                
                # 4. Backpropagation
                optimizer.zero_grad()
                loss = criterion(current_q_values, target_f)
                loss.backward()
                optimizer.step()
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if (e+1) % 50 == 0:
            print(f"Episode: {e+1}/{episodes} | Total Reward: {total_reward} | Epsilon: {epsilon:.2f} | RAM Crashes: {'Yes' if done else 'No'}")

    print("\n✅ RL Agent Trained Successfully!")
    torch.save(agent.state_dict(), "rl_memory_agent.pth")
    print("🧠 Brain saved as 'rl_memory_agent.pth'")

if __name__ == "__main__":
    train_rl_agent()