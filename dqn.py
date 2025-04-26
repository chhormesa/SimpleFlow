import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import namedtuple # 値とフィールド名をペアで格納するnamedtupleを使用
import os
from statistics import mean
import pandas as pd

import traci
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

COMMON_CONFIG = {
    'MAX_STEPS': 1200,
    'NUM_EPISODES': 5000,
    'BATCH_SIZE': 20,
    'CAPACITY': 1000
}

SUMO_BINARY='sumo'
SUMO_PATH = 'sumo'

SUMO_CONFIG = {
    "SUMOCFG_PATH": f"./{SUMO_PATH}/1",
    "SUMO_FILE": "main.sumocfg",  # ← change this per traffic scenario
    "EASTBOUND_LANE_ID": "Node1_2_EB_0",
    "SOUTHBOUND_LANE_ID": "Node3_2_SB_0",
    "TRAFFIC_LIGHT_NODE": "Node2",
}

MODEL_CONFIGS = {
    1: {
    "STATE_SIZE": 5,               # e.g., [left_q, right_q, prev_left, prev_right, current_phase]
    "ACTION_SIZE": 2,              # 0: keep, 1: switch
    "NUM_EPISODES": 100,           # training runs
    "MAX_STEPS": 120000,            # total SUMO steps per episode (20 minutes at 0.1s step)
    "DECISION_STEP": 5,
    "LOST_TIME_STEPS": 5,         # 5 seconds = 1 steps (SUMO step-length = 0.1s)
    "EPSILON": 0.1,                # exploration rate for ε-greedy
    "GAMMA": 0.95,
    "STEP": 1,
    "EPSILON_TYPE": 'linear'
},
}

# Options for ε-greedy function
EPSILON_FUNCTIONS = {
    'quadratic': lambda episode, num_episodes: 1 - (episode**2/num_episodes**2),            
    'yoshizawa': lambda episode, num_episodes: 5.0 * 10**-5 * (episode - num_episodes)**2,  
    'linear': lambda episode, num_episodes: 1 - (episode/num_episodes)                      
}

MODELS_TO_RUN = [ 1 ] 

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DataSaver:
    def __init__(self, base_dir= r"C:\Users\khrti\OneDrive\ドキュメント\Project\Sumo\simpleflow\results" , model_path = None):
        self.base_dir = base_dir
        self.model_path = model_path if model_path is not None else ""
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        model_dir = os.path.join(self.base_dir, self.model_path)
        if os.path.exists(model_dir):
            self.output_dir = self._create_next_subfolder()
        else:
            self.output_dir = model_dir
            os.makedirs(self.output_dir)
        
    def _create_next_subfolder(self):
        base_name = self.model_path[:-1] 
        current_num = int(self.model_path[-1])  
        next_num = current_num + 1
        folder_name = f"{base_name}{next_num}_Attention"
        
        output_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(output_dir)
        return output_dir
    
    def save_config(self, config, model_number):
        config_filename = f"model_{model_number}_config.txt"
        config_path = os.path.join(self.output_dir, config_filename)
        
        with open(config_path, 'w') as f:
            import sys
            script_name = sys.argv[0]

            f.write(f"Configuration for Model {model_number}\n")
            f.write("="*50 + "\n\n")

            f.write("Execution Information:\n")
            f.write("-"*20 + "\n")
            f.write(f"Script filename: {script_name}\n\n")
            
            f.write("Basic Configuration:\n")
            f.write("-"*20 + "\n")
            for key, value in config.items():
                if key != 'NETWORK': 
                    f.write(f"{key}: {value}\n")

            if hasattr(self, 'network_structure'):
                f.write("\nNeural Network Architecture:\n")
                f.write("-"*20 + "\n")
                f.write(self.network_structure)
            
            f.write("\nNotes:\n")
            f.write("-"*20 + "\n")
            f.write("- DISTRIBUTION_TYPE: 'uniform' means constant inflow, 'poisson' means random inflow\n")
            f.write("- *_INFLOW: Number of vehicles entering per time step\n")
            f.write("- *_OUTFLOW: Maximum number of vehicles that can exit per time step\n")
            f.write("- GAMMA: Discount factor for future rewards\n")
            f.write("- MAX_STEPS: Number of time steps per episode\n")
            f.write("- NUM_EPISODES: Total number of training episodes\n")
            f.write("- BATCH_SIZE: Number of samples used for each training step\n")
            f.write("- CAPACITY: Size of the replay memory buffer\n")
            
            from datetime import datetime
            f.write(f"\nConfiguration saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def update_network_structure(self, model):
        import io
        output = io.StringIO()
        print(model, file=output)
        structure = output.getvalue()
        
        self.network_structure = (
            "Network Structure:\n"
            f"{structure}\n"
            "\nLayer Details:\n"
        )
        
        total_params = 0
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            self.network_structure += f"- {name}: {list(param.shape)} ({param_count:,} parameters)\n"
        
        self.network_structure += f"\nTotal Parameters: {total_params:,}"

    def save_plot(self, fig, filename):
        save_filename = f"{self.model_path}_{filename}"
        
        path = os.path.join(self.output_dir, f"{save_filename}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"saved plot as: {path}")

    def save_data(self, data, filename):
        save_filename = f"{self.model_path}_{filename}"
        
        path = os.path.join(self.output_dir, f"{save_filename}.csv")
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=True)
        else:
            pd.DataFrame(data).to_csv(path, index=True)
        print(f"saved data as: {path}")

    def save_model(self, model, filename):
        save_filename = f"{self.model_path}_{filename}"
        
        path = os.path.join(self.output_dir, f"{save_filename}.pth")
        torch.save(model.state_dict(), path)
        print(f"saved model as: {path}")


# Class for replay memory to enable mini-batch learning
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # Maximum length of the memory
        self.memory = []          # Variable to store experiences
        self.index = 0            # Index indicating where to save the next experience

    def push(self, state, action, state_next, reward):
        ''' Save a transition into memory '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # If memory is not full, append a new slot
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity  # Move the save index forward by one

    def sample(self, batch_size):
        ''' Randomly retrieve a batch of experiences '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        ''' Return the current length of the memory '''
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, n_in, n_mid1, n_mid2, n_mid3, n_mid4, n_out, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_mid1)
        self.fc2 = nn.Linear(n_mid1, n_mid2)
        self.fc3 = nn.Linear(n_mid2, n_mid3)
        self.fc4 = nn.Linear(n_mid3, n_mid4)
        
        # Dueling Network
        self.fc5_adv = nn.Linear(n_mid4, n_out)  # Advantage stream
        self.fc5_v = nn.Linear(n_mid4, 1)        # Value stream

        # Optional dropout layer
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):   # Not using dropout
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))

        adv = self.fc5_adv(h4)  # No ReLU for advantage output
        val = self.fc5_v(h4).expand(-1, adv.size(1))  # No ReLU for value output

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output
    
# Class that serves as the brain of the DQN agent
class Brain:
    def __init__(self, num_states, num_actions, config):
        self.num_actions = num_actions # Get the number of actions
        self.config = config
        self.memory = ReplayMemory(self.config['CAPACITY'])  # Create a memory object to store experiences

        # Dueling DQN architecture
        self.model = Net(num_states, 128, 128, 64, 64, num_actions)
        print(self.model)                       # Output the model structure
        # Optimizer setting (using Adam with a learning rate of 0.0001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.epsilon_func = EPSILON_FUNCTIONS[self.config.get('EPSILON_TYPE', 'linear')]

    def replay(self):
        '''Learn the neural network parameters using Experience Replay'''

        # 1. Check if memory is large enough (do nothing if memory size is smaller than mini-batch size)
        if len(self.memory) < self.config['BATCH_SIZE']:
            return

        # 2. Create a mini-batch
        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = Transition(*zip(*transitions)) # Convert from Numpy to Torch.Tensor
        
        state_batch = torch.cat(batch.state) # Reshape tensors
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # 3. Compute the target Q-values
        self.model.eval() # Switch to evaluation mode
        state_action_values = self.model(state_batch).gather(1, action_batch) # Extract Q-values for chosen actions

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state))) # Check if next state exists
        next_state_values = torch.zeros(self.config['BATCH_SIZE'])
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach() # Max Q-value for next state

        expected_state_action_values = reward_batch + self.config['GAMMA'] * next_state_values # Compute target Q-values using Q-learning formula

        # 4. Update the model parameters
        self.model.train() # Switch to training mode
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # Compute loss

        self.optimizer.zero_grad() # Reset gradients
        loss.backward() # Backpropagation
        self.optimizer.step()# Update parameters

    def decide_action(self, state, episode):
        '''Decide an action based on the current state'''
        epsilon = self.epsilon_func(episode, self.config['NUM_EPISODES'])
        if epsilon <= np.random.uniform(0, 1): # Choose the best action
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else: # Choose a random action
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action
    
    def get_q_values(self, state):
        '''Get Q-values for the current state'''
        self.model.eval()
        with torch.no_grad():
            return self.model(state).numpy()

# Agent class
class Agent:
    def __init__(self, num_states, num_actions, config):
        self.config = config  # Store configuration
        self.brain = Brain(num_states, num_actions, config) # Create the brain for the agent to decide actions

    def update_q_function(self):
        '''Update the Q-function'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''Decide an action'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''Save the transition to the memory object'''
        self.brain.memory.push(state, action, state_next, reward)
    
    def get_q_values(self, state):
        '''Get Q-values for the current state from the Brain class'''
        return self.brain.get_q_values(state)

# Environment class (modified from original)s
class SUMOEnvironment:
    def __init__(self, model_number, model_specific_config, state_size=5, action_size=2):
        self.step_count = 0
        self.next_decision_step = 0
        self.green_phases = [0, 2]             # Real green phases
        self.yellow_phase = 1                  # Yellow phase for all transitions
        self.current_phase_index = 0          # Index into self.green_phases, not direct phase value
        self.current_phase = self.green_phases[self.current_phase_index]

        self.model_number = model_number
        self.config = {
            **COMMON_CONFIG,         # Common settings
            **model_specific_config,  # Model-specific settings
            **SUMO_CONFIG
        }

        # Get lost time from config (default is 2 if not specified)
        self.lost_time_steps = self.config.get('LOST_TIME_STEPS', 1)

        # Set model path
        self.model_path = f'trained_model_{model_number}.pth'
        self.saving_model_path = f'trained_model_{model_number}'

        # Initialize DataSaver
        self.data_saver = DataSaver(model_path=self.saving_model_path)  
        self.data_saver.save_config(self.config, self.model_number)  # Save configuration info

        self.state_size = state_size  # Number of state features (e.g., recent history of left/right queues)
        self.action_size = action_size  # Number of actions
        self.agent = Agent(self.state_size, self.action_size, self.config) # Create agent that operates in the environment
        self.state = np.zeros(self.state_size)
        self.q_value_history = []  # List to store Q-value history
        
        # Update network structure information
        self.data_saver.update_network_structure(self.agent.brain.model)
        
        # Save configuration (including network structure)
        self.data_saver.save_config(self.config, self.model_number)

        # Get number of episodes from config
        NUM_EPISODES = self.config['NUM_EPISODES']
        self.episodes_to_plot = [0, 
                               NUM_EPISODES-5, 
                               NUM_EPISODES-4, 
                               NUM_EPISODES-3, 
                               NUM_EPISODES-2, 
                               NUM_EPISODES-1]
        self.action_results = []  # List to store action outcomes
        
        self.lost_time_remaining = 0
        self.current_lane_history = []
        self.an_episode_actions = []
        self.all_episode_action_results = []
        self.episode_current_lanes = []
        self.all_episode_total_delays = []  

        # Queue-related variables
        self.left_queue = 0
        self.right_queue = 0
        self.left_queue_history = []
        self.right_queue_history = []

        # For evaluating total delay time
        self.total_collected = 0
        self.total_delay_time = 0

        # Variable used in the reward function
        self.previous_total_queue = 0

        # Calculate theoretical maximum reward
        self.theoretical_rewards = 6826
        # self.theoretical_rewards = self.calculate_theoretical_rewards()
        print(f"Theoretical Maximum Reward: {self.theoretical_rewards}")
        print("---")

    def step(self, episode, current_step):
        state_array = self._get_state()
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
        q_values = self.agent.get_q_values(state_tensor)

        action = torch.tensor([[0]])  # default keep
        took_action = False

        # === Only decide if at decision step ===
        if self.step_count >= self.next_decision_step:
            action = self.agent.get_action(state_tensor, episode)
            took_action = True

            if action.item() == 1:  # SWITCH
                print("\n it is switch -> ", self.step_count, "\n")
                # Yellow phase
                traci.trafficlight.setPhase(self.config["TRAFFIC_LIGHT_NODE"], self.yellow_phase)
                for _ in range(self.lost_time_steps):
                    traci.simulationStep()
                    self.step_count += 1

                # Next green phase
                self.current_phase_index = (self.current_phase_index + 1) % len(self.green_phases)
                self.current_phase = self.green_phases[self.current_phase_index]
                traci.trafficlight.setPhase(self.config["TRAFFIC_LIGHT_NODE"], self.current_phase)

                self.next_decision_step = self.step_count + self.config["DECISION_STEP"]
            else:  # KEEP
                print("\n it is keep -> ", self.step_count, "\n")
                self.next_decision_step = self.step_count + self.config["DECISION_STEP"]

        # Continue normal step (either just stepped or after switching)
        traci.simulationStep()
        self.step_count += 1

        # Get next state and reward
        next_state_array = self._get_state()
        reward_value = -sum(next_state_array[:2])
        reward_tensor = torch.FloatTensor([reward_value])
        next_state_tensor = torch.FloatTensor(next_state_array).unsqueeze(0)

        # Only memorize if action was taken
        if took_action:
            self.agent.memorize(state_tensor, action, next_state_tensor, reward_tensor)
            self.agent.update_q_function()

        return reward_value, str(action.item()), {
            'q_values': q_values[0],
            'left_queue': next_state_array[0],
            'right_queue': next_state_array[1]
        }

    def _get_state(self):
        left_q = traci.lane.getLastStepHaltingNumber(self.config['EASTBOUND_LANE_ID'])
        right_q = traci.lane.getLastStepHaltingNumber(self.config['SOUTHBOUND_LANE_ID'])
        prev_left = self.left_queue if hasattr(self, 'left_queue') else 0
        prev_right = self.right_queue if hasattr(self, 'right_queue') else 0

        self.left_queue = left_q
        self.right_queue = right_q

        return [left_q, right_q, prev_left, prev_right, self.current_phase]

    def run(self):
        # Config
        NUM_EPISODES = self.config['NUM_EPISODES']
        MAX_STEPS = self.config['MAX_STEPS']

        # Result containers
        total_rewards = []
        self.q_value_history = []
        self.left_queue_history = []
        self.right_queue_history = []
        self.all_episode_action_results = []
        self.all_episode_total_delays = []
        self.all_episode_current_lanes = []

        for episode in range(NUM_EPISODES):
            # Start SUMO
            traci.start([
                SUMO_BINARY,
                "-c", f"{self.config["SUMOCFG_PATH"]}/{self.config["SUMO_FILE"]}",
                "--step-length", str(self.config.get("STEP", "0.1")),
                "--start",
                "--quit-on-end",
                "--delay", "0",
                "--lateral-resolution", "0",
                "--duration-log.statistics", 
                "--statistic-output", f"{self.config["SUMOCFG_PATH"]}/stats_summary_{episode}.xml"
            ])

            self.current_phase = 0
            self.last_switch_step = 0
            self.total_delay_time = 0
            self.left_queue = 0
            self.right_queue = 0

            episode_q_values = []
            episode_actions = []
            episode_left_queue = []
            episode_right_queue = []
            current_phase_record = []

            while True:
                # Step through simulation
                reward, action_result, info = self.step(episode, self.step_count)

                # Save episode info
                episode_q_values.append(info["q_values"])
                episode_actions.append(action_result)
                episode_left_queue.append(info["left_queue"])
                episode_right_queue.append(info["right_queue"])
                current_phase_record.append(self.current_phase)
                self.total_delay_time += -reward  # reward is negative total queue

                if traci.vehicle.getIDCount() == 0:
                    # No vehicles left, end episode
                    break

            # Close SUMO
            traci.close()

            # Save per-episode results
            total = 0
            for q_tensor in episode_q_values:
                q_values = q_tensor.tolist()  # list of Q-values (for both actions)
                total -= sum(q_values)

            total_rewards.append(total)
            self.q_value_history.append(episode_q_values)
            self.left_queue_history.append(episode_left_queue)
            self.right_queue_history.append(episode_right_queue)
            self.all_episode_action_results.append(episode_actions)
            self.all_episode_total_delays.append(self.total_delay_time)
            self.all_episode_current_lanes.append(current_phase_record)

            # Console log
            if episode % 10 == 9 or episode == 0 or episode >= (NUM_EPISODES - 10):
                print(f"Episode {episode}: Total Reward = {total_rewards[-1]:.2f}, Total Delay = {self.total_delay_time:.2f}")
                print(f"Action Results: {episode_actions[:20]} ...")
                print("---")

        print("Training finished.")

        # === Save Results ===
        self._save_training_results(total_rewards)

        return total_rewards, self.all_episode_action_results, [], self.all_episode_action_results, self.all_episode_total_delays

    def plot_q_values(self):
        '''Output a graph of the output Q value over all episodes and steps'''
        # Creating a plot
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111)

        # Collect all Q values ​​of all steps of all episodes into one array
        all_q_values = np.concatenate(self.q_value_history)
        
        # Calculate the total number of steps
        total_steps = len(all_q_values)
        
        # Generates x-axis values ​​(including episode and step information)
        x = np.arange(total_steps)
        episodes = x // self.config['MAX_STEPS']
        steps = x % self.config['MAX_STEPS']
        
        # Plot by behavior
        action_labels = ['0:Keep', '1:Switch']
        for i in range(2):
            ax.plot(x, all_q_values[:, i], label=f'Action {i}: {action_labels[i]}', alpha=0.7)
        
        ax.set_title('Q-values for All Episodes and Steps')
        ax.set_xlabel('Total Steps')
        ax.set_ylabel('Q-value')
        ax.legend()
        
        # Sets the tick to the first step of the specified episode.
        tick_episodes = [0, 49] + list(range(99, self.config['NUM_EPISODES'], 50))
        tick_locations = [ep * self.config['MAX_STEPS'] for ep in tick_episodes]
        
        ax.set_xticks(tick_locations)
        ax.set_xticklabels([f'Ep {ep}, Step 0' for ep in tick_episodes], rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig

    def plot_q_value(self):
        '''Output a graph of the output Q value transition in a specific episode'''
        fig = plt.figure(figsize=(20, 15))
        action_labels = ['0:Keep', '1:Switch']
        for i, episode in enumerate(self.episodes_to_plot):
            if episode < len(self.q_value_history):
                q_values = np.array(self.q_value_history[episode])
                ax = fig.add_subplot(len(self.episodes_to_plot), 1, i+1)
                steps = np.arange(len(q_values))
                for j in range(2):
                    ax.plot(steps, q_values[:, j], label=f'Action {j}: {action_labels[j]}')
                ax.set_title(f'Q-values for Episode {episode}')
                ax.set_xlabel('Step')
                ax.set_ylabel('Q-value')
                ax.legend()
        plt.tight_layout()
        return fig

    def plot_queue_lengths(self):
        '''Print graphs of queue transitions, selected lanes, and action timings for a given episode'''
        fig = plt.figure(figsize=(20, 15))
        queue_labels = ['queue (route_we)', 'queue (route_ns)']
        for i, episode in enumerate(self.episodes_to_plot):
            if episode < len(self.left_queue_history):
                ax = fig.add_subplot(len(self.episodes_to_plot), 1, i+1)
                queue_steps = range(len(self.left_queue_history[episode]))
                steps = range(len(self.left_queue_history[episode])-1)
                
                # Plot queue lengths
                ax.plot(queue_steps, self.left_queue_history[episode], label=queue_labels[0])
                ax.plot(queue_steps, self.right_queue_history[episode], label=queue_labels[1])
                
                # Fill selected lane
                selected_lane = self.all_episode_current_lanes[episode]
                for j in range(1, len(steps)+1):
                    if selected_lane[j] == 0:
                        ax.axvspan(j-1, j, facecolor='blue', alpha=0.1, linewidth=0)
                    else:
                        ax.axvspan(j-1, j, facecolor='red', alpha=0.1, linewidth=0)
                
                # Show 'S' and 'L' actions
                for step, action in enumerate(self.all_episode_action_results[episode]):
                    if action in ['S', 'L']:
                        ax.axvspan(step, step+1, facecolor='grey', alpha=0.6, linewidth=0)
                
                # Dummy plot for legend
                ax.plot([], [], color='blue', alpha=0.1, linewidth=10, label='route_we selected')
                ax.plot([], [], color='red', alpha=0.1, linewidth=10, label='route_ns selected')
                ax.plot([], [], color='grey', alpha=0.6, linewidth=10, label='Switch and Lost time')
                
                ax.set_title(f'Queue length and actions in episode {episode}')
                ax.set_xlabel('Step')
                ax.set_ylabel('Queue length / Current lane')
                ax.legend()
                ax.set_xlim(0, len(steps))
                ax.set_ylim(-0.5, max(max(self.left_queue_history[episode]), max(self.right_queue_history[episode])) + 0.5)
        plt.tight_layout()
        return fig

    def plot_total_delays(self):
        '''Output a graph of the total delay time'''
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111)
        
        ax.plot(self.all_episode_total_delays)
        ax.set_title('Total Delay Time per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Delay Time')
        
        plt.tight_layout()
        return fig

    def _save_training_results(self, total_rewards):
        rewards_df = pd.DataFrame({
            'episode': range(len(total_rewards)),
            'total_reward': total_rewards,
            'total_delay': self.all_episode_total_delays
        })
        self.data_saver.save_data(rewards_df, 'rewards')

        # Q-values
        q_values_data = []
        for episode, q_values in enumerate(self.q_value_history):
            episode_data = pd.DataFrame(q_values, columns=['Q_keep', 'Q_switch'])
            episode_data['episode'] = episode
            episode_data['step'] = range(len(q_values))
            q_values_data.append(episode_data)
        q_values_df = pd.concat(q_values_data, ignore_index=True)
        self.data_saver.save_data(q_values_df, 'q_values')

        # Queue
        queue_data = []
        for episode in range(len(self.left_queue_history)):
            episode_data = pd.DataFrame({
                'queue_left': self.left_queue_history[episode],
                'queue_right': self.right_queue_history[episode],
                'current_phase': self.all_episode_current_lanes[episode],
                'action': self.all_episode_action_results[episode],
                'episode': episode,
                'step': range(len(self.left_queue_history[episode]))
            })
            queue_data.append(episode_data)
        queue_df = pd.concat(queue_data, ignore_index=True)
        self.data_saver.save_data(queue_df, 'queue_history')

        # Plot
        fig_rewards = plt.figure(figsize=(20, 10))
        plt.plot(total_rewards)
        plt.title("Total Rewards over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        self.data_saver.save_plot(fig_rewards, 'learning_curve')

# Defined independently of the Environment class
def run_model(model_number):
        print(f"\nRunning Model {model_number}")
        print("=====================================")
        
        # Retrieve model configuration
        if model_number not in MODEL_CONFIGS:
            print(f"Error: No configuration found for model number {model_number}")
            raise SystemExit("Program terminated due to missing model configuration.")
        
        # Check if model path already exists
        config = MODEL_CONFIGS[model_number]
        
        # Initialize environment (if you want to change lost time, modify lost_time_steps=2 here)
        model_path = f'trained_model_{model_number}.pth'
        if os.path.exists(model_path):
            print(f"Error: Model already exists at {model_path}")
            print("Please delete the existing model file or specify a different path.")
            raise SystemExit("Program terminated to prevent overwriting existing model.")   
    
        env = SUMOEnvironment(model_number, config) #If you want to change the lost time, change it as follows: (lost_time_steps=2)
        rewards, all_actions, all_optimal_actions, all_action_results, total_delays = env.run()  

        # Save the trained model
        torch.save(env.agent.brain.model.state_dict(), model_path)
        print(f"Training completed and model {model_number} saved.")

        # Print average total delay time for the last few episodes
        print(f'Average of total delay time for the last 10 episodes = {np.mean(total_delays[-10:])}')
        print(f'Average of total delay time for the last 8 episodes = {np.mean(total_delays[-8:])}')
        print(f'Average of total delay time for the last 5 episodes = {np.mean(total_delays[-5:])}')
        print("=====================================\n")


if __name__ == "__main__":
    for model_num in MODELS_TO_RUN:
        run_model(model_num)