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
MODEL_CONFIGS = {
    1: {
    "SUMOCFG_PATH": "./hh.sumocfg",  # ← change this per traffic scenario
    "STATE_SIZE": 5,               # e.g., [left_q, right_q, prev_left, prev_right, current_phase]
    "ACTION_SIZE": 2,              # 0: keep, 1: switch
    "NUM_EPISODES": 100,           # training runs
    "MAX_STEPS": 12000,            # total SUMO steps per episode (20 minutes at 0.1s step)
    "DECISION_STEP": 50,
    "LOST_TIME_STEPS": 50,         # 5 seconds = 50 steps (SUMO step-length = 0.1s)
    "EPSILON": 0.1,                # exploration rate for ε-greedy
    "GAMMA": 0.95,
    "EASTBOUND_LANE_ID": "Node1_2_EB_0",
    "SOUTHBOUND_LANE_ID": "Node3_2_SB_0",
    "TRAFFIC_LIGHT_NODE": "Node2",
    "STEP": "0.1"
},
}

MODELS_TO_RUN = [ 1 ] 

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DataSaver:
    def __init__(self, base_dir= r"C:\Users\khrti\OneDrive\ドキュメント\Project\Sumo\Mimic\results" , model_path = None):
        """
        Parameters:
        - base_dir: 基本となるディレクトリ名
        - model_path: モデルの基本名 (例: trained_model_4)
        """
        self.base_dir = base_dir
        self.model_path = model_path if model_path is not None else ""
        
        # 基本ディレクトリの作成
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # モデル名のフォルダが既に存在するかチェック
        model_dir = os.path.join(self.base_dir, self.model_path)
        if os.path.exists(model_dir):
            # 同名のフォルダが存在する場合、新しい番号のフォルダを作成
            self.output_dir = self._create_next_subfolder()
        else:
            # 存在しない場合、そのままのモデル名でフォルダを作成
            self.output_dir = model_dir
            os.makedirs(self.output_dir)
        
    def _create_next_subfolder(self):
        """同名のモデルが存在する場合、番号を増やしてAttentionを付加"""
        base_name = self.model_path[:-1]  # 'trained_GANmodel_' を取得
        current_num = int(self.model_path[-1])  # 現在の番号を取得
        next_num = current_num + 1
        folder_name = f"{base_name}{next_num}_Attention"
        
        # 新しいフォルダのパスを作成
        output_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(output_dir)
        return output_dir
    
    def save_config(self, config, model_number):
        """設定情報をテキストファイルとして保存"""
        config_filename = f"model_{model_number}_config.txt"
        config_path = os.path.join(self.output_dir, config_filename)
        
        with open(config_path, 'w') as f:
            # 実行ファイル名の取得と書き込み
            import sys
            script_name = sys.argv[0]  # 実行中のスクリプトのファイル名

            f.write(f"Configuration for Model {model_number}\n")
            f.write("="*50 + "\n\n")

            # 実行ファイル名を追加
            f.write("Execution Information:\n")
            f.write("-"*20 + "\n")
            f.write(f"Script filename: {script_name}\n\n")
            
            # 基本設定の書き込み
            f.write("Basic Configuration:\n")
            f.write("-"*20 + "\n")
            for key, value in config.items():
                if key != 'NETWORK':  # NETWORKは後で別途書き込む
                    f.write(f"{key}: {value}\n")
            
            # ネットワーク構造の詳細を書き込み
            if hasattr(self, 'network_structure'):
                f.write("\nNeural Network Architecture:\n")
                f.write("-"*20 + "\n")
                f.write(self.network_structure)
            
            # 追加の説明や注釈
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
            
            # 実行日時も記録
            from datetime import datetime
            f.write(f"\nConfiguration saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def update_network_structure(self, model):
        """ネットワーク構造の文字列を保存"""
        import io
        # 標準出力をキャプチャ
        output = io.StringIO()
        print(model, file=output)
        structure = output.getvalue()
        
        # より読みやすい形式に整形
        self.network_structure = (
            "Network Structure:\n"
            f"{structure}\n"
            "\nLayer Details:\n"
        )
        
        # 各層のパラメータ数を計算
        total_params = 0
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            self.network_structure += f"- {name}: {list(param.shape)} ({param_count:,} parameters)\n"
        
        self.network_structure += f"\nTotal Parameters: {total_params:,}"

    def save_plot(self, fig, filename):
        """プロットを画像として保存"""
        save_filename = f"{self.model_path}_{filename}"
        
        path = os.path.join(self.output_dir, f"{save_filename}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"saved plot as: {path}")

    def save_data(self, data, filename):
        """データをCSVとして保存"""
        save_filename = f"{self.model_path}_{filename}"
        
        path = os.path.join(self.output_dir, f"{save_filename}.csv")
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=True)
        else:
            pd.DataFrame(data).to_csv(path, index=True)
        print(f"saved data as: {path}")

    def save_model(self, model, filename):
        """モデルの保存"""
        save_filename = f"{self.model_path}_{filename}"
        
        path = os.path.join(self.output_dir, f"{save_filename}.pth")
        torch.save(model.state_dict(), path)
        print(f"saved model as: {path}")


# ミニバッチ学習を実現するために、経験を保存するメモリクラス
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # メモリの最大長さ
        self.memory = [] # 経験を保存する変数
        self.index = 0 # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        ''' transitionをメモリに保存する '''
        if len(self.memory) < self.capacity:
            self.memory.append(None) # メモリが満タンでないときは足す
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity # 保存するindexを1つずらす

    def sample(self, batch_size):
        ''' batch_size分だけランダムに保存内容を取り出す '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        ''' 関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, n_in, n_mid1, n_mid2, n_mid3, n_mid4,  n_out, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_mid1)
        self.fc2 = nn.Linear(n_mid1, n_mid2)
        self.fc3 = nn.Linear(n_mid2, n_mid3)
        self.fc4 = nn.Linear(n_mid3, n_mid4)
        # Dueling Network
        self.fc5_adv = nn.Linear(n_mid4, n_out)  # Advantage側
        self.fc5_v = nn.Linear(n_mid4, 1)  # 価値V側

        # Dropout層の追加
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):   # dropoutを使用しない
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))

        adv = self.fc5_adv(h4)  # この出力はReLUしない
        val = self.fc5_v(h4).expand(-1, adv.size(1))  # この出力はReLUしない

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output
    
# DQNエージェントの脳となるクラス
class Brain:
    def __init__(self, num_states, num_actions, config):
        self.num_actions = num_actions # 行動数を取得
        self.config = config
        self.memory = ReplayMemory(self.config['CAPACITY']) # 経験を記憶するメモリオブジェクトを生成

        # Dueling DQN アーキテクチャ
        self.model = Net(num_states, 128, 128, 64, 64, num_actions)
        print(self.model)                       # 上記のネットワークモデル設定内容を出力
        # 最適化手法の設定 (今回はadam、学習率は0.0001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)


    def replay(self):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # 1. メモリのサイズ確認(メモリサイズがミニバッチより小さい間は何もしない)
        if len(self.memory) < self.config['BATCH_SIZE']:
            return

        # 2. ミニバッチ作成
        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = Transition(*zip(*transitions)) # Numpy型からTorch.Tensor型へ変換
        
        state_batch = torch.cat(batch.state) # テンソルサイズの変換
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # 3. 教師信号となるQ値を求める
        self.model.eval() # 推論モードに切り替え
        state_action_values = self.model(state_batch).gather(1, action_batch) # 各行動に対応するQ値を特定

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state))) # 次の状態があるか確認
        next_state_values = torch.zeros(self.config['BATCH_SIZE'])
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach() # 次の状態があるindexの最大Q値を求める

        expected_state_action_values = reward_batch + self.config['GAMMA'] * next_state_values # Q学習の式で教師Q値を求める

        # 4. 結合パラメータの更新
        self.model.train() # 訓練モードに切り替え
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # 損失関数の計算

        self.optimizer.zero_grad() # 勾配リセット
        loss.backward() # バックプロバゲージョンを計算
        self.optimizer.step() # 結合パラメータの更新

    def decide_action(self, state, episode):
        ''' 現在の状態に応じて行動を決定する '''
        # epsilon = 5.0 * 10**-5 * (episode - self.config['NUM_EPISODES'])**2    # 吉澤修論  
        # epsilon = (1 - (episode**2/self.config['NUM_EPISODES'] ** 2))   # 二次関数低減
        epsilon = 1 - (episode/self.config['NUM_EPISODES'])
        if epsilon <= np.random.uniform(0, 1): # 最適行動を選択
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else: # ランダムに選択
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action
    
    def get_q_values(self, state):
        '''現在の状態に対するQ値を取得する'''
        self.model.eval()
        with torch.no_grad():
            return self.model(state).numpy()

# エージェントクラス
class Agent:
    def __init__(self, num_states, num_actions, config):
        self.config = config  # configを保存
        self.brain = Brain(num_states, num_actions, config) # エージェントが行動を決定するための頭脳生成

    def update_q_function(self):
        ''' Q関数を更新'''
        self.brain.replay()

    def get_action(self, state, episode):
        ''' 行動を決定する '''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        ''' memoryオブジェクトにtransitionの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)
    
    def get_q_values(self, state):
        '''Brainクラスから現在の状態に対するQ値を取得する'''
        return self.brain.get_q_values(state)

# 環境クラス (本からの変更あり)
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
            **COMMON_CONFIG,         # 共通設定
            **model_specific_config  # モデル固有の設定
        }

        # lost_timeをconfigから取得（デフォルト値は2）
        self.lost_time_steps = self.config.get('LOST_TIME_STEPS', 1)

        # モデルのパス設定
        self.model_path = f'trained_model_{model_number}.pth'
        self.saving_model_path = f'trained_model_{model_number}'

        # DataSaverの初期化を追加
        self.data_saver = DataSaver(model_path=self.saving_model_path)  
        self.data_saver.save_config(self.config, self.model_number)  # 設定情報を保存
        self.generator = None  # 初期化時にはGeneratorを作成しない
        self.state_size = state_size  # 状態数 (過去10回分の左右のcandy履歴を状態として使用)
        self.action_size = action_size  # 行動数
        self.agent = Agent(self.state_size, self.action_size, self.config) # 環境内で行動するエージェント生成
        self.state = np.zeros(self.state_size)
        self.q_value_history = []  # Q値の履歴を保存するリスト
        
        # ネットワーク構造の情報を更新
        self.data_saver.update_network_structure(self.agent.brain.model)
        
        # 設定情報を保存（ネットワーク構造を含む）
        self.data_saver.save_config(self.config, self.model_number)

        # configから値を取得
        NUM_EPISODES = self.config['NUM_EPISODES']
        self.episodes_to_plot = [0, 
                               NUM_EPISODES-5, 
                               NUM_EPISODES-4, 
                               NUM_EPISODES-3, 
                               NUM_EPISODES-2, 
                               NUM_EPISODES-1]
        self.action_results = []  # 行動結果を保存するリスト
        
        self.lost_time_remaining = 0
        self.current_lane_history = []
        self.an_episode_actions = []
        self.all_episode_action_results = []
        self.episode_current_lanes = []
        self.all_episode_total_delays = []  

        #待ち行列関連
        self.left_queue = 0
        self.right_queue = 0
        self.left_queue_history = []
        self.right_queue_history = []

        # 総遅れ時間の評価用
        self.total_collected = 0
        self.total_delay_time = 0

        # 報酬関数のための変数
        self.previous_total_queue = 0

        # 理論上の最大報酬を計算
        self.theoretical_rewards = 6826
        # self.theoretical_rewards = self.calculate_theoretical_rewards()
        print(f"Theoretical Maximum Reward: {self.theoretical_rewards}")
        print("---")

    def calculate_theoretical_rewards(self):
        """各ステップでのCandyの最大値の累計を計算"""
        return sum(max(candy) for candy in self.generator.candy_pattern)

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

    def analyze_actions(self, total_reward): 
        accuracy = total_reward / self.theoretical_rewards
        
        return {
            "overall_accuracy": accuracy
        }


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
                "-c", self.config["SUMOCFG_PATH"],
                "--step-length", str(self.config.get("STEP", "0.1")),
                "--start",
                "--quit-on-end"
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
        '''全エピソード全ステップの出力Q値の推移グラフを出力'''
        # プロットの作成
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111)

        # 全エピソードの全ステップのQ値を1つの配列にまとめる
        all_q_values = np.concatenate(self.q_value_history)
        
        # 総ステップ数を計算
        total_steps = len(all_q_values)
        
        # x軸の値を生成（エピソードとステップの情報を含む）
        x = np.arange(total_steps)
        episodes = x // self.config['MAX_STEPS']
        steps = x % self.config['MAX_STEPS']
        
        # 行動別にプロット
        action_labels = ['0:Keep', '1:Switch']
        for i in range(2):
            ax.plot(x, all_q_values[:, i], label=f'Action {i}: {action_labels[i]}', alpha=0.7)
        
        ax.set_title('Q-values for All Episodes and Steps')
        ax.set_xlabel('Total Steps')
        ax.set_ylabel('Q-value')
        ax.legend()
        
        # 指定されたエピソードの第1ステップにティックを設定
        tick_episodes = [0, 49] + list(range(99, self.config['NUM_EPISODES'], 50))
        tick_locations = [ep * self.config['MAX_STEPS'] for ep in tick_episodes]
        
        ax.set_xticks(tick_locations)
        ax.set_xticklabels([f'Ep {ep}, Step 0' for ep in tick_episodes], rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig

    def plot_q_value(self):
        '''特定のエピソードにおける出力Q値の推移グラフを出力'''
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
        '''特定のエピソードにおける待ち行列の推移、選択レーン、およびアクションタイミングのグラフを出力'''
        fig = plt.figure(figsize=(20, 15))
        queue_labels = ['queue (route_we)', 'queue (route_ns)']
        for i, episode in enumerate(self.episodes_to_plot):
            if episode < len(self.left_queue_history):
                ax = fig.add_subplot(len(self.episodes_to_plot), 1, i+1)
                queue_steps = range(len(self.left_queue_history[episode]))
                steps = range(len(self.left_queue_history[episode])-1)
                
                # 待ち行列の長さをプロット
                ax.plot(queue_steps, self.left_queue_history[episode], label=queue_labels[0])
                ax.plot(queue_steps, self.right_queue_history[episode], label=queue_labels[1])
                
                # 選択されたレーンを塗りつぶし表示
                selected_lane = self.all_episode_current_lanes[episode]
                for j in range(1, len(steps)+1):
                    if selected_lane[j] == 0:
                        ax.axvspan(j-1, j, facecolor='blue', alpha=0.1, linewidth=0)
                    else:
                        ax.axvspan(j-1, j, facecolor='red', alpha=0.1, linewidth=0)
                
                # 'S'と'L'のアクションを表示
                for step, action in enumerate(self.all_episode_action_results[episode]):
                    if action in ['S', 'L']:
                        ax.axvspan(step, step+1, facecolor='grey', alpha=0.6, linewidth=0)
                
                # 凡例用のダミープロット
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
        '''総遅延時間の推移グラフを出力'''
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

# Environmentクラスとは独立して定義している
def run_model(model_number):
        print(f"\nRunning Model {model_number}")
        print("=====================================")
        
        # モデル設定の取得
        if model_number not in MODEL_CONFIGS:
            print(f"Error: No configuration found for model number {model_number}")
            raise SystemExit("Program terminated due to missing model configuration.")
        
        # モデル設定の取得
        config = MODEL_CONFIGS[model_number]
        
        # モデルの保存パスをチェック
        model_path = f'trained_model_{model_number}.pth'
        if os.path.exists(model_path):
            print(f"Error: Model already exists at {model_path}")
            print("Please delete the existing model file or specify a different path.")
            raise SystemExit("Program terminated to prevent overwriting existing model.")   
    
        env = SUMOEnvironment(model_number, config) #ロスタイムを変更する場合は(lost_time_steps=2)のように変更すること
        rewards, all_actions, all_optimal_actions, all_action_results, total_delays = env.run()  

        # 学習済みモデルの保存
        torch.save(env.agent.brain.model.state_dict(), model_path)
        print(f"Training completed and model {model_number} saved.")

        # 最新10エピソード分の総遅れ時間の平均値
        print(f'Average of total delay time for the last 10 episodes = {np.mean(total_delays[-10:])}')
        print(f'Average of total delay time for the last 8 episodes = {np.mean(total_delays[-8:])}')
        print(f'Average of total delay time for the last 5 episodes = {np.mean(total_delays[-5:])}')
        print("=====================================\n")


if __name__ == "__main__":
    for model_num in MODELS_TO_RUN:
        run_model(model_num)