import os
import sys
import subprocess
import threading
import time
import socket
import http.server
import socketserver
import shutil

try:
    import webview
    import json
    import numpy as np
    import threading
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    import math
except Exception as e:
    print('\nMissing Python dependency. Make sure required packages are installed:')
    print('  pip install -r requirements.txt\n')
    raise


def find_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def serve_dist(dist_dir):
    port = find_free_port()
    handler = http.server.SimpleHTTPRequestHandler
    # Serve from the dist directory
    cwd = os.getcwd()
    os.chdir(dist_dir)
    httpd = socketserver.TCPServer(("", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    os.chdir(cwd)
    return httpd, port


def start_npm_dev():
    # Start the dev server (expects `npm run dev` to be defined, e.g. Vite)
    proc = subprocess.Popen([shutil.which('npm') or 'npm', 'run', 'dev'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True)
    return proc


def wait_for_url(url, timeout=30):
    import urllib.request

    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


class JSApi:
    def __init__(self, window=None, store_path='training_data.json'):
        self.window = window
        self.store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), store_path)
        # Training state
        self._training_thread = None
        self._stop_event = threading.Event()
        self.running = False
        self.episode = 0
        self.last_reward = 0.0
        self.reward_history = []
        # PyTorch model and optimizer
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pth')
        self.device = torch.device('cpu')
        self.model = PolicyNet(8, 16, 4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99

    def save_training_data(self, json_str):
        """Save training snapshot from JS (JSON string)."""
        try:
            with open(self.store_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return {'ok': True, 'path': self.store_path}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def load_training_data(self):
        """Return stored training data as string or empty string if missing."""
        try:
            if not os.path.exists(self.store_path):
                return ''
            with open(self.store_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def python_ping(self):
        return 'pong'

    def save_model(self, path: str = None):
        """Save model state dict to disk."""
        try:
            p = path or self.model_path
            torch.save(self.model.state_dict(), p)
            return {'ok': True, 'path': p}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def load_model(self, path: str = None):
        try:
            p = path or self.model_path
            if not os.path.exists(p):
                return {'ok': False, 'error': 'file not found'}
            state = torch.load(p, map_location=self.device)
            self.model.load_state_dict(state)
            return {'ok': True, 'path': p}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    # --- Training control methods exposed to JS ---
    def start_training(self):
        """Start a lightweight simulated training loop in a background thread."""
        if self.running:
            return {'ok': False, 'error': 'training already running'}

        self._stop_event.clear()
        self._training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.running = True
        self._training_thread.start()
        return {'ok': True}

    def stop_training(self):
        if not self.running:
            return {'ok': False, 'error': 'not running'}
        self._stop_event.set()
        if self._training_thread is not None:
            self._training_thread.join(timeout=5)
        self.running = False
        return {'ok': True}

    def get_status(self):
        return {
            'running': bool(self.running),
            'episode': int(self.episode),
            'last_reward': float(self.last_reward),
            'avg_reward': float(np.mean(self.reward_history) if self.reward_history else 0.0)
        }

    def _push_update(self, payload: dict):
        """Push a JSON payload into the frontend by calling a global handler `window.onPythonMessage`."""
        try:
            if self.window:
                js = f"window.onPythonMessage({json.dumps(payload)})"
                # evaluate_js returns the result, but we ignore it
                self.window.evaluate_js(js)
        except Exception:
            pass
    def _training_loop(self):
        """Training loop using PyTorch and a Python reimplementation of the JS environment."""
        env = Game()

        while not self._stop_event.is_set():
            self.episode += 1
            state = env.reset()
            log_probs = []
            rewards = []
            episode_reward = 0.0

            # run episode
            for t in range(1000):
                s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.model(s_tensor)
                probs = torch.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                action = int(m.sample().item())
                logp = m.log_prob(torch.tensor(action, device=self.device))

                reward, done, next_state = env.step(action)
                log_probs.append(logp)
                rewards.append(reward)
                episode_reward += reward

                state = next_state
                if done or self._stop_event.is_set():
                    break

            # compute returns and loss (REINFORCE)
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            if len(returns) > 0:
                # normalize
                returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
                loss = 0.0
                for lp, R in zip(log_probs, returns):
                    loss = loss - lp * R

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # record and push
            self.last_reward = float(episode_reward)
            self.reward_history.append(self.last_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)

            payload = {
                'type': 'training_update',
                'episode': int(self.episode),
                'last_reward': float(self.last_reward),
                'avg_reward': float(np.mean(self.reward_history))
            }
            self._push_update(payload)

            # short sleep to allow UI responsiveness
            for _ in range(5):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

        # training stopped; send final status
        self.running = False
        payload = {'type': 'training_stopped', 'episode': int(self.episode)}
        self._push_update(payload)


class PolicyNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


class Game:
    """Python re-implementation of the JS Game environment used by the TSX demo.
    The implementation mirrors the logic (positions, projectiles, simple player AI).
    """
    def __init__(self):
        self.width = 600
        self.height = 400
        self.reset()

    def reset(self):
        self.player = {
            'x': 100.0,
            'y': self.height / 2.0,
            'health': 100.0,
            'speed': 3.0,
            'attackCooldown': 0,
            'attackRange': 60.0
        }
        self.npc = {
            'x': 500.0,
            'y': self.height / 2.0,
            'health': 100.0,
            'speed': 2.5,
            'attackCooldown': 0,
            'attackRange': 60.0
        }
        self.projectiles = []
        self.done = False
        self.totalReward = 0.0
        return self.get_state()

    def get_state(self):
        dx = self.player['x'] - self.npc['x']
        dy = self.player['y'] - self.npc['y']
        distance = math.sqrt(dx * dx + dy * dy)
        return [
            dx / self.width,
            dy / self.height,
            distance / 500.0,
            self.npc['health'] / 100.0,
            self.player['health'] / 100.0,
            self.npc['attackCooldown'] / 30.0,
            1.0 if (self.player['y'] < self.npc['y']) else 0.0,
            1.0 if (distance < self.npc['attackRange']) else 0.0
        ]

    def step(self, action: int):
        if self.done:
            return 0.0, True, self.get_state()

        reward = 0.0

        # NPC actions: 0=move up, 1=move down, 2=move toward, 3=attack
        if action == 0 and self.npc['y'] > 30:
            self.npc['y'] -= self.npc['speed']
        elif action == 1 and self.npc['y'] < self.height - 30:
            self.npc['y'] += self.npc['speed']
        elif action == 2:
            dx = self.player['x'] - self.npc['x']
            if abs(dx) > 70:
                self.npc['x'] += self.npc['speed'] if dx > 0 else -self.npc['speed']
            reward += 0.01
        elif action == 3 and self.npc['attackCooldown'] == 0:
            dx = self.player['x'] - self.npc['x']
            dy = self.player['y'] - self.npc['y']
            distance = math.sqrt(dx * dx + dy * dy) if (dx != 0 or dy != 0) else 1.0
            if distance < self.npc['attackRange']:
                self.projectiles.append({
                    'x': self.npc['x'],
                    'y': self.npc['y'],
                    'vx': (dx / distance) * 5.0,
                    'vy': (dy / distance) * 5.0,
                    'owner': 'npc'
                })
                self.npc['attackCooldown'] = 30
                reward += 0.1
            else:
                reward -= 0.05

        # Simple player AI
        if random.random() < 0.02:
            self.player['y'] += (random.random() - 0.5) * 10.0
        self.player['y'] = max(30.0, min(self.height - 30.0, self.player['y']))

        if self.player['attackCooldown'] == 0 and random.random() < 0.05:
            dx = self.npc['x'] - self.player['x']
            dy = self.npc['y'] - self.player['y']
            distance = math.sqrt(dx * dx + dy * dy) if (dx != 0 or dy != 0) else 1.0
            if distance < 200.0:
                self.projectiles.append({
                    'x': self.player['x'],
                    'y': self.player['y'],
                    'vx': (dx / distance) * 5.0,
                    'vy': (dy / distance) * 5.0,
                    'owner': 'player'
                })
                self.player['attackCooldown'] = 30

        # Update projectiles and collisions
        new_projectiles = []
        for p in self.projectiles:
            p['x'] += p['vx']
            p['y'] += p['vy']

            if p['owner'] == 'player':
                dist = math.hypot(p['x'] - self.npc['x'], p['y'] - self.npc['y'])
                if dist < 20.0:
                    self.npc['health'] -= 20.0
                    reward -= 1.0
                    continue

            if p['owner'] == 'npc':
                dist = math.hypot(p['x'] - self.player['x'], p['y'] - self.player['y'])
                if dist < 20.0:
                    self.player['health'] -= 20.0
                    reward += 1.0
                    continue

            if 0 < p['x'] < self.width and 0 < p['y'] < self.height:
                new_projectiles.append(p)

        self.projectiles = new_projectiles

        # cooldowns
        if self.npc['attackCooldown'] > 0:
            self.npc['attackCooldown'] -= 1
        if self.player['attackCooldown'] > 0:
            self.player['attackCooldown'] -= 1

        # check win/loss
        if self.npc['health'] <= 0:
            reward -= 5.0
            self.done = True
            self.winner = 'player'
        elif self.player['health'] <= 0:
            reward += 5.0
            self.done = True
            self.winner = 'npc'

        reward += 0.005
        self.totalReward += reward
        return reward, self.done, self.get_state()


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    dist = os.path.join(cwd, 'dist')
    server_proc = None
    httpd = None

    if os.path.isdir(dist) and os.path.exists(os.path.join(dist, 'index.html')):
        print('Found production build in `dist/`. Serving it locally.')
        httpd, port = serve_dist(dist)
        url = f'http://127.0.0.1:{port}/index.html'
    else:
        # No dist; try to start dev server
        if shutil.which('npm') is None:
            print("No 'dist' directory and 'npm' not found.\nPlease either build the frontend into `dist/` or install Node.js and run the dev server.")
            sys.exit(1)

        print('No `dist/` found. Starting frontend dev server with `npm run dev`.')
        server_proc = start_npm_dev()
        dev_url = 'http://127.0.0.1:5173/'
        print(f'Waiting for dev server at {dev_url} ...')
        if not wait_for_url(dev_url, timeout=60):
            print('Dev server did not start in time. Check terminal output for errors.')
            try:
                server_proc.terminate()
            except Exception:
                pass
            sys.exit(1)
        url = dev_url

    try:
        api = JSApi()
        # create window with API exposed to JS as `window.pywebview.api`
        window = webview.create_window('Intelligent NPC', url, js_api=api)
        # attach window back-reference so API can evaluate JS later if needed
        api.window = window
        webview.start()
    finally:
        if server_proc:
            try:
                server_proc.terminate()
            except Exception:
                pass
        if httpd:
            try:
                httpd.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
