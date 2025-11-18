"""
Deep Reinforcement Learning Bullet Dodging NPC - Desktop Application
Requires: pip install pywebview
"""

import webview
import os
import sys

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DRL Bullet Dodging NPC</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            background: rgba(26, 26, 46, 0.8);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            max-width: 900px;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        h1 {
            font-size: 28px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .brain-icon {
            color: #00ff88;
            font-size: 32px;
        }
        
        .controls {
            display: flex;
            gap: 10px;
        }
        
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .btn-train {
            background: #00ff88;
            color: #1a1a2e;
        }
        
        .btn-train.active {
            background: #ff4757;
            color: white;
        }
        
        .btn-reset {
            background: #4a9eff;
            color: white;
        }
        
        canvas {
            border: 4px solid #2d3561;
            border-radius: 8px;
            display: block;
            margin-bottom: 20px;
            background: #0f0f1e;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(45, 53, 97, 0.6);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(74, 158, 255, 0.3);
        }
        
        .stat-label {
            font-size: 12px;
            color: #aaa;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00ff88;
        }
        
        .info {
            margin-top: 20px;
            padding: 15px;
            background: rgba(45, 53, 97, 0.4);
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.6;
            color: #ccc;
        }
        
        .info strong {
            color: #00ff88;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <span class="brain-icon">🧠</span>
                DRL Bullet Dodging NPC
            </h1>
            <div class="controls">
                <button id="trainBtn" class="btn-train" onclick="toggleTraining()">
                    <span>▶</span> Train
                </button>
                <button class="btn-reset" onclick="resetGame()">
                    <span>↻</span> Reset
                </button>
            </div>
        </div>
        
        <canvas id="gameCanvas" width="840" height="600"></canvas>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Episodes</div>
                <div class="stat-value" id="episodes">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Reward</div>
                <div class="stat-value" id="avgReward">0.0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Exploration (ε)</div>
                <div class="stat-value" id="epsilon">1.000</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value" id="successRate">0%</div>
            </div>
        </div>
        
        <div class="info">
            <p><strong>How it works:</strong> The NPC uses Deep Q-Learning (DQN) to learn optimal dodging strategies.</p>
            <p style="margin-top: 8px;"><strong>State:</strong> Relative position and velocity of nearest bullet</p>
            <p><strong>Actions:</strong> Move up, down, left, or right</p>
            <p><strong>Reward:</strong> +0.1 per frame survived, +distance bonus, -10 for getting hit</p>
        </div>
    </div>

    <script>
        // Deep Q-Network
        class DQN {
            constructor() {
                this.learningRate = 0.001;
                this.gamma = 0.95;
                this.epsilonDecay = 0.995;
                this.epsilonMin = 0.01;
                this.memory = [];
                this.maxMemory = 2000;
                this.batchSize = 32;
                
                this.w1 = this.initWeights(4, 8);
                this.b1 = new Array(8).fill(0);
                this.w2 = this.initWeights(8, 4);
                this.b2 = new Array(4).fill(0);
            }
            
            initWeights(input, output) {
                const weights = [];
                for (let i = 0; i < input; i++) {
                    weights[i] = [];
                    for (let j = 0; j < output; j++) {
                        weights[i][j] = (Math.random() - 0.5) * 0.5;
                    }
                }
                return weights;
            }
            
            relu(x) {
                return Math.max(0, x);
            }
            
            forward(state) {
                const hidden = [];
                for (let i = 0; i < 8; i++) {
                    let sum = this.b1[i];
                    for (let j = 0; j < 4; j++) {
                        sum += state[j] * this.w1[j][i];
                    }
                    hidden[i] = this.relu(sum);
                }
                
                const output = [];
                for (let i = 0; i < 4; i++) {
                    let sum = this.b2[i];
                    for (let j = 0; j < 8; j++) {
                        sum += hidden[j] * this.w2[j][i];
                    }
                    output[i] = sum;
                }
                
                return output;
            }
            
            predict(state) {
                return this.forward(state);
            }
            
            remember(state, action, reward, nextState, done) {
                this.memory.push({ state, action, reward, nextState, done });
                if (this.memory.length > this.maxMemory) {
                    this.memory.shift();
                }
            }
            
            replay() {
                if (this.memory.length < this.batchSize) return;
                
                const batch = [];
                for (let i = 0; i < this.batchSize; i++) {
                    const idx = Math.floor(Math.random() * this.memory.length);
                    batch.push(this.memory[idx]);
                }
                
                for (const exp of batch) {
                    const { state, action, reward, nextState, done } = exp;
                    
                    let target = reward;
                    if (!done) {
                        const nextQ = this.predict(nextState);
                        target = reward + this.gamma * Math.max(...nextQ);
                    }
                    
                    const currentQ = this.predict(state);
                    const error = target - currentQ[action];
                    
                    this.updateWeights(state, action, error);
                }
            }
            
            updateWeights(state, action, error) {
                const lr = this.learningRate;
                
                const hidden = [];
                for (let i = 0; i < 8; i++) {
                    let sum = this.b1[i];
                    for (let j = 0; j < 4; j++) {
                        sum += state[j] * this.w1[j][i];
                    }
                    hidden[i] = this.relu(sum);
                }
                
                for (let j = 0; j < 8; j++) {
                    this.w2[j][action] += lr * error * hidden[j];
                }
                this.b2[action] += lr * error;
                
                for (let i = 0; i < 4; i++) {
                    for (let j = 0; j < 8; j++) {
                        const hiddenError = error * this.w2[j][action] * (hidden[j] > 0 ? 1 : 0);
                        this.w1[i][j] += lr * hiddenError * state[i] * 0.1;
                    }
                }
            }
        }

        // Game state
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        
        let dqn = new DQN();
        let training = false;
        let epsilon = 1.0;
        let episode = 0;
        let totalReward = 0;
        let episodeRewards = [];
        let stats = { hits: 0, dodges: 0 };
        
        let game = {
            npc: { x: 420, y: 300, size: 20, vx: 0, vy: 0 },
            bullets: [],
            frameCount: 0
        };

        function getState() {
            const npc = game.npc;
            let closestBullet = null;
            let minDist = Infinity;
            
            for (const bullet of game.bullets) {
                const dist = Math.sqrt(
                    Math.pow(bullet.x - npc.x, 2) + Math.pow(bullet.y - npc.y, 2)
                );
                if (dist < minDist) {
                    minDist = dist;
                    closestBullet = bullet;
                }
            }
            
            if (!closestBullet) {
                return [0, 0, 0, 0];
            }
            
            const dx = (closestBullet.x - npc.x) / 840;
            const dy = (closestBullet.y - npc.y) / 600;
            const dvx = closestBullet.vx / 5;
            const dvy = closestBullet.vy / 5;
            
            return [dx, dy, dvx, dvy];
        }

        function takeAction(action) {
            const speed = 4;
            game.npc.vx = 0;
            game.npc.vy = 0;
            
            if (action === 0) game.npc.vy = -speed;
            if (action === 1) game.npc.vy = speed;
            if (action === 2) game.npc.vx = -speed;
            if (action === 3) game.npc.vx = speed;
        }

        function spawnBullet() {
            const side = Math.floor(Math.random() * 4);
            let x, y, vx, vy;
            
            if (side === 0) {
                x = Math.random() * 840;
                y = 0;
                vx = (Math.random() - 0.5) * 4;
                vy = 2 + Math.random() * 2;
            } else if (side === 1) {
                x = Math.random() * 840;
                y = 600;
                vx = (Math.random() - 0.5) * 4;
                vy = -(2 + Math.random() * 2);
            } else if (side === 2) {
                x = 0;
                y = Math.random() * 600;
                vx = 2 + Math.random() * 2;
                vy = (Math.random() - 0.5) * 4;
            } else {
                x = 840;
                y = Math.random() * 600;
                vx = -(2 + Math.random() * 2);
                vy = (Math.random() - 0.5) * 4;
            }
            
            game.bullets.push({ x, y, vx, vy, size: 8 });
        }

        function checkCollision() {
            const npc = game.npc;
            for (const bullet of game.bullets) {
                const dist = Math.sqrt(
                    Math.pow(bullet.x - npc.x, 2) + Math.pow(bullet.y - npc.y, 2)
                );
                if (dist < npc.size + bullet.size) {
                    return true;
                }
            }
            return false;
        }

        function resetGameState() {
            game.npc = { x: 420, y: 300, size: 20, vx: 0, vy: 0 };
            game.bullets = [];
            game.frameCount = 0;
        }

        function updateUI() {
            document.getElementById('episodes').textContent = episode;
            document.getElementById('avgReward').textContent = 
                (episodeRewards.length > 0 ? 
                    (episodeRewards.reduce((a, b) => a + b, 0) / episodeRewards.length).toFixed(1) : 
                    '0.0');
            document.getElementById('epsilon').textContent = epsilon.toFixed(3);
            const total = stats.hits + stats.dodges;
            document.getElementById('successRate').textContent = 
                total > 0 ? Math.round((stats.dodges / total) * 100) + '%' : '0%';
        }

        function gameLoop() {
            const state = getState();
            
            let action;
            if (training && Math.random() < epsilon) {
                action = Math.floor(Math.random() * 4);
            } else {
                const qValues = dqn.predict(state);
                action = qValues.indexOf(Math.max(...qValues));
            }
            
            takeAction(action);
            
            game.npc.x += game.npc.vx;
            game.npc.y += game.npc.vy;
            game.npc.x = Math.max(game.npc.size, Math.min(840 - game.npc.size, game.npc.x));
            game.npc.y = Math.max(game.npc.size, Math.min(600 - game.npc.size, game.npc.y));
            
            game.bullets = game.bullets.filter(bullet => {
                bullet.x += bullet.vx;
                bullet.y += bullet.vy;
                return bullet.x > -50 && bullet.x < 890 && bullet.y > -50 && bullet.y < 650;
            });
            
            if (game.frameCount % 60 === 0 && game.bullets.length < 8) {
                spawnBullet();
            }
            
            let r = 0.1;
            let done = false;
            
            if (checkCollision()) {
                r = -10;
                done = true;
                stats.hits++;
            } else {
                const minDist = Math.min(...game.bullets.map(b => 
                    Math.sqrt(Math.pow(b.x - game.npc.x, 2) + Math.pow(b.y - game.npc.y, 2))
                ), 1000);
                r += minDist / 1000;
            }
            
            totalReward += r;
            const nextState = getState();
            
            if (training) {
                dqn.remember(state, action, r, nextState, done);
                dqn.replay();
            }
            
            if (done) {
                episodeRewards.push(totalReward);
                if (episodeRewards.length > 100) {
                    episodeRewards.shift();
                }
                
                episode++;
                stats.dodges++;
                totalReward = 0;
                resetGameState();
                
                if (training) {
                    epsilon = Math.max(dqn.epsilonMin, epsilon * dqn.epsilonDecay);
                }
                
                updateUI();
            }
            
            game.frameCount++;
            
            // Draw
            ctx.fillStyle = '#0f0f1e';
            ctx.fillRect(0, 0, 840, 600);
            
            ctx.fillStyle = training ? '#00ff88' : '#4a9eff';
            ctx.beginPath();
            ctx.arc(game.npc.x, game.npc.y, game.npc.size, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = '#ff4757';
            for (const bullet of game.bullets) {
                ctx.beginPath();
                ctx.arc(bullet.x, bullet.y, bullet.size, 0, Math.PI * 2);
                ctx.fill();
            }
            
            ctx.fillStyle = '#fff';
            ctx.font = '14px monospace';
            ctx.fillText(`Episode: ${episode}`, 10, 20);
            ctx.fillText(`Reward: ${totalReward.toFixed(1)}`, 10, 40);
            ctx.fillText(`ε: ${epsilon.toFixed(3)}`, 10, 60);
            
            requestAnimationFrame(gameLoop);
        }

        function toggleTraining() {
            training = !training;
            const btn = document.getElementById('trainBtn');
            if (training) {
                btn.innerHTML = '<span>⏸</span> Pause';
                btn.classList.add('active');
            } else {
                btn.innerHTML = '<span>▶</span> Train';
                btn.classList.remove('active');
            }
        }

        function resetGame() {
            dqn = new DQN();
            training = false;
            epsilon = 1.0;
            episode = 0;
            totalReward = 0;
            episodeRewards = [];
            stats = { hits: 0, dodges: 0 };
            resetGameState();
            updateUI();
            
            const btn = document.getElementById('trainBtn');
            btn.innerHTML = '<span>▶</span> Train';
            btn.classList.remove('active');
        }

        // Start game loop
        gameLoop();
    </script>
</body>
</html>
"""


class API:
    """API class for Python-JavaScript communication"""
    
    def get_title(self):
        return "DRL Bullet Dodging NPC"
    
    def save_model(self):
        """Placeholder for saving model weights"""
        return {"status": "success", "message": "Model saved successfully!"}
    
    def load_model(self):
        """Placeholder for loading model weights"""
        return {"status": "success", "message": "Model loaded successfully!"}


def main():
    """Main entry point for the application"""
    
    api = API()
    
    # Create window
    window = webview.create_window(
        'DRL Bullet Dodging NPC',
        html=HTML_CONTENT,
        width=950,
        height=900,
        resizable=True,
        js_api=api
    )
    
    # Start the application
    webview.start(debug=True)


if __name__ == '__main__':
    main()
