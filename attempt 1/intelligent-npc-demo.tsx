import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Brain, Zap } from 'lucide-react';

const IntelligentNPC = () => {
  const canvasRef = useRef(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [score, setScore] = useState(0);
  const [avgReward, setAvgReward] = useState(0);
  const [npcWins, setNpcWins] = useState(0);
  const [playerWins, setPlayerWins] = useState(0);
  
  const gameRef = useRef(null);
  const animationRef = useRef(null);

  // Simple Neural Network
  class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
      this.w1 = this.randomMatrix(inputSize, hiddenSize);
      this.b1 = this.randomMatrix(1, hiddenSize)[0];
      this.w2 = this.randomMatrix(hiddenSize, outputSize);
      this.b2 = this.randomMatrix(1, outputSize)[0];
      this.learningRate = 0.01;
    }

    randomMatrix(rows, cols) {
      return Array(rows).fill(0).map(() => 
        Array(cols).fill(0).map(() => (Math.random() - 0.5) * 0.5)
      );
    }

    relu(x) {
      return Math.max(0, x);
    }

    reluDerivative(x) {
      return x > 0 ? 1 : 0;
    }

    forward(input) {
      // Hidden layer
      this.hidden = input.map((val, i) => {
        let sum = this.b1[i % this.b1.length];
        for (let j = 0; j < input.length; j++) {
          sum += input[j] * this.w1[j][i % this.w1[0].length];
        }
        return this.relu(sum);
      });

      // Output layer
      this.output = Array(this.w2[0].length).fill(0).map((_, i) => {
        let sum = this.b2[i];
        for (let j = 0; j < this.hidden.length; j++) {
          sum += this.hidden[j] * this.w2[j][i];
        }
        return sum;
      });

      return this.output;
    }

    train(input, targetAction, reward) {
      // Forward pass
      const output = this.forward(input);
      
      // Calculate error
      const error = Array(output.length).fill(0);
      error[targetAction] = reward;

      // Backprop (simplified)
      for (let i = 0; i < this.w2.length; i++) {
        for (let j = 0; j < this.w2[0].length; j++) {
          this.w2[i][j] += this.learningRate * error[j] * this.hidden[i];
        }
      }

      for (let i = 0; i < this.b2.length; i++) {
        this.b2[i] += this.learningRate * error[i];
      }
    }

    getAction(state, epsilon = 0.1) {
      if (Math.random() < epsilon) {
        return Math.floor(Math.random() * 4); // Random exploration
      }
      const output = this.forward(state);
      return output.indexOf(Math.max(...output));
    }
  }

  // Game State
  class Game {
    constructor() {
      this.width = 600;
      this.height = 400;
      this.reset();
      this.nn = new NeuralNetwork(8, 16, 4); // 8 inputs, 16 hidden, 4 actions
      this.rewardHistory = [];
      this.epsilon = 0.3; // Exploration rate
    }

    reset() {
      this.player = {
        x: 100,
        y: this.height / 2,
        health: 100,
        speed: 3,
        attackCooldown: 0,
        attackRange: 60
      };

      this.npc = {
        x: 500,
        y: this.height / 2,
        health: 100,
        speed: 2.5,
        attackCooldown: 0,
        attackRange: 60
      };

      this.projectiles = [];
      this.done = false;
      this.totalReward = 0;
    }

    getState() {
      const dx = this.player.x - this.npc.x;
      const dy = this.player.y - this.npc.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      return [
        dx / this.width,                    // Relative X position
        dy / this.height,                   // Relative Y position
        distance / 500,                     // Distance to player
        this.npc.health / 100,              // NPC health
        this.player.health / 100,           // Player health
        this.npc.attackCooldown / 30,       // Attack cooldown status
        (this.player.y < this.npc.y) ? 1 : 0, // Player above/below
        (distance < this.npc.attackRange) ? 1 : 0 // In attack range
      ];
    }

    step(action) {
      if (this.done) return 0;

      let reward = 0;

      // NPC actions: 0=move up, 1=move down, 2=move toward, 3=attack
      if (action === 0 && this.npc.y > 30) {
        this.npc.y -= this.npc.speed;
      } else if (action === 1 && this.npc.y < this.height - 30) {
        this.npc.y += this.npc.speed;
      } else if (action === 2) {
        const dx = this.player.x - this.npc.x;
        if (Math.abs(dx) > 70) {
          this.npc.x += dx > 0 ? this.npc.speed : -this.npc.speed;
        }
        reward += 0.01; // Small reward for engaging
      } else if (action === 3 && this.npc.attackCooldown === 0) {
        const dx = this.player.x - this.npc.x;
        const dy = this.player.y - this.npc.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < this.npc.attackRange) {
          this.projectiles.push({
            x: this.npc.x,
            y: this.npc.y,
            vx: (dx / distance) * 5,
            vy: (dy / distance) * 5,
            owner: 'npc'
          });
          this.npc.attackCooldown = 30;
          reward += 0.1; // Reward for attacking in range
        } else {
          reward -= 0.05; // Penalty for attacking out of range
        }
      }

      // Simple player AI (moves randomly and attacks)
      if (Math.random() < 0.02) {
        this.player.y += (Math.random() - 0.5) * 10;
      }
      this.player.y = Math.max(30, Math.min(this.height - 30, this.player.y));

      if (this.player.attackCooldown === 0 && Math.random() < 0.05) {
        const dx = this.npc.x - this.player.x;
        const dy = this.npc.y - this.player.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 200) {
          this.projectiles.push({
            x: this.player.x,
            y: this.player.y,
            vx: (dx / distance) * 5,
            vy: (dy / distance) * 5,
            owner: 'player'
          });
          this.player.attackCooldown = 30;
        }
      }

      // Update projectiles
      this.projectiles = this.projectiles.filter(p => {
        p.x += p.vx;
        p.y += p.vy;

        // Check collision with NPC
        if (p.owner === 'player') {
          const dist = Math.sqrt((p.x - this.npc.x) ** 2 + (p.y - this.npc.y) ** 2);
          if (dist < 20) {
            this.npc.health -= 20;
            reward -= 1; // Penalty for getting hit
            return false;
          }
        }

        // Check collision with Player
        if (p.owner === 'npc') {
          const dist = Math.sqrt((p.x - this.player.x) ** 2 + (p.y - this.player.y) ** 2);
          if (dist < 20) {
            this.player.health -= 20;
            reward += 1; // Big reward for hitting player
            return false;
          }
        }

        return p.x > 0 && p.x < this.width && p.y > 0 && p.y < this.height;
      });

      // Cooldowns
      if (this.npc.attackCooldown > 0) this.npc.attackCooldown--;
      if (this.player.attackCooldown > 0) this.player.attackCooldown--;

      // Check win/loss
      if (this.npc.health <= 0) {
        reward -= 5;
        this.done = true;
        this.winner = 'player';
      } else if (this.player.health <= 0) {
        reward += 5;
        this.done = true;
        this.winner = 'npc';
      }

      // Small reward for staying alive
      reward += 0.005;

      this.totalReward += reward;
      return reward;
    }

    draw(ctx) {
      // Clear canvas
      ctx.fillStyle = '#1a1a2e';
      ctx.fillRect(0, 0, this.width, this.height);

      // Draw grid
      ctx.strokeStyle = '#16213e20';
      for (let i = 0; i < this.width; i += 50) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, this.height);
        ctx.stroke();
      }
      for (let i = 0; i < this.height; i += 50) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(this.width, i);
        ctx.stroke();
      }

      // Draw player
      ctx.fillStyle = '#3498db';
      ctx.beginPath();
      ctx.arc(this.player.x, this.player.y, 20, 0, Math.PI * 2);
      ctx.fill();
      
      // Player health bar
      ctx.fillStyle = '#2ecc71';
      ctx.fillRect(this.player.x - 25, this.player.y - 35, (this.player.health / 100) * 50, 5);
      ctx.strokeStyle = '#fff';
      ctx.strokeRect(this.player.x - 25, this.player.y - 35, 50, 5);

      // Draw NPC
      ctx.fillStyle = '#e74c3c';
      ctx.beginPath();
      ctx.arc(this.npc.x, this.npc.y, 20, 0, Math.PI * 2);
      ctx.fill();

      // NPC health bar
      ctx.fillStyle = '#2ecc71';
      ctx.fillRect(this.npc.x - 25, this.npc.y - 35, (this.npc.health / 100) * 50, 5);
      ctx.strokeStyle = '#fff';
      ctx.strokeRect(this.npc.x - 25, this.npc.y - 35, 50, 5);

      // Draw projectiles
      this.projectiles.forEach(p => {
        ctx.fillStyle = p.owner === 'player' ? '#3498db' : '#e74c3c';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw labels
      ctx.fillStyle = '#fff';
      ctx.font = '12px monospace';
      ctx.fillText('PLAYER', this.player.x - 20, this.player.y + 40);
      ctx.fillText('NPC (AI)', this.npc.x - 25, this.npc.y + 40);

      // Draw attack range indicators
      if (this.npc.attackCooldown === 0) {
        ctx.strokeStyle = '#e74c3c30';
        ctx.beginPath();
        ctx.arc(this.npc.x, this.npc.y, this.npc.attackRange, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    gameRef.current = new Game();

    let episodeCount = 0;
    let frameCount = 0;

    const gameLoop = () => {
      if (!isPaused) {
        const game = gameRef.current;
        
        // Get state and action
        const state = game.getState();
        const action = game.nn.getAction(state, game.epsilon);
        
        // Take action and get reward
        const reward = game.step(action);
        
        // Train network
        game.nn.train(state, action, reward);
        
        // Draw
        game.draw(ctx);

        frameCount++;

        // Episode management
        if (game.done) {
          episodeCount++;
          game.rewardHistory.push(game.totalReward);
          if (game.rewardHistory.length > 100) game.rewardHistory.shift();
          
          const avg = game.rewardHistory.reduce((a, b) => a + b, 0) / game.rewardHistory.length;
          
          setEpisode(episodeCount);
          setScore(Math.round(game.totalReward * 10) / 10);
          setAvgReward(Math.round(avg * 10) / 10);
          
          if (game.winner === 'npc') {
            setNpcWins(prev => prev + 1);
          } else {
            setPlayerWins(prev => prev + 1);
          }

          // Decay exploration rate
          game.epsilon = Math.max(0.05, game.epsilon * 0.995);
          
          game.reset();
        }
      }

      animationRef.current = requestAnimationFrame(gameLoop);
    };

    if (isTraining) {
      gameLoop();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isTraining, isPaused]);

  const handleStart = () => {
    setIsTraining(true);
    setIsPaused(false);
  };

  const handlePause = () => {
    setIsPaused(!isPaused);
  };

  const handleReset = () => {
    setIsTraining(false);
    setIsPaused(false);
    setEpisode(0);
    setScore(0);
    setAvgReward(0);
    setNpcWins(0);
    setPlayerWins(0);
    if (gameRef.current) {
      gameRef.current = new Game();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg shadow-2xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
          <Brain className="text-purple-400" />
          Intelligent NPC with Neural Network
        </h1>
        <p className="text-slate-300">Watch the NPC learn combat through reinforcement learning</p>
      </div>

      <div className="bg-slate-800 rounded-lg p-4 mb-4">
        <canvas
          ref={canvasRef}
          width={600}
          height={400}
          className="w-full border-2 border-slate-700 rounded"
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-slate-700 p-4 rounded-lg">
          <div className="text-slate-400 text-sm">Episode</div>
          <div className="text-2xl font-bold text-white">{episode}</div>
        </div>
        <div className="bg-slate-700 p-4 rounded-lg">
          <div className="text-slate-400 text-sm">Episode Reward</div>
          <div className="text-2xl font-bold text-green-400">{score}</div>
        </div>
        <div className="bg-slate-700 p-4 rounded-lg">
          <div className="text-slate-400 text-sm">Avg Reward (100)</div>
          <div className="text-2xl font-bold text-blue-400">{avgReward}</div>
        </div>
        <div className="bg-slate-700 p-4 rounded-lg">
          <div className="text-slate-400 text-sm">Win Rate</div>
          <div className="text-xl font-bold text-purple-400">
            {npcWins} - {playerWins}
          </div>
        </div>
      </div>

      <div className="flex gap-3">
        <button
          onClick={handleStart}
          disabled={isTraining && !isPaused}
          className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-slate-600 text-white py-3 px-6 rounded-lg font-semibold flex items-center justify-center gap-2 transition"
        >
          <Play size={20} />
          Start Training
        </button>
        <button
          onClick={handlePause}
          disabled={!isTraining}
          className="flex-1 bg-yellow-600 hover:bg-yellow-700 disabled:bg-slate-600 text-white py-3 px-6 rounded-lg font-semibold flex items-center justify-center gap-2 transition"
        >
          {isPaused ? <Play size={20} /> : <Pause size={20} />}
          {isPaused ? 'Resume' : 'Pause'}
        </button>
        <button
          onClick={handleReset}
          className="flex-1 bg-red-600 hover:bg-red-700 text-white py-3 px-6 rounded-lg font-semibold flex items-center justify-center gap-2 transition"
        >
          <RotateCcw size={20} />
          Reset
        </button>
      </div>

      <div className="mt-6 bg-slate-700 p-4 rounded-lg">
        <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
          <Zap className="text-yellow-400" size={20} />
          How it Works
        </h3>
        <ul className="text-slate-300 text-sm space-y-1">
          <li>• <strong>Blue circle</strong>: Player with basic AI</li>
          <li>• <strong>Red circle</strong>: NPC learning with neural network</li>
          <li>• NPC learns 4 actions: move up, move down, move toward player, attack</li>
          <li>• Rewards: +1 for hitting player, -1 for getting hit, +5/-5 for win/loss</li>
          <li>• Network uses 8 inputs (positions, health, distance) → 16 hidden neurons → 4 outputs</li>
          <li>• Exploration rate decays over time as NPC gets better</li>
        </ul>
      </div>
    </div>
  );
};

export default IntelligentNPC;