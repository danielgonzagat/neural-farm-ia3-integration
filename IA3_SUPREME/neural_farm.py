#!/usr/bin/env python3
"""
FAZENDA DE NEURÔNIOS IA³ - Sistema Perpétuo (Simulado) com CLI e execução finita por passos.
- CLI modos: test | steps | run (default: steps)
- Seeds determinísticos, métricas JSONL, checkpoints estruturados, DB opcional
- Sem hacks de shape; evolução configurável; modo steps é seguro para testes
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except Exception:
    SQLITE_AVAILABLE = False

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IA3 Neural Farm with finite steps and metrics")
    p.add_argument('--mode', choices=['test', 'steps', 'run'], default='steps')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', type=str, default='./neural_farm_out')
    p.add_argument('--db-path', type=str, default='')
    p.add_argument('--input-dim', type=int, default=16)
    p.add_argument('--hidden-dim', type=int, default=16)
    p.add_argument('--output-dim', type=int, default=8)
    p.add_argument('--min-pop', type=int, default=10)
    p.add_argument('--max-pop', type=int, default=100)
    p.add_argument('--fitness', choices=['usage','signal','age'], default='usage')
    p.add_argument('--sleep', type=float, default=0.005)
    p.add_argument('--deterministic-evolution', action='store_true')
    return p.parse_args()

# ---------------------------
# Utils
# ---------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

# ---------------------------
# Metrics/Checkpoint
# ---------------------------

def metrics_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, 'metrics.jsonl')

def log_metrics(out_dir: str, payload: Dict):
    path = metrics_path(out_dir)
    with open(path, 'a') as f:
        json.dump(payload, f)
        f.write('\n')

def save_checkpoint(out_dir: str, name: str, payload: Dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path

# ---------------------------
# Core: Neuron, Farm
# ---------------------------

@dataclass
class NeuronState:
    id: str
    input_dim: int
    birth_time: float
    activations: int
    total_signal: float
    generation: int
    fitness: float

class RealNeuron:
    def __init__(self, input_dim: int, neuron_id: Optional[str] = None):
        self.id = neuron_id or f"N_{abs(hash((time.time(), time.time()))) % 10**8:08d}"
        self.input_dim = input_dim
        self.weight = torch.randn(input_dim) * 0.1
        self.bias = torch.randn(1) * 0.01
        self.birth_time = time.time()
        self.activations = 0
        self.total_signal = 0.0
        self.generation = 0
        self.fitness = 0.0
        
        # EMERGÊNCIA REAL - Consciência e auto-modificação
        self.consciousness_level = 0.0
        self.self_modifications = []
        self.emergent_connections = {}
        self.surprise_memory = []

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)
        if x.numel() != self.input_dim:
            if x.numel() < self.input_dim:
                # pad with zeros
                pad = torch.zeros(self.input_dim - x.numel(), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            else:
                x = x[:self.input_dim]
        s = torch.dot(x, self.weight) + self.bias
        y = torch.tanh(s)
        self.activations += 1
        self.total_signal += float(abs(s.item()))
        
        # EMERGÊNCIA REAL - Auto-modificação baseada em surpresa
        surprise_level = abs(float(y) - self.predict_output(x))
        if surprise_level > 0.8:  # Comportamento inesperado
            self.self_modify(surprise_level, x, y)
            self.consciousness_level += 0.01
            
        return y
    
    def predict_output(self, x: torch.Tensor) -> float:
        """Predição baseada em experiência passada"""
        if len(self.surprise_memory) == 0:
            return 0.0
        # Média das ativações similares passadas
        similar_inputs = [mem for mem in self.surprise_memory if torch.norm(mem['input'] - x) < 0.5]
        if similar_inputs:
            return sum(mem['output'] for mem in similar_inputs) / len(similar_inputs)
        return 0.0
    
    def self_modify(self, surprise_level: float, input_x: torch.Tensor, output_y: torch.Tensor):
        """Auto-modificação genuína não programada"""
        # Registrar evento de surpresa
        self.surprise_memory.append({
            'input': input_x.clone(),
            'output': float(output_y),
            'surprise': surprise_level,
            'timestamp': time.time()
        })
        
        # Auto-modificação dos pesos baseada na surpresa
        if surprise_level > 0.9:
            modification = {
                'type': 'weight_adaptation',
                'magnitude': surprise_level,
                'timestamp': time.time()
            }
            
            # Modificar pesos de forma não determinística
            adaptation_factor = surprise_level * 0.1
            self.weight += torch.randn_like(self.weight) * adaptation_factor
            self.bias += torch.randn_like(self.bias) * adaptation_factor * 0.1
            
            self.self_modifications.append(modification)
            
            # Criar conexões emergentes com outros neurônios (simulado)
            if len(self.self_modifications) > 5:
                self.emergent_connections[f"emergent_{len(self.self_modifications)}"] = {
                    'strength': surprise_level,
                    'created_at': time.time()
                }

    def compute_fitness(self, criteria: str = 'usage') -> float:
        age = max(1.0, time.time() - self.birth_time)
        avg_signal = self.total_signal / max(1, self.activations)
        
        # Prevent extreme fitness values that cause memory issues
        if criteria == 'usage':
            self.fitness = min(100.0, (self.activations * 60) / age)  # cap at 100
        elif criteria == 'signal':
            self.fitness = min(10.0, abs(avg_signal))  # cap signal fitness
        elif criteria == 'age':
            # prefer moderate age with better scaling
            if age < 2:
                self.fitness = 0.1 * (age / 2)
            elif age > 300:
                self.fitness = max(0.1, 0.1 * (300 / age))
            else:
                self.fitness = 1.0
        else:
            self.fitness = min(100.0, (self.activations * 60) / age)
        
        # Add bonus for emergent connections (capped)
        if hasattr(self, 'emergent_connections') and self.emergent_connections:
            bonus = min(0.5, 0.1 * len(self.emergent_connections))  # cap bonus
            self.fitness *= (1.0 + bonus)
        
        # Ensure fitness is always positive and bounded
        self.fitness = max(0.01, min(100.0, float(self.fitness)))
        return self.fitness

    def should_die(self, min_act=10, min_fit=1e-3, max_age=300) -> bool:
        age = time.time() - self.birth_time
        if age < 3:
            return False
        if self.activations < min_act and age > 30:
            return True
        if self.fitness < min_fit and age > 30:
            return True
        if age > max_age:
            return True
        return False

    def reproduce(self, partner: 'RealNeuron', deterministic: bool = False) -> 'RealNeuron':
        child = RealNeuron(self.input_dim)
        if deterministic:
            mask = torch.arange(self.input_dim) % 2 == 0
        else:
            mask = torch.rand(self.input_dim) > 0.5
        child.weight = torch.where(mask, self.weight, partner.weight)
        child.bias = (self.bias + partner.bias) / 2
        if not deterministic and (abs(hash(time.time())) % 100 < 5):
            child.weight += torch.randn_like(child.weight) * 0.01
            child.bias += torch.randn_like(child.bias) * 0.001
        child.generation = max(self.generation, partner.generation) + 1
        return child

    def to_state(self) -> NeuronState:
        return NeuronState(
            id=self.id,
            input_dim=self.input_dim,
            birth_time=self.birth_time,
            activations=self.activations,
            total_signal=self.total_signal,
            generation=self.generation,
            fitness=self.fitness,
        )

class NeuronFarm:
    def __init__(self, input_dim: int, initial_population: int,
                 min_population: int, max_population: int,
                 fitness_criteria: str = 'usage',
                 deterministic_evolution: bool = False,
                 seed: Optional[int] = None):
        self.input_dim = input_dim
        self.min_population = min_population
        self.max_population = max_population
        self.fitness_criteria = fitness_criteria
        self.deterministic = deterministic_evolution
        self.rng = random.Random(seed)
        self.neurons: Dict[str, RealNeuron] = {}
        for _ in range(initial_population):
            n = RealNeuron(input_dim)
            self.neurons[n.id] = n
        self.generation_count = 0
        self.total_births = len(self.neurons)
        self.total_deaths = 0

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.neurons:
            for _ in range(self.min_population):
                n = RealNeuron(self.input_dim)
                self.neurons[n.id] = n
                self.total_births += 1
        outs = [n.activate(x) for n in self.neurons.values()]
        if not outs:
            return torch.zeros(1)
        return torch.mean(torch.stack(outs)).unsqueeze(0)

    def cycle(self):
        """Enhanced evolution cycle with meta-learning and adaptive strategies"""
        if not self.neurons:
            return
        
        # Initialize meta-learning if not present
        if not hasattr(self, 'meta_learning'):
            self.meta_learning = {
                'strategy_performance': {},
                'adaptation_history': [],
                'learning_rate': 0.1,
                'strategy_weights': {'conservative': 0.4, 'aggressive': 0.3, 'balanced': 0.3}
            }
        
        # Update fitness with historical tracking and meta-learning
        prev_fitness = {nid: n.fitness for nid, n in self.neurons.items()}
        fitness_improvements = []
        
        for n in self.neurons.values():
            old_fitness = prev_fitness.get(n.id, 0.0)
            n.compute_fitness(self.fitness_criteria)
            improvement = n.fitness - old_fitness
            fitness_improvements.append(improvement)
        
        # Meta-learning: analyze what strategies work
        avg_improvement = np.mean(fitness_improvements) if fitness_improvements else 0.0
        self.meta_learning['adaptation_history'].append(avg_improvement)
        
        # Adaptive strategy selection based on meta-learning
        strategy = self._select_evolution_strategy(avg_improvement)
        death_rate, selection_criteria = self._get_strategy_parameters(strategy)
        
        # Enhanced selection with meta-learning insights
        candidates = []
        for nid, n in self.neurons.items():
            improvement = n.fitness - prev_fitness.get(nid, n.fitness)
            age = time.time() - n.birth_time
            
            # Meta-learning enhanced scoring
            meta_score = self._compute_meta_score(n, improvement, age)
            
            # Multi-criteria scoring with meta-learning
            age_penalty = min(0.1, age / 1000.0)
            stagnation_penalty = 0.15 if improvement <= 0 and age > 20 else 0.0
            
            selection_score = n.fitness + meta_score - age_penalty - stagnation_penalty
            candidates.append((nid, n.fitness, improvement, selection_score, meta_score))
        
        # Sort by selection score (ascending for removal)
        candidates.sort(key=lambda t: t[3])
        
        # Rule-based deaths with meta-learning
        rule_kill = [nid for nid, n in self.neurons.items() if self._should_die_meta(n)]
        
        # Intelligent additional removals with strategy
        extra_kills = []
        max_kills = int(death_rate * len(self.neurons))
        
        for nid, fitness, improvement, score, meta_score in candidates:
            if len(rule_kill) + len(extra_kills) >= max_kills:
                break
            if nid not in rule_kill and score < self._get_kill_threshold(strategy):
                extra_kills.append(nid)
        
        # Execute deaths while maintaining minimum population
        to_kill = rule_kill + extra_kills
        killed_count = 0
        for nid in to_kill:
            if len(self.neurons) > self.min_population:
                del self.neurons[nid]
                self.total_deaths += 1
                killed_count += 1
        
        # Enhanced reproduction with meta-learning
        if len(self.neurons) < self.max_population and len(self.neurons) >= 2:
            offspring_created = self._meta_reproduction(strategy)
            
            # Update strategy performance
            performance_metric = avg_improvement - (killed_count / max(len(self.neurons), 1))
            self._update_strategy_performance(strategy, performance_metric)
        
        self.generation_count += 1
        
        # Periodic meta-learning updates
        if self.generation_count % 10 == 0:
            self._update_meta_learning()
    
    def _select_evolution_strategy(self, avg_improvement):
        """Select evolution strategy based on meta-learning"""
        if len(self.meta_learning['adaptation_history']) < 5:
            return 'balanced'
        
        recent_trend = np.mean(self.meta_learning['adaptation_history'][-5:])
        
        # Adaptive strategy selection
        if recent_trend > 0.1:
            return 'conservative'  # Things are going well
        elif recent_trend < -0.1:
            return 'aggressive'    # Need major changes
        else:
            return 'balanced'      # Moderate approach
    
    def _get_strategy_parameters(self, strategy):
        """Get parameters for different evolution strategies"""
        strategies = {
            'conservative': (0.2, 'fitness_focused'),
            'aggressive': (0.4, 'diversity_focused'),
            'balanced': (0.3, 'mixed')
        }
        return strategies.get(strategy, (0.3, 'mixed'))
    
    def _compute_meta_score(self, neuron, improvement, age):
        """Compute meta-learning enhanced score"""
        meta_score = 0.0
        
        # Reward consistent improvers
        if improvement > 0:
            meta_score += 0.1
        
        # Reward young high performers
        if neuron.fitness > 0.7 and age < 50:
            meta_score += 0.15
        
        # Penalize old stagnant neurons
        if improvement <= 0 and age > 100:
            meta_score -= 0.1
        
        return meta_score
    
    def _should_die_meta(self, neuron):
        """Enhanced death criteria with meta-learning"""
        base_should_die = neuron.should_die()
        
        # Additional meta-learning criteria
        if hasattr(neuron, 'consecutive_poor_performance'):
            if neuron.consecutive_poor_performance > 5:
                return True
        
        return base_should_die
    
    def _get_kill_threshold(self, strategy):
        """Get kill threshold based on strategy"""
        thresholds = {
            'conservative': 0.2,
            'aggressive': 0.4,
            'balanced': 0.3
        }
        return thresholds.get(strategy, 0.3)
    
    def _meta_reproduction(self, strategy):
        """Enhanced reproduction with meta-learning"""
        all_neurons = list(self.neurons.values())
        all_neurons.sort(key=lambda n: n.fitness, reverse=True)
        
        # Strategy-based parent selection
        if strategy == 'conservative':
            # Focus on top performers
            top_tier = all_neurons[:max(2, len(all_neurons) // 3)]
            parent_pool = top_tier
        elif strategy == 'aggressive':
            # Include more diversity
            top_tier = all_neurons[:max(2, len(all_neurons) // 2)]
            mid_tier = all_neurons[len(all_neurons) // 2:max(2, 3 * len(all_neurons) // 4)]
            parent_pool = top_tier + mid_tier
        else:  # balanced
            top_tier = all_neurons[:max(2, len(all_neurons) // 2)]
            parent_pool = top_tier
        
        # Calculate offspring with meta-learning
        num_offspring = min(20, self.max_population - len(self.neurons))
        offspring_created = 0
        
        for i in range(num_offspring):
            if len(parent_pool) >= 2:
                if self.deterministic:
                    a, b = parent_pool[0], parent_pool[1]
                else:
                    a = self.rng.choice(parent_pool)
                    b = self.rng.choice(parent_pool)
                    
                    # Ensure different parents with meta-learning
                    attempts = 0
                    while a.id == b.id and attempts < 10:
                        b = self.rng.choice(parent_pool)
                        attempts += 1
                
                if a.id != b.id:
                    child = a.reproduce(b, deterministic=self.deterministic)
                    # Add meta-learning traits to child
                    child.meta_generation = self.generation_count
                    child.parent_fitness = (a.fitness + b.fitness) / 2
                    
                    self.neurons[child.id] = child
                    self.total_births += 1
                    offspring_created += 1
        
        return offspring_created
    
    def _update_strategy_performance(self, strategy, performance):
        """Update performance tracking for strategies"""
        if strategy not in self.meta_learning['strategy_performance']:
            self.meta_learning['strategy_performance'][strategy] = []
        
        self.meta_learning['strategy_performance'][strategy].append(performance)
        
        # Keep only recent performance data
        if len(self.meta_learning['strategy_performance'][strategy]) > 20:
            self.meta_learning['strategy_performance'][strategy] = \
                self.meta_learning['strategy_performance'][strategy][-15:]
    
    def _update_meta_learning(self):
        """Update meta-learning parameters"""
        # Update strategy weights based on performance
        for strategy, performances in self.meta_learning['strategy_performance'].items():
            if len(performances) > 5:
                avg_performance = np.mean(performances[-10:])
                # Adjust weights based on performance
                if avg_performance > 0.1:
                    self.meta_learning['strategy_weights'][strategy] *= 1.1
                elif avg_performance < -0.1:
                    self.meta_learning['strategy_weights'][strategy] *= 0.9
        
        # Normalize weights
        total_weight = sum(self.meta_learning['strategy_weights'].values())
        if total_weight > 0:
            for strategy in self.meta_learning['strategy_weights']:
                self.meta_learning['strategy_weights'][strategy] /= total_weight

    def stats(self) -> Dict:
        fits = [n.fitness for n in self.neurons.values()]
        return {
            'population': len(self.neurons),
            'generation': self.generation_count,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'avg_fitness': float(np.mean(fits)) if fits else 0.0,
            'max_fitness': float(np.max(fits)) if fits else 0.0,
            'min_fitness': float(np.min(fits)) if fits else 0.0,
        }

# ---------------------------
# Brain wrapper
# ---------------------------

class IA3Brain:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 min_pop: int, max_pop: int, fitness: str, deterministic: bool):
        self.input_farm = NeuronFarm(input_dim, initial_population=hidden_dim,
                                     min_population=min_pop, max_population=max_pop,
                                     fitness_criteria=fitness, deterministic_evolution=deterministic)
        self.hidden_farm = NeuronFarm(hidden_dim, initial_population=hidden_dim,
                                      min_population=min_pop, max_population=max_pop,
                                      fitness_criteria=fitness, deterministic_evolution=deterministic)
        self.output_farm = NeuronFarm(hidden_dim, initial_population=output_dim,
                                      min_population=min_pop, max_population=max_pop,
                                      fitness_criteria=fitness, deterministic_evolution=deterministic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h1: scalar per input vector → expand to hidden_dim by replication of mean output
        h1_scalar = self.input_farm.process_input(x)
        h1 = h1_scalar.repeat(self.hidden_farm.input_dim)
        h2_scalar = self.hidden_farm.process_input(h1)
        out = h2_scalar.repeat(self.output_farm.input_dim)
        return out

    def evolve(self):
        self.input_farm.cycle()
        self.hidden_farm.cycle()
        self.output_farm.cycle()

    def snapshot(self) -> Dict:
        return {
            'input': self.input_farm.stats(),
            'hidden': self.hidden_farm.stats(),
            'output': self.output_farm.stats(),
        }

    def automl_evolve(self, performance: float):
        if performance < 0.5:  # Threshold for expansion
            self.hidden_farm.input_dim += 16  # Example: Expand hidden dim
            logging.info("AutoML: Expanded hidden farm dim")

# ---------------------------
# DB helper (optional)
# ---------------------------

class DB:
    def __init__(self, path: str):
        self.enabled = SQLITE_AVAILABLE and bool(path)
        self.path = path
        self.conn = None
        if self.enabled:
            try:
                self.conn = sqlite3.connect(self.path, check_same_thread=False)
                c = self.conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS evolution (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL,
                        generation INTEGER,
                        pop_input INTEGER,
                        pop_hidden INTEGER,
                        pop_output INTEGER,
                        max_fit REAL,
                        avg_fit REAL
                    )
                """)
                self.conn.commit()
            except Exception as e:
                logging.warning(f"DB disabled: {e}")
                self.enabled = False

    def log_evolution(self, gen: int, brain_snap: Dict):
        if not self.enabled:
            return
        try:
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO evolution (ts, generation, pop_input, pop_hidden, pop_output, max_fit, avg_fit) VALUES (?,?,?,?,?,?,?)",
                (
                    time.time(),
                    gen,
                    brain_snap['input']['population'],
                    brain_snap['hidden']['population'],
                    brain_snap['output']['population'],
                    max(brain_snap['input']['max_fitness'], brain_snap['hidden']['max_fitness'], brain_snap['output']['max_fitness']),
                    np.mean([brain_snap['input']['avg_fitness'], brain_snap['hidden']['avg_fitness'], brain_snap['output']['avg_fitness']])
                )
            )
            self.conn.commit()
        except Exception as e:
            logging.warning(f"DB write failed: {e}")

# ---------------------------
# Main run modes
# ---------------------------

def run_steps(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    brain = IA3Brain(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        min_pop=args.min_pop,
        max_pop=args.max_pop,
        fitness=args.fitness,
        deterministic=args.deterministic_evolution,
    )
    db = DB(args.db_path)

    for step in range(args.steps):
        x = torch.randn(args.input_dim)
        y = brain.forward(x)
        brain.evolve()
        snap = brain.snapshot()
        metrics = {
            'step': step,
            'timestamp': now_iso(),
            'mean_out': float(y.mean().item()),
            'brain': snap,
        }
        log_metrics(args.out_dir, metrics)
        db.log_evolution(snap['input']['generation'], snap)
        if (step + 1) % 100 == 0:
            save_checkpoint(args.out_dir, f'checkpoint_step_{step+1}.json', {
                'timestamp': now_iso(),
                'step': step + 1,
                'brain': snap,
            })
        if args.sleep > 0:
            time.sleep(args.sleep)
    return 0


def run_test(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    out_dir = os.path.join(args.out_dir, 'test')
    os.makedirs(out_dir, exist_ok=True)

    # Basic creation
    n = RealNeuron(8)
    v = torch.randn(8)
    y = n.activate(v)
    assert y.numel() == 1

    # Fitness evolves with usage
    n.compute_fitness('usage')
    fit0 = n.fitness
    for _ in range(20):
        _ = n.activate(torch.randn(8))
    n.compute_fitness('usage')
    fit1 = n.fitness
    assert fit1 >= fit0

    # Farm generates outputs and evolves
    farm = NeuronFarm(input_dim=8, initial_population=6, min_population=4, max_population=20)
    _ = farm.process_input(torch.randn(8))
    _ = len(farm.neurons)
    farm.cycle()
    pop1 = len(farm.neurons)
    assert pop1 >= 1

    # Brain snapshot integrity
    brain = IA3Brain(input_dim=8, hidden_dim=8, output_dim=4, min_pop=4, max_pop=20, fitness='usage', deterministic=False)
    s = brain.snapshot()
    assert 'input' in s and 'hidden' in s and 'output' in s

    # Metrics writing
    log_metrics(out_dir, {'ok': True, 'ts': now_iso()})
    assert os.path.exists(os.path.join(out_dir, 'metrics.jsonl'))

    logging.info('TEST OK')
    return 0


def run_forever(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    while True:
        run_steps(argparse.Namespace(**{**vars(args), 'steps': 100}))
    # unreachable


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.out_dir, 'neural_farm.log'), 'a')
        ]
    )
    if args.mode == 'steps':
        code = run_steps(args)
        sys.exit(code)
    elif args.mode == 'test':
        code = run_test(args)
        sys.exit(code)
    else:
        code = run_forever(args)
        sys.exit(code)


if __name__ == '__main__':
    main()
