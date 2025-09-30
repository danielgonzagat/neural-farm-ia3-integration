#!/usr/bin/env python3
"""
NEURON FARM OPTIMIZED v2.0
Versão otimizada e simplificada da Fazenda de Neurônios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import sqlite3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('NeuronFarm')

class EvolvingNeuron:
    """Neurônio evolutivo simplificado"""
    
    async def __init__(self, input_size: int, neuron_id: str):
        self.id = neuron_id
        self.input_size = input_size
        self.weights = torch.randn(input_size) * 0.1
        self.bias = torch.randn(1) * 0.1
        self.activation = np.random.choice(['relu', 'tanh', 'sigmoid'])
        self.fitness = 0.0
        self.age = 0
        self.plasticity = random.uniform(0.01, 0.1)
        
    async def forward(self, x):
        """Processa entrada"""
        output = torch.matmul(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            output = F.relu(output)
        elif self.activation == 'tanh':
            output = torch.tanh(output)
        else:
            output = torch.sigmoid(output)
            
        return await output
    
    async def mutate(self, rate: float = 0.01):
        """Muta neurônio"""
        # Muta pesos
        self.weights += torch.randn_like(self.weights) * rate
        self.bias += torch.randn_like(self.bias) * rate
        
        # Chance de mudar ativação
        if np.random.random() < 0.1:
            self.activation = np.random.choice(['relu', 'tanh', 'sigmoid'])
            
    async def crossover(self, other: 'EvolvingNeuron') -> 'EvolvingNeuron':
        """Crossover com outro neurônio"""
        child = EvolvingNeuron(self.input_size, f"{self.id}_x_{other.id}")
        
        # Mistura pesos
        mask = torch.rand_like(self.weights) > 0.5
        child.weights = torch.where(mask, self.weights, other.weights)
        child.bias = (self.bias + other.bias) / 2
        
        # Herda ativação
        child.activation = np.random.choice([self.activation, other.activation])
        child.plasticity = (self.plasticity + other.plasticity) / 2
        
        return await child

class NeuronLayer:
    """Camada de neurônios evolutivos"""
    
    async def __init__(self, input_size: int, num_neurons: int, layer_id: int):
        self.neurons = []
        for i in range(num_neurons):
            neuron = EvolvingNeuron(input_size, f"L{layer_id}_N{i}")
            self.neurons.append(neuron)
            
    async def forward(self, x):
        """Processa através da camada"""
        outputs = []
        for neuron in self.neurons:
            out = neuron.forward(x)
            outputs.append(out.unsqueeze(-1))
            
        return await torch.cat(outputs, dim=-1)
    
    async def evolve(self):
        """Evolui neurônios da camada"""
        # Remove neurônios fracos
        self.neurons = [n for n in self.neurons if n.fitness > 0.1 or n.age < 10]
        
        # Garante mínimo de neurônios
        while len(self.neurons) < 5:
            # Cria novo neurônio
            new_neuron = EvolvingNeuron(
                self.neurons[0].input_size if self.neurons else 784,
                f"L_NEW_{np.random.randint(1000,9999)}"
            )
            self.neurons.append(new_neuron)
            
        # Envelhece neurônios
        for n in self.neurons:
            n.age += 1

class EvolvedNetwork(nn.Module):
    """Rede neural com neurônios evolutivos"""
    
    async def __init__(self):
        super().__init__()
        
        # Camadas de neurônios evolutivos
        self.layer1 = NeuronLayer(784, 50, 0)  # Menos neurônios
        self.layer2 = NeuronLayer(50, 30, 1)
        self.layer3 = NeuronLayer(30, 10, 2)
        
        # Camada de saída fixa para estabilidade
        self.output = nn.Linear(10, 10)
        
        self.fitness = 0.0
        
    async def forward(self, x):
        """Forward pass"""
        x = x.view(x.size(0), -1)
        
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.output(x)
        
        return await x
    
    async def mutate(self, rate: float = 0.01):
        """Muta a rede"""
        for layer in [self.layer1, self.layer2, self.layer3]:
            for neuron in layer.neurons:
                if np.random.random() < 0.3:
                    neuron.mutate(rate)
                    
        # Muta camada de saída
        with torch.no_grad():
            self.output.weight += torch.randn_like(self.output.weight) * rate * 0.1
            
    async def crossover(self, other: 'EvolvedNetwork') -> 'EvolvedNetwork':
        """Crossover com outra rede"""
        child = EvolvedNetwork()
        
        # Crossover de neurônios por camada
        for child_layer, parent1_layer, parent2_layer in [
            (child.layer1, self.layer1, other.layer1),
            (child.layer2, self.layer2, other.layer2),
            (child.layer3, self.layer3, other.layer3)
        ]:
            child_layer.neurons = []
            max_neurons = max(len(parent1_layer.neurons), len(parent2_layer.neurons))
            
            for i in range(max_neurons):
                if i < len(parent1_layer.neurons) and i < len(parent2_layer.neurons):
                    # Crossover entre neurônios correspondentes
                    if np.random.random() < 0.5:
                        child_neuron = parent1_layer.neurons[i].crossover(parent2_layer.neurons[i])
                    else:
                        # Copia direto de um dos pais
                        parent_neuron = np.random.choice([parent1_layer.neurons[i], parent2_layer.neurons[i]])
                        child_neuron = EvolvingNeuron(parent_neuron.input_size, f"child_{i}")
                        child_neuron.weights = parent_neuron.weights.clone()
                        child_neuron.bias = parent_neuron.bias.clone()
                        child_neuron.activation = parent_neuron.activation
                else:
                    # Pega de quem tiver
                    if i < len(parent1_layer.neurons):
                        parent_neuron = parent1_layer.neurons[i]
                    else:
                        parent_neuron = parent2_layer.neurons[i]
                    child_neuron = EvolvingNeuron(parent_neuron.input_size, f"child_{i}")
                    child_neuron.weights = parent_neuron.weights.clone()
                    child_neuron.bias = parent_neuron.bias.clone()
                    
                child_layer.neurons.append(child_neuron)
                
        # Crossover da camada de saída
        with torch.no_grad():
            mask = torch.rand_like(self.output.weight) > 0.5
            child.output.weight.data = torch.where(mask, self.output.weight, other.output.weight)
            
        return await child
    
    async def evolve_topology(self):
        """Evolui topologia"""
        for layer in [self.layer1, self.layer2, self.layer3]:
            layer.evolve()

class NeuronFarmOptimized:
    """Fazenda otimizada de neurônios"""
    
    async def __init__(self):
        self.population_size = 20  # Menor população
        self.elite_size = 4
        self.mutation_rate = 0.1
        self.batch_size = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generation = 0
        self.best_accuracy = 0
        
        # Setup
        self.setup_data()
        self.population = self.create_population()
        self.setup_database()
        
        logger.info(f"🧬 Fazenda de Neurônios iniciada no {self.device}")
        
    async def setup_data(self):
        """Carrega MNIST"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        
        # Usa subset para treino rápido
        train_subset = torch.utils.data.Subset(train_data, range(10000))
        test_subset = torch.utils.data.Subset(test_data, range(2000))
        
        self.train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_subset, batch_size=self.batch_size)
        
    async def setup_database(self):
        """Cria database"""
        self.conn = sqlite3.connect('neuron_farm_optimized.db')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS evolution (
                generation INTEGER PRIMARY KEY,
                timestamp TEXT,
                best_accuracy REAL,
                avg_accuracy REAL,
                total_neurons INTEGER,
                topology TEXT
            )
        ''')
        self.conn.commit()
        
    async def create_population(self) -> List[EvolvedNetwork]:
        """Cria população inicial"""
        population = []
        for _ in range(self.population_size):
            net = EvolvedNetwork().to(self.device)
            population.append(net)
        return await population
        
    async def train_and_evaluate(self, network: EvolvedNetwork) -> float:
        """Treina e avalia rede"""
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Treino rápido (poucas épocas)
        for epoch in range(2):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if batch_idx > 50:  # Limita batches
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        # Avaliação
        network.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = network(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
                
        accuracy = correct / total
        network.fitness = accuracy
        
        # Atualiza fitness dos neurônios
        for layer in [network.layer1, network.layer2, network.layer3]:
            for neuron in layer.neurons:
                neuron.fitness = accuracy
                
        return await accuracy
        
    async def evolve_generation(self):
        """Evolui uma geração"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Geração {self.generation}")
        
        # Treina e avalia
        accuracies = []
        for i, net in enumerate(self.population):
            acc = self.train_and_evaluate(net)
            accuracies.append(acc)
            
            # Evolui topologia
            net.evolve_topology()
            
            # Conta neurônios
            total_neurons = (
                len(net.layer1.neurons) + 
                len(net.layer2.neurons) + 
                len(net.layer3.neurons)
            )
            
            logger.info(f"Rede {i}: {acc:.4f} accuracy, {total_neurons} neurônios")
            
            # GC a cada 5 redes
            if i % 5 == 0:
                gc.collect()
                
        # Estatísticas
        best_idx = np.argmax(accuracies)
        self.best_accuracy = accuracies[best_idx]
        avg_accuracy = np.mean(accuracies)
        
        logger.info(f"Melhor: {self.best_accuracy:.4f}")
        logger.info(f"Média: {avg_accuracy:.4f}")
        
        # Salva no database
        best_net = self.population[best_idx]
        total_neurons = (
            len(best_net.layer1.neurons) + 
            len(best_net.layer2.neurons) + 
            len(best_net.layer3.neurons)
        )
        
        topology = {
            'layer1': len(best_net.layer1.neurons),
            'layer2': len(best_net.layer2.neurons),
            'layer3': len(best_net.layer3.neurons)
        }
        
        self.conn.execute(
            'INSERT INTO evolution VALUES (?, ?, ?, ?, ?, ?)',
            (self.generation, datetime.now().isoformat(), 
             self.best_accuracy, avg_accuracy, total_neurons, json.dumps(topology))
        )
        self.conn.commit()
        
        # Nova população
        self.population = self.reproduce(accuracies)
        
        # Checkpoint
        if self.generation % 10 == 0:
            self.save_checkpoint()
            
        self.generation += 1
        
    async def reproduce(self, accuracies: List[float]) -> List[EvolvedNetwork]:
        """Cria nova geração"""
        # Ordena por fitness
        sorted_idx = np.argsort(accuracies)[::-1]
        
        new_population = []
        
        # Elite
        for i in range(self.elite_size):
            elite = self.population[sorted_idx[i]]
            new_population.append(elite)
            
        # Reprodução
        while len(new_population) < self.population_size:
            # Seleção por torneio
            parent1_idx = self.tournament_select(accuracies)
            parent2_idx = self.tournament_select(accuracies)
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            child = parent1.crossover(parent2)
            child = child.to(self.device)
            
            # Mutação
            if np.random.random() < self.mutation_rate:
                child.mutate(0.01)
                
            new_population.append(child)
            
        return await new_population
        
    async def tournament_select(self, fitnesses: List[float], k: int = 3) -> int:
        """Seleção por torneio"""
        indices = random.sample(range(len(fitnesses)), min(k, len(fitnesses)))
        return await max(indices, key=lambda i: fitnesses[i])
        
    async def save_checkpoint(self):
        """Salva checkpoint"""
        checkpoint_dir = Path('neuron_farm_checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Salva melhor rede
        best_idx = 0
        best_fitness = 0
        for i, net in enumerate(self.population):
            if net.fitness > best_fitness:
                best_fitness = net.fitness
                best_idx = i
                
        checkpoint = {
            'generation': self.generation,
            'best_accuracy': self.best_accuracy,
            'network_state': self.population[best_idx].state_dict(),
            'topology': {
                'layer1': len(self.population[best_idx].layer1.neurons),
                'layer2': len(self.population[best_idx].layer2.neurons),
                'layer3': len(self.population[best_idx].layer3.neurons)
            }
        }
        
        torch.save(checkpoint, checkpoint_dir / f'gen_{self.generation}.pth')
        logger.info(f"💾 Checkpoint salvo: gen_{self.generation}.pth")
        
    async def run(self, max_generations: int = 100):
        """Loop principal"""
        logger.info("🚀 Iniciando evolução de neurônios")
        logger.info(f"População: {self.population_size}")
        logger.info(f"Meta: 95% accuracy")
        
        try:
            while self.generation < max_generations and self.best_accuracy < 0.95:
                self.evolve_generation()
                
                # Para se estagnar muito
                if self.generation > 20 and self.best_accuracy < 0.5:
                    logger.warning("⚠️ Reiniciando população estagnada")
                    self.population = self.create_population()
                    
        except KeyboardInterrupt:
            logger.info("\n⏸️ Evolução pausada")
            
        finally:
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ Evolução finalizada!")
            logger.info(f"Melhor accuracy: {self.best_accuracy:.4f}")
            logger.info(f"Gerações: {self.generation}")
            self.conn.close()

if __name__ == "__main__":
    farm = NeuronFarmOptimized()
    farm.run()