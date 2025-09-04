"""
Advanced AI Agents for Fleet Resource Optimization
Implements reinforcement learning, MCP integration, and agentic AI for continuous allocation adjustment
"""

import numpy as np
import random
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import aiohttp
from collections import deque
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class State:
    """Environment state for reinforcement learning"""
    available_vehicles: int
    pending_demands: int
    high_priority_demands: int
    average_wait_time: float
    utilization_rate: float
    weather_condition: str
    traffic_level: str
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for RL"""
        # Normalize features
        weather_encoding = {'clear': 0, 'clouds': 1, 'rain': 2, 'snow': 3}.get(self.weather_condition, 0)
        traffic_encoding = {'low': 0, 'medium': 1, 'high': 2, 'severe': 3}.get(self.traffic_level, 0)
        
        return np.array([
            self.available_vehicles / 50.0,  # Normalize by max vehicles
            self.pending_demands / 20.0,     # Normalize by typical max demands
            self.high_priority_demands / 10.0,
            min(self.average_wait_time / 30.0, 1.0),  # Cap at 30 minutes
            self.utilization_rate / 100.0,
            weather_encoding / 3.0,
            traffic_encoding / 3.0,
            self.time_of_day / 23.0,
            self.day_of_week / 6.0
        ])

@dataclass
class Action:
    """Action for reinforcement learning"""
    action_type: str  # 'greedy', 'priority_first', 'distance_optimized', 'ai_guided'
    parameters: Dict[str, Any]
    
    def to_vector(self) -> np.ndarray:
        """Convert action to feature vector"""
        action_encoding = {'greedy': 0, 'priority_first': 1, 'distance_optimized': 2, 'ai_guided': 3}
        return np.array([
            action_encoding.get(self.action_type, 0) / 3.0,
            self.parameters.get('priority_weight', 0.5),
            self.parameters.get('distance_weight', 0.5),
            self.parameters.get('utilization_weight', 0.5)
        ])

@dataclass
class Reward:
    """Reward signal for reinforcement learning"""
    assignment_success: float
    priority_satisfaction: float
    utilization_improvement: float
    wait_time_reduction: float
    total_reward: float
    
    def __post_init__(self):
        if self.total_reward == 0:
            self.total_reward = (
                self.assignment_success * 0.3 +
                self.priority_satisfaction * 0.25 +
                self.utilization_improvement * 0.25 +
                self.wait_time_reduction * 0.2
            )

class MCPAgent:
    """Model Context Protocol Agent for multi-source data coordination"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.data_sources = {
            'weather': 'openweather',
            'traffic': 'google_maps',
            'demand': 'simulated',
            'fleet': 'internal'
        }
        self.context_buffer = deque(maxlen=100)
        
    async def fetch_multi_source_data(self, locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Fetch data from multiple sources using MCP pattern"""
        tasks = []
        
        # Weather data
        for location in locations:
            tasks.append(self._fetch_weather_data(location))
        
        # Traffic data
        for location in locations:
            tasks.append(self._fetch_traffic_data(location))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        weather_data = {}
        traffic_data = {}
        
        for i, result in enumerate(results):
            if i < len(locations):
                if not isinstance(result, Exception):
                    weather_data[locations[i]] = result
            else:
                if not isinstance(result, Exception):
                    traffic_data[locations[i - len(locations)]] = result
        
        return {
            'weather': weather_data,
            'traffic': traffic_data,
            'timestamp': datetime.now(),
            'sources': self.data_sources
        }
    
    async def _fetch_weather_data(self, location: Tuple[float, float]) -> Dict[str, Any]:
        """Fetch weather data for a location"""
        try:
            weather = self.api_client.get_weather_data(location)
            return {
                'condition': weather.condition,
                'temperature': weather.temperature,
                'humidity': weather.humidity,
                'wind_speed': weather.wind_speed
            }
        except Exception as e:
            logger.error(f"Weather fetch error: {e}")
            return {'condition': 'unknown', 'temperature': 20, 'humidity': 50, 'wind_speed': 5}
    
    async def _fetch_traffic_data(self, location: Tuple[float, float]) -> Dict[str, Any]:
        """Fetch traffic data for a location"""
        try:
            traffic = self.api_client.get_traffic_data(location)
            return {
                'congestion_level': traffic.congestion_level,
                'speed': traffic.average_speed,
                'delay': traffic.delay_minutes
            }
        except Exception as e:
            logger.error(f"Traffic fetch error: {e}")
            return {'congestion_level': 'medium', 'speed': 30, 'delay': 5}

class ReinforcementLearningAgent:
    """Reinforcement Learning Agent for fleet optimization"""
    
    def __init__(self, state_size: int = 9, action_size: int = 4, learning_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Q-learning parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Q-table (state-action values)
        self.q_table = {}
        self.memory = deque(maxlen=10000)
        
        # Performance tracking
        self.episode_rewards = []
        self.performance_history = []
        
        # Load existing model if available
        self.load_model()
    
    def get_state_key(self, state: State) -> str:
        """Convert state to hashable key for Q-table"""
        # Discretize continuous values for Q-table
        vector = state.to_vector()
        discrete_state = tuple(np.round(vector * 10).astype(int))
        return str(discrete_state)
    
    def choose_action(self, state: State) -> Action:
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        if np.random.random() <= self.epsilon:
            # Explore: random action
            action_type = random.choice(['greedy', 'priority_first', 'distance_optimized', 'ai_guided'])
            parameters = {
                'priority_weight': random.uniform(0.1, 0.9),
                'distance_weight': random.uniform(0.1, 0.9),
                'utilization_weight': random.uniform(0.1, 0.9)
            }
        else:
            # Exploit: best known action
            if state_key in self.q_table:
                action_idx = np.argmax(self.q_table[state_key])
            else:
                action_idx = random.randint(0, self.action_size - 1)
            
            action_types = ['greedy', 'priority_first', 'distance_optimized', 'ai_guided']
            action_type = action_types[action_idx]
            parameters = {
                'priority_weight': 0.5,
                'distance_weight': 0.5,
                'utilization_weight': 0.5
            }
        
        return Action(action_type=action_type, parameters=parameters)
    
    def update_q_table(self, state: State, action: Action, reward: Reward, next_state: State):
        """Update Q-table using Q-learning"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-table entries if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Get action index
        action_types = ['greedy', 'priority_first', 'distance_optimized', 'ai_guided']
        action_idx = action_types.index(action.action_type)
        
        # Q-learning update
        current_q = self.q_table[state_key][action_idx]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward.total_reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state_key][action_idx] = new_q
        
        # Store experience
        self.memory.append((state, action, reward, next_state))
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, state: Dict, action: Dict, success: bool = True) -> float:
        """Calculate reward for an action"""
        base_reward = 0.0
        
        # Success reward
        if success:
            base_reward += 10.0
            
            # Priority bonus
            priority = action.get('priority', 1)
            base_reward += priority * 2.0
            
            # Distance penalty (shorter is better)
            distance = action.get('distance', 0)
            if distance > 0:
                base_reward += max(0, 5.0 - distance)
            
            # Utilization bonus
            utilization = state.get('utilization_rate', 0)
            base_reward += utilization * 0.1
            
        else:
            base_reward -= 5.0
        
        # Wait time penalty
        wait_time = state.get('average_wait_time', 0)
        base_reward -= wait_time * 0.1
        
        return base_reward
    
    def select_action(self, state: Dict) -> Dict:
        """Select action based on current state"""
        # Simple action selection for testing
        return {
            'vehicle_id': f"V{random.randint(1, 10)}",
            'demand_id': f"R{random.randint(1, 5)}",
            'priority': random.randint(1, 5),
            'distance': random.uniform(1.0, 10.0)
        }
    
    def update_q_value(self, state: Dict, action: Dict, reward: float, next_state: Dict):
        """Update Q-value for state-action pair"""
        # Simple Q-learning update for testing
        state_key = str(sorted(state.items()))
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Update Q-value (simplified)
        action_idx = hash(str(action)) % self.action_size
        self.q_table[state_key][action_idx] += self.learning_rate * reward

    def save_model(self, filepath: str = "rl_agent_model.pkl"):
        """Save the trained model"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'performance_history': self.performance_history,
            'episode_rewards': self.episode_rewards
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"RL model saved to {filepath}")
    
    def load_model(self, filepath: str = "rl_agent_model.pkl"):
        """Load a trained model"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                self.q_table = model_data.get('q_table', {})
                self.epsilon = model_data.get('epsilon', 0.1)
                self.performance_history = model_data.get('performance_history', [])
                self.episode_rewards = model_data.get('episode_rewards', [])
                logger.info(f"RL model loaded from {filepath}")
            except Exception as e:
                logger.error(f"Error loading RL model: {e}")

class GeneticAlgorithmOptimizer:
    """Genetic Algorithm for fleet optimization"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
    def optimize_allocation(self, vehicles: List, demands: List) -> List[Tuple[int, int]]:
        """Optimize vehicle-demand allocation using genetic algorithm"""
        if not demands or not vehicles:
            return []
        
        # Initialize population
        population = self._initialize_population(vehicles, demands)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual, vehicles, demands) for individual in population]
            
            # Selection
            parents = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i], parents[i + 1])
                    child1 = self._mutate(child1, vehicles, demands)
                    child2 = self._mutate(child2, vehicles, demands)
                    offspring.extend([child1, child2])
            
            # Replace population
            population = offspring[:self.population_size]
        
        # Return best solution
        final_fitness = [self._evaluate_fitness(individual, vehicles, demands) for individual in population]
        best_idx = np.argmax(final_fitness)
        return population[best_idx]
    
    def _initialize_population(self, vehicles: List, demands: List) -> List[List[Tuple[int, int]]]:
        """Initialize random population of allocations"""
        population = []
        for _ in range(self.population_size):
            individual = []
            available_vehicles = list(range(len(vehicles)))
            random.shuffle(available_vehicles)
            
            for demand_idx in range(len(demands)):
                if available_vehicles:
                    vehicle_idx = available_vehicles.pop(0)
                    individual.append((vehicle_idx, demand_idx))
                else:
                    individual.append((-1, demand_idx))  # Unassigned
            
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual: List[Tuple[int, int]], vehicles: List, demands: List) -> float:
        """Evaluate fitness of an allocation"""
        total_score = 0.0
        
        for vehicle_idx, demand_idx in individual:
            if vehicle_idx == -1:  # Unassigned
                total_score -= 10  # Penalty for unassigned demands
                continue
            
            if vehicle_idx >= len(vehicles) or demand_idx >= len(demands):
                continue
            
            vehicle = vehicles[vehicle_idx]
            demand = demands[demand_idx]
            
            # Distance-based score
            distance = self._calculate_distance(vehicle.location, demand.pickup_location)
            distance_score = max(0, 10 - distance / 5)  # Higher score for closer assignments
            
            # Priority-based score
            priority_score = demand.priority * 2
            
            # Capacity-based score
            capacity_score = 5 if vehicle.capacity >= demand.passengers else 0
            
            total_score += distance_score + priority_score + capacity_score
        
        return total_score
    
    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2) * 111  # Approximate km
    
    def _selection(self, population: List, fitness_scores: List[float]) -> List:
        """Tournament selection"""
        parents = []
        for _ in range(len(population)):
            # Tournament of size 3
            tournament = random.sample(list(zip(population, fitness_scores)), 3)
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents
    
    def _crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def _mutate(self, individual: List, vehicles: List, demands: List) -> List:
        """Mutate individual by swapping assignments"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.copy()
        if len(mutated) > 1:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated

class AgenticFleetOptimizer:
    """Main agentic AI system for continuous fleet optimization"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.mcp_agent = MCPAgent(api_client)
        self.rl_agent = ReinforcementLearningAgent()
        self.ga_optimizer = GeneticAlgorithmOptimizer()
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = {
            'total_assignments': 0,
            'successful_assignments': 0,
            'average_wait_time': 0.0,
            'utilization_rate': 0.0,
            'priority_satisfaction': 0.0
        }
        
        # AI decision making
        self.decision_context = {}
        self.learning_enabled = True
        
    async def optimize_fleet_allocation(self, vehicles: List, demands: List, 
                                      weather_data: Dict, traffic_data: Dict) -> Dict[str, Any]:
        """Main optimization function using multiple AI approaches"""
        
        # 1. Gather multi-source data using MCP
        locations = [v.location for v in vehicles] + [d.pickup_location for d in demands]
        mcp_data = await self.mcp_agent.fetch_multi_source_data(locations)
        
        # 2. Create environment state for RL
        state = self._create_state(vehicles, demands, weather_data, traffic_data)
        
        # 3. Get AI recommendation
        if self.learning_enabled:
            action = self.rl_agent.choose_action(state)
        else:
            action = Action('ai_guided', {'priority_weight': 0.7, 'distance_weight': 0.3, 'utilization_weight': 0.5})
        
        # 4. Apply optimization algorithm based on action
        if action.action_type == 'ai_guided':
            allocations = self.ga_optimizer.optimize_allocation(vehicles, demands)
        else:
            allocations = self._apply_heuristic_optimization(vehicles, demands, action)
        
        # 5. Calculate performance metrics
        metrics = self._calculate_performance_metrics(vehicles, demands, allocations)
        
        # 6. Update RL agent if learning is enabled
        if self.learning_enabled and len(self.optimization_history) > 0:
            prev_state, prev_action, prev_metrics = self.optimization_history[-1]
            reward = self._calculate_reward(prev_metrics, metrics)
            self.rl_agent.update_q_table(prev_state, prev_action, reward, state)
        
        # 7. Store optimization history
        self.optimization_history.append((state, action, metrics))
        
        # 8. Update performance metrics
        self._update_performance_metrics(metrics)
        
        return {
            'allocations': allocations,
            'action_taken': action,
            'performance_metrics': metrics,
            'mcp_data': mcp_data,
            'ai_recommendation': self._generate_ai_recommendation(action, metrics)
        }
    
    def _create_state(self, vehicles: List, demands: List, weather_data: Dict, traffic_data: Dict) -> State:
        """Create environment state from current situation"""
        available_vehicles = len([v for v in vehicles if v.status == 'available'])
        pending_demands = len([d for d in demands if d.status == 'pending'])
        high_priority_demands = len([d for d in demands if d.status == 'pending' and d.priority >= 4])
        
        # Calculate average wait time
        wait_times = [d.estimated_wait_time for d in demands if d.status == 'pending' and d.estimated_wait_time is not None]
        avg_wait_time = np.mean(wait_times) if wait_times else 0.0
        
        # Calculate utilization rate
        busy_vehicles = len([v for v in vehicles if v.status == 'busy'])
        utilization_rate = (busy_vehicles / len(vehicles) * 100) if vehicles else 0.0
        
        # Get weather and traffic conditions
        weather_condition = 'clear'
        traffic_level = 'medium'
        
        if weather_data:
            weather_condition = list(weather_data.values())[0].condition if weather_data else 'clear'
        if traffic_data:
            traffic_level = list(traffic_data.values())[0].congestion_level if traffic_data else 'medium'
        
        return State(
            available_vehicles=available_vehicles,
            pending_demands=pending_demands,
            high_priority_demands=high_priority_demands,
            average_wait_time=avg_wait_time,
            utilization_rate=utilization_rate,
            weather_condition=weather_condition,
            traffic_level=traffic_level,
            time_of_day=datetime.now().hour,
            day_of_week=datetime.now().weekday()
        )
    
    def _apply_heuristic_optimization(self, vehicles: List, demands: List, action: Action) -> List[Tuple[int, int]]:
        """Apply heuristic optimization based on action type"""
        allocations = []
        available_vehicles = [i for i, v in enumerate(vehicles) if v.status == 'available']
        pending_demands = [i for i, d in enumerate(demands) if d.status == 'pending']
        
        if action.action_type == 'priority_first':
            # Sort demands by priority (highest first)
            pending_demands.sort(key=lambda i: demands[i].priority, reverse=True)
        elif action.action_type == 'distance_optimized':
            # Sort demands by distance to nearest available vehicle
            def distance_to_nearest(demand_idx):
                demand = demands[demand_idx]
                min_distance = float('inf')
                for vehicle_idx in available_vehicles:
                    vehicle = vehicles[vehicle_idx]
                    distance = np.sqrt((vehicle.location[0] - demand.pickup_location[0])**2 + 
                                     (vehicle.location[1] - demand.pickup_location[1])**2)
                    min_distance = min(min_distance, distance)
                return min_distance
            
            pending_demands.sort(key=distance_to_nearest)
        
        # Assign vehicles to demands
        for demand_idx in pending_demands:
            if not available_vehicles:
                break
            
            demand = demands[demand_idx]
            best_vehicle_idx = None
            best_score = -1
            
            for vehicle_idx in available_vehicles:
                vehicle = vehicles[vehicle_idx]
                
                # Check capacity
                if vehicle.capacity < demand.passengers:
                    continue
                
                # Calculate score based on action parameters
                distance = np.sqrt((vehicle.location[0] - demand.pickup_location[0])**2 + 
                                 (vehicle.location[1] - demand.pickup_location[1])**2)
                
                score = (
                    action.parameters.get('priority_weight', 0.5) * demand.priority +
                    action.parameters.get('distance_weight', 0.5) * (10 - distance) +
                    action.parameters.get('utilization_weight', 0.5) * (vehicle.capacity - vehicle.current_load)
                )
                
                if score > best_score:
                    best_score = score
                    best_vehicle_idx = vehicle_idx
            
            if best_vehicle_idx is not None:
                allocations.append((best_vehicle_idx, demand_idx))
                available_vehicles.remove(best_vehicle_idx)
        
        return allocations
    
    def _calculate_performance_metrics(self, vehicles: List, demands: List, allocations: List[Tuple[int, int]]) -> Dict[str, float]:
        """Calculate performance metrics for the allocation"""
        total_demands = len([d for d in demands if d.status == 'pending'])
        assigned_demands = len(allocations)
        
        assignment_rate = (assigned_demands / total_demands * 100) if total_demands > 0 else 0
        
        # Calculate priority satisfaction
        high_priority_assigned = 0
        high_priority_total = len([d for d in demands if d.status == 'pending' and d.priority >= 4])
        
        for vehicle_idx, demand_idx in allocations:
            if demand_idx < len(demands) and demands[demand_idx].priority >= 4:
                high_priority_assigned += 1
        
        priority_satisfaction = (high_priority_assigned / high_priority_total * 100) if high_priority_total > 0 else 0
        
        # Calculate utilization improvement
        busy_vehicles = len([v for v in vehicles if v.status == 'busy']) + len(allocations)
        utilization_rate = (busy_vehicles / len(vehicles) * 100) if vehicles else 0
        
        return {
            'assignment_rate': assignment_rate,
            'priority_satisfaction': priority_satisfaction,
            'utilization_rate': utilization_rate,
            'total_assignments': assigned_demands,
            'high_priority_assignments': high_priority_assigned
        }
    
    def _calculate_reward(self, prev_metrics: Dict, current_metrics: Dict) -> Reward:
        """Calculate reward signal for RL agent"""
        assignment_improvement = current_metrics['assignment_rate'] - prev_metrics.get('assignment_rate', 0)
        priority_improvement = current_metrics['priority_satisfaction'] - prev_metrics.get('priority_satisfaction', 0)
        utilization_improvement = current_metrics['utilization_rate'] - prev_metrics.get('utilization_rate', 0)
        
        # Wait time reduction (simplified)
        wait_time_reduction = max(0, assignment_improvement / 10)  # Approximate
        
        return Reward(
            assignment_success=assignment_improvement / 100,
            priority_satisfaction=priority_improvement / 100,
            utilization_improvement=utilization_improvement / 100,
            wait_time_reduction=wait_time_reduction,
            total_reward=0  # Will be calculated in __post_init__
        )
    
    def _generate_ai_recommendation(self, action: Action, metrics: Dict) -> str:
        """Generate human-readable AI recommendation"""
        recommendations = []
        
        if action.action_type == 'ai_guided':
            recommendations.append("🤖 AI-guided genetic algorithm optimization applied")
        elif action.action_type == 'priority_first':
            recommendations.append("🎯 Priority-first assignment strategy used")
        elif action.action_type == 'distance_optimized':
            recommendations.append("📍 Distance-optimized assignment strategy used")
        else:
            recommendations.append("⚡ Greedy assignment strategy used")
        
        if metrics['assignment_rate'] > 80:
            recommendations.append("✅ Excellent assignment rate achieved")
        elif metrics['assignment_rate'] > 60:
            recommendations.append("👍 Good assignment rate")
        else:
            recommendations.append("⚠️ Assignment rate could be improved")
        
        if metrics['priority_satisfaction'] > 90:
            recommendations.append("🏆 Outstanding priority satisfaction")
        elif metrics['priority_satisfaction'] > 70:
            recommendations.append("✅ Good priority handling")
        else:
            recommendations.append("⚠️ High-priority demands need attention")
        
        return "\n".join(recommendations)
    
    def _update_performance_metrics(self, metrics: Dict):
        """Update overall performance metrics"""
        self.performance_metrics['total_assignments'] += metrics.get('total_assignments', 0)
        self.performance_metrics['successful_assignments'] += 1 if metrics.get('assignment_rate', 0) > 50 else 0
        self.performance_metrics['average_wait_time'] = metrics.get('average_wait_time', 0)
        self.performance_metrics['utilization_rate'] = metrics.get('utilization_rate', 0)
        self.performance_metrics['priority_satisfaction'] = metrics.get('priority_satisfaction', 0)
    
    def get_ai_insights(self) -> Dict[str, Any]:
        """Get comprehensive AI insights and recommendations"""
        return {
            'rl_agent_status': {
                'epsilon': self.rl_agent.epsilon,
                'episodes_trained': len(self.rl_agent.episode_rewards),
                'average_reward': np.mean(self.rl_agent.episode_rewards[-10:]) if self.rl_agent.episode_rewards else 0,
                'learning_enabled': self.learning_enabled
            },
            'performance_summary': self.performance_metrics,
            'optimization_history_count': len(self.optimization_history),
            'recommendations': self._generate_system_recommendations()
        }
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        if self.performance_metrics['utilization_rate'] < 70:
            recommendations.append("💡 Consider increasing fleet size or optimizing vehicle distribution")
        
        if self.performance_metrics['priority_satisfaction'] < 80:
            recommendations.append("🎯 Implement priority-based pre-allocation for high-priority demands")
        
        if self.rl_agent.epsilon > 0.5:
            recommendations.append("🧠 RL agent is still exploring - let it learn more patterns")
        else:
            recommendations.append("🎓 RL agent has learned optimal patterns - good performance expected")
        
        return recommendations
    
    def save_ai_state(self, filepath: str = "ai_agent_state.json"):
        """Save AI agent state"""
        state_data = {
            'performance_metrics': self.performance_metrics,
            'optimization_history_count': len(self.optimization_history),
            'learning_enabled': self.learning_enabled,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        # Save RL model
        self.rl_agent.save_model()
        
        logger.info(f"AI agent state saved to {filepath}")
    
    def load_ai_state(self, filepath: str = "ai_agent_state.json"):
        """Load AI agent state"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    state_data = json.load(f)
                
                self.performance_metrics = state_data.get('performance_metrics', self.performance_metrics)
                self.learning_enabled = state_data.get('learning_enabled', True)
                
                logger.info(f"AI agent state loaded from {filepath}")
            except Exception as e:
                logger.error(f"Error loading AI agent state: {e}")

# Global AI agent instance
ai_agent = None

def initialize_ai_agent(api_client):
    """Initialize the global AI agent"""
    global ai_agent
    ai_agent = AgenticFleetOptimizer(api_client)
    return ai_agent

def get_ai_agent():
    """Get the global AI agent instance"""
    return ai_agent
