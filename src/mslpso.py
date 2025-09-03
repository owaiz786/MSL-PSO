# src/mslpso.py
import numpy as np
from tqdm import tqdm

class MSLPSO:
    def __init__(self, fitness_evaluator, num_particles, num_features, num_swarms, generations):
        self.evaluator = fitness_evaluator
        self.num_particles = num_particles
        self.num_features = num_features
        self.num_swarms = num_swarms
        self.generations = generations

        # PSO parameters
        self.w = 0.5
        self.c1 = 0.8
        self.c2 = 0.9

        self.swarms = self._initialize_swarms()
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        self.convergence_history = []
        
        # ✅ ---- NEW: Add a list to store the full history for animation ----
        self.animation_history = []

    def _initialize_swarms(self):
      swarms = []
      for _ in range(self.num_swarms):
        positions = np.random.randint(0, 2, (self.num_particles, self.num_features))
        velocities = np.random.randn(self.num_particles, self.num_features) * 0.1
        swarm = {
            'positions': positions,
            'velocities': velocities,
            'pbest_positions': np.copy(positions),
            'pbest_fitness': np.full(self.num_particles, -np.inf),
            'swarm_best_position': None,
            'swarm_best_fitness': -np.inf
        }
        swarms.append(swarm)
      return swarms

    def _update_velocity(self, swarm):
        r1 = np.random.rand(self.num_particles, self.num_features)
        r2 = np.random.rand(self.num_particles, self.num_features)
        
        cognitive_velocity = self.c1 * r1 * (swarm['pbest_positions'] - swarm['positions'])
        social_velocity = self.c2 * r2 * (swarm['swarm_best_position'] - swarm['positions'])
        
        swarm['velocities'] = self.w * swarm['velocities'] + cognitive_velocity + social_velocity

    def _update_position(self, swarm):
        probabilities = 1 / (1 + np.exp(-swarm['velocities']))
        random_values = np.random.rand(self.num_particles, self.num_features)
        swarm['positions'] = (probabilities > random_values).astype(int)

    def optimize(self):
        for gen in tqdm(range(self.generations), desc="Optimizing"):
            # ✅ ---- NEW: Record the state for the current generation ----
            current_gen_positions = []
            current_gen_swarm_ids = []
            current_gen_fitness = []
            for s_idx, swarm in enumerate(self.swarms):
                current_gen_positions.append(swarm['positions'])
                current_gen_swarm_ids.extend([s_idx] * self.num_particles)
                current_gen_fitness.append(swarm['pbest_fitness'])

            self.animation_history.append({
                'positions': np.vstack(current_gen_positions),
                'swarm_ids': np.array(current_gen_swarm_ids),
                'fitness': np.concatenate(current_gen_fitness)
            })
            # ✅ ---- END NEW ----

            for s_idx, swarm in enumerate(self.swarms):
                for p_idx in range(self.num_particles):
                    fitness = self.evaluator.evaluate(swarm['positions'][p_idx])
                    
                    if fitness > swarm['pbest_fitness'][p_idx]:
                        swarm['pbest_fitness'][p_idx] = fitness
                        swarm['pbest_positions'][p_idx] = np.copy(swarm['positions'][p_idx])
                
                best_particle_in_swarm_idx = np.argmax(swarm['pbest_fitness'])
                if swarm['pbest_fitness'][best_particle_in_swarm_idx] > swarm['swarm_best_fitness']:
                    swarm['swarm_best_fitness'] = swarm['pbest_fitness'][best_particle_in_swarm_idx]
                    swarm['swarm_best_position'] = np.copy(swarm['pbest_positions'][best_particle_in_swarm_idx])
            
            for swarm in self.swarms:
                if swarm['swarm_best_fitness'] > self.global_best_fitness:
                    self.global_best_fitness = swarm['swarm_best_fitness']
                    self.global_best_position = np.copy(swarm['swarm_best_position'])
            
            self.convergence_history.append(self.global_best_fitness)
            
            for swarm in self.swarms:
                self._update_velocity(swarm)
                self._update_position(swarm)

        print(f"\nOptimization Finished!")
        print(f"Best Fitness: {self.global_best_fitness}")
        print(f"Number of features selected: {np.sum(self.global_best_position)}")
        
        # ✅ ---- NEW: Return the animation history ----
        return self.global_best_position, self.convergence_history, self.animation_history