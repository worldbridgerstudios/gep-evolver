"""
GEP EVOLUTION ENGINE
====================
The main loop that drives evolution.

Population → Evaluate → Select → Reproduce → Mutate → Repeat

Based on Ferreira's architecture with configurable parameters.
"""

import random
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import math
import time

from karva import Gene, random_gene, evaluate_gene, gene_to_expression, effective_length
from operators import (
    mutate_gene, transpose_is, transpose_ris,
    recombine_one_point, recombine_two_point
)
from fitness import FitnessEvaluator, FitnessResult, tournament_select, elitism_select

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GEPConfig:
    """Configuration for GEP evolution."""
    # Target
    target: float = 1.2020569031595942  # ζ(3)
    
    # Population
    population_size: int = 100
    head_length: int = 8
    
    # Evolution
    generations: int = 500
    mutation_rate: float = 0.05
    is_transposition_rate: float = 0.1
    ris_transposition_rate: float = 0.1
    one_point_recomb_rate: float = 0.3
    two_point_recomb_rate: float = 0.3
    
    # Selection
    tournament_size: int = 3
    elitism_count: int = 1
    
    # Fitness
    max_fitness: float = 1000.0
    parsimony_weight: float = 0.02
    use_log_scale: bool = True
    
    # Stopping
    target_fitness: float = 999.0  # Stop if we hit this
    stagnation_limit: int = 100    # Stop if no improvement for N generations

# =============================================================================
# EVOLUTION STATE
# =============================================================================

@dataclass
class EvolutionStats:
    """Statistics from evolution run."""
    generation: int = 0
    best_fitness: float = 0.0
    best_value: float = 0.0
    best_error: float = float('inf')
    best_expression: str = ""
    best_gene: Gene = None
    avg_fitness: float = 0.0
    generations_without_improvement: int = 0
    start_time: float = 0.0
    elapsed_time: float = 0.0
    history: List[Tuple[int, float, float, str]] = field(default_factory=list)

# =============================================================================
# EVOLUTION ENGINE
# =============================================================================

class GEPEvolver:
    """Main GEP evolution engine."""
    
    def __init__(self, config: GEPConfig = None):
        self.config = config or GEPConfig()
        self.evaluator = FitnessEvaluator(
            target=self.config.target,
            max_fitness=self.config.max_fitness,
            parsimony_weight=self.config.parsimony_weight,
            use_log_scale=self.config.use_log_scale
        )
        self.population: List[Gene] = []
        self.fitnesses: List[float] = []
        self.results: List[FitnessResult] = []
        self.stats = EvolutionStats()
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = [
            random_gene(head_length=self.config.head_length)
            for _ in range(self.config.population_size)
        ]
    
    def evaluate_population(self):
        """Evaluate fitness of all individuals."""
        self.results = [self.evaluator.evaluate(gene) for gene in self.population]
        self.fitnesses = [r.raw_fitness for r in self.results]
    
    def update_stats(self):
        """Update evolution statistics."""
        best_idx = max(range(len(self.fitnesses)), key=lambda i: self.fitnesses[i])
        best_result = self.results[best_idx]
        
        # Check for improvement
        if best_result.raw_fitness > self.stats.best_fitness:
            self.stats.best_fitness = best_result.raw_fitness
            self.stats.best_value = best_result.value
            self.stats.best_error = best_result.error
            self.stats.best_expression = best_result.expression
            self.stats.best_gene = self.population[best_idx]
            self.stats.generations_without_improvement = 0
        else:
            self.stats.generations_without_improvement += 1
        
        self.stats.avg_fitness = sum(self.fitnesses) / len(self.fitnesses)
        self.stats.elapsed_time = time.time() - self.stats.start_time
        
        # Record history
        self.stats.history.append((
            self.stats.generation,
            self.stats.best_fitness,
            self.stats.best_error,
            self.stats.best_expression
        ))
    
    def select_parent(self) -> Gene:
        """Select a parent for reproduction."""
        idx = tournament_select(self.population, self.fitnesses, self.config.tournament_size)
        return self.population[idx]
    
    def create_offspring(self, parent: Gene) -> Gene:
        """Create offspring from parent through genetic operations."""
        offspring = Gene(sequence=parent.sequence, head_length=parent.head_length)
        
        # Mutation
        if random.random() < self.config.mutation_rate:
            offspring = mutate_gene(offspring, mutation_rate=0.1)
        
        # IS Transposition
        if random.random() < self.config.is_transposition_rate:
            offspring = transpose_is(offspring)
        
        # RIS Transposition
        if random.random() < self.config.ris_transposition_rate:
            offspring = transpose_ris(offspring)
        
        return offspring
    
    def evolve_generation(self):
        """Evolve one generation."""
        new_population = []
        
        # Elitism: copy best individuals unchanged
        elite = elitism_select(self.population, self.fitnesses, self.config.elitism_count)
        new_population.extend(elite)
        
        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Recombination?
            if random.random() < self.config.one_point_recomb_rate:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child1, child2 = recombine_one_point(parent1, parent2)
                new_population.append(self.create_offspring(child1))
                if len(new_population) < self.config.population_size:
                    new_population.append(self.create_offspring(child2))
            elif random.random() < self.config.two_point_recomb_rate:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child1, child2 = recombine_two_point(parent1, parent2)
                new_population.append(self.create_offspring(child1))
                if len(new_population) < self.config.population_size:
                    new_population.append(self.create_offspring(child2))
            else:
                # Asexual reproduction
                parent = self.select_parent()
                new_population.append(self.create_offspring(parent))
        
        self.population = new_population[:self.config.population_size]
    
    def should_stop(self) -> bool:
        """Check stopping conditions."""
        if self.stats.best_fitness >= self.config.target_fitness:
            return True
        if self.stats.generations_without_improvement >= self.config.stagnation_limit:
            return True
        return False
    
    def run(self, verbose: bool = True, log_interval: int = 50) -> EvolutionStats:
        """Run the full evolution."""
        self.stats = EvolutionStats()
        self.stats.start_time = time.time()
        
        if verbose:
            print(f"GEP Evolution targeting {self.config.target}")
            print(f"Population: {self.config.population_size}, Head length: {self.config.head_length}")
            print("-" * 70)
        
        # Initialize
        self.initialize_population()
        
        for gen in range(self.config.generations):
            self.stats.generation = gen
            
            # Evaluate
            self.evaluate_population()
            self.update_stats()
            
            # Log progress
            if verbose and (gen % log_interval == 0 or gen < 10):
                rel_err = self.stats.best_error / abs(self.config.target) * 100
                print(f"Gen {gen:4d} | Best: {self.stats.best_value:.10f} | "
                      f"Error: {rel_err:.6f}% | "
                      f"Fitness: {self.stats.best_fitness:.2f}")
            
            # Check stopping
            if self.should_stop():
                if verbose:
                    print(f"\nStopping at generation {gen}")
                break
            
            # Evolve
            self.evolve_generation()
        
        # Final evaluation
        self.evaluate_population()
        self.update_stats()
        
        if verbose:
            print("-" * 70)
            print(f"BEST RESULT:")
            print(f"  Value: {self.stats.best_value:.15f}")
            print(f"  Target: {self.config.target:.15f}")
            print(f"  Error: {self.stats.best_error:.2e}")
            print(f"  Relative Error: {self.stats.best_error/abs(self.config.target)*100:.8f}%")
            print(f"  Expression: {self.stats.best_expression}")
            print(f"  Gene: {self.stats.best_gene.sequence if self.stats.best_gene else 'N/A'}")
            print(f"  Time: {self.stats.elapsed_time:.2f}s")
        
        return self.stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GEP FORMULA EVOLVER")
    print("=" * 70)
    print()
    
    # Configure for ζ(3)
    config = GEPConfig(
        target=1.2020569031595942853997381615114,  # ζ(3)
        population_size=200,
        head_length=10,
        generations=1000,
        mutation_rate=0.044,
        is_transposition_rate=0.1,
        ris_transposition_rate=0.1,
        one_point_recomb_rate=0.3,
        two_point_recomb_rate=0.3,
        tournament_size=3,
        elitism_count=2,
        parsimony_weight=0.02,
        use_log_scale=True,
        stagnation_limit=200
    )
    
    evolver = GEPEvolver(config)
    stats = evolver.run(verbose=True, log_interval=50)
    
    print("\n" + "=" * 70)
    print("TOP 5 FORMULAS FOUND:")
    print("=" * 70)
    
    # Sort by fitness and show top 5
    sorted_results = sorted(
        zip(evolver.population, evolver.results),
        key=lambda x: x[1].raw_fitness,
        reverse=True
    )
    
    for i, (gene, result) in enumerate(sorted_results[:5]):
        rel_err = result.relative_error * 100
        print(f"\n{i+1}. Fitness: {result.raw_fitness:.2f}")
        print(f"   Value: {result.value:.12f}")
        print(f"   Error: {rel_err:.8f}%")
        print(f"   Expr:  {result.expression}")
        print(f"   Gene:  {gene.sequence}")
