"""
FITNESS EVALUATION
==================
How do we know if a formula is "good"?

For constant approximation (like ζ(3)):
- Primary: how close is the result to target?
- Secondary: how simple is the expression? (parsimony)
- Optional: does it use "preferred" structures? (triangular bonus)

Fitness is typically scaled 0-1000 for tournament selection.
"""

import math
from typing import Callable, Optional
from dataclasses import dataclass

from karva import Gene, evaluate_gene, effective_length, karva_to_tree, tree_depth

# =============================================================================
# TARGET CONSTANTS
# =============================================================================

# Apéry's constant
ZETA_3 = 1.2020569031595942853997381615114

# Other targets we might want
ZETA_5 = 1.0369277551433699263313654864570
ZETA_7 = 1.0083492773819228268397975498497
EULER_MASCHERONI = 0.5772156649015328606065120900824

# =============================================================================
# FITNESS FUNCTIONS
# =============================================================================

@dataclass
class FitnessResult:
    """Complete fitness evaluation result."""
    raw_fitness: float      # 0 to max_fitness
    value: float            # Expression result
    error: float            # Absolute error from target
    relative_error: float   # Relative error
    complexity: int         # Expression complexity
    expression: str         # Human-readable expression

def is_valid_number(value) -> bool:
    """Check if value is a valid real number."""
    if isinstance(value, complex):
        return False
    try:
        return not (math.isnan(value) or math.isinf(value))
    except (TypeError, ValueError):
        return False

def fitness_proximity(value: float, target: float, max_fitness: float = 1000.0) -> float:
    """
    Proximity-based fitness.
    
    Uses Ferreira's formula: fitness = max_fitness / (1 + |error|)
    Perfect match → max_fitness
    Larger errors → lower fitness
    """
    if not is_valid_number(value):
        return 0.0
    
    error = abs(value - target)
    return max_fitness / (1.0 + error)

def fitness_relative(value: float, target: float, max_fitness: float = 1000.0) -> float:
    """
    Relative error fitness (better for small targets).
    
    fitness = max_fitness / (1 + relative_error)
    """
    if not is_valid_number(value):
        return 0.0
    
    if target == 0:
        return fitness_proximity(value, target, max_fitness)
    
    rel_error = abs(value - target) / abs(target)
    return max_fitness / (1.0 + rel_error * 100)  # Scale relative error

def fitness_log_error(value: float, target: float, max_fitness: float = 1000.0) -> float:
    """
    Logarithmic error fitness (rewards precision exponentially).
    
    fitness = max_fitness * (1 - log10(1 + relative_error))
    Capped to non-negative.
    """
    if not is_valid_number(value):
        return 0.0
    
    if target == 0:
        error = abs(value)
    else:
        error = abs(value - target) / abs(target)
    
    # Log scale: error of 10^-8 gives fitness ≈ 800
    log_penalty = math.log10(1 + error * 1e8) / 8
    return max(0, max_fitness * (1 - log_penalty))

def fitness_with_parsimony(value: float, target: float, complexity: int,
                           parsimony_weight: float = 0.1,
                           max_fitness: float = 1000.0) -> float:
    """
    Fitness with parsimony pressure.
    
    Simpler expressions get a bonus.
    """
    base_fitness = fitness_proximity(value, target, max_fitness)
    
    # Parsimony bonus: smaller complexity = higher bonus
    # Max complexity ~15, so bonus ranges 0-1.5
    parsimony_bonus = parsimony_weight * (15 - min(complexity, 15))
    
    return base_fitness * (1 + parsimony_bonus)

# =============================================================================
# COMPLETE EVALUATOR
# =============================================================================

class FitnessEvaluator:
    """
    Configurable fitness evaluator.
    """
    def __init__(self, 
                 target: float,
                 max_fitness: float = 1000.0,
                 parsimony_weight: float = 0.05,
                 use_log_scale: bool = True):
        self.target = target
        self.max_fitness = max_fitness
        self.parsimony_weight = parsimony_weight
        self.use_log_scale = use_log_scale
    
    def evaluate(self, gene: Gene) -> FitnessResult:
        """Full evaluation of a gene."""
        from karva import gene_to_expression
        
        value = evaluate_gene(gene)
        complexity = effective_length(gene)
        
        if not is_valid_number(value):
            return FitnessResult(
                raw_fitness=0.0,
                value=float('nan'),
                error=float('inf'),
                relative_error=float('inf'),
                complexity=complexity,
                expression=gene_to_expression(gene)
            )
        
        error = abs(value - self.target)
        rel_error = error / abs(self.target) if self.target != 0 else error
        
        # Calculate fitness
        if self.use_log_scale:
            base_fitness = fitness_log_error(value, self.target, self.max_fitness)
        else:
            base_fitness = fitness_proximity(value, self.target, self.max_fitness)
        
        # Apply parsimony
        parsimony_bonus = self.parsimony_weight * (15 - min(complexity, 15))
        raw_fitness = base_fitness * (1 + parsimony_bonus)
        
        return FitnessResult(
            raw_fitness=raw_fitness,
            value=value,
            error=error,
            relative_error=rel_error,
            complexity=complexity,
            expression=gene_to_expression(gene)
        )

# =============================================================================
# SELECTION
# =============================================================================

def tournament_select(population: list, fitnesses: list, tournament_size: int = 3) -> int:
    """
    Tournament selection.
    
    Randomly select tournament_size individuals, return the fittest.
    Returns index into population.
    """
    import random
    
    contestants = random.sample(range(len(population)), tournament_size)
    winner = max(contestants, key=lambda i: fitnesses[i])
    return winner

def roulette_select(fitnesses: list) -> int:
    """
    Roulette wheel selection (fitness-proportionate).
    
    Probability of selection proportional to fitness.
    """
    import random
    
    total = sum(fitnesses)
    if total == 0:
        return random.randint(0, len(fitnesses) - 1)
    
    r = random.random() * total
    cumulative = 0
    for i, f in enumerate(fitnesses):
        cumulative += f
        if cumulative >= r:
            return i
    
    return len(fitnesses) - 1

def elitism_select(population: list, fitnesses: list, n_elite: int = 1) -> list:
    """
    Select the top n_elite individuals.
    
    These are copied unchanged to the next generation.
    """
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    return [population[i] for i in sorted_indices[:n_elite]]
