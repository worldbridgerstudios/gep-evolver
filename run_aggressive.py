"""
GEP EVOLVER — Aggressive Push for 10^-8
========================================
Based on the discovery that the formula uses 36, 43, 6 - let's seed heavily.
"""

from evolve import GEPEvolver, GEPConfig
from karva import Gene, TERMINAL_SYMBOLS, OPERATORS, evaluate_gene, gene_to_expression
import math
import random

# Target 
ZETA_3 = 1.2020569031595942853997381615114

def verify_formula():
    """Verify our known formula."""
    # Our formula: π³ × √(66/43913)
    our_value = (math.pi ** 3) * math.sqrt(66 / 43913)
    error = abs(our_value - ZETA_3) / ZETA_3
    print(f"Our formula: π³√(66/43913) = {our_value:.15f}")
    print(f"Target ζ(3):               = {ZETA_3:.15f}")
    print(f"Relative error: {error:.2e}")
    print()

def create_diverse_seeds(head_length: int, count: int = 50) -> list:
    """Create diverse seeds focusing on triangular numbers."""
    seeds = []
    operators = list(OPERATORS.keys())
    terminals = TERMINAL_SYMBOLS
    tail_length = head_length + 1  # For max_arity = 2
    
    # Key terminals: a=π, e=3, f=6, j=36, k=43, l=66, n=666, o=43913
    key_terms = ['a', 'e', 'f', 'j', 'k', 'l', 'n', 'o']
    
    for _ in range(count):
        # Random head with bias toward our key operators and terms
        head = []
        for i in range(head_length):
            if i == 0:
                # Start with operator
                head.append(random.choice(operators))
            elif random.random() < 0.5:
                head.append(random.choice(operators))
            else:
                head.append(random.choice(key_terms))
        
        # Random tail from key terms
        tail = [random.choice(key_terms) for _ in range(tail_length)]
        
        seq = ''.join(head) + ''.join(tail)
        seeds.append(Gene(sequence=seq, head_length=head_length))
    
    return seeds

if __name__ == "__main__":
    print("=" * 70)
    print("GEP FORMULA EVOLVER — AGGRESSIVE PUSH")
    print("=" * 70)
    print()
    
    verify_formula()
    
    # Very aggressive config
    config = GEPConfig(
        target=ZETA_3,
        population_size=1000,     # Big population
        head_length=15,           # Long head
        generations=5000,         # Many generations
        mutation_rate=0.1,        # High mutation
        is_transposition_rate=0.2,
        ris_transposition_rate=0.2,
        one_point_recomb_rate=0.5,
        two_point_recomb_rate=0.3,
        tournament_size=7,        # Strong selection
        elitism_count=5,
        parsimony_weight=0.005,   # Very low parsimony (allow complex)
        use_log_scale=True,
        stagnation_limit=1000     # Patient
    )
    
    evolver = GEPEvolver(config)
    evolver.initialize_population()
    
    # Seed with diverse formulas
    seeds = create_diverse_seeds(config.head_length, count=100)
    for i, seed in enumerate(seeds):
        if i < len(evolver.population):
            evolver.population[i] = seed
    
    print(f"Population: {config.population_size}, Head length: {config.head_length}")
    print(f"Running up to {config.generations} generations...")
    print("-" * 70)
    
    stats = evolver.run(verbose=True, log_interval=200)
    
    print("\n" + "=" * 70)
    print("TOP 15 UNIQUE FORMULAS:")
    print("=" * 70)
    
    # Deduplicate
    seen = set()
    unique = []
    for gene, result in zip(evolver.population, evolver.results):
        expr = result.expression
        if expr not in seen:
            seen.add(expr)
            unique.append((gene, result))
    
    unique.sort(key=lambda x: x[1].raw_fitness, reverse=True)
    
    for i, (gene, result) in enumerate(unique[:15]):
        rel_err = result.relative_error * 100
        print(f"\n{i+1}. Error: {rel_err:.10f}% | Fitness: {result.raw_fitness:.2f}")
        print(f"   = {result.value:.15f}")
        print(f"   {result.expression}")
