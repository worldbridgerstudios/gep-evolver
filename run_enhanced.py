"""
GEP EVOLVER — Enhanced Run
===========================
Targeting ζ(3) with better parameters and seeded population.
"""

from evolve import GEPEvolver, GEPConfig
from karva import Gene, TERMINAL_SYMBOLS, OPERATORS
import math

# Known good formula: π³ × √(66/43913)
# In K-expression we need: * (^ π 3) (Q (/ 66 43913))
# Terminals: a=π, e=3, l=66, o=43913

def seed_known_formulas(head_length: int) -> list:
    """Create genes encoding known good approximations."""
    seeds = []
    
    # We can't easily encode the exact formula due to structure,
    # but we can seed with partial structures that include the key numbers
    
    # Build some seeds that use π, 3, 66, 43913
    operators = list(OPERATORS.keys())
    terminals = TERMINAL_SYMBOLS
    
    # Seed 1: Start with multiplication and power involving π
    # *^Qa/lo... (aiming for π^3 * sqrt(66/43913))
    seed1 = "*^aQ/lo" + "e" * (head_length - 7)  # π^something * sqrt(66/43913)
    tail_len = head_length * 1 + 1  # MAX_ARITY=2, so tail = h*(2-1)+1 = h+1
    seed1 += "aelo" + "a" * (tail_len - 4)
    
    # Seed 2: Try Q*^a... structure
    seed2 = "Q/*^alo" + "e" * (head_length - 7)
    seed2 += "aelo" + "a" * (tail_len - 4)
    
    # Seed 3: Focus on the 66/43913 ratio
    seed3 = "*a/Q^lo" + "e" * (head_length - 7)
    seed3 += "aelo" + "a" * (tail_len - 4)
    
    for seq in [seed1, seed2, seed3]:
        if len(seq) >= head_length + tail_len:
            seq = seq[:head_length + tail_len]
            seeds.append(Gene(sequence=seq, head_length=head_length))
    
    return seeds

if __name__ == "__main__":
    print("=" * 70)
    print("GEP FORMULA EVOLVER — ENHANCED RUN")
    print("=" * 70)
    print()
    
    # Aggressive exploration config
    config = GEPConfig(
        target=1.2020569031595942853997381615114,  # ζ(3)
        population_size=500,      # Larger population
        head_length=12,           # Longer head = more complex expressions
        generations=2000,         # More generations
        mutation_rate=0.08,       # Higher mutation
        is_transposition_rate=0.15,
        ris_transposition_rate=0.15,
        one_point_recomb_rate=0.4,
        two_point_recomb_rate=0.3,
        tournament_size=5,        # Stronger selection pressure
        elitism_count=3,
        parsimony_weight=0.01,    # Less parsimony pressure (allow complex)
        use_log_scale=True,
        stagnation_limit=500      # Don't give up early
    )
    
    evolver = GEPEvolver(config)
    
    # Inject seeded formulas
    evolver.initialize_population()
    seeds = seed_known_formulas(config.head_length)
    for i, seed in enumerate(seeds):
        if i < len(evolver.population):
            evolver.population[i] = seed
    
    # Run with more verbose output
    stats = evolver.run(verbose=True, log_interval=100)
    
    print("\n" + "=" * 70)
    print("TOP 10 UNIQUE FORMULAS:")
    print("=" * 70)
    
    # Deduplicate by expression
    seen_exprs = set()
    unique_results = []
    for gene, result in zip(evolver.population, evolver.results):
        expr = result.expression
        if expr not in seen_exprs:
            seen_exprs.add(expr)
            unique_results.append((gene, result))
    
    # Sort by fitness
    unique_results.sort(key=lambda x: x[1].raw_fitness, reverse=True)
    
    for i, (gene, result) in enumerate(unique_results[:10]):
        rel_err = result.relative_error * 100
        print(f"\n{i+1}. Fitness: {result.raw_fitness:.2f} | Error: {rel_err:.8f}%")
        print(f"   Value: {result.value:.15f}")
        print(f"   Expr:  {result.expression}")
