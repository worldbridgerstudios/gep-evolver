"""
GEP EVOLVER — Seeded with Known Formula
========================================
Inject the π³√(66/43913) formula and see if GEP can improve it.
"""

from evolve import GEPEvolver, GEPConfig
from karva import Gene, TERMINAL_SYMBOLS, OPERATORS, evaluate_gene, gene_to_expression
import math
import random

ZETA_3 = 1.2020569031595942853997381615114

def create_seeded_population(head_length: int, pop_size: int) -> list:
    """Create population seeded with known good formulas."""
    population = []
    operators = list(OPERATORS.keys())
    terminals = TERMINAL_SYMBOLS
    tail_length = head_length + 1
    
    # THE EXACT FORMULA: *^Qae/lo (π³ × √(66/43913))
    # Terminals: a=π, e=3, l=66, o=43913
    exact_head = "*^Qae/lo"
    
    # Pad head to required length
    exact_head = exact_head + random.choices(terminals, k=head_length - len(exact_head))
    exact_head = ''.join(exact_head[:head_length])
    exact_tail = ''.join(random.choices(terminals, k=tail_length))
    exact_gene = Gene(sequence=exact_head + exact_tail, head_length=head_length)
    
    # Add exact formula multiple times
    for _ in range(20):
        head = "*^Qae/lo" + ''.join(random.choices(terminals, k=head_length - 8))
        tail = ''.join(random.choices(terminals, k=tail_length))
        population.append(Gene(sequence=head + tail, head_length=head_length))
    
    # Add variations of the formula
    variations = [
        "*^Q/loae",  # sqrt first, different order
        "*Q/^loae",  # different structure
        "^*Qae/lo",  # swap * and ^
        "*^aQ/loe",  # move a (π) around
        "/Q*^aelo",  # different root
    ]
    for var in variations:
        for _ in range(5):
            head = var + ''.join(random.choices(terminals, k=head_length - len(var)))
            tail = ''.join(random.choices(terminals, k=tail_length))
            population.append(Gene(sequence=head + tail, head_length=head_length))
    
    # Add random individuals focusing on key operators: *, ^, Q, /
    # and key terminals: a(π), e(3), l(66), o(43913), j(36), k(43)
    key_ops = ['*', '^', 'Q', '/']
    key_terms = ['a', 'e', 'l', 'o', 'j', 'k', 'n']  # n=666
    
    while len(population) < pop_size:
        head = []
        for i in range(head_length):
            if i < 4:
                head.append(random.choice(key_ops))
            elif random.random() < 0.4:
                head.append(random.choice(key_ops))
            else:
                head.append(random.choice(key_terms))
        tail = [random.choice(key_terms) for _ in range(tail_length)]
        population.append(Gene(sequence=''.join(head) + ''.join(tail), head_length=head_length))
    
    return population[:pop_size]

if __name__ == "__main__":
    print("=" * 70)
    print("GEP EVOLVER — SEEDED WITH KNOWN FORMULA")
    print("=" * 70)
    print()
    
    # Our formula baseline
    our_value = (math.pi ** 3) * math.sqrt(66 / 43913)
    our_error = abs(our_value - ZETA_3) / ZETA_3
    print(f"Our formula: π³√(66/43913)")
    print(f"  Value: {our_value:.15f}")
    print(f"  Error: {our_error:.2e}")
    print()
    
    # Config: big population, many generations, aggressive
    config = GEPConfig(
        target=ZETA_3,
        population_size=2000,
        head_length=12,
        generations=3000,
        mutation_rate=0.05,       # Lower mutation to preserve good structures
        is_transposition_rate=0.1,
        ris_transposition_rate=0.1,
        one_point_recomb_rate=0.6,  # High recombination to mix good pieces
        two_point_recomb_rate=0.3,
        tournament_size=5,
        elitism_count=10,         # Keep more elite
        parsimony_weight=0.002,
        use_log_scale=True,
        stagnation_limit=1500
    )
    
    evolver = GEPEvolver(config)
    evolver.population = create_seeded_population(config.head_length, config.population_size)
    
    # Check seeds
    print("Checking seeded formulas...")
    for i in range(5):
        gene = evolver.population[i]
        val = evaluate_gene(gene)
        expr = gene_to_expression(gene)
        err = abs(val - ZETA_3) / ZETA_3 if val != 0 else float('inf')
        print(f"  Seed {i}: {val:.10f} (err={err:.2e}) - {expr[:60]}...")
    print()
    
    print(f"Population: {config.population_size}, Head: {config.head_length}")
    print("-" * 70)
    
    stats = evolver.run(verbose=True, log_interval=200)
    
    print("\n" + "=" * 70)
    print("TOP 20 UNIQUE FORMULAS:")
    print("=" * 70)
    
    # Deduplicate by value (formulas with same value within 1e-15)
    seen_values = []
    unique = []
    for gene, result in zip(evolver.population, evolver.results):
        is_new = True
        for sv in seen_values:
            if abs(result.value - sv) < 1e-12:
                is_new = False
                break
        if is_new:
            seen_values.append(result.value)
            unique.append((gene, result))
    
    unique.sort(key=lambda x: x[1].raw_fitness, reverse=True)
    
    for i, (gene, result) in enumerate(unique[:20]):
        rel_err = result.relative_error
        print(f"\n{i+1}. Error: {rel_err:.2e} | Fitness: {result.raw_fitness:.2f}")
        print(f"   = {result.value:.15f}")
        print(f"   {result.expression}")
        print(f"   Gene: {gene.sequence}")
