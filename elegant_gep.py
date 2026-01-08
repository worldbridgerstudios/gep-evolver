"""
GEP WITH ELEGANT PRIMITIVES
============================
Constrain terminals to small integers and let GEP BUILD the structure.

Instead of: find 43913, then realize it's 66×666-43
We want:    find T(11) × T(T(8)) - 43 directly

Terminal set (elegant primitives):
- π, φ (transcendentals worth keeping)
- 1, 2, 3, 4, 5, 6, 8, 11, 36 (small integers, key triangular indices)

Operators:
- +, -, *, /, ^, √
- T() — triangular function (this is KEY)

If GEP finds T(T(8)), it's already elegant!
"""

import math
import random
from dataclasses import dataclass
from typing import List, Callable

# =============================================================================
# ELEGANT PRIMITIVES
# =============================================================================

@dataclass
class Operator:
    symbol: str
    arity: int
    func: Callable
    
EPSILON = 1e-10
MAX_VALUE = 1e15

def protected_div(a, b):
    if abs(b) < EPSILON: return 1.0
    return max(-MAX_VALUE, min(MAX_VALUE, a / b))

def protected_pow(a, b):
    b = max(-20, min(20, b))
    if b < 0 and abs(a) < EPSILON: return 1.0
    if a < 0 and b != int(b): a = abs(a)
    try:
        result = a ** b
        if isinstance(result, complex): return abs(result)
        return max(-MAX_VALUE, min(MAX_VALUE, result))
    except: return 1.0

def triangular(n):
    """T(n) = n(n+1)/2"""
    n = int(max(0, min(1000, abs(n))))
    return n * (n + 1) // 2

OPERATORS = {
    '+': Operator('+', 2, lambda a,b: a+b),
    '-': Operator('-', 2, lambda a,b: a-b),
    '*': Operator('*', 2, lambda a,b: a*b),
    '/': Operator('/', 2, protected_div),
    '^': Operator('^', 2, protected_pow),
    'Q': Operator('Q', 1, lambda a: math.sqrt(abs(a))),
    'T': Operator('T', 1, triangular),  # THE KEY OPERATOR
}

MAX_ARITY = 2

# ELEGANT TERMINALS — small integers only!
CONSTANTS = {
    'π': math.pi,
    'φ': (1 + math.sqrt(5)) / 2,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '8': 8,
    '11': 11,
    '36': 36,  # T(8), useful shortcut
}

TERMINAL_SYMBOLS = list('abcdefghijk')[:len(CONSTANTS)]
TERMINAL_MAP = dict(zip(TERMINAL_SYMBOLS, CONSTANTS.values()))
TERMINAL_NAMES = dict(zip(TERMINAL_SYMBOLS, CONSTANTS.keys()))

# =============================================================================
# GENE STRUCTURE (from karva.py but self-contained)
# =============================================================================

@dataclass
class Gene:
    sequence: str
    head_length: int
    
    @property
    def tail_length(self): return self.head_length * (MAX_ARITY - 1) + 1
    @property
    def head(self): return self.sequence[:self.head_length]
    @property
    def tail(self): return self.sequence[self.head_length:]

@dataclass 
class TreeNode:
    symbol: str
    children: List['TreeNode'] = None
    def __post_init__(self):
        if self.children is None: self.children = []
    @property
    def is_terminal(self): return self.symbol in TERMINAL_SYMBOLS
    @property
    def arity(self):
        if self.is_terminal: return 0
        return OPERATORS.get(self.symbol, Operator('?', 0, lambda: 0)).arity

def random_gene(head_length: int) -> Gene:
    operators = list(OPERATORS.keys())
    terminals = TERMINAL_SYMBOLS
    tail_length = head_length + 1
    head = ''.join(random.choice(operators + terminals) for _ in range(head_length))
    tail = ''.join(random.choice(terminals) for _ in range(tail_length))
    return Gene(sequence=head + tail, head_length=head_length)

def karva_to_tree(gene: Gene) -> TreeNode:
    seq = gene.sequence
    if not seq: return None
    root = TreeNode(seq[0])
    queue = [root]
    pos = 1
    while queue and pos < len(seq):
        current = queue.pop(0)
        for _ in range(current.arity):
            if pos >= len(seq): break
            child = TreeNode(seq[pos])
            current.children.append(child)
            if child.arity > 0: queue.append(child)
            pos += 1
    return root

def evaluate_tree(node: TreeNode) -> float:
    if node is None: return 0.0
    try:
        if node.is_terminal: return TERMINAL_MAP.get(node.symbol, 0)
        op = OPERATORS.get(node.symbol)
        if op is None: return 0.0
        children = [evaluate_tree(c) for c in node.children]
        result = op.func(*children)
        if isinstance(result, complex): return abs(result)
        return result
    except: return 0.0

def evaluate_gene(gene: Gene) -> float:
    return evaluate_tree(karva_to_tree(gene))

def tree_to_infix(node: TreeNode) -> str:
    if node is None: return "?"
    if node.is_terminal: return TERMINAL_NAMES.get(node.symbol, node.symbol)
    sym = node.symbol
    if len(node.children) == 1:
        child = tree_to_infix(node.children[0])
        if sym == 'Q': return f"√({child})"
        if sym == 'T': return f"T({child})"
        return f"{sym}({child})"
    if len(node.children) == 2:
        left = tree_to_infix(node.children[0])
        right = tree_to_infix(node.children[1])
        if sym == '^': return f"({left})^({right})"
        return f"({left} {sym} {right})"
    return f"{sym}(?)"

def gene_to_expression(gene: Gene) -> str:
    return tree_to_infix(karva_to_tree(gene))

# =============================================================================
# GENETIC OPERATORS
# =============================================================================

def mutate_gene(gene: Gene, rate: float = 0.05) -> Gene:
    seq = list(gene.sequence)
    ops = list(OPERATORS.keys())
    terms = TERMINAL_SYMBOLS
    for i in range(len(seq)):
        if random.random() < rate:
            if i < gene.head_length:
                seq[i] = random.choice(ops + terms)
            else:
                seq[i] = random.choice(terms)
    return Gene(''.join(seq), gene.head_length)

def transpose_is(gene: Gene, length: int = 3) -> Gene:
    seq = gene.sequence
    h = gene.head_length
    if h < 2: return gene
    src = random.randint(0, len(seq) - length)
    segment = seq[src:src + length]
    ins = random.randint(1, h - 1)
    new_head = (seq[:ins] + segment + seq[ins:h])[:h]
    return Gene(new_head + gene.tail, h)

def recombine(g1: Gene, g2: Gene) -> tuple:
    pt = random.randint(1, len(g1.sequence) - 1)
    return (Gene(g1.sequence[:pt] + g2.sequence[pt:], g1.head_length),
            Gene(g2.sequence[:pt] + g1.sequence[pt:], g2.head_length))

# =============================================================================
# EVOLUTION
# =============================================================================

ZETA_3 = 1.2020569031595942853997381615114

def fitness(value: float, target: float = ZETA_3) -> float:
    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        return 0.0
    error = abs(value - target) / abs(target)
    # Log scale fitness
    return max(0, 1000 * (1 - math.log10(1 + error * 1e10) / 10))

def evolve(pop_size=500, head_len=12, generations=2000, verbose=True):
    population = [random_gene(head_len) for _ in range(pop_size)]
    best_ever = (0, None, None, "")
    stagnant = 0
    
    for gen in range(generations):
        # Evaluate
        results = [(g, evaluate_gene(g)) for g in population]
        fits = [fitness(v) for g, v in results]
        
        # Find best
        best_idx = max(range(len(fits)), key=lambda i: fits[i])
        best_gene, best_val = results[best_idx]
        best_fit = fits[best_idx]
        best_expr = gene_to_expression(best_gene)
        
        if best_fit > best_ever[0]:
            best_ever = (best_fit, best_gene, best_val, best_expr)
            stagnant = 0
        else:
            stagnant += 1
        
        if verbose and (gen % 100 == 0 or gen < 10):
            err = abs(best_val - ZETA_3) / ZETA_3 * 100
            print(f"Gen {gen:4d} | {best_val:.12f} | err={err:.8f}% | fit={best_fit:.1f}")
        
        if stagnant > 500:
            if verbose: print(f"Stagnant at gen {gen}")
            break
        
        # Selection + reproduction
        new_pop = [best_gene]  # Elitism
        while len(new_pop) < pop_size:
            # Tournament
            t = random.sample(range(pop_size), 3)
            winner = max(t, key=lambda i: fits[i])
            parent = population[winner]
            
            # Genetic ops
            child = mutate_gene(parent, 0.08)
            if random.random() < 0.1:
                child = transpose_is(child)
            if random.random() < 0.4:
                t2 = random.sample(range(pop_size), 3)
                w2 = max(t2, key=lambda i: fits[i])
                child, _ = recombine(child, population[w2])
            
            new_pop.append(child)
        
        population = new_pop[:pop_size]
    
    return best_ever, population

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GEP WITH ELEGANT PRIMITIVES")
    print("=" * 70)
    print()
    print("Terminals:", list(CONSTANTS.keys()))
    print("Operators:", list(OPERATORS.keys()))
    print()
    print(f"Target: ζ(3) = {ZETA_3}")
    print()
    
    # Our formula for reference
    our = (math.pi ** 3) * math.sqrt(66 / 43913)
    print(f"Our formula π³√(66/43913) = {our:.15f}")
    print(f"  66 = T(11), 666 = T(36) = T(T(8))")
    print(f"  43913 = 66×666 - 43 = T(11)×T(T(8)) - 43")
    print()
    
    print("-" * 70)
    best, pop = evolve(pop_size=1000, head_len=15, generations=3000)
    
    print("-" * 70)
    print(f"\nBEST FOUND:")
    print(f"  Fitness: {best[0]:.2f}")
    print(f"  Value:   {best[2]:.15f}")
    print(f"  Error:   {abs(best[2] - ZETA_3) / ZETA_3:.2e}")
    print(f"  Formula: {best[3]}")
    print(f"  Gene:    {best[1].sequence}")
    
    # Show top 10 unique
    print("\n" + "=" * 70)
    print("TOP 10 UNIQUE FORMULAS:")
    seen = set()
    results = [(g, evaluate_gene(g), gene_to_expression(g)) for g in pop]
    results.sort(key=lambda x: fitness(x[1]), reverse=True)
    
    count = 0
    for g, v, expr in results:
        if expr in seen: continue
        seen.add(expr)
        err = abs(v - ZETA_3) / ZETA_3
        print(f"\n{count+1}. err={err:.2e} | {v:.12f}")
        print(f"   {expr}")
        count += 1
        if count >= 10: break
