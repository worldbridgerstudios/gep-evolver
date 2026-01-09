"""
GLYPH-AWARE GEP ENGINE
=======================
GEP engine that operates on frozen glyphs instead of raw numbers.

Any formula produced is inherently elegant because the atoms ARE elegant.

Supports custom operator registration for domain-specific extensions.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional

from .glyphs import GlyphSet, Glyph

# =============================================================================
# PROTECTED OPERATORS
# =============================================================================

EPSILON = 1e-10
MAX_VALUE = 1e15


def protected_div(a, b):
    if abs(b) < EPSILON:
        return 1.0
    return max(-MAX_VALUE, min(MAX_VALUE, a / b))


def protected_pow(a, b):
    b = max(-50, min(50, b))
    if b < 0 and abs(a) < EPSILON:
        return 1.0
    if a < 0 and b != int(b):
        a = abs(a)
    try:
        result = a ** b
        if isinstance(result, complex):
            return abs(result)
        return max(-MAX_VALUE, min(MAX_VALUE, result))
    except:
        return 1.0


def protected_sqrt(a):
    return math.sqrt(abs(a))


@dataclass
class Operator:
    """Defines a GEP operator."""
    symbol: str
    arity: int
    func: Callable
    display: str  # How to display in expressions


# Default operators
DEFAULT_OPERATORS = {
    '+': Operator('+', 2, lambda a, b: a + b, '+'),
    '-': Operator('-', 2, lambda a, b: a - b, '-'),
    '*': Operator('*', 2, lambda a, b: a * b, '×'),
    '/': Operator('/', 2, protected_div, '/'),
    '^': Operator('^', 2, protected_pow, '^'),
    'Q': Operator('Q', 1, protected_sqrt, '√'),
}


def get_max_arity(operators: Dict[str, Operator]) -> int:
    """Get maximum arity from operator set."""
    return max(op.arity for op in operators.values()) if operators else 2


# =============================================================================
# GENE STRUCTURE
# =============================================================================

@dataclass
class Gene:
    """A GEP gene with head/tail structure."""
    sequence: str
    head_length: int

    @property
    def tail_length(self):
        max_arity = 2  # Default assumption
        return self.head_length * (max_arity - 1) + 1

    @property
    def head(self):
        return self.sequence[:self.head_length]

    @property
    def tail(self):
        return self.sequence[self.head_length:]


@dataclass
class TreeNode:
    """Node in an expression tree."""
    symbol: str
    children: List['TreeNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


# =============================================================================
# GLYPH-AWARE ENGINE
# =============================================================================

class GlyphGEP:
    """
    GEP engine that uses a GlyphSet for terminals.
    
    All terminal symbols come from the glyph set.
    Evaluation looks up values from glyphs.
    Display uses glyph names for elegant output.
    
    Supports custom operators for domain-specific extensions.
    """

    def __init__(self, glyph_set: GlyphSet, operators: Dict[str, Operator] = None):
        """
        Initialize the GEP engine.
        
        Args:
            glyph_set: GlyphSet defining available terminals
            operators: Custom operators dict (defaults to standard arithmetic)
        """
        self.glyphs = glyph_set
        self.operators = operators if operators is not None else DEFAULT_OPERATORS.copy()
        self.terminal_symbols = glyph_set.symbols
        self.operator_symbols = list(self.operators.keys())
        self.all_symbols = self.operator_symbols + self.terminal_symbols
        self._max_arity = get_max_arity(self.operators)

    def register_operator(self, symbol: str, arity: int, func: Callable, display: str = None):
        """
        Register a custom operator.
        
        Args:
            symbol: Single character symbol for the operator
            arity: Number of arguments (1 for unary, 2 for binary)
            func: The function to apply
            display: Display string (defaults to symbol)
        """
        if len(symbol) != 1:
            raise ValueError(f"Operator symbol must be single char, got '{symbol}'")
        self.operators[symbol] = Operator(symbol, arity, func, display or symbol)
        self.operator_symbols = list(self.operators.keys())
        self.all_symbols = self.operator_symbols + self.terminal_symbols
        self._max_arity = get_max_arity(self.operators)

    def is_terminal(self, symbol: str) -> bool:
        return symbol in self.terminal_symbols

    def is_operator(self, symbol: str) -> bool:
        return symbol in self.operator_symbols

    def arity(self, symbol: str) -> int:
        if self.is_terminal(symbol):
            return 0
        op = self.operators.get(symbol)
        return op.arity if op else 0

    # =========================================================================
    # GENE OPERATIONS
    # =========================================================================

    def random_gene(self, head_length: int) -> Gene:
        """Create random gene using only glyph terminals."""
        tail_length = head_length * (self._max_arity - 1) + 1

        # Head: operators + terminals
        head = ''.join(random.choice(self.all_symbols) for _ in range(head_length))

        # Tail: terminals only (from glyphs)
        tail = ''.join(random.choice(self.terminal_symbols) for _ in range(tail_length))

        return Gene(sequence=head + tail, head_length=head_length)

    def mutate(self, gene: Gene, rate: float = 0.05) -> Gene:
        """Point mutation respecting glyph constraints."""
        seq = list(gene.sequence)
        for i in range(len(seq)):
            if random.random() < rate:
                if i < gene.head_length:
                    seq[i] = random.choice(self.all_symbols)
                else:
                    seq[i] = random.choice(self.terminal_symbols)
        return Gene(''.join(seq), gene.head_length)

    def transpose_is(self, gene: Gene, length: int = 3) -> Gene:
        """IS transposition."""
        seq = gene.sequence
        h = gene.head_length
        if h < 2:
            return gene
        src = random.randint(0, len(seq) - length)
        segment = seq[src:src + length]
        ins = random.randint(1, h - 1)
        new_head = (seq[:ins] + segment + seq[ins:h])[:h]
        return Gene(new_head + gene.tail, h)

    def recombine(self, g1: Gene, g2: Gene) -> Tuple[Gene, Gene]:
        """One-point crossover."""
        pt = random.randint(1, len(g1.sequence) - 1)
        return (
            Gene(g1.sequence[:pt] + g2.sequence[pt:], g1.head_length),
            Gene(g2.sequence[:pt] + g1.sequence[pt:], g2.head_length)
        )

    # =========================================================================
    # TREE OPERATIONS
    # =========================================================================

    def to_tree(self, gene: Gene) -> TreeNode:
        """Convert K-expression to tree."""
        seq = gene.sequence
        if not seq:
            return None

        root = TreeNode(seq[0])
        queue = [root]
        pos = 1

        while queue and pos < len(seq):
            current = queue.pop(0)
            for _ in range(self.arity(current.symbol)):
                if pos >= len(seq):
                    break
                child = TreeNode(seq[pos])
                current.children.append(child)
                if self.arity(child.symbol) > 0:
                    queue.append(child)
                pos += 1

        return root

    def evaluate_tree(self, node: TreeNode) -> float:
        """Evaluate tree using glyph values."""
        if node is None:
            return 0.0

        try:
            if self.is_terminal(node.symbol):
                return self.glyphs.value(node.symbol)

            op = self.operators.get(node.symbol)
            if op is None:
                return 0.0

            children = [self.evaluate_tree(c) for c in node.children]
            result = op.func(*children)

            if isinstance(result, complex):
                return abs(result)
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except:
            return 0.0

    def evaluate(self, gene: Gene) -> float:
        """Evaluate a gene."""
        return self.evaluate_tree(self.to_tree(gene))

    # =========================================================================
    # DISPLAY (using glyph names for elegance)
    # =========================================================================

    def tree_to_elegant(self, node: TreeNode) -> str:
        """Convert tree to elegant expression using glyph names."""
        if node is None:
            return "?"

        if self.is_terminal(node.symbol):
            return self.glyphs.display(node.symbol)

        op = self.operators.get(node.symbol)
        if op is None:
            return "?"

        if len(node.children) == 1:
            child = self.tree_to_elegant(node.children[0])
            if node.symbol == 'Q':
                return f"√({child})"
            return f"{op.display}({child})"

        if len(node.children) == 2:
            left = self.tree_to_elegant(node.children[0])
            right = self.tree_to_elegant(node.children[1])
            if node.symbol == '^':
                return f"({left})^({right})"
            return f"({left} {op.display} {right})"

        return "?"

    def to_elegant(self, gene: Gene) -> str:
        """Convert gene to elegant expression."""
        return self.tree_to_elegant(self.to_tree(gene))


# =============================================================================
# EVOLUTION WITH GLYPHS
# =============================================================================

@dataclass
class EvolutionResult:
    """Result from a GEP evolution run."""
    fitness: float
    gene: Gene
    value: float
    expression: str
    generations_run: int
    converged: bool


def evolve_with_glyphs(
    glyph_set: GlyphSet,
    target: float,
    pop_size: int = 500,
    head_len: int = 12,
    generations: int = 2000,
    mutation_rate: float = 0.08,
    crossover_rate: float = 0.4,
    transpose_rate: float = 0.1,
    stagnation_limit: int = 500,
    operators: Dict[str, Operator] = None,
    verbose: bool = True,
    seed: int = None,
) -> Tuple[EvolutionResult, List[Gene], GlyphGEP]:
    """
    Run GEP evolution using a frozen glyph set.
    
    Args:
        glyph_set: GlyphSet defining available terminals
        target: Target value to evolve towards
        pop_size: Population size
        head_len: Gene head length
        generations: Maximum generations
        mutation_rate: Point mutation probability
        crossover_rate: Crossover probability
        transpose_rate: IS transposition probability
        stagnation_limit: Stop after this many generations without improvement
        operators: Custom operators (uses defaults if None)
        verbose: Print progress
        seed: Random seed for reproducibility
    
    Returns:
        (EvolutionResult, population, engine)
    """
    if seed is not None:
        random.seed(seed)

    engine = GlyphGEP(glyph_set, operators)
    population = [engine.random_gene(head_len) for _ in range(pop_size)]

    def fitness(value):
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return 0.0
        error = abs(value - target) / abs(target)
        return max(0, 1000 * (1 - math.log10(1 + error * 1e10) / 10))

    best_ever = EvolutionResult(0, None, 0, "", 0, False)
    stagnant = 0

    for gen in range(generations):
        # Evaluate
        results = [(g, engine.evaluate(g)) for g in population]
        fits = [fitness(v) for g, v in results]

        # Find best
        best_idx = max(range(len(fits)), key=lambda i: fits[i])
        best_gene, best_val = results[best_idx]
        best_fit = fits[best_idx]
        best_expr = engine.to_elegant(best_gene)

        if best_fit > best_ever.fitness:
            best_ever = EvolutionResult(best_fit, best_gene, best_val, best_expr, gen, False)
            stagnant = 0
        else:
            stagnant += 1

        if verbose and (gen % 100 == 0 or gen < 10):
            err = abs(best_val - target) / abs(target) * 100
            print(f"Gen {gen:4d} | {best_val:.12f} | err={err:.8f}% | fit={best_fit:.1f}")

        if stagnant > stagnation_limit:
            if verbose:
                print(f"Stagnant at gen {gen}")
            best_ever = EvolutionResult(
                best_ever.fitness, best_ever.gene, best_ever.value,
                best_ever.expression, gen, True
            )
            break

        # Selection + reproduction
        new_pop = [best_gene]  # Elitism
        while len(new_pop) < pop_size:
            # Tournament
            t = random.sample(range(pop_size), 3)
            winner = max(t, key=lambda i: fits[i])
            parent = population[winner]

            # Genetic ops
            child = engine.mutate(parent, mutation_rate)
            if random.random() < transpose_rate:
                child = engine.transpose_is(child)
            if random.random() < crossover_rate:
                t2 = random.sample(range(pop_size), 3)
                w2 = max(t2, key=lambda i: fits[i])
                child, _ = engine.recombine(child, population[w2])

            new_pop.append(child)

        population = new_pop[:pop_size]

    # Ensure final generation count
    best_ever = EvolutionResult(
        best_ever.fitness, best_ever.gene, best_ever.value,
        best_ever.expression, generations if not best_ever.converged else best_ever.generations_run,
        best_ever.converged
    )

    return best_ever, population, engine
