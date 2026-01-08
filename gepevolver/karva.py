r"""
KARVA ENCODING — The Heart of GEP
==================================
Cândida Ferreira's brilliant insight: a linear string that ALWAYS produces valid expression trees.

The K-expression is read left-to-right. Each operator knows its arity (how many arguments it needs).
Arguments are consumed from what follows. The TAIL region ensures we never run out of terminals.

Example: +*ab-cd with head_length=3
         +           <- root, arity 2, needs 2 args
        / \
       *   -         <- positions 1,2 fill root's args
      / \ / \
     a  b c  d       <- positions 3,4,5,6 fill their args

Gene structure:
  HEAD: can contain operators OR terminals (length h)
  TAIL: can ONLY contain terminals (length t = h*(n-1)+1 where n = max arity)

This guarantees: no matter what's in the head, we always have enough terminals to complete.

Reference: Ferreira, C. (2001) "Gene Expression Programming: A New Adaptive Algorithm
           for Solving Problems" arXiv:cs/0102027
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any
import random
import math

# =============================================================================
# OPERATORS
# =============================================================================

@dataclass
class Operator:
    """An operator with its symbol, arity, and evaluation function."""
    symbol: str
    arity: int          # 0 = terminal, 1 = unary, 2 = binary
    func: Callable
    
    def __repr__(self):
        return self.symbol

# =============================================================================
# PROTECTED OPERATORS
# =============================================================================
# Ferreira's approach: operators return bounded values, never NaN/inf.
# This keeps organisms alive but with reduced fitness for weird behavior.

EPSILON = 1e-10
MAX_VALUE = 1e15
MIN_VALUE = -1e15

def protected_div(a: float, b: float) -> float:
    """Protected division: returns 1 if denominator too small."""
    if abs(b) < EPSILON:
        return 1.0
    result = a / b
    return max(MIN_VALUE, min(MAX_VALUE, result))

def protected_log(a: float) -> float:
    """Protected log: returns 0 for non-positive inputs."""
    if abs(a) < EPSILON:
        return 0.0
    return max(MIN_VALUE, math.log(abs(a)))

def protected_pow(a: float, b: float) -> float:
    """Protected power: caps inputs and outputs."""
    # Cap exponent to prevent overflow
    b = max(-20, min(20, b))
    # Cap base for negative exponents
    if b < 0 and abs(a) < EPSILON:
        return 1.0
    # Handle negative base with non-integer exponent (would give complex)
    if a < 0 and b != int(b):
        a = abs(a)
    try:
        result = a ** b
        if isinstance(result, complex):
            return abs(result)
        return max(MIN_VALUE, min(MAX_VALUE, result))
    except (ValueError, OverflowError):
        return 1.0

def protected_sqrt(a: float) -> float:
    """Protected sqrt: uses abs."""
    return math.sqrt(abs(a))

def protected_triangular(a: float) -> float:
    """Protected triangular: caps input."""
    n = int(max(0, min(1000, a)))
    return n * (n + 1) // 2

# Standard operators with protection
OPERATORS = {
    '+': Operator('+', 2, lambda a, b: max(MIN_VALUE, min(MAX_VALUE, a + b))),
    '-': Operator('-', 2, lambda a, b: max(MIN_VALUE, min(MAX_VALUE, a - b))),
    '*': Operator('*', 2, lambda a, b: max(MIN_VALUE, min(MAX_VALUE, a * b))),
    '/': Operator('/', 2, protected_div),
    '^': Operator('^', 2, protected_pow),
    'Q': Operator('Q', 1, protected_sqrt),
    'L': Operator('L', 1, protected_log),
    'T': Operator('T', 1, protected_triangular),
    'S': Operator('S', 1, lambda a: math.sin(a)),
    'C': Operator('C', 1, lambda a: math.cos(a)),
}

# Maximum arity across all operators (needed for tail length calculation)
MAX_ARITY = max(op.arity for op in OPERATORS.values())

# =============================================================================
# CONSTANTS (TERMINALS)
# =============================================================================

# Our special constants pool for triangular prime theory
CONSTANTS = {
    'π': math.pi,
    'e': math.e,
    'φ': (1 + math.sqrt(5)) / 2,  # golden ratio
    '2': 2,
    '3': 3,
    '6': 6,
    '8': 8,
    '11': 11,
    '24': 24,
    '36': 36,
    '43': 43,
    '66': 66,
    '108': 108,
    '666': 666,
    '43913': 43913,
}

# Terminal symbols (single characters for Karva encoding)
TERMINAL_SYMBOLS = list('abcdefghijklmno')[:len(CONSTANTS)]
TERMINAL_MAP = dict(zip(TERMINAL_SYMBOLS, CONSTANTS.values()))
TERMINAL_NAMES = dict(zip(TERMINAL_SYMBOLS, CONSTANTS.keys()))

def get_terminal_value(symbol: str) -> float:
    """Get the numeric value of a terminal symbol."""
    return TERMINAL_MAP.get(symbol, float('nan'))

def get_terminal_name(symbol: str) -> str:
    """Get the human-readable name of a terminal."""
    return TERMINAL_NAMES.get(symbol, symbol)

# =============================================================================
# GENE STRUCTURE
# =============================================================================

@dataclass
class Gene:
    """
    A GEP gene with head/tail structure.
    
    head_length: h
    tail_length: t = h * (max_arity - 1) + 1
    total_length: h + t
    
    The head can contain any symbol (operators or terminals).
    The tail can ONLY contain terminals.
    """
    sequence: str
    head_length: int
    
    @property
    def tail_length(self) -> int:
        return self.head_length * (MAX_ARITY - 1) + 1
    
    @property
    def head(self) -> str:
        return self.sequence[:self.head_length]
    
    @property
    def tail(self) -> str:
        return self.sequence[self.head_length:]
    
    def __repr__(self):
        return f"Gene(head='{self.head}', tail='{self.tail}')"

def random_gene(head_length: int, operators: List[str] = None, terminals: List[str] = None) -> Gene:
    """Create a random gene with valid head/tail structure."""
    if operators is None:
        operators = list(OPERATORS.keys())
    if terminals is None:
        terminals = TERMINAL_SYMBOLS
    
    tail_length = head_length * (MAX_ARITY - 1) + 1
    
    # Head: mix of operators and terminals
    head_alphabet = operators + terminals
    head = ''.join(random.choice(head_alphabet) for _ in range(head_length))
    
    # Tail: terminals only
    tail = ''.join(random.choice(terminals) for _ in range(tail_length))
    
    return Gene(sequence=head + tail, head_length=head_length)

# =============================================================================
# EXPRESSION TREE
# =============================================================================

@dataclass 
class TreeNode:
    """A node in the expression tree."""
    symbol: str
    children: List['TreeNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def is_terminal(self) -> bool:
        return self.symbol in TERMINAL_SYMBOLS
    
    @property
    def arity(self) -> int:
        if self.is_terminal:
            return 0
        return OPERATORS.get(self.symbol, Operator('?', 0, lambda: 0)).arity

def karva_to_tree(gene: Gene) -> TreeNode:
    """
    Convert K-expression to expression tree.
    
    This is Ferreira's key algorithm:
    1. Start with root = first symbol
    2. Process symbols left-to-right
    3. Each operator gets its children from the queue
    4. Stop when all nodes are complete (no pending children)
    """
    sequence = gene.sequence
    if not sequence:
        return None
    
    # Create root node
    root = TreeNode(sequence[0])
    
    # Queue of nodes that need children
    queue = [root]
    
    # Current position in K-expression (start after root)
    pos = 1
    
    while queue and pos < len(sequence):
        # Get next node that needs children
        current = queue.pop(0)
        
        # How many children does this node need?
        arity = current.arity
        
        for _ in range(arity):
            if pos >= len(sequence):
                break
            
            # Create child node
            child = TreeNode(sequence[pos])
            current.children.append(child)
            
            # If child is operator, it needs children too
            if child.arity > 0:
                queue.append(child)
            
            pos += 1
    
    return root


# =============================================================================
# TREE EVALUATION
# =============================================================================

def evaluate_tree(node: TreeNode) -> float:
    """
    Recursively evaluate an expression tree.
    Protected operators ensure bounded output.
    """
    if node is None:
        return 0.0
    
    try:
        # Terminal: return its value
        if node.is_terminal:
            return get_terminal_value(node.symbol)
        
        # Operator: evaluate children first, then apply operator
        op = OPERATORS.get(node.symbol)
        if op is None:
            return 0.0
        
        child_values = [evaluate_tree(child) for child in node.children]
        result = op.func(*child_values)
        
        # Final safety: handle any remaining complex numbers
        if isinstance(result, complex):
            return abs(result)
        return result
    
    except Exception:
        return 0.0

def evaluate_gene(gene: Gene) -> float:
    """Evaluate a gene by converting to tree and evaluating."""
    tree = karva_to_tree(gene)
    return evaluate_tree(tree)

# =============================================================================
# EXPRESSION PRINTING
# =============================================================================

def tree_to_infix(node: TreeNode) -> str:
    """Convert expression tree to human-readable infix notation."""
    if node is None:
        return "?"
    
    # Terminal: return its readable name
    if node.is_terminal:
        return get_terminal_name(node.symbol)
    
    symbol = node.symbol
    
    # Unary operator
    if len(node.children) == 1:
        child = tree_to_infix(node.children[0])
        if symbol == 'Q':
            return f"√({child})"
        elif symbol == 'L':
            return f"log({child})"
        elif symbol == 'T':
            return f"T({child})"
        elif symbol == 'S':
            return f"sin({child})"
        elif symbol == 'C':
            return f"cos({child})"
        else:
            return f"{symbol}({child})"
    
    # Binary operator
    if len(node.children) == 2:
        left = tree_to_infix(node.children[0])
        right = tree_to_infix(node.children[1])
        
        if symbol == '^':
            return f"({left})^({right})"
        else:
            return f"({left} {symbol} {right})"
    
    return f"{symbol}(?)"

def gene_to_expression(gene: Gene) -> str:
    """Convert gene to human-readable expression."""
    tree = karva_to_tree(gene)
    return tree_to_infix(tree)

# =============================================================================
# TREE COMPLEXITY
# =============================================================================

def tree_depth(node: TreeNode) -> int:
    """Calculate depth of expression tree."""
    if node is None or node.is_terminal:
        return 1
    if not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)

def tree_size(node: TreeNode) -> int:
    """Count total nodes in expression tree."""
    if node is None:
        return 0
    return 1 + sum(tree_size(child) for child in node.children)

def effective_length(gene: Gene) -> int:
    """
    Calculate how much of the gene is actually used.
    The K-expression may not use all of the tail.
    """
    tree = karva_to_tree(gene)
    return tree_size(tree)

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("KARVA ENCODING DEMO")
    print("=" * 60)
    
    # Show terminal mapping
    print("\nTerminal Mapping (symbol → name → value):")
    for sym in TERMINAL_SYMBOLS:
        name = get_terminal_name(sym)
        val = get_terminal_value(sym)
        print(f"  {sym} → {name:>6} → {val}")
    
    print("\nOperators:", list(OPERATORS.keys()))
    print(f"Max arity: {MAX_ARITY}")
    
    # Create a random gene
    print("\n" + "-" * 60)
    print("Random Gene Demo:")
    gene = random_gene(head_length=5)
    print(f"  Gene: {gene}")
    print(f"  K-expression: {gene.sequence}")
    
    tree = karva_to_tree(gene)
    print(f"  Expression: {gene_to_expression(gene)}")
    
    result = evaluate_gene(gene)
    print(f"  Result: {result}")
    print(f"  Effective length: {effective_length(gene)}")
    
    # Demo: manually construct a gene that should give ζ(3) ≈ π³√(66/43913)
    # We need: ^π3 then Q then /66,43913
    # K-expression: ^*πQa/lm where a=3, l=66, m=43913
    # Actually let's try: *^π3Q/lm
    print("\n" + "-" * 60)
    print("Manual Gene Demo (targeting ζ(3)):")
    
    # For this we need to be more careful with construction
    # Target: π³ × √(66/43913)
    # That's: * (^ π 3) (Q (/ 66 43913))
    # In K-expression: *^Q/πcalm... where c=3, a=π, l=66, m=43913
    # Wait, need to check terminal mappings
    
    print("\n  Checking terminals for target formula:")
    print(f"    π = {TERMINAL_NAMES.get('a', '?')} (symbol 'a')")
    print(f"    3 = {TERMINAL_NAMES.get('d', '?')} (symbol 'd')")  
    print(f"    66 = {TERMINAL_NAMES.get('l', '?')} (symbol 'l')")
    print(f"    43913 = {TERMINAL_NAMES.get('o', '?')} (symbol 'o')")
