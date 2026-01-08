"""
FORMULA VERIFICATION & TARGETED SEARCH
=======================================
Let's verify our formula encodes correctly, then seed it.
"""

import math
from karva import (
    Gene, karva_to_tree, evaluate_gene, gene_to_expression,
    TERMINAL_SYMBOLS, TERMINAL_MAP, TERMINAL_NAMES, OPERATORS
)

ZETA_3 = 1.2020569031595942853997381615114

def show_terminals():
    """Show terminal mappings."""
    print("TERMINAL MAPPINGS:")
    print("-" * 40)
    for sym in TERMINAL_SYMBOLS:
        name = TERMINAL_NAMES.get(sym, sym)
        val = TERMINAL_MAP.get(sym, '?')
        print(f"  {sym} = {name:>6} = {val}")
    print()

def test_formula(expr_desc: str, karva: str, head_len: int):
    """Test a Karva expression."""
    gene = Gene(sequence=karva, head_length=head_len)
    value = evaluate_gene(gene)
    expr = gene_to_expression(gene)
    error = abs(value - ZETA_3) / ZETA_3
    
    print(f"{expr_desc}:")
    print(f"  Karva: {karva[:head_len]}|{karva[head_len:]}")
    print(f"  Expr:  {expr}")
    print(f"  Value: {value:.15f}")
    print(f"  Error: {error:.2e}")
    print()
    return value, error

if __name__ == "__main__":
    print("=" * 70)
    print("FORMULA VERIFICATION")
    print("=" * 70)
    print()
    
    show_terminals()
    
    # Our formula: π³ × √(66/43913)
    # = (* (^ π 3) (Q (/ 66 43913)))
    # In K-expression, we need:
    # * ^ Q / a e l o + padding
    # Where: a=π, e=3, l=66, o=43913
    
    print("TARGET FORMULA: π³ × √(66/43913)")
    target = (math.pi ** 3) * math.sqrt(66 / 43913)
    print(f"  Computed: {target:.15f}")
    print(f"  ζ(3):     {ZETA_3:.15f}")
    print(f"  Error:    {abs(target - ZETA_3) / ZETA_3:.2e}")
    print()
    
    # K-expression interpretation:
    # Position 0: * (root, needs 2 args)
    # Position 1: ^ (first arg of *, needs 2 args)
    # Position 2: Q (second arg of *, needs 1 arg)
    # Position 3: a (first arg of ^, which is π)
    # Position 4: e (second arg of ^, which is 3)
    # Position 5: / (arg of Q, needs 2 args)
    # Position 6: l (first arg of /, which is 66)
    # Position 7: o (second arg of /, which is 43913)
    
    # So K-expression is: *^Qae/lo + tail padding
    
    head = "*^Qae/lo"
    head_len = 8
    tail_len = head_len + 1  # = 9 for max_arity=2
    tail = "aaaaaaaaa"[:tail_len]
    
    test_formula("π³ × √(66/43913) attempt 1", head + tail, head_len)
    
    # Try longer head
    head_len = 10
    tail_len = head_len + 1
    head = "*^Qae/lo" + "aa"  # padding in head
    tail = "a" * tail_len
    test_formula("π³ × √(66/43913) attempt 2", head + tail, head_len)
    
    # Try with different structure: Q first
    # √(66/43913) × π³
    # = (* (Q (/ 66 43913)) (^ π 3))
    # K: *Q^/loae
    head = "*Q^/loae" + "aa"
    test_formula("√(66/43913) × π³", head + tail, head_len)
    
    # Test GEP's best formula from last run
    print("=" * 70)
    print("TESTING GEP DISCOVERIES")
    print("=" * 70)
    print()
    
    # The (((2 / (36 / (43 / (6 + e)))) + 43) / 36) formula
    # This is: / (+ (/ 2 (/ 36 (/ 43 (+ 6 e)))) 43) 36
    # K: /+j/d/j/k+feaaaa...
    head_len = 12
    tail_len = head_len + 1
    karva = "/+/d/j/k+fej" + "j" * tail_len  # padding tail with j=36
    gene = Gene(sequence=karva, head_length=head_len)
    print(f"Testing: {gene_to_expression(gene)}")
    print(f"Value: {evaluate_gene(gene):.15f}")
