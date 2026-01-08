"""
GENETIC OPERATORS — Evolution Machinery
========================================
Ferreira's GEP uses operators inspired by real molecular biology:

1. MUTATION — Random point changes (like SNPs in DNA)
2. TRANSPOSITION — Gene segments jump to new positions
3. ROOT TRANSPOSITION — Special case: segment becomes new root
4. GENE TRANSPOSITION — Whole genes swap positions (multi-gene only)
5. RECOMBINATION — Sexual mixing of genetic material

The key insight: because of Karva encoding, ALL these operations
preserve validity. You cannot create a malformed expression.

Reference: Ferreira (2001) arXiv:cs/0102027
"""

import random
from typing import List, Tuple
from karva import (
    Gene, random_gene, OPERATORS, TERMINAL_SYMBOLS,
    MAX_ARITY, karva_to_tree, evaluate_gene
)

# =============================================================================
# MUTATION
# =============================================================================

def mutate_gene(gene: Gene, mutation_rate: float = 0.05) -> Gene:
    """
    Point mutation: randomly change symbols.
    
    Rules:
    - In HEAD: can mutate to any symbol (operator or terminal)
    - In TAIL: can only mutate to terminals
    
    This preserves validity.
    """
    sequence = list(gene.sequence)
    head_length = gene.head_length
    
    operators = list(OPERATORS.keys())
    terminals = TERMINAL_SYMBOLS
    head_alphabet = operators + terminals
    
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            if i < head_length:
                # Head: any symbol
                sequence[i] = random.choice(head_alphabet)
            else:
                # Tail: terminals only
                sequence[i] = random.choice(terminals)
    
    return Gene(sequence=''.join(sequence), head_length=head_length)

# =============================================================================
# TRANSPOSITION (IS Transposition)
# =============================================================================

def transpose_is(gene: Gene, is_length: int = 3) -> Gene:
    """
    IS (Insertion Sequence) Transposition.
    
    A random segment from anywhere in the gene copies itself
    and inserts at a random position in the HEAD (not position 0).
    
    The tail automatically adjusts (pushed out and truncated).
    """
    sequence = gene.sequence
    head_length = gene.head_length
    total_length = len(sequence)
    
    # Choose random source position and length
    source_start = random.randint(0, total_length - is_length)
    segment = sequence[source_start:source_start + is_length]
    
    # Choose insertion point in head (positions 1 to head_length-1)
    if head_length < 2:
        return gene  # Can't do IS transposition with tiny head
    
    insert_pos = random.randint(1, head_length - 1)
    
    # Insert segment, truncate to maintain length
    new_head = sequence[:insert_pos] + segment + sequence[insert_pos:head_length]
    new_head = new_head[:head_length]  # Truncate to head_length
    
    # Tail stays same length (or regenerate if needed)
    new_sequence = new_head + gene.tail
    
    return Gene(sequence=new_sequence, head_length=head_length)

# =============================================================================
# ROOT TRANSPOSITION (RIS)
# =============================================================================

def transpose_ris(gene: Gene, ris_length: int = 3) -> Gene:
    """
    RIS (Root IS) Transposition.
    
    Like IS transposition, but the segment is inserted at position 0,
    becoming the new root of the expression tree.
    
    The segment MUST start with an operator (function) symbol.
    """
    sequence = gene.sequence
    head_length = gene.head_length
    
    operators = list(OPERATORS.keys())
    
    # Find all positions in HEAD that contain operators
    operator_positions = [i for i in range(head_length) if sequence[i] in operators]
    
    if not operator_positions:
        return gene  # No operators to transpose
    
    # Choose random operator position as segment start
    source_start = random.choice(operator_positions)
    
    # Extract segment (from operator to end of valid range)
    segment_end = min(source_start + ris_length, len(sequence))
    segment = sequence[source_start:segment_end]
    
    # Insert at root (position 0)
    new_head = segment + sequence[:head_length]
    new_head = new_head[:head_length]  # Truncate
    
    new_sequence = new_head + gene.tail
    
    return Gene(sequence=new_sequence, head_length=head_length)

# =============================================================================
# ONE-POINT RECOMBINATION
# =============================================================================

def recombine_one_point(gene1: Gene, gene2: Gene) -> Tuple[Gene, Gene]:
    """
    One-point crossover.
    
    Choose a random point, swap everything after that point.
    Both genes must have same head_length.
    """
    if gene1.head_length != gene2.head_length:
        raise ValueError("Genes must have same head_length")
    
    seq1, seq2 = gene1.sequence, gene2.sequence
    
    # Choose crossover point
    point = random.randint(1, len(seq1) - 1)
    
    # Swap
    new_seq1 = seq1[:point] + seq2[point:]
    new_seq2 = seq2[:point] + seq1[point:]
    
    return (
        Gene(sequence=new_seq1, head_length=gene1.head_length),
        Gene(sequence=new_seq2, head_length=gene2.head_length)
    )

# =============================================================================
# TWO-POINT RECOMBINATION
# =============================================================================

def recombine_two_point(gene1: Gene, gene2: Gene) -> Tuple[Gene, Gene]:
    """
    Two-point crossover.
    
    Choose two random points, swap the segment between them.
    """
    if gene1.head_length != gene2.head_length:
        raise ValueError("Genes must have same head_length")
    
    seq1, seq2 = gene1.sequence, gene2.sequence
    length = len(seq1)
    
    # Choose two points
    p1 = random.randint(0, length - 2)
    p2 = random.randint(p1 + 1, length - 1)
    
    # Swap middle segment
    new_seq1 = seq1[:p1] + seq2[p1:p2] + seq1[p2:]
    new_seq2 = seq2[:p1] + seq1[p1:p2] + seq2[p2:]
    
    return (
        Gene(sequence=new_seq1, head_length=gene1.head_length),
        Gene(sequence=new_seq2, head_length=gene2.head_length)
    )

# =============================================================================
# GENE RECOMBINATION (for multi-gene chromosomes)
# =============================================================================

def recombine_gene(genes1: List[Gene], genes2: List[Gene]) -> Tuple[List[Gene], List[Gene]]:
    """
    Gene recombination for multi-gene chromosomes.
    
    Swap entire genes between two chromosomes.
    """
    if len(genes1) != len(genes2):
        raise ValueError("Chromosomes must have same number of genes")
    
    if len(genes1) < 2:
        return genes1.copy(), genes2.copy()
    
    # Choose which genes to swap
    point = random.randint(1, len(genes1) - 1)
    
    new1 = genes1[:point] + genes2[point:]
    new2 = genes2[:point] + genes1[point:]
    
    return new1, new2

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("GENETIC OPERATORS DEMO")
    print("=" * 60)
    
    # Create a gene
    gene = random_gene(head_length=7)
    print(f"Original: {gene.sequence}")
    
    # Mutation
    mutated = mutate_gene(gene, mutation_rate=0.3)
    print(f"Mutated:  {mutated.sequence}")
    
    # IS Transposition
    transposed = transpose_is(gene, is_length=3)
    print(f"IS Trans: {transposed.sequence}")
    
    # RIS Transposition
    ris = transpose_ris(gene, ris_length=3)
    print(f"RIS:      {ris.sequence}")
    
    # Recombination
    gene2 = random_gene(head_length=7)
    print(f"\nGene1: {gene.sequence}")
    print(f"Gene2: {gene2.sequence}")
    
    child1, child2 = recombine_one_point(gene, gene2)
    print(f"1-pt recomb:")
    print(f"  Child1: {child1.sequence}")
    print(f"  Child2: {child2.sequence}")
    
    child1, child2 = recombine_two_point(gene, gene2)
    print(f"2-pt recomb:")
    print(f"  Child1: {child1.sequence}")
    print(f"  Child2: {child2.sequence}")
