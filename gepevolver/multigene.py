"""
Multi-Gene GEP Extension
========================
Support for evolving multiple genes as a single unit.

The chromosome (set of genes) is the unit of selection.
Fitness evaluates the chromosome as a whole.
"""

import random
from dataclasses import dataclass
from typing import List, Callable, Tuple, Dict, Any

from .engine import GlyphGEP, Gene, Operator
from .glyphs import GlyphSet


@dataclass
class Chromosome:
    """A multi-gene chromosome."""
    genes: List[Gene]
    
    def __len__(self):
        return len(self.genes)
    
    def __getitem__(self, idx):
        return self.genes[idx]


@dataclass 
class MultiGeneResult:
    """Result from multi-gene evolution."""
    fitness: float
    chromosome: Chromosome
    expressions: List[str]
    generations_run: int
    converged: bool
    metadata: Dict[str, Any] = None


class MultiGeneGEP:
    """
    Multi-gene GEP engine.
    
    Each individual is a Chromosome containing multiple genes.
    Fitness function receives the full chromosome.
    """
    
    def __init__(self, glyph_set: GlyphSet, num_genes: int, operators: Dict[str, Operator] = None):
        self.engine = GlyphGEP(glyph_set, operators)
        self.num_genes = num_genes
        self.glyphs = glyph_set
    
    def random_chromosome(self, head_length: int) -> Chromosome:
        """Create random chromosome with num_genes genes."""
        genes = [self.engine.random_gene(head_length) for _ in range(self.num_genes)]
        return Chromosome(genes)
    
    def mutate(self, chrom: Chromosome, rate: float = 0.05) -> Chromosome:
        """Mutate each gene independently."""
        new_genes = [self.engine.mutate(g, rate) for g in chrom.genes]
        return Chromosome(new_genes)
    
    def transpose(self, chrom: Chromosome, length: int = 3) -> Chromosome:
        """Apply IS transposition to one random gene."""
        new_genes = list(chrom.genes)
        idx = random.randint(0, len(new_genes) - 1)
        new_genes[idx] = self.engine.transpose_is(new_genes[idx], length)
        return Chromosome(new_genes)
    
    def recombine(self, c1: Chromosome, c2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Recombine corresponding genes."""
        new1, new2 = [], []
        for g1, g2 in zip(c1.genes, c2.genes):
            r1, r2 = self.engine.recombine(g1, g2)
            new1.append(r1)
            new2.append(r2)
        return Chromosome(new1), Chromosome(new2)
    
    def gene_swap(self, c1: Chromosome, c2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Swap one gene between chromosomes."""
        idx = random.randint(0, self.num_genes - 1)
        new1 = [g for g in c1.genes]
        new2 = [g for g in c2.genes]
        new1[idx], new2[idx] = new2[idx], new1[idx]
        return Chromosome(new1), Chromosome(new2)
    
    def evaluate_gene(self, gene: Gene) -> float:
        """Evaluate single gene."""
        return self.engine.evaluate(gene)
    
    def to_expression(self, gene: Gene) -> str:
        """Convert gene to expression string."""
        return self.engine.to_elegant(gene)
    
    def to_expressions(self, chrom: Chromosome) -> List[str]:
        """Convert all genes to expressions."""
        return [self.to_expression(g) for g in chrom.genes]


def evolve_multigene(
    glyph_set: GlyphSet,
    num_genes: int,
    fitness_fn: Callable[[MultiGeneGEP, Chromosome], float],
    pop_size: int = 200,
    head_len: int = 8,
    generations: int = 500,
    mutation_rate: float = 0.08,
    crossover_rate: float = 0.4,
    transpose_rate: float = 0.1,
    gene_swap_rate: float = 0.1,
    stagnation_limit: int = 100,
    operators: Dict[str, Operator] = None,
    verbose: bool = True,
    seed: int = None,
) -> Tuple[MultiGeneResult, List[Chromosome], MultiGeneGEP]:
    """
    Evolve multi-gene chromosomes.
    
    Args:
        glyph_set: Terminal symbols
        num_genes: Number of genes per chromosome
        fitness_fn: Function(engine, chromosome) -> fitness score
        pop_size: Population size
        head_len: Gene head length
        generations: Max generations
        mutation_rate: Point mutation probability
        crossover_rate: Gene crossover probability
        transpose_rate: IS transposition probability
        gene_swap_rate: Probability of swapping genes between chromosomes
        stagnation_limit: Stop after this many gens without improvement
        operators: Custom operators
        verbose: Print progress
        seed: Random seed
    
    Returns:
        (MultiGeneResult, population, engine)
    """
    if seed is not None:
        random.seed(seed)
    
    engine = MultiGeneGEP(glyph_set, num_genes, operators)
    population = [engine.random_chromosome(head_len) for _ in range(pop_size)]
    
    best_ever = MultiGeneResult(0, None, [], 0, False)
    stagnant = 0
    
    for gen in range(generations):
        # Evaluate
        fits = [fitness_fn(engine, chrom) for chrom in population]
        
        # Find best
        best_idx = max(range(len(fits)), key=lambda i: fits[i])
        best_chrom = population[best_idx]
        best_fit = fits[best_idx]
        
        if best_fit > best_ever.fitness:
            best_ever = MultiGeneResult(
                best_fit, best_chrom,
                engine.to_expressions(best_chrom),
                gen, False
            )
            stagnant = 0
        else:
            stagnant += 1
        
        if verbose and (gen % 50 == 0 or gen < 10):
            print(f"Gen {gen:4d} | fitness={best_fit:.4f}")
        
        if stagnant > stagnation_limit:
            if verbose:
                print(f"Converged at gen {gen}")
            best_ever.converged = True
            best_ever.generations_run = gen
            break
        
        # Selection + reproduction
        new_pop = [best_chrom]  # Elitism
        
        while len(new_pop) < pop_size:
            # Tournament selection
            t = random.sample(range(pop_size), 3)
            winner_idx = max(t, key=lambda i: fits[i])
            parent = population[winner_idx]
            
            # Mutation
            child = engine.mutate(parent, mutation_rate)
            
            # Transposition
            if random.random() < transpose_rate:
                child = engine.transpose(child)
            
            # Crossover
            if random.random() < crossover_rate:
                t2 = random.sample(range(pop_size), 3)
                w2 = max(t2, key=lambda i: fits[i])
                child, _ = engine.recombine(child, population[w2])
            
            # Gene swap
            if random.random() < gene_swap_rate:
                t3 = random.sample(range(pop_size), 3)
                w3 = max(t3, key=lambda i: fits[i])
                child, _ = engine.gene_swap(child, population[w3])
            
            new_pop.append(child)
        
        population = new_pop[:pop_size]
    
    if not best_ever.converged:
        best_ever.generations_run = generations
    
    return best_ever, population, engine
