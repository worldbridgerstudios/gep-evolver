"""
GEPEvolver - Gene Expression Programming for Elegant Formula Discovery

A GEP engine based on Cândida Ferreira's 2001 architecture, enhanced with
Frozen Glyphs — a system that constrains the search space to elegant primitives.

Usage:
    from gepevolver import GlyphSet, evolve_with_glyphs
    
    # Create a glyph set (constrained terminals)
    glyphs = GlyphSet.cubes_and_triangulars()
    
    # Evolve towards a target value
    result, population, engine = evolve_with_glyphs(
        glyph_set=glyphs,
        target=1.2020569,  # ζ(3)
        pop_size=500,
        generations=2000
    )
    
    print(f"Expression: {result[3]}")
    print(f"Value: {result[2]}")
"""

__version__ = "0.1.0"

from .glyphs import GlyphSet, Glyph
from .engine import GlyphGEP, evolve_with_glyphs
from .karva import (
    Gene,
    TreeNode,
    random_gene,
    karva_to_tree,
    evaluate_gene,
    gene_to_expression,
    OPERATORS,
    TERMINAL_SYMBOLS,
)

__all__ = [
    # Version
    "__version__",
    # Glyphs
    "GlyphSet",
    "Glyph",
    # Engine
    "GlyphGEP",
    "evolve_with_glyphs",
    # Core Karva
    "Gene",
    "TreeNode",
    "random_gene",
    "karva_to_tree",
    "evaluate_gene",
    "gene_to_expression",
    "OPERATORS",
    "TERMINAL_SYMBOLS",
]
