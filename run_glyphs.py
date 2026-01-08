"""
RUN GEP WITH FROZEN GLYPHS
===========================
Search for ζ(3) formulas using cubes, triangulars, and key constants.

Any formula found is automatically elegant — the atoms ARE the elegant forms.
"""

import math
from glyphs import GlyphSet
from glyph_gep import evolve_with_glyphs, GlyphGEP

ZETA_3 = 1.2020569031595942853997381615114

def T(n): return n * (n + 1) // 2

if __name__ == "__main__":
    print("=" * 70)
    print("GEP WITH FROZEN GLYPHS")
    print("=" * 70)
    print()
    
    # Define our elegant glyph set
    glyphs = GlyphSet.custom([
        # Transcendentals
        ('p', 'π', math.pi, 'pi'),
        ('e', 'e', math.e, 'e'),
        
        # Cubes (Nicomachus connection)
        ('A', '2³', 8),
        ('B', '3³', 27),
        ('C', '4³', 64),
        ('D', '5³', 125),
        ('E', '6³', 216),
        ('F', '7³', 343),
        ('G', '8³', 512),
        ('H', '9³', 729),
        ('I', '10³', 1000),
        ('J', '11³', 1331),
        
        # Key triangulars
        ('a', 'T₃', T(3)),        # 6
        ('b', 'T₄', T(4)),        # 10
        ('c', 'T₈', T(8)),        # 36
        ('d', 'T₁₁', T(11)),      # 66
        ('f', 'T₃₆', T(36)),      # 666 = T(T₈)
        
        # Sri Yantra and denominator
        ('s', '43', 43),          # Sri Yantra regions
        ('z', '43913', 43913),    # 66×666-43
        
        # Small integers for fine-tuning
        ('1', '1', 1),
        ('2', '2', 2),
        ('3', '3', 3),
        ('6', '6', 6),
    ])
    
    print("GLYPH SET:")
    print(glyphs.describe())
    print()
    
    # Reference formulas
    print("REFERENCE:")
    our_formula = (math.pi ** 3) * math.sqrt(66 / 43913)
    our_error = abs(our_formula - ZETA_3) / ZETA_3
    print(f"  Our formula: π³ × √(T₁₁/43913) = {our_formula:.15f}")
    print(f"  Target ζ(3):                   = {ZETA_3:.15f}")
    print(f"  Error: {our_error:.2e}")
    print()
    
    print("-" * 70)
    print("EVOLVING...")
    print("-" * 70)
    
    best, population, engine = evolve_with_glyphs(
        glyph_set=glyphs,
        target=ZETA_3,
        pop_size=1000,
        head_len=14,
        generations=3000,
        verbose=True
    )
    
    print("-" * 70)
    print(f"\nBEST FOUND:")
    print(f"  Fitness: {best[0]:.2f}")
    print(f"  Value:   {best[2]:.15f}")
    print(f"  Error:   {abs(best[2] - ZETA_3) / ZETA_3:.2e}")
    print(f"  Formula: {best[3]}")
    print(f"  Gene:    {best[1].sequence}")
    
    # Show top 15 unique
    print("\n" + "=" * 70)
    print("TOP 15 UNIQUE ELEGANT FORMULAS:")
    print("=" * 70)
    
    seen = set()
    results = [(g, engine.evaluate(g), engine.to_elegant(g)) for g in population]
    
    def fit(v):
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            return 0
        return -abs(v - ZETA_3)
    
    results.sort(key=lambda x: fit(x[1]), reverse=True)
    
    count = 0
    for g, v, expr in results:
        if expr in seen: continue
        if v == 0 or math.isnan(v) or math.isinf(v): continue
        seen.add(expr)
        err = abs(v - ZETA_3) / ZETA_3
        print(f"\n{count+1}. err={err:.2e} | {v:.12f}")
        print(f"   {expr}")
        count += 1
        if count >= 15: break
