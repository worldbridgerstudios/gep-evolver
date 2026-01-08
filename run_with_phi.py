"""
RUN GEP WITH FULL CONSTANTS
============================
Now includes φ (golden ratio) and explores ζ relationships.
"""

import math
from glyphs import GlyphSet
from glyph_gep import evolve_with_glyphs, GlyphGEP

ZETA_3 = 1.2020569031595942853997381615114
ZETA_5 = 1.0369277551433699263313654864570
PHI = (1 + math.sqrt(5)) / 2

def T(n): return n * (n + 1) // 2

if __name__ == "__main__":
    print("=" * 70)
    print("GEP WITH φ (GOLDEN RATIO) ADDED")
    print("=" * 70)
    print()
    
    # Enhanced glyph set with φ
    glyphs = GlyphSet.custom([
        # Transcendentals - NOW INCLUDING φ
        ('p', 'π', math.pi),
        ('e', 'e', math.e),
        ('f', 'φ', PHI),           # GOLDEN RATIO!
        
        # Cubes
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
        ('g', 'T₃₆', T(36)),      # 666
        
        # Special numbers
        ('s', '43', 43),
        ('z', '43913', 43913),
        
        # Small integers
        ('1', '1', 1),
        ('2', '2', 2),
        ('3', '3', 3),
        ('6', '6', 6),
    ])
    
    print("GLYPH SET:")
    print(glyphs.describe())
    print()
    
    # Show key relationships
    print("KEY CONSTANTS:")
    print(f"  π = {math.pi:.10f}")
    print(f"  e = {math.e:.10f}")
    print(f"  φ = {PHI:.10f}")
    print(f"  φ + π = {PHI + math.pi:.10f}")
    print(f"  φ × π = {PHI * math.pi:.10f}")
    print()
    
    print("REFERENCE:")
    our_formula = (math.pi ** 3) * math.sqrt(66 / 43913)
    our_error = abs(our_formula - ZETA_3) / ZETA_3
    print(f"  π³ × √(T₁₁/43913) = {our_formula:.15f}")
    print(f"  Target ζ(3):       = {ZETA_3:.15f}")
    print(f"  Error: {our_error:.2e}")
    print()
    
    print("-" * 70)
    print("EVOLVING WITH φ...")
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
    
    # Show top 20 unique
    print("\n" + "=" * 70)
    print("TOP 20 UNIQUE FORMULAS:")
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
        
        # Flag formulas containing φ
        phi_flag = " ★φ" if 'φ' in expr else ""
        print(f"\n{count+1}. err={err:.2e}{phi_flag}")
        print(f"   {v:.12f}")
        print(f"   {expr}")
        count += 1
        if count >= 20: break
