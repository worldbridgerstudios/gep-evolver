"""
FROZEN GLYPHS — Symbolic Number Sets for GEP
=============================================
Bolt-on system for constraining GEP to elegant solution spaces.

A GlyphSet defines:
- Named symbols (e.g., "T₁₁", "8³", "π")
- Their computed values (66, 512, 3.14159...)
- Display forms for human-readable output

The GEP engine operates on symbols. Evaluation looks up values.
Any expression produced is automatically elegant.

Usage:
    glyphs = GlyphSet.cubes_and_triangulars()
    engine = GEPEngine(glyph_set=glyphs)
    # Now any formula uses only elegant atoms
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any

# =============================================================================
# GLYPH DEFINITION
# =============================================================================

@dataclass
class Glyph:
    """A frozen symbolic constant."""
    symbol: str      # Internal symbol (single char for Karva)
    name: str        # Human-readable name (e.g., "T₁₁", "8³")
    value: float     # Computed numeric value
    formula: str     # How it's computed (e.g., "T(11)", "8**3")
    
    def __repr__(self):
        return f"{self.name}={self.value}"

# =============================================================================
# GLYPH SET
# =============================================================================

class GlyphSet:
    """
    A configurable set of frozen glyphs.
    
    Provides:
    - Symbol → Value mapping (for evaluation)
    - Symbol → Name mapping (for display)
    - Validation of which symbols are allowed
    """
    
    def __init__(self, name: str = "custom"):
        self.name = name
        self.glyphs: Dict[str, Glyph] = {}
        self._symbols: List[str] = []
    
    def add(self, symbol: str, name: str, value: float, formula: str = "") -> 'GlyphSet':
        """Add a glyph. Returns self for chaining."""
        if len(symbol) != 1:
            raise ValueError(f"Symbol must be single char, got '{symbol}'")
        glyph = Glyph(symbol=symbol, name=name, value=value, formula=formula or name)
        self.glyphs[symbol] = glyph
        if symbol not in self._symbols:
            self._symbols.append(symbol)
        return self
    
    @property
    def symbols(self) -> List[str]:
        """List of valid terminal symbols."""
        return self._symbols.copy()
    
    def value(self, symbol: str) -> float:
        """Get numeric value of a symbol."""
        if symbol in self.glyphs:
            return self.glyphs[symbol].value
        return float('nan')
    
    def display(self, symbol: str) -> str:
        """Get human-readable name of a symbol."""
        if symbol in self.glyphs:
            return self.glyphs[symbol].name
        return symbol
    
    def __contains__(self, symbol: str) -> bool:
        return symbol in self.glyphs
    
    def __len__(self) -> int:
        return len(self.glyphs)
    
    def __repr__(self):
        return f"GlyphSet({self.name}, {len(self)} glyphs)"
    
    def describe(self) -> str:
        """Full description of all glyphs."""
        lines = [f"GlyphSet: {self.name}", "-" * 40]
        for sym in self._symbols:
            g = self.glyphs[sym]
            lines.append(f"  {sym} = {g.name:>10} = {g.value:>15.6f}  [{g.formula}]")
        return "\n".join(lines)
    
    # =========================================================================
    # PRESET GLYPH SETS
    # =========================================================================
    
    @classmethod
    def cubes(cls, max_n: int = 11) -> 'GlyphSet':
        """Cubes of small integers: 1³, 2³, ..., n³"""
        gs = cls("cubes")
        symbols = "abcdefghijk"
        for i, n in enumerate(range(1, max_n + 1)):
            if i >= len(symbols):
                break
            gs.add(symbols[i], f"{n}³", n**3, f"{n}**3")
        return gs
    
    @classmethod
    def triangulars(cls) -> 'GlyphSet':
        """Key triangular numbers."""
        gs = cls("triangulars")
        def T(n): return n * (n + 1) // 2
        
        gs.add('a', 'T₂', T(2), 'T(2)')      # 3
        gs.add('b', 'T₃', T(3), 'T(3)')      # 6
        gs.add('c', 'T₄', T(4), 'T(4)')      # 10
        gs.add('d', 'T₅', T(5), 'T(5)')      # 15
        gs.add('e', 'T₆', T(6), 'T(6)')      # 21
        gs.add('f', 'T₇', T(7), 'T(7)')      # 28
        gs.add('g', 'T₈', T(8), 'T(8)')      # 36
        gs.add('h', 'T₉', T(9), 'T(9)')      # 45
        gs.add('i', 'T₁₀', T(10), 'T(10)')   # 55
        gs.add('j', 'T₁₁', T(11), 'T(11)')   # 66
        gs.add('k', 'T₃₆', T(36), 'T(36)')   # 666 = T(T(8))
        return gs
    
    @classmethod
    def transcendentals(cls) -> 'GlyphSet':
        """Mathematical constants."""
        gs = cls("transcendentals")
        gs.add('p', 'π', math.pi, 'pi')
        gs.add('e', 'e', math.e, 'e')
        gs.add('f', 'φ', (1 + math.sqrt(5)) / 2, 'phi')
        gs.add('z', 'ζ₃', 1.2020569031595942, 'zeta(3)')
        gs.add('y', 'ζ₂', math.pi**2 / 6, 'zeta(2)')
        gs.add('g', 'γ', 0.5772156649015329, 'euler_gamma')
        return gs
    
    @classmethod
    def zeta3_relevant(cls) -> 'GlyphSet':
        """
        Numbers relevant to our ζ(3) research.
        The "magic numbers" we've discovered.
        """
        gs = cls("zeta3_relevant")
        def T(n): return n * (n + 1) // 2
        
        # Transcendentals
        gs.add('p', 'π', math.pi, 'pi')
        gs.add('f', 'φ', (1 + math.sqrt(5)) / 2, 'phi')
        
        # Small integers (building blocks)
        gs.add('2', '2', 2, '2')
        gs.add('3', '3', 3, '3')
        gs.add('6', '6', 6, '6')
        
        # Key triangulars
        gs.add('a', 'T₄', T(4), 'T(4)')      # 10
        gs.add('b', 'T₈', T(8), 'T(8)')      # 36
        gs.add('c', 'T₁₁', T(11), 'T(11)')   # 66
        gs.add('d', 'T₃₆', T(36), 'T(36)')   # 666
        
        # Sri Yantra correction
        gs.add('s', '43', 43, 'sri_yantra')  # T₈+T₄-T₂
        
        # The denominator
        gs.add('D', '43913', 43913, 'T₁₁×T₃₆-43')
        
        return gs
    
    @classmethod
    def cubes_and_triangulars(cls) -> 'GlyphSet':
        """
        Combined set: cubes + triangulars + key transcendentals.
        The full elegant search space.
        """
        gs = cls("cubes_and_triangulars")
        def T(n): return n * (n + 1) // 2
        
        # Transcendentals
        gs.add('p', 'π', math.pi, 'pi')
        gs.add('e', 'e', math.e, 'e')
        
        # Cubes 2³ through 8³ (1³=1 not interesting)
        gs.add('A', '2³', 8, '2**3')
        gs.add('B', '3³', 27, '3**3')
        gs.add('C', '4³', 64, '4**3')
        gs.add('D', '5³', 125, '5**3')
        gs.add('E', '6³', 216, '6**3')
        gs.add('F', '7³', 343, '7**3')
        gs.add('G', '8³', 512, '8**3')
        
        # Key triangulars
        gs.add('a', 'T₃', T(3), 'T(3)')      # 6
        gs.add('b', 'T₄', T(4), 'T(4)')      # 10
        gs.add('c', 'T₈', T(8), 'T(8)')      # 36
        gs.add('d', 'T₁₁', T(11), 'T(11)')   # 66
        gs.add('f', 'T₃₆', T(36), 'T(36)')   # 666
        
        # Small integers for fine-tuning
        gs.add('1', '1', 1, '1')
        gs.add('2', '2', 2, '2')
        gs.add('3', '3', 3, '3')
        
        return gs
    
    @classmethod
    def custom(cls, specs: List[tuple]) -> 'GlyphSet':
        """
        Build from list of (symbol, name, value) tuples.
        
        Example:
            GlyphSet.custom([
                ('a', 'π', math.pi),
                ('b', '36', 36),
                ('c', '66', 66),
            ])
        """
        gs = cls("custom")
        for spec in specs:
            if len(spec) == 3:
                sym, name, val = spec
                gs.add(sym, name, val, name)
            elif len(spec) == 4:
                sym, name, val, formula = spec
                gs.add(sym, name, val, formula)
        return gs


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FROZEN GLYPHS — Symbolic Number Sets")
    print("=" * 60)
    
    print("\n--- PRESET: Cubes ---")
    cubes = GlyphSet.cubes(8)
    print(cubes.describe())
    
    print("\n--- PRESET: Triangulars ---")
    tri = GlyphSet.triangulars()
    print(tri.describe())
    
    print("\n--- PRESET: ζ(3) Relevant ---")
    z3 = GlyphSet.zeta3_relevant()
    print(z3.describe())
    
    print("\n--- PRESET: Cubes + Triangulars ---")
    combined = GlyphSet.cubes_and_triangulars()
    print(combined.describe())
    
    print("\n--- Custom Set ---")
    custom = GlyphSet.custom([
        ('x', 'π³', math.pi**3),
        ('y', '√(66/43913)', math.sqrt(66/43913)),
    ])
    print(custom.describe())
    
    # Test evaluation
    print("\n--- Value Lookup ---")
    gs = GlyphSet.zeta3_relevant()
    for sym in gs.symbols:
        print(f"  {sym} → {gs.display(sym)} = {gs.value(sym)}")
