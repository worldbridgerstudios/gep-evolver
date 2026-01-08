"""
FORMULA PARSER
==============
Parse human-friendly formulas like "π³ × √(T₁₁/43913)" into Gene sequences.

Supports:
- Subscripts: T₁₁, T₈, T₃₆
- Superscripts: π³, 2³, n²
- Operators: +, -, ×, /, √, ^
- Constants: π, e, φ
"""

import re
import math
from typing import Tuple, Optional, List
from glyphs import GlyphSet

# Unicode mappings
SUPERSCRIPT_MAP = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', 
                   '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}
SUBSCRIPT_MAP = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
                 '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'}

def parse_subscript(s: str) -> str:
    """Convert subscript digits to regular: T₁₁ → T11"""
    result = []
    for c in s:
        result.append(SUBSCRIPT_MAP.get(c, c))
    return ''.join(result)

def parse_superscript(s: str) -> str:
    """Convert superscript digits to regular: ³ → 3"""
    result = []
    for c in s:
        result.append(SUPERSCRIPT_MAP.get(c, c))
    return ''.join(result)

def normalize_formula(formula: str) -> str:
    """
    Normalize a human formula to parseable form.
    
    π³ × √(T₁₁/43913) → (pi**3) * sqrt(T(11)/43913)
    """
    s = formula
    
    # Replace Unicode operators
    s = s.replace('×', '*').replace('÷', '/')
    s = s.replace('√', 'sqrt')
    
    # Handle superscripts: π³ → pi**3, 2³ → 2**3
    superscript_pattern = r'([a-zA-Zπφ\d]+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)'
    def replace_super(m):
        base = m.group(1)
        exp = parse_superscript(m.group(2))
        return f'({base}**{exp})'
    s = re.sub(superscript_pattern, replace_super, s)
    
    # Handle ^ to ** (Python exponentiation)
    s = s.replace('^', '**')
    
    # Handle T subscripts: T₁₁ → T(11), T₈ → T(8)
    t_pattern = r'T([₀₁₂₃₄₅₆₇₈₉]+)'
    def replace_t(m):
        n = parse_subscript(m.group(1))
        return f'T({n})'
    s = re.sub(t_pattern, replace_t, s)
    
    # Handle standalone subscripts in other contexts
    for sub, digit in SUBSCRIPT_MAP.items():
        s = s.replace(sub, digit)
    
    return s

def tokenize(formula: str) -> List[str]:
    """Tokenize a normalized formula."""
    # Add spaces around operators
    for op in ['(', ')', '+', '-', '*', '/', '^', ',']:
        formula = formula.replace(op, f' {op} ')
    
    tokens = formula.split()
    return [t for t in tokens if t]

class FormulaParser:
    """
    Parse formulas into executable form and optionally into Karva genes.
    """
    
    def __init__(self, glyph_set: GlyphSet = None):
        self.glyphs = glyph_set
        self.constants = {
            'pi': math.pi, 'π': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2, 'φ': (1 + math.sqrt(5)) / 2,
            'zeta3': 1.2020569031595942,
        }
    
    def T(self, n: int) -> float:
        """Triangular number."""
        return n * (n + 1) // 2
    
    def evaluate(self, formula: str) -> Tuple[float, str]:
        """
        Evaluate a human formula and return (value, normalized_form).
        
        Examples:
            "π³ × √(T₁₁/43913)" → (1.2020569..., "(pi^3) * sqrt(T(11)/43913)")
            "11³ / 10³" → (1.331, "(11^3) / (10^3)")
        """
        normalized = normalize_formula(formula)
        
        # Build evaluation context
        context = {
            'pi': math.pi, 'π': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2, 'φ': (1 + math.sqrt(5)) / 2,
            'sqrt': math.sqrt,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'T': self.T,
            'abs': abs,
        }
        
        try:
            # Safe eval with restricted context
            value = eval(normalized, {"__builtins__": {}}, context)
            return float(value), normalized
        except Exception as ex:
            return float('nan'), f"Error: {ex}"
    
    def validate(self, formula: str) -> Tuple[bool, str]:
        """Check if formula is valid. Returns (is_valid, message)."""
        value, norm = self.evaluate(formula)
        if math.isnan(value):
            return False, norm
        return True, f"{norm} = {value}"


# =============================================================================
# QUICK INPUT HELPERS
# =============================================================================

def quick_formula(s: str) -> Tuple[float, str]:
    """Quick evaluate a formula string."""
    parser = FormulaParser()
    return parser.evaluate(s)

def test_formulas():
    """Test various formula formats."""
    formulas = [
        "π³ × √(T₁₁/43913)",
        "π^3 * sqrt(66/43913)",
        "11³ / (10³ + (2³ + e) × T₄)",
        "T₈ × T₃",
        "2³ + 3³ + 4³",
        "√(T₁₁ / (T₁₁ × T₃₆ - 43))",
    ]
    
    parser = FormulaParser()
    for f in formulas:
        val, norm = parser.evaluate(f)
        print(f"{f}")
        print(f"  → {norm}")
        print(f"  = {val:.10f}")
        print()


if __name__ == "__main__":
    print("FORMULA PARSER TEST")
    print("=" * 60)
    test_formulas()
