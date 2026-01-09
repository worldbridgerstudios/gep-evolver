# GEPEvolver

Gene Expression Programming engine for discovering elegant mathematical formulas.

## Installation

```bash
pip install gepevolver
```

Or from source:
```bash
git clone https://github.com/worldbridgerstudios/GEPEvolver
cd GEPEvolver
pip install -e .
```

## What Is This?

A **Gene Expression Programming** (GEP) engine based on Cândida Ferreira's 2001 architecture, enhanced with **Frozen Glyphs** — a system that constrains the search space to elegant primitives.

### The Key Insight

Instead of searching over raw numbers (where GEP might find `43913` and you have to realize it's `66×666-43`), we constrain terminals to **elegant atoms**:

- Cubes: 1³, 2³, 3³, ...
- Triangulars: T(3), T(4), T(8), T(11), T(36)
- Transcendentals: π, e, φ

Any formula GEP produces is **automatically elegant** because the atoms ARE elegant.

## Quick Start

```python
from gepevolver import GlyphSet, evolve_with_glyphs

# Use preset glyph set
glyphs = GlyphSet.cubes_and_triangulars()

# Evolve expression for target value
result, population, engine = evolve_with_glyphs(
    glyph_set=glyphs,
    target=137.036,  # Fine structure constant inverse
    pop_size=300,
    head_len=12,
    generations=1000,
)

print(f"Expression: {result.expression}")
print(f"Value: {result.value}")
print(f"Error: {abs(result.value - 137.036) / 137.036 * 100:.4f}%")
```

## Custom Glyph Sets

```python
from gepevolver import GlyphSet

# Create custom terminals
glyphs = GlyphSet.custom([
    ('a', 'T₁₆', 136),      # T(16) = 136
    ('b', '36', 36),
    ('c', '66', 66),
    ('p', 'π', 3.14159265),
    ('f', 'φ', 1.61803399),
])

# Or build programmatically
glyphs = GlyphSet("my_set")
glyphs.add('a', 'T₁₆', 136, formula='T(16)')
glyphs.add('b', 'cubed', 27, formula='3³')
```

## Custom Operators

```python
from gepevolver import GlyphSet, GlyphGEP, Operator, DEFAULT_OPERATORS

# Define custom operator
T_op = Operator(
    symbol='T',
    arity=1,
    func=lambda n: int(n) * (int(n) + 1) // 2,
    display='T'
)

# Create operator set
operators = {**DEFAULT_OPERATORS, 'T': T_op}

# Use with engine
glyphs = GlyphSet.custom([('a', '16', 16)])
engine = GlyphGEP(glyphs, operators)
```

## Preset Glyph Sets

```python
GlyphSet.cubes(n=10)              # 1³ through 10³
GlyphSet.triangulars()            # T(1) through T(11)
GlyphSet.transcendentals()        # π, e, φ
GlyphSet.cubes_and_triangulars()  # Combined set with transcendentals
```

## Package Structure

```
gepevolver/
├── __init__.py    # Public API exports
├── engine.py      # GlyphGEP engine and evolution
├── glyphs.py      # GlyphSet and Glyph classes
└── karva.py       # Karva encoding utilities
```

## Theory

Based on:
- **Ferreira (2001)**: Gene Expression Programming (arXiv:cs/0102027)
- **Nicomachus (~100 CE)**: Σk³ = T(n)² — cubes collapse to squared triangulars

The triangular numbers appear to be fundamental atoms of number-theoretic structure.

## License

CC0 1.0 Universal — Public Domain

---

*The glyphs freeze elegance. GEP finds the pattern.*
