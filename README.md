# GEP Formula Evolver

Gene Expression Programming engine for discovering elegant mathematical formulas.

## Quick Start

```bash
cd ~/Projects/GEPEvolver
python3 gui/app.py
```

Or:
```bash
./run_gui.sh
```

## What Is This?

A **Gene Expression Programming** (GEP) engine based on Cândida Ferreira's 2001 architecture, enhanced with **Frozen Glyphs** — a system that constrains the search space to elegant primitives.

### The Key Insight

Instead of searching over raw numbers (where GEP might find `43913` and you have to realize it's `66×666-43`), we constrain terminals to **elegant atoms**:

- Cubes: 2³, 3³, 4³, ... 11³
- Triangulars: T₃, T₄, T₈, T₁₁, T₃₆
- Transcendentals: π, e, φ

Any formula GEP produces is **automatically elegant** because the atoms ARE elegant.

## Files

```
GEPEvolver/
├── gui/
│   ├── app.py              # Main GUI application
│   └── formula_parser.py   # Parse formulas like "π³ × √(T₁₁/43913)"
├── glyphs.py               # Frozen glyph system
├── glyph_gep.py            # Glyph-aware GEP engine
├── karva.py                # Core Karva encoding
├── operators.py            # Genetic operators
├── fitness.py              # Fitness evaluation
├── evolve.py               # Evolution engine
├── run_gui.sh              # Launch script
└── run_glyphs.py           # CLI runner
```

## GUI Features

### Parameter Panel (Left)
- **Target**: Choose ζ(3), ζ(5), ζ(7), π, e, φ, or enter custom
- **Glyph Set**: Cubes+Triangulars, Pure Triangulars, Pure Cubes, Small Integers
- **Evolution params**: Population, head length, generations, mutation rate

### Formula Input (Center)
- **Symbol palette**: Click to insert π, √, ³, T₁₁, etc.
- **Formula entry**: Type formulas like `π³ × √(T₁₁/43913)`
- **Evaluate**: Check formula value instantly
- **Seeds**: Add formulas to seed the initial population

### Results (Right)
- **Best formula**: The best discovered formula
- **Top unique**: List of top 20 unique formulas with errors

## Formula Syntax

The parser accepts:

| Input | Parsed as |
|-------|-----------|
| `π³` | `pi**3` |
| `√(x)` | `sqrt(x)` |
| `T₁₁` | `T(11)` (triangular) |
| `2³` | `2**3` |
| `×` | `*` |
| `÷` | `/` |

Examples:
```
π³ × √(T₁₁/43913)        → our ζ(3) formula
11³ / (10³ + (2³+e)×T₄)  → GEP's discovery
T₈ × T₃                   → 36 × 6 = 216
```

## Command Line

For batch runs without GUI:

```bash
python3 run_glyphs.py
```

Or run with custom glyph sets:

```python
from glyphs import GlyphSet
from glyph_gep import evolve_with_glyphs

glyphs = GlyphSet.custom([
    ('p', 'π', 3.14159),
    ('a', 'T₈', 36),
    # ... more glyphs
])

best, population, engine = evolve_with_glyphs(
    glyph_set=glyphs,
    target=1.2020569,
    pop_size=1000,
    head_len=14,
    generations=2000
)
```

## Key Discoveries

### Our Formula (10⁻⁸)
```
ζ(3) ≈ π³ × √(T₁₁/43913)
     = π³ × √(66/43913)
     where 43913 = T₁₁ × T₃₆ - 43
```

### GEP's Discovery (4.75×10⁻⁸)
```
ζ(3) ≈ 11³ / (10³ + (2³+e)(T₄ + √3/(T₈×T₃)))
```

Different structure, same neighborhood — cubes and triangulars throughout!

## Theory

Based on:
- **Ferreira (2001)**: Gene Expression Programming (arXiv:cs/0102027)
- **Nicomachus (~100 CE)**: Σk³ = T(n)² — cubes collapse to squared triangulars
- **Plichta (1997)**: Prime Number Cross on 24-wheel

The triangular numbers appear to be fundamental atoms of number-theoretic structure.

---

*The glyphs freeze elegance. GEP finds the pattern.*
