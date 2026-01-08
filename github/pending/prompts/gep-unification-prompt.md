# GEP Evolver: Unify Karva Engine with Optional Glyph Support

## Context

I have a GEP (Gene Expression Programming) formula evolver at `/Users/nick/Projects/GEPEvolver/`.

Currently there are **two separate implementations**:
1. `karva.py` — main Karva encoding engine with hardcoded `CONSTANTS` dict (raw numbers like π, 66, 43913)
2. `glyph_gep.py` — separate engine that uses `GlyphSet` from `glyphs.py` for "frozen glyphs"

These are not integrated. I want them unified.

## The Task

Modify `karva.py` to **optionally** accept a `GlyphSet` for terminals, while preserving the ability to use raw numeric constants when no glyph set is provided.

### Requirements

1. **Backward compatible**: If no `GlyphSet` passed, use existing `CONSTANTS` dict behavior
2. **Optional glyph mode**: If `GlyphSet` provided, terminals come from glyph symbols/values
3. **Display integration**: `tree_to_infix()` should use glyph display names when in glyph mode
4. **Single engine**: Remove the need for separate `glyph_gep.py` — one unified `karva.py`

### Suggested Approach

```python
# In karva.py, modify random_gene() and evaluate functions to accept optional glyph_set:

def random_gene(head_length: int, operators: List[str] = None, 
                terminals: List[str] = None, glyph_set: GlyphSet = None) -> Gene:
    """
    If glyph_set provided, terminals come from glyph_set.symbols
    Otherwise use default TERMINAL_SYMBOLS
    """
    ...

def evaluate_tree(node: TreeNode, glyph_set: GlyphSet = None) -> float:
    """
    If glyph_set provided, look up terminal values from glyph_set.value()
    Otherwise use TERMINAL_MAP
    """
    ...

def tree_to_infix(node: TreeNode, glyph_set: GlyphSet = None) -> str:
    """
    If glyph_set provided, use glyph_set.display() for terminal names
    Otherwise use TERMINAL_NAMES
    """
    ...
```

### Files to Examine

- `/Users/nick/Projects/GEPEvolver/karva.py` — main engine (433 lines)
- `/Users/nick/Projects/GEPEvolver/glyphs.py` — GlyphSet class (276 lines)
- `/Users/nick/Projects/GEPEvolver/glyph_gep.py` — separate glyph engine (can be deprecated after unification)
- `/Users/nick/Projects/GEPEvolver/evolve.py` — evolution loop (may need glyph_set parameter threading)
- `/Users/nick/Projects/GEPEvolver/fitness.py` — fitness evaluation

### Glyph System Summary

`GlyphSet` (from `glyphs.py`) provides:
- `.symbols` — list of single-char terminal symbols
- `.value(symbol)` — numeric value lookup
- `.display(symbol)` — human-readable name (e.g., "T₁₁", "8³", "π")
- Preset factories: `GlyphSet.cubes()`, `.triangulars()`, `.transcendentals()`, `.cubes_and_triangulars()`, `.zeta3_relevant()`

### Why This Matters

**Frozen Glyphs** constrain the search space to elegant primitives. Instead of GEP finding raw `43913` (requiring human recognition that it's `66×666-43`), terminals are constrained to elegant atoms like cubes, triangulars, transcendentals. Any formula GEP produces is *automatically elegant* because the atoms ARE elegant.

But sometimes we want raw numeric search too — hence optional, not mandatory.

## Deliverable

Modified `karva.py` with optional `glyph_set` parameter threading through:
- `random_gene()`
- `evaluate_tree()` / `evaluate_gene()`
- `tree_to_infix()`

Update `evolve.py` to accept and pass through `glyph_set`.

After unification, `glyph_gep.py` can be marked deprecated or removed.
