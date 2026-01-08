#!/usr/bin/env python3
"""
GEP EVOLVER — Web GUI
======================
Flask-based web interface for the GEP formula evolver.

Run: python3 gui/web_app.py
Then open: http://localhost:5000
"""

import sys
import os
import math
import json
import threading
import queue
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template_string, jsonify, request
from glyphs import GlyphSet
from glyph_gep import GlyphGEP, Gene

app = Flask(__name__)

# =============================================================================
# CONSTANTS & PRESETS
# =============================================================================

ZETA_3 = 1.2020569031595942853997381615114
ZETA_5 = 1.0369277551433699263313654864570
ZETA_7 = 1.0083492773819228268397975498497

def T(n): return n * (n + 1) // 2

GLYPH_PRESETS = {
    "cubes_tri": {
        "name": "Cubes + Triangulars",
        "glyphs": [
            ('p', 'π', math.pi), ('e', 'e', math.e),
            ('A', '2³', 8), ('B', '3³', 27), ('C', '4³', 64), ('D', '5³', 125),
            ('E', '6³', 216), ('F', '7³', 343), ('G', '8³', 512), ('H', '9³', 729),
            ('I', '10³', 1000), ('J', '11³', 1331),
            ('a', 'T₃', T(3)), ('b', 'T₄', T(4)), ('c', 'T₈', T(8)),
            ('d', 'T₁₁', T(11)), ('f', 'T₃₆', T(36)),
            ('s', '43', 43), ('z', '43913', 43913),
            ('1', '1', 1), ('2', '2', 2), ('3', '3', 3), ('6', '6', 6),
        ]
    },
    "pure_tri": {
        "name": "Pure Triangulars",
        "glyphs": [
            ('p', 'π', math.pi), ('e', 'e', math.e),
            ('a', 'T₂', T(2)), ('b', 'T₃', T(3)), ('c', 'T₄', T(4)),
            ('d', 'T₅', T(5)), ('f', 'T₆', T(6)), ('g', 'T₇', T(7)),
            ('h', 'T₈', T(8)), ('i', 'T₉', T(9)), ('j', 'T₁₀', T(10)),
            ('k', 'T₁₁', T(11)), ('l', 'T₁₂', T(12)), ('m', 'T₃₆', T(36)),
            ('1', '1', 1), ('2', '2', 2), ('3', '3', 3),
        ]
    },
    "pure_cubes": {
        "name": "Pure Cubes",
        "glyphs": [
            ('p', 'π', math.pi), ('e', 'e', math.e),
            ('1', '1³', 1), ('A', '2³', 8), ('B', '3³', 27), ('C', '4³', 64),
            ('D', '5³', 125), ('E', '6³', 216), ('F', '7³', 343), ('G', '8³', 512),
            ('H', '9³', 729), ('I', '10³', 1000), ('J', '11³', 1331), ('K', '12³', 1728),
            ('2', '2', 2), ('3', '3', 3), ('6', '6', 6),
        ]
    },
    "small_int": {
        "name": "Small Integers",
        "glyphs": [
            ('p', 'π', math.pi), ('e', 'e', math.e), 
            ('f', 'φ', (1+math.sqrt(5))/2),
            ('1', '1', 1), ('2', '2', 2), ('3', '3', 3), ('4', '4', 4),
            ('5', '5', 5), ('6', '6', 6), ('7', '7', 7), ('8', '8', 8),
            ('9', '9', 9), ('a', '10', 10), ('b', '11', 11), ('c', '12', 12),
        ]
    },
}

# Global state for evolution
evolution_state = {
    "running": False,
    "results": [],
    "best": None,
    "generation": 0,
    "progress": []
}

# =============================================================================
# FORMULA PARSER
# =============================================================================

def parse_formula(formula):
    """Parse and evaluate a formula."""
    import re
    s = formula
    
    # Unicode replacements
    s = s.replace('×', '*').replace('÷', '/')
    s = s.replace('√', 'sqrt')
    
    # Superscripts
    super_map = {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'}
    for sup, dig in super_map.items():
        s = s.replace(sup, f'**{dig}')
    
    # Subscripts in T notation
    sub_map = {'₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9'}
    for sub, dig in sub_map.items():
        s = s.replace(f'T{sub}', f'T({dig}')
        # Handle multi-digit
        if f'({dig}' in s and ')' not in s[s.find(f'({dig}')+2:s.find(f'({dig}')+5]:
            # Find end of number
            pass
    
    # Simple T replacement
    s = re.sub(r'T\((\d+)(?!\))', r'T(\1)', s)
    s = s.replace('^', '**')
    
    context = {
        'pi': math.pi, 'π': math.pi,
        'e': math.e,
        'phi': (1+math.sqrt(5))/2, 'φ': (1+math.sqrt(5))/2,
        'sqrt': math.sqrt,
        'T': lambda n: n*(n+1)//2,
        'log': math.log,
        'sin': math.sin,
        'cos': math.cos,
    }
    
    try:
        value = eval(s, {"__builtins__": {}}, context)
        return float(value), s, None
    except Exception as ex:
        return None, s, str(ex)

# =============================================================================
# EVOLUTION
# =============================================================================

def run_evolution(glyph_preset, target, params, stop_flag):
    """Run evolution (called in background thread)."""
    global evolution_state
    import random
    
    # Build glyph set
    preset = GLYPH_PRESETS.get(glyph_preset, GLYPH_PRESETS["cubes_tri"])
    glyphs = GlyphSet.custom(preset["glyphs"])
    engine = GlyphGEP(glyphs)
    
    pop_size = params.get('pop_size', 500)
    head_len = params.get('head_len', 12)
    generations = params.get('generations', 1000)
    mutation_rate = params.get('mutation_rate', 0.08)
    
    population = [engine.random_gene(head_len) for _ in range(pop_size)]
    
    def fitness(value):
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return 0.0
        error = abs(value - target) / abs(target) if target != 0 else abs(value)
        return max(0, 1000 * (1 - math.log10(1 + error * 1e10) / 10))
    
    best_ever = (0, None, 0.0, "")
    
    for gen in range(generations):
        if stop_flag[0]:
            break
        
        results = [(g, engine.evaluate(g)) for g in population]
        fits = [fitness(v) for g, v in results]
        
        best_idx = max(range(len(fits)), key=lambda i: fits[i])
        best_gene, best_val = results[best_idx]
        best_fit = fits[best_idx]
        best_expr = engine.to_elegant(best_gene)
        
        if best_fit > best_ever[0]:
            best_ever = (best_fit, best_gene, best_val, best_expr)
        
        # Update state
        evolution_state["generation"] = gen
        evolution_state["best"] = {
            "value": best_val,
            "fitness": best_fit,
            "expression": best_expr,
            "error": abs(best_val - target) / abs(target) if target != 0 else 0
        }
        
        if gen % 50 == 0:
            evolution_state["progress"].append({
                "gen": gen,
                "value": best_val,
                "error": abs(best_val - target) / abs(target) * 100
            })
        
        # Evolve
        new_pop = [best_gene]
        while len(new_pop) < pop_size:
            t = random.sample(range(pop_size), 3)
            winner = max(t, key=lambda i: fits[i])
            parent = population[winner]
            
            child = engine.mutate(parent, mutation_rate)
            if random.random() < 0.1:
                child = engine.transpose_is(child)
            if random.random() < 0.4:
                t2 = random.sample(range(pop_size), 3)
                w2 = max(t2, key=lambda i: fits[i])
                child, _ = engine.recombine(child, population[w2])
            
            new_pop.append(child)
        
        population = new_pop[:pop_size]
    
    # Final results - deduplicated
    results = [(g, engine.evaluate(g), engine.to_elegant(g)) for g in population]
    results.sort(key=lambda x: fitness(x[1]), reverse=True)
    
    seen = set()
    unique = []
    for g, v, expr in results:
        if expr not in seen and not math.isnan(v):
            seen.add(expr)
            err = abs(v - target) / abs(target) if target != 0 else v
            unique.append({
                "value": v,
                "expression": expr,
                "error": err,
                "fitness": fitness(v)
            })
    
    evolution_state["results"] = unique[:30]
    evolution_state["running"] = False

stop_flag = [False]
evolution_thread = None
