#!/usr/bin/env python3
"""
GEP EVOLVER GUI
===============
Interactive GUI for exploring formula evolution with frozen glyphs.

Run: python3 gui/app.py
"""

import sys
import os
import math
import threading
import queue
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional, List, Callable

from glyphs import GlyphSet
from glyph_gep import GlyphGEP, Gene
from formula_parser import FormulaParser, normalize_formula

# =============================================================================
# CONSTANTS
# =============================================================================

ZETA_3 = 1.2020569031595942853997381615114
ZETA_5 = 1.0369277551433699263313654864570
ZETA_7 = 1.0083492773819228268397975498497

TARGETS = {
    "ζ(3) Apéry": ZETA_3,
    "ζ(5)": ZETA_5,
    "ζ(7)": ZETA_7,
    "π": math.pi,
    "e": math.e,
    "φ Golden": (1 + math.sqrt(5)) / 2,
}

# =============================================================================
# PRESET GLYPH SETS
# =============================================================================

def T(n): return n * (n + 1) // 2

GLYPH_PRESETS = {
    "Cubes + Triangulars": GlyphSet.custom([
        ('p', 'π', math.pi), ('e', 'e', math.e),
        ('A', '2³', 8), ('B', '3³', 27), ('C', '4³', 64), ('D', '5³', 125),
        ('E', '6³', 216), ('F', '7³', 343), ('G', '8³', 512), ('H', '9³', 729),
        ('I', '10³', 1000), ('J', '11³', 1331),
        ('a', 'T₃', T(3)), ('b', 'T₄', T(4)), ('c', 'T₈', T(8)),
        ('d', 'T₁₁', T(11)), ('f', 'T₃₆', T(36)),
        ('s', '43', 43), ('z', '43913', 43913),
        ('1', '1', 1), ('2', '2', 2), ('3', '3', 3), ('6', '6', 6),
    ]),
    "Pure Triangulars": GlyphSet.custom([
        ('p', 'π', math.pi), ('e', 'e', math.e),
        ('a', 'T₂', T(2)), ('b', 'T₃', T(3)), ('c', 'T₄', T(4)),
        ('d', 'T₅', T(5)), ('f', 'T₆', T(6)), ('g', 'T₇', T(7)),
        ('h', 'T₈', T(8)), ('i', 'T₉', T(9)), ('j', 'T₁₀', T(10)),
        ('k', 'T₁₁', T(11)), ('l', 'T₁₂', T(12)),
        ('m', 'T₃₆', T(36)),
        ('1', '1', 1), ('2', '2', 2), ('3', '3', 3),
    ]),
    "Pure Cubes": GlyphSet.custom([
        ('p', 'π', math.pi), ('e', 'e', math.e),
        ('1', '1³', 1), ('A', '2³', 8), ('B', '3³', 27), ('C', '4³', 64),
        ('D', '5³', 125), ('E', '6³', 216), ('F', '7³', 343), ('G', '8³', 512),
        ('H', '9³', 729), ('I', '10³', 1000), ('J', '11³', 1331), ('K', '12³', 1728),
        ('2', '2', 2), ('3', '3', 3), ('6', '6', 6),
    ]),
    "Small Integers": GlyphSet.custom([
        ('p', 'π', math.pi), ('e', 'e', math.e), ('f', 'φ', (1+math.sqrt(5))/2),
        ('1', '1', 1), ('2', '2', 2), ('3', '3', 3), ('4', '4', 4),
        ('5', '5', 5), ('6', '6', 6), ('7', '7', 7), ('8', '8', 8),
        ('9', '9', 9), ('a', '10', 10), ('b', '11', 11), ('c', '12', 12),
    ]),
}

# =============================================================================
# SYMBOL PALETTE
# =============================================================================

SYMBOL_PALETTE = [
    ('π', 'pi'), ('e', 'e'), ('φ', 'phi'), ('√', 'sqrt('),
    ('²', '^2'), ('³', '^3'), ('×', '*'), ('÷', '/'),
    ('T₂', 'T(2)'), ('T₃', 'T(3)'), ('T₄', 'T(4)'), ('T₈', 'T(8)'),
    ('T₁₁', 'T(11)'), ('T₃₆', 'T(36)'),
    ('2³', '8'), ('3³', '27'), ('11³', '1331'),
]


# =============================================================================
# EVOLUTION WORKER (runs in background thread)
# =============================================================================

class EvolutionWorker:
    """Runs evolution in background, sends updates via queue."""
    
    def __init__(self, msg_queue: queue.Queue):
        self.queue = msg_queue
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self, glyph_set: GlyphSet, target: float, params: dict,
              seed_genes: List[Gene] = None):
        """Start evolution in background thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            args=(glyph_set, target, params, seed_genes),
            daemon=True
        )
        self.thread.start()
    
    def stop(self):
        """Signal evolution to stop."""
        self.running = False
    
    def _run(self, glyph_set: GlyphSet, target: float, params: dict,
             seed_genes: List[Gene]):
        """Evolution loop."""
        import random
        
        engine = GlyphGEP(glyph_set)
        pop_size = params.get('pop_size', 500)
        head_len = params.get('head_len', 12)
        generations = params.get('generations', 2000)
        mutation_rate = params.get('mutation_rate', 0.08)
        
        # Initialize population
        population = [engine.random_gene(head_len) for _ in range(pop_size)]
        
        # Inject seeds
        if seed_genes:
            for i, seed in enumerate(seed_genes[:min(len(seed_genes), pop_size//4)]):
                population[i] = seed
        
        def fitness(value):
            if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                return 0.0
            error = abs(value - target) / abs(target) if target != 0 else abs(value)
            return max(0, 1000 * (1 - math.log10(1 + error * 1e10) / 10))
        
        best_ever = (0, None, 0.0, "")
        
        for gen in range(generations):
            if not self.running:
                break
            
            # Evaluate
            results = [(g, engine.evaluate(g)) for g in population]
            fits = [fitness(v) for g, v in results]
            
            # Find best
            best_idx = max(range(len(fits)), key=lambda i: fits[i])
            best_gene, best_val = results[best_idx]
            best_fit = fits[best_idx]
            best_expr = engine.to_elegant(best_gene)
            
            if best_fit > best_ever[0]:
                best_ever = (best_fit, best_gene, best_val, best_expr)
            
            # Send update every 10 generations or on improvement
            if gen % 10 == 0 or best_fit > best_ever[0]:
                err = abs(best_val - target) / abs(target) * 100 if target != 0 else abs(best_val) * 100
                self.queue.put(('progress', {
                    'gen': gen,
                    'best_val': best_val,
                    'best_fit': best_fit,
                    'best_expr': best_expr,
                    'error_pct': err,
                }))
            
            # Selection + reproduction
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
        
        # Final results
        results = [(g, engine.evaluate(g), engine.to_elegant(g)) for g in population]
        results.sort(key=lambda x: fitness(x[1]), reverse=True)
        
        # Deduplicate
        seen = set()
        unique = []
        for g, v, expr in results:
            if expr not in seen and not math.isnan(v):
                seen.add(expr)
                unique.append((g, v, expr, fitness(v)))
        
        self.queue.put(('done', {
            'best': best_ever,
            'top_formulas': unique[:20],
        }))
        self.running = False


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class GEPApp:
    """Main GUI application."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GEP Formula Evolver — Frozen Glyphs")
        self.root.geometry("1200x800")
        
        # State
        self.msg_queue = queue.Queue()
        self.worker = EvolutionWorker(self.msg_queue)
        self.parser = FormulaParser()
        self.current_glyphs: Optional[GlyphSet] = None
        
        # Build UI
        self._build_ui()
        
        # Start queue processor
        self._process_queue()
    
    def _build_ui(self):
        """Build the main UI."""
        # Main container with 3 columns
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # LEFT: Parameters
        left = ttk.Frame(main, width=280)
        main.add(left, weight=1)
        self._build_params_panel(left)
        
        # CENTER: Evolution + Formula Input
        center = ttk.Frame(main, width=500)
        main.add(center, weight=2)
        self._build_center_panel(center)
        
        # RIGHT: Results
        right = ttk.Frame(main, width=400)
        main.add(right, weight=2)
        self._build_results_panel(right)
    
    def _build_params_panel(self, parent):
        """Build parameters panel."""
        ttk.Label(parent, text="⚙️ Parameters", font=('Helvetica', 14, 'bold')).pack(pady=5)
        
        # Target selection
        ttk.Label(parent, text="Target:").pack(anchor='w', padx=5)
        self.target_var = tk.StringVar(value="ζ(3) Apéry")
        target_combo = ttk.Combobox(parent, textvariable=self.target_var,
                                     values=list(TARGETS.keys()), state='readonly')
        target_combo.pack(fill='x', padx=5, pady=2)
        
        # Custom target
        ttk.Label(parent, text="Custom target:").pack(anchor='w', padx=5, pady=(10,0))
        self.custom_target = ttk.Entry(parent)
        self.custom_target.pack(fill='x', padx=5, pady=2)
        self.custom_target.insert(0, "1.2020569031595942")
        
        # Glyph set
        ttk.Label(parent, text="Glyph Set:").pack(anchor='w', padx=5, pady=(10,0))
        self.glyph_var = tk.StringVar(value="Cubes + Triangulars")
        glyph_combo = ttk.Combobox(parent, textvariable=self.glyph_var,
                                    values=list(GLYPH_PRESETS.keys()), state='readonly')
        glyph_combo.pack(fill='x', padx=5, pady=2)
        glyph_combo.bind('<<ComboboxSelected>>', self._on_glyph_change)
        
        # Glyph display
        self.glyph_display = scrolledtext.ScrolledText(parent, height=8, width=30,
                                                        font=('Courier', 9))
        self.glyph_display.pack(fill='x', padx=5, pady=5)
        self._update_glyph_display()
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)
        
        # Evolution parameters
        params_frame = ttk.LabelFrame(parent, text="Evolution")
        params_frame.pack(fill='x', padx=5, pady=5)
        
        # Population
        ttk.Label(params_frame, text="Population:").grid(row=0, column=0, sticky='w', padx=5)
        self.pop_size = ttk.Entry(params_frame, width=10)
        self.pop_size.grid(row=0, column=1, padx=5, pady=2)
        self.pop_size.insert(0, "1000")
        
        # Head length
        ttk.Label(params_frame, text="Head length:").grid(row=1, column=0, sticky='w', padx=5)
        self.head_len = ttk.Entry(params_frame, width=10)
        self.head_len.grid(row=1, column=1, padx=5, pady=2)
        self.head_len.insert(0, "14")
        
        # Generations
        ttk.Label(params_frame, text="Generations:").grid(row=2, column=0, sticky='w', padx=5)
        self.generations = ttk.Entry(params_frame, width=10)
        self.generations.grid(row=2, column=1, padx=5, pady=2)
        self.generations.insert(0, "2000")
        
        # Mutation rate
        ttk.Label(params_frame, text="Mutation rate:").grid(row=3, column=0, sticky='w', padx=5)
        self.mutation = ttk.Entry(params_frame, width=10)
        self.mutation.grid(row=3, column=1, padx=5, pady=2)
        self.mutation.insert(0, "0.08")
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill='x', padx=5, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start Evolution",
                                     command=self._start_evolution)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop",
                                    command=self._stop_evolution, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
    
    def _build_center_panel(self, parent):
        """Build center panel with formula input and progress."""
        # Formula input section
        input_frame = ttk.LabelFrame(parent, text="🧮 Formula Input (seed population)")
        input_frame.pack(fill='x', padx=5, pady=5)
        
        # Symbol palette
        palette = ttk.Frame(input_frame)
        palette.pack(fill='x', padx=5, pady=2)
        
        for i, (display, insert) in enumerate(SYMBOL_PALETTE):
            btn = ttk.Button(palette, text=display, width=4,
                            command=lambda v=insert: self._insert_symbol(v))
            btn.grid(row=i//9, column=i%9, padx=1, pady=1)
        
        # Formula entry
        self.formula_entry = ttk.Entry(input_frame, font=('Courier', 12))
        self.formula_entry.pack(fill='x', padx=5, pady=5)
        self.formula_entry.insert(0, "π³ × √(T₁₁/43913)")
        self.formula_entry.bind('<Return>', lambda e: self._evaluate_formula())
        
        # Formula buttons
        formula_btns = ttk.Frame(input_frame)
        formula_btns.pack(fill='x', padx=5, pady=2)
        
        ttk.Button(formula_btns, text="Evaluate", 
                   command=self._evaluate_formula).pack(side='left', padx=2)
        ttk.Button(formula_btns, text="Add to Seeds",
                   command=self._add_seed).pack(side='left', padx=2)
        ttk.Button(formula_btns, text="Clear Seeds",
                   command=self._clear_seeds).pack(side='left', padx=2)
        
        # Formula result
        self.formula_result = ttk.Label(input_frame, text="", font=('Courier', 10))
        self.formula_result.pack(fill='x', padx=5, pady=5)
        
        # Seed list
        ttk.Label(input_frame, text="Seeds:").pack(anchor='w', padx=5)
        self.seed_list = scrolledtext.ScrolledText(input_frame, height=4, width=50,
                                                    font=('Courier', 9))
        self.seed_list.pack(fill='x', padx=5, pady=2)
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)
        
        # Progress section
        progress_frame = ttk.LabelFrame(parent, text="📈 Evolution Progress")
        progress_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Current best
        self.current_best = ttk.Label(progress_frame, text="Ready to evolve...",
                                       font=('Courier', 11))
        self.current_best.pack(fill='x', padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill='x', padx=5, pady=5)
        
        # Log
        self.log = scrolledtext.ScrolledText(progress_frame, height=15, width=60,
                                              font=('Courier', 9))
        self.log.pack(fill='both', expand=True, padx=5, pady=5)
    
    def _build_results_panel(self, parent):
        """Build results panel."""
        ttk.Label(parent, text="🏆 Results", font=('Helvetica', 14, 'bold')).pack(pady=5)
        
        # Best formula
        best_frame = ttk.LabelFrame(parent, text="Best Formula")
        best_frame.pack(fill='x', padx=5, pady=5)
        
        self.best_formula = scrolledtext.ScrolledText(best_frame, height=4, width=45,
                                                       font=('Courier', 11))
        self.best_formula.pack(fill='x', padx=5, pady=5)
        
        # Top formulas list
        ttk.Label(parent, text="Top Unique Formulas:").pack(anchor='w', padx=5, pady=(10,0))
        
        self.results_list = scrolledtext.ScrolledText(parent, height=30, width=45,
                                                       font=('Courier', 9))
        self.results_list.pack(fill='both', expand=True, padx=5, pady=5)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_glyph_change(self, event=None):
        self._update_glyph_display()
    
    def _update_glyph_display(self):
        name = self.glyph_var.get()
        glyphs = GLYPH_PRESETS.get(name)
        if glyphs:
            self.glyph_display.delete('1.0', tk.END)
            for sym in glyphs.symbols:
                g = glyphs.glyphs[sym]
                self.glyph_display.insert(tk.END, f"{sym}: {g.name} = {g.value:.4g}\n")
    
    def _insert_symbol(self, symbol: str):
        self.formula_entry.insert(tk.INSERT, symbol)
        self.formula_entry.focus()
    
    def _evaluate_formula(self):
        formula = self.formula_entry.get()
        value, normalized = self.parser.evaluate(formula)
        
        if math.isnan(value):
            self.formula_result.config(text=f"❌ {normalized}", foreground='red')
        else:
            target = self._get_target()
            error = abs(value - target) / abs(target) * 100 if target != 0 else 0
            self.formula_result.config(
                text=f"✓ {value:.15f} (err: {error:.2e}%)",
                foreground='green'
            )
    
    def _add_seed(self):
        formula = self.formula_entry.get()
        value, _ = self.parser.evaluate(formula)
        if not math.isnan(value):
            self.seed_list.insert(tk.END, f"{formula}\n")
    
    def _clear_seeds(self):
        self.seed_list.delete('1.0', tk.END)
    
    def _get_target(self) -> float:
        custom = self.custom_target.get().strip()
        if custom:
            try:
                return float(custom)
            except:
                pass
        return TARGETS.get(self.target_var.get(), ZETA_3)
    
    def _start_evolution(self):
        if self.worker.running:
            return
        
        # Get parameters
        glyphs = GLYPH_PRESETS.get(self.glyph_var.get())
        target = self._get_target()
        
        params = {
            'pop_size': int(self.pop_size.get()),
            'head_len': int(self.head_len.get()),
            'generations': int(self.generations.get()),
            'mutation_rate': float(self.mutation.get()),
        }
        
        # Clear previous results
        self.log.delete('1.0', tk.END)
        self.results_list.delete('1.0', tk.END)
        self.best_formula.delete('1.0', tk.END)
        self.progress['value'] = 0
        self.progress['maximum'] = params['generations']
        
        # Update UI
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        self.log.insert(tk.END, f"Starting evolution...\n")
        self.log.insert(tk.END, f"Target: {target:.15f}\n")
        self.log.insert(tk.END, f"Glyphs: {self.glyph_var.get()}\n")
        self.log.insert(tk.END, f"Params: pop={params['pop_size']}, head={params['head_len']}\n")
        self.log.insert(tk.END, "-" * 50 + "\n")
        
        # Start worker
        self.worker.start(glyphs, target, params)
    
    def _stop_evolution(self):
        self.worker.stop()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log.insert(tk.END, "\n⏹ Stopped by user\n")
    
    def _process_queue(self):
        """Process messages from worker thread."""
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()
                
                if msg_type == 'progress':
                    self.progress['value'] = data['gen']
                    self.current_best.config(
                        text=f"Gen {data['gen']}: {data['best_val']:.12f} "
                             f"(err: {data['error_pct']:.6f}%)"
                    )
                    if data['gen'] % 100 == 0:
                        self.log.insert(tk.END, 
                            f"Gen {data['gen']:4d} | {data['best_val']:.10f} | "
                            f"err={data['error_pct']:.6f}%\n")
                        self.log.see(tk.END)
                
                elif msg_type == 'done':
                    self.start_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    
                    best = data['best']
                    self.best_formula.delete('1.0', tk.END)
                    self.best_formula.insert(tk.END, 
                        f"Value: {best[2]:.15f}\n"
                        f"Error: {abs(best[2]-self._get_target())/abs(self._get_target()):.2e}\n"
                        f"{best[3]}")
                    
                    # Show top formulas
                    self.results_list.delete('1.0', tk.END)
                    target = self._get_target()
                    for i, (g, v, expr, fit) in enumerate(data['top_formulas'][:20]):
                        err = abs(v - target) / abs(target) if target != 0 else v
                        self.results_list.insert(tk.END,
                            f"{i+1}. err={err:.2e}\n"
                            f"   {v:.12f}\n"
                            f"   {expr}\n\n")
                    
                    self.log.insert(tk.END, "\n✓ Evolution complete!\n")
                    self.log.see(tk.END)
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._process_queue)


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()
    app = GEPApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
