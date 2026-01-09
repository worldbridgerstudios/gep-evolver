"""Tests for GEPEvolver package."""

import math
import pytest
from gepevolver import (
    GlyphSet,
    Glyph,
    GlyphGEP,
    Gene,
    TreeNode,
    Operator,
    evolve_with_glyphs,
    EvolutionResult,
    DEFAULT_OPERATORS,
    protected_div,
    protected_pow,
    protected_sqrt,
)


class TestGlyph:
    """Tests for Glyph dataclass."""

    def test_glyph_creation(self):
        g = Glyph(symbol='a', name='T₁₆', value=136.0, formula='T(16)')
        assert g.symbol == 'a'
        assert g.name == 'T₁₆'
        assert g.value == 136.0
        assert g.formula == 'T(16)'

    def test_glyph_repr(self):
        g = Glyph(symbol='a', name='T₁₆', value=136.0, formula='T(16)')
        assert 'T₁₆=136' in repr(g)


class TestGlyphSet:
    """Tests for GlyphSet class."""

    def test_create_empty(self):
        gs = GlyphSet("test")
        assert len(gs) == 0
        assert gs.name == "test"

    def test_add_glyph(self):
        gs = GlyphSet("test")
        gs.add('a', 'alpha', 137.036, 'α⁻¹')
        assert len(gs) == 1
        assert 'a' in gs
        assert gs.value('a') == 137.036
        assert gs.display('a') == 'alpha'

    def test_add_requires_single_char(self):
        gs = GlyphSet("test")
        with pytest.raises(ValueError):
            gs.add('ab', 'invalid', 1.0)

    def test_chained_add(self):
        gs = GlyphSet("test").add('a', 'one', 1).add('b', 'two', 2)
        assert len(gs) == 2

    def test_symbols_property(self):
        gs = GlyphSet("test")
        gs.add('a', 'one', 1)
        gs.add('b', 'two', 2)
        symbols = gs.symbols
        assert 'a' in symbols
        assert 'b' in symbols
        # Should return copy
        symbols.append('c')
        assert 'c' not in gs.symbols

    def test_value_unknown_symbol(self):
        gs = GlyphSet("test")
        val = gs.value('x')
        assert math.isnan(val)

    def test_display_unknown_symbol(self):
        gs = GlyphSet("test")
        assert gs.display('x') == 'x'

    def test_preset_cubes(self):
        gs = GlyphSet.cubes(5)
        assert len(gs) == 5
        # 1³=1, 2³=8, 3³=27, 4³=64, 5³=125
        assert gs.value('a') == 1
        assert gs.value('e') == 125

    def test_preset_triangulars(self):
        gs = GlyphSet.triangulars()
        # T(8) = 36
        assert gs.value('g') == 36
        # T(11) = 66
        assert gs.value('j') == 66

    def test_preset_transcendentals(self):
        gs = GlyphSet.transcendentals()
        assert abs(gs.value('p') - math.pi) < 1e-10
        assert abs(gs.value('e') - math.e) < 1e-10

    def test_preset_cubes_and_triangulars(self):
        gs = GlyphSet.cubes_and_triangulars()
        assert len(gs) > 10
        assert abs(gs.value('p') - math.pi) < 1e-10

    def test_custom_creation(self):
        gs = GlyphSet.custom([
            ('a', 'π', math.pi),
            ('b', '36', 36),
            ('c', '136', 136, 'T(16)'),
        ])
        assert len(gs) == 3
        assert gs.value('c') == 136

    def test_describe(self):
        gs = GlyphSet("test").add('a', 'one', 1)
        desc = gs.describe()
        assert 'test' in desc
        assert 'one' in desc


class TestProtectedOperators:
    """Tests for protected math operators."""

    def test_protected_div_normal(self):
        assert protected_div(10, 2) == 5

    def test_protected_div_by_zero(self):
        assert protected_div(10, 0) == 1.0

    def test_protected_div_small_denominator(self):
        assert protected_div(10, 1e-15) == 1.0

    def test_protected_pow_normal(self):
        assert protected_pow(2, 3) == 8

    def test_protected_pow_negative_base_fractional(self):
        # Should use abs for negative base with fractional exponent
        result = protected_pow(-2, 0.5)
        assert abs(result - math.sqrt(2)) < 1e-10

    def test_protected_pow_overflow(self):
        result = protected_pow(10, 100)
        assert result <= 1e15

    def test_protected_sqrt(self):
        assert protected_sqrt(4) == 2

    def test_protected_sqrt_negative(self):
        assert protected_sqrt(-4) == 2  # Uses abs


class TestGene:
    """Tests for Gene dataclass."""

    def test_gene_creation(self):
        gene = Gene(sequence='+ab', head_length=1)
        assert gene.sequence == '+ab'
        assert gene.head_length == 1
        assert gene.head == '+'
        assert gene.tail == 'ab'


class TestGlyphGEP:
    """Tests for GlyphGEP engine."""

    @pytest.fixture
    def simple_engine(self):
        gs = GlyphSet("test")
        gs.add('a', '1', 1)
        gs.add('b', '2', 2)
        gs.add('c', '3', 3)
        return GlyphGEP(gs)

    def test_is_terminal(self, simple_engine):
        assert simple_engine.is_terminal('a')
        assert not simple_engine.is_terminal('+')

    def test_is_operator(self, simple_engine):
        assert simple_engine.is_operator('+')
        assert not simple_engine.is_operator('a')

    def test_arity(self, simple_engine):
        assert simple_engine.arity('a') == 0
        assert simple_engine.arity('+') == 2
        assert simple_engine.arity('Q') == 1

    def test_random_gene(self, simple_engine):
        gene = simple_engine.random_gene(head_length=5)
        assert len(gene.sequence) == 5 + 6  # head + tail
        # All tail chars should be terminals
        for c in gene.tail:
            assert simple_engine.is_terminal(c)

    def test_mutate(self, simple_engine):
        gene = Gene(sequence='+abcdef', head_length=3)
        mutated = simple_engine.mutate(gene, rate=1.0)  # 100% mutation
        assert mutated.sequence != gene.sequence

    def test_recombine(self, simple_engine):
        g1 = Gene(sequence='+abcde', head_length=3)
        g2 = Gene(sequence='-fghij', head_length=3)
        c1, c2 = simple_engine.recombine(g1, g2)
        # Both children should have valid structure
        assert len(c1.sequence) == len(g1.sequence)
        assert len(c2.sequence) == len(g2.sequence)

    def test_to_tree(self, simple_engine):
        gene = Gene(sequence='+ab', head_length=1)
        tree = simple_engine.to_tree(gene)
        assert tree.symbol == '+'
        assert len(tree.children) == 2
        assert tree.children[0].symbol == 'a'
        assert tree.children[1].symbol == 'b'

    def test_evaluate_simple(self, simple_engine):
        gene = Gene(sequence='+abccc', head_length=1)  # 1 + 2 = 3
        result = simple_engine.evaluate(gene)
        assert result == 3

    def test_evaluate_nested(self, simple_engine):
        # Karva notation: *+abccc with head=2
        # Tree: * has children [+, a], + has children [b, c]
        # = (b + c) * a = (2 + 3) * 1 = 5
        gene = Gene(sequence='*+abccc', head_length=2)
        result = simple_engine.evaluate(gene)
        assert result == 5

    def test_to_elegant(self, simple_engine):
        gene = Gene(sequence='+abccc', head_length=1)
        expr = simple_engine.to_elegant(gene)
        assert '1' in expr
        assert '2' in expr
        assert '+' in expr

    def test_register_operator(self, simple_engine):
        simple_engine.register_operator('T', 1, lambda n: n*(n+1)//2, 'T')
        assert simple_engine.is_operator('T')
        assert simple_engine.arity('T') == 1


class TestEvolveWithGlyphs:
    """Tests for the evolution function."""

    def test_evolve_simple_target(self):
        gs = GlyphSet.custom([
            ('a', '1', 1),
            ('b', '2', 2),
            ('c', '3', 3),
            ('d', '4', 4),
        ])
        # Target: 5 (should be easy: 2+3 or 4+1)
        result, pop, engine = evolve_with_glyphs(
            glyph_set=gs,
            target=5.0,
            pop_size=50,
            head_len=3,
            generations=100,
            verbose=False,
        )
        assert isinstance(result, EvolutionResult)
        assert abs(result.value - 5.0) < 0.1

    def test_evolve_returns_population(self):
        gs = GlyphSet.cubes(3)
        result, pop, engine = evolve_with_glyphs(
            glyph_set=gs,
            target=10.0,
            pop_size=20,
            generations=10,
            verbose=False,
        )
        assert len(pop) == 20
        assert all(isinstance(g, Gene) for g in pop)

    def test_evolve_returns_engine(self):
        gs = GlyphSet.cubes(3)
        result, pop, engine = evolve_with_glyphs(
            glyph_set=gs,
            target=10.0,
            pop_size=20,
            generations=10,
            verbose=False,
        )
        assert isinstance(engine, GlyphGEP)

    def test_evolve_with_seed(self):
        gs = GlyphSet.cubes(5)
        r1, _, _ = evolve_with_glyphs(
            glyph_set=gs, target=100.0, pop_size=20,
            generations=50, verbose=False, seed=42
        )
        r2, _, _ = evolve_with_glyphs(
            glyph_set=gs, target=100.0, pop_size=20,
            generations=50, verbose=False, seed=42
        )
        # Same seed should give same result
        assert r1.value == r2.value

    def test_evolve_with_custom_operators(self):
        gs = GlyphSet.custom([('a', '16', 16)])
        T_op = Operator('T', 1, lambda n: n*(n+1)//2, 'T')
        result, _, _ = evolve_with_glyphs(
            glyph_set=gs,
            target=136.0,  # T(16) = 136
            pop_size=50,
            generations=100,
            operators={'T': T_op, '+': DEFAULT_OPERATORS['+']},
            verbose=False,
        )
        # Should find T(16) = 136
        assert abs(result.value - 136.0) < 1.0


class TestEvolutionResult:
    """Tests for EvolutionResult dataclass."""

    def test_evolution_result_fields(self):
        gene = Gene(sequence='+ab', head_length=1)
        result = EvolutionResult(
            fitness=100.0,
            gene=gene,
            value=3.0,
            expression='(1 + 2)',
            generations_run=50,
            converged=True,
        )
        assert result.fitness == 100.0
        assert result.value == 3.0
        assert result.converged


class TestDefaultOperators:
    """Tests for default operator set."""

    def test_all_operators_present(self):
        assert '+' in DEFAULT_OPERATORS
        assert '-' in DEFAULT_OPERATORS
        assert '*' in DEFAULT_OPERATORS
        assert '/' in DEFAULT_OPERATORS
        assert '^' in DEFAULT_OPERATORS
        assert 'Q' in DEFAULT_OPERATORS

    def test_operators_are_operator_instances(self):
        for op in DEFAULT_OPERATORS.values():
            assert isinstance(op, Operator)
            assert hasattr(op, 'symbol')
            assert hasattr(op, 'arity')
            assert hasattr(op, 'func')
            assert hasattr(op, 'display')
