from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.propagator_snapshot import (
    atom_key,
    collect_atoms,
    collect_clause_atoms,
    preprocess_assertions,
    prepare_propagator_formula,
    build_bool_snapshot,
    clear_bool_snapshot_cache,
)


def test_collect_atoms_and_clause_cooccurrence():
    x = z3.Int("x")
    a, b, c = x >= 5, x <= 2, z3.Bool("c")
    asserts = [z3.Or(a, b), z3.Or(c, z3.Not(a))]
    atoms = collect_atoms(asserts)
    keys = {atom_key(t) for t in atoms}
    assert {atom_key(a), atom_key(b), atom_key(c)} <= keys

    snap, amap = build_bool_snapshot(asserts)
    bkeys = {bv.var_id for bv in snap.bool_vars}
    assert {atom_key(a), atom_key(b), atom_key(c)} <= bkeys
    # 每个顶层 assertion 的原子共现为一个 clause
    assert len(snap.clauses) == 2
    lits0 = {vid for vid, _ in snap.clauses[0].literals}
    assert lits0 == {atom_key(a), atom_key(b)}
    # ¬a 的极性为 False
    pol = dict((vid, pos) for vid, pos in snap.clauses[1].literals)
    assert pol[atom_key(a)] is False
    # 映射能取回 z3 原子
    assert amap[atom_key(a)] is not None


def test_collect_clause_atoms_skips_unit_keeps_disjunction():
    """注册原子：单元盒界不收录，析取子句中的原子收录。"""
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    box_lo, box_hi = x >= 0, x <= 10
    asserts = [box_lo, box_hi, z3.Or(a, b)]
    all_keys = {atom_key(t) for t in collect_atoms(asserts)}
    reg_keys = {atom_key(t) for t in collect_clause_atoms(asserts)}
    assert {atom_key(box_lo), atom_key(box_hi), atom_key(a), atom_key(b)} <= all_keys
    assert atom_key(a) in reg_keys and atom_key(b) in reg_keys
    assert atom_key(box_lo) not in reg_keys
    assert atom_key(box_hi) not in reg_keys


def test_preprocess_then_clause_atoms_via_prepare():
    """预处理 + 析取原子：prepare_propagator_formula 返回同 ctx 公式与注册集。"""
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    # 冗余真值单元：simplify/propagate 后应被消掉或弱化
    asserts = [x >= 0, x <= 10, z3.Or(a, b), z3.BoolVal(True)]
    pp, atoms = prepare_propagator_formula(asserts)
    assert pp, "预处理后应仍有断言"
    assert all(hasattr(f, "ctx") and f.ctx == asserts[0].ctx for f in pp)
    reg = {atom_key(t) for t in atoms}
    # 析取分支原子应仍在注册集（或预处理改写后仍为 ≥2 元子句的原子）
    assert reg <= {atom_key(t) for t in collect_atoms(pp)}
    # 单元盒界即使仍在 pp 中也不应单独构成注册（除非被并进多元 Or）
    assert atom_key(x >= 0) not in reg or len(collect_clause_atoms([x >= 0])) == 0


def test_preprocess_assertions_idempotent_sat():
    x = z3.Int("x")
    asserts = [x >= 0, x <= 3, z3.Or(x >= 1, x <= 0)]
    pp = preprocess_assertions(asserts)
    s = z3.Solver()
    s.add(*pp)
    assert s.check() == z3.sat


def test_assignment_and_candidates():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    snap, _ = build_bool_snapshot([z3.Or(a, b)], assignment={atom_key(a): True})
    amap = {bv.var_id: bv for bv in snap.bool_vars}
    assert amap[atom_key(a)].assignment is True
    assert amap[atom_key(a)].is_candidate is False
    assert amap[atom_key(b)].assignment is None
    assert amap[atom_key(b)].is_candidate is True
    # 已定原子退出候选；子句已满足 → 无 literal 边
    assert set(snap.candidate_bool_ids) == {atom_key(b)}
    assert len(snap.clauses) == 1
    assert snap.clauses[0].is_satisfied is True
    assert snap.clauses[0].literals == []


def test_boolean_bcp_forces_unit_and_prunes_edges():
    """假文字使子句变单元 → BCP 强制另一原子，并投影边。"""
    clear_bool_snapshot_cache()
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    snap, _ = build_bool_snapshot(
        [z3.Or(a, b)], assignment={atom_key(a): False}
    )
    by_id = {bv.var_id: bv for bv in snap.bool_vars}
    assert by_id[atom_key(a)].assignment is False
    assert by_id[atom_key(b)].assignment is True   # BCP 强制
    assert snap.candidate_bool_ids == []
    assert snap.clauses[0].is_satisfied is True
    assert snap.search_state.trail_length == 2


def test_project_drops_falsified_literals():
    """未满足子句去掉假文字，保留未定文字。"""
    clear_bool_snapshot_cache()
    x = z3.Int("x")
    a, b, c = x >= 5, x <= 2, x >= 1
    # 赋值 a=False：Or(a,b,c) 投影为 (b,c)，且不会单元传播
    snap, _ = build_bool_snapshot(
        [z3.Or(a, b, c)], assignment={atom_key(a): False}
    )
    assert set(snap.candidate_bool_ids) == {atom_key(b), atom_key(c)}
    assert snap.clauses[0].is_satisfied is None
    assert {vid for vid, _ in snap.clauses[0].literals} == {
        atom_key(b), atom_key(c)
    }


def test_build_bool_snapshot_theory_features():
    """SMT(LIA) 应产生理论原子/数值变量结构特征；纯 SAT 实例保持为空（不变量）。"""
    from omt_branching.solver.sat_instances import generate_hard_smt_lia, generate_rand_3sat
    from omt_branching.input.graph_builder import GraphBuilder
    from omt_branching.interfaces import EdgeType

    # SMT(LIA)：理论原子 + 数值变量被填充，且 var_coeffs 键 ∈ numeric_vars
    atoms, clauses = generate_hard_smt_lia(6, 12, 3, 6, 4, seed=3)
    snap, _ = build_bool_snapshot(clauses)
    assert snap.theory_atoms, "SMT(LIA) 应产生理论原子节点"
    assert snap.numeric_vars, "SMT(LIA) 应产生数值变量节点"
    numvar_ids = {n.num_var_id for n in snap.numeric_vars}
    ta = snap.theory_atoms[0]
    assert ta.var_coeffs and all(v in numvar_ids for v in ta.var_coeffs)  # variable_in_atom 边连得上
    assert ta.bool_var_id in {b.var_id for b in snap.bool_vars}           # atom_abstracted_by 边连得上

    # 端到端连边断言：结构特征真的进了图（atom_abstracted_by / variable_in_atom 边数 > 0）
    g = GraphBuilder().build(snap)
    assert g.num_edges(EdgeType.ATOM_ABSTRACTED_BY) > 0
    assert g.num_edges(EdgeType.VARIABLE_IN_ATOM) > 0

    # SAT 不变性：纯布尔常量实例不产生理论/数值节点（保护 SAT 正结果）
    sat_atoms, sat_clauses = generate_rand_3sat(20, seed=7)
    ssnap, _ = build_bool_snapshot(sat_clauses)
    assert ssnap.theory_atoms == [] and ssnap.numeric_vars == []


def test_linear_decomposition_branches():
    """直接单测 _linear 的 SUB/UMINUS/有理常数分支，覆盖 ADD-of-MUL 之外的形状。"""
    import z3
    from omt_branching.solver.propagator_snapshot import _linear
    x, y = z3.Ints("x y")
    # SUB: 3*x - 2*y  -> {x:3, y:-2}, const 0
    c, k = _linear(3 * x - 2 * y)
    assert c[str(x)] == 3.0 and c[str(y)] == -2.0 and k == 0.0
    # UMINUS: -x -> {x:-1}
    c, k = _linear(-x)
    assert c[str(x)] == -1.0
    # 有理常数 (Real): x/2 形式的系数与常数项
    r = z3.Real("r")
    c, k = _linear(r + z3.RealVal("3/2"))
    assert c[str(r)] == 1.0 and abs(k - 1.5) < 1e-9


def test_static_cache_reuses_structure_updates_assignment():
    """同 assertions 二次调用应命中静态缓存；有赋值时投影子句为新对象。"""
    clear_bool_snapshot_cache()
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [z3.Or(a, b)]
    snap1, amap1 = build_bool_snapshot(asserts)
    snap2, amap2 = build_bool_snapshot(
        asserts, assignment={atom_key(a): True}, stats={"conflicts": 3}
    )
    assert amap1 is amap2  # 静态 amap 复用
    assert snap1.theory_atoms is snap2.theory_atoms
    # 空赋值复用静态子句；有赋值则投影为新列表
    assert snap1.clauses is not snap2.clauses
    snap0, _ = build_bool_snapshot(asserts)
    assert snap0.clauses is snap1.clauses
    by_id = {bv.var_id: bv for bv in snap2.bool_vars}
    assert by_id[atom_key(a)].assignment is True
    assert by_id[atom_key(b)].assignment is None
    assert snap2.search_state.conflict_count == 3
    assert snap1.search_state.conflict_count == 0


def test_projected_graph_drops_satisfied_clause_edges():
    """投影后 GraphBuilder 不再为已满足子句连 literal 边。"""
    from omt_branching.input.graph_builder import GraphBuilder
    from omt_branching.interfaces import EdgeType

    clear_bool_snapshot_cache()
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [z3.Or(a, b)]
    g0 = GraphBuilder().build(build_bool_snapshot(asserts)[0])
    g1 = GraphBuilder().build(
        build_bool_snapshot(asserts, assignment={atom_key(a): True})[0]
    )
    assert g0.num_edges(EdgeType.LITERAL_IN_CLAUSE) == 2
    assert g1.num_edges(EdgeType.LITERAL_IN_CLAUSE) == 0
    assert g1.meta["candidate_bool_ids"] == [atom_key(b)]
