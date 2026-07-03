#!/usr/bin/env python3
"""
Generate random LRA optimization benchmarks with Boolean structure (and/or/not).

Instances are witness-driven: a satisfying assignment is sampled first, then
linear constraints and Boolean combinations are built so that the witness
satisfies every assertion. This keeps the feasible region (and thus the
optimization problem) SAT with high probability.

Output format matches benchmarks/lra_opt (QF_LRA-opt-benchmark-v1).

Usage:
    python scripts/gen_lra_bool_opt_benchmark.py --count 6 --output-dir benchmarks/lra_opt
    python scripts/gen_lra_bool_opt_benchmark.py --count 3 --seed 42 --validate-z3 path/to/z3
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Rational / SMT helpers
# ---------------------------------------------------------------------------

Rat = Fraction


def rat(x: Fraction | int | float) -> str:
    f = Fraction(x).limit_denominator(10_000)
    if f.denominator == 1:
        return str(f.numerator)
    return f"(/ {f.numerator} {f.denominator})"


def lin_value(coeffs: Sequence[Rat], witness: Sequence[Rat]) -> Rat:
    return sum(c * w for c, w in zip(coeffs, witness))


def fmt_lin(coeffs: Sequence[Rat], names: Sequence[str]) -> str:
    parts: List[str] = []
    for c, name in zip(coeffs, names):
        if c == 0:
            continue
        if c == 1:
            parts.append(name)
        elif c == -1:
            parts.append(f"(- {name})")
        else:
            parts.append(f"(* {rat(c)} {name})")
    if not parts:
        return "0"
    if len(parts) == 1:
        return parts[0]
    return f"(+ {' '.join(parts)})"


def fmt_cmp(op: str, coeffs: Sequence[Rat], names: Sequence[str], bound: Rat) -> str:
    return f"({op} {fmt_lin(coeffs, names)} {rat(bound)})"


def fmt_eq(coeffs: Sequence[Rat], names: Sequence[str], bound: Rat) -> str:
    return fmt_cmp("=", coeffs, names, bound)


# ---------------------------------------------------------------------------
# Boolean atoms evaluated on witness
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    op: str  # >=, <=, =
    coeffs: Tuple[Rat, ...]
    bound: Rat

    def holds(self, witness: Sequence[Rat]) -> bool:
        v = lin_value(self.coeffs, witness)
        if self.op == ">=":
            return v >= self.bound
        if self.op == "<=":
            return v <= self.bound
        if self.op == "=":
            return v == self.bound
        raise ValueError(self.op)

    def smt(self, names: Sequence[str]) -> str:
        return fmt_cmp(self.op, self.coeffs, names, self.bound)


@dataclass
class BoolExpr:
    kind: str  # atom, not, and, or
    atom: Optional[Atom] = None
    children: Tuple["BoolExpr", ...] = field(default_factory=tuple)

    def holds(self, witness: Sequence[Rat]) -> bool:
        if self.kind == "atom":
            assert self.atom is not None
            return self.atom.holds(witness)
        if self.kind == "not":
            return not self.children[0].holds(witness)
        if self.kind == "and":
            return all(c.holds(witness) for c in self.children)
        if self.kind == "or":
            return any(c.holds(witness) for c in self.children)
        raise ValueError(self.kind)

    def smt(self, names: Sequence[str]) -> str:
        if self.kind == "atom":
            assert self.atom is not None
            return self.atom.smt(names)
        if self.kind == "not":
            return f"(not {self.children[0].smt(names)})"
        if self.kind == "and":
            return f"(and {' '.join(c.smt(names) for c in self.children)})"
        if self.kind == "or":
            return f"(or {' '.join(c.smt(names) for c in self.children)})"
        raise ValueError(self.kind)


# ---------------------------------------------------------------------------
# Random generation primitives
# ---------------------------------------------------------------------------

def random_coeffs(rng: random.Random, n: int) -> Tuple[Rat, ...]:
    return tuple(Rat(rng.randint(-6, 6)) for _ in range(n))


def random_atom(
    rng: random.Random,
    n: int,
    witness: Sequence[Rat],
    *,
    force: Optional[bool] = None,
) -> Atom:
    coeffs = random_coeffs(rng, n)
    if all(c == 0 for c in coeffs):
        coeffs = tuple(Rat(1) if i == 0 else Rat(0) for i in range(n))
    v = lin_value(coeffs, witness)
    op = rng.choice([">=", "<=", "="])
    slack = Rat(rng.randint(1, 8))
    if force is True:
        if op == ">=":
            bound = v - slack
        elif op == "<=":
            bound = v + slack
        else:
            bound = v
    elif force is False:
        if op == ">=":
            bound = v + slack
        elif op == "<=":
            bound = v - slack
        else:
            # make equality false
            bound = v + Rat(1, 2)
    else:
        delta = Rat(rng.randint(-5, 5), rng.randint(1, 4))
        bound = v + delta
    return Atom(op, coeffs, bound)


def random_bool_expr(
    rng: random.Random,
    n: int,
    witness: Sequence[Rat],
    depth: int,
    *,
    must_hold: bool,
) -> BoolExpr:
    if depth <= 0 or rng.random() < 0.35:
        return BoolExpr("atom", atom=random_atom(rng, n, witness, force=must_hold))

    op = rng.choice(["and", "or", "not"])
    if op == "not":
        child = random_bool_expr(rng, n, witness, depth - 1, must_hold=not must_hold)
        return BoolExpr("not", children=(child,))

    arity = rng.randint(2, 3)
    if op == "and":
        if must_hold:
            children = tuple(
                random_bool_expr(rng, n, witness, depth - 1, must_hold=True)
                for _ in range(arity)
            )
        else:
            # all false
            children = tuple(
                random_bool_expr(rng, n, witness, depth - 1, must_hold=False)
                for _ in range(arity)
            )
        return BoolExpr("and", children=children)

    # or
    if must_hold:
        idx = rng.randrange(arity)
        children = []
        for i in range(arity):
            children.append(
                random_bool_expr(
                    rng, n, witness, depth - 1, must_hold=(i == idx)
                )
            )
        return BoolExpr("or", children=tuple(children))

    children = tuple(
        random_bool_expr(rng, n, witness, depth - 1, must_hold=False)
        for _ in range(arity)
    )
    return BoolExpr("or", children=children)


def implication_atom(
    rng: random.Random, n: int, witness: Sequence[Rat]
) -> BoolExpr:
    """Generate (or (not P) Q) satisfied by witness."""
    p = random_atom(rng, n, witness)
    p_holds = p.holds(witness)
    q = random_atom(rng, n, witness, force=True if p_holds else None)
    return BoolExpr(
        "or",
        children=(
            BoolExpr("not", children=(BoolExpr("atom", atom=p),)),
            BoolExpr("atom", atom=q),
        ),
    )


def piecewise_or(
    rng: random.Random,
    names: Sequence[str],
    witness: Sequence[Rat],
    num_branches: int,
) -> BoolExpr:
    """
    (or (and range_0 side_0 ...) (and range_1 side_1 ...) ...)
    Witness falls in exactly one range; that branch's side constraints hold.
    """
    n = len(names)
    sel = random_coeffs(rng, n)
    if all(c == 0 for c in sel):
        sel = tuple(Rat(1) if i == 0 else Rat(0) for i in range(n))
    center = lin_value(sel, witness)

    width = Rat(rng.randint(2, 6))
    gaps = [Rat(rng.randint(1, 4)) for _ in range(num_branches - 1)]
    intervals: List[Tuple[Rat, Rat]] = []
    lo = center - width * Rat(num_branches // 2 + 1)
    for i in range(num_branches):
        hi = lo + width
        intervals.append((lo, hi))
        lo = hi + (gaps[i] if i < len(gaps) else Rat(0))

    # place witness in the closest interval, then widen that interval
    active = min(
        range(num_branches),
        key=lambda i: abs(lin_value(sel, witness) - (intervals[i][0] + intervals[i][1]) / 2),
    )
    lo_a, hi_a = intervals[active]
    if not (lo_a <= center <= hi_a):
        mid = (lo_a + hi_a) / 2
        shift = center - mid
        intervals[active] = (lo_a + shift, hi_a + shift)

    branches: List[BoolExpr] = []
    for i, (lo_i, hi_i) in enumerate(intervals):
        range_atom = Atom(">=", sel, lo_i)
        range_atom2 = Atom("<=", sel, hi_i)
        children: List[BoolExpr] = [
            BoolExpr("atom", atom=range_atom),
            BoolExpr("atom", atom=range_atom2),
        ]
        if i == active:
            # pin one variable to witness value inside active branch
            pin = rng.randrange(n)
            pin_coeffs = tuple(Rat(1) if j == pin else Rat(0) for j in range(n))
            children.append(
                BoolExpr(
                    "atom",
                    atom=Atom("=", pin_coeffs, witness[pin]),
                )
            )
            # optional extra linear equality on witness
            if n > 1 and rng.random() < 0.7:
                combo = random_coeffs(rng, n)
                children.append(
                    BoolExpr(
                        "atom",
                        atom=Atom("=", combo, lin_value(combo, witness)),
                    )
                )
        else:
            # inactive branches: harmless bounds still linear
            for _ in range(rng.randint(0, 2)):
                children.append(
                    BoolExpr(
                        "atom",
                        atom=random_atom(rng, n, witness, force=None),
                    )
                )
        branches.append(BoolExpr("and", children=tuple(children)))
    return BoolExpr("or", children=tuple(branches))


def halfspace_satisfied(
    rng: random.Random, n: int, witness: Sequence[Rat], op: str
) -> Atom:
    coeffs = random_coeffs(rng, n)
    if all(c == 0 for c in coeffs):
        coeffs = tuple(Rat(1) if i == 0 else Rat(0) for i in range(n))
    v = lin_value(coeffs, witness)
    slack = Rat(rng.randint(1, 20))
    if op == "<=":
        return Atom("<=", coeffs, v + slack)
    return Atom(">=", coeffs, v - slack)


# ---------------------------------------------------------------------------
# Benchmark assembly
# ---------------------------------------------------------------------------

@dataclass
class Benchmark:
    bench_id: str
    family: str
    names: List[str]
    witness: List[Rat]
    assertions: List[str]
    objective: str
    obj_coeffs: List[Rat]
    seed: int
    description: str

    @property
    def num_constraints(self) -> int:
        return len(self.assertions)

    def to_smt2(self) -> str:
        lines = [
            f"; QF_LRA optimization benchmark: {self.bench_id}",
            f"; family={self.family}  vars={len(self.names)}  constraints={self.num_constraints}",
            f"; seed={self.seed}  objective={self.objective}",
            f"; {self.description}",
            "",
            "(set-logic QF_LRA)",
            "",
        ]
        for name in self.names:
            lines.append(f"(declare-fun {name} () Real)")
        lines.append("")
        for a in self.assertions:
            lines.append(f"(assert {a})")
        lines.append("")
        obj = fmt_lin(self.obj_coeffs, self.names)
        lines.append(f"({self.objective} {obj})")
        lines.append("")
        lines.append("(check-sat)")
        lines.append("(get-objectives)")
        lines.append("(get-model)")
        lines.append("")
        return "\n".join(lines)


def generate_benchmark(
    rng: random.Random,
    bench_id: str,
    family: str,
    num_vars: int,
    *,
    num_branches: int,
    bool_depth: int,
    num_implications: int,
    num_halfspaces: int,
) -> Benchmark:
    names = [f"x{i}" for i in range(num_vars)]
    witness = [Rat(rng.randint(1, 12), rng.randint(1, 6)) for _ in range(num_vars)]

    assertions: List[str] = []

    # non-negativity + loose upper box (witness-driven)
    for i, w in enumerate(witness):
        assertions.append(fmt_cmp(">=", [Rat(1) if j == i else Rat(0) for j in range(num_vars)], names, Rat(0)))
        assertions.append(
            fmt_cmp(
                "<=",
                [Rat(1) if j == i else Rat(0) for j in range(num_vars)],
                names,
                w + Rat(rng.randint(5, 30)),
            )
        )

    # random half-spaces containing witness
    for _ in range(num_halfspaces):
        atom = halfspace_satisfied(rng, num_vars, witness, rng.choice(["<=", ">="]))
        assertions.append(atom.smt(names))

    if family == "piecewise":
        expr = piecewise_or(rng, names, witness, num_branches)
        assertions.append(expr.smt(names))
        desc = (
            f"piecewise OR-of-AND with {num_branches} branches over linear selector, "
            f"{num_vars} vars"
        )
    elif family == "nested":
        expr = random_bool_expr(rng, num_vars, witness, bool_depth, must_hold=True)
        assertions.append(expr.smt(names))
        for _ in range(rng.randint(1, 2)):
            assertions.append(
                random_bool_expr(rng, num_vars, witness, max(1, bool_depth - 1), must_hold=True).smt(
                    names
                )
            )
        desc = f"nested and/or/not depth~{bool_depth}, {num_vars} vars"
    elif family == "imply":
        for _ in range(num_implications):
            assertions.append(implication_atom(rng, num_vars, witness).smt(names))
        # add one piecewise disjunction for mixing
        assertions.append(piecewise_or(rng, names, witness, max(2, num_branches // 2)).smt(names))
        desc = f"{num_implications} (or (not P) Q) implications plus piecewise guard"
    elif family == "mixed":
        assertions.append(piecewise_or(rng, names, witness, num_branches).smt(names))
        assertions.append(
            random_bool_expr(rng, num_vars, witness, bool_depth, must_hold=True).smt(names)
        )
        for _ in range(num_implications):
            assertions.append(implication_atom(rng, num_vars, witness).smt(names))
        desc = f"mixed piecewise + nested bool + implications, {num_vars} vars"
    else:
        raise ValueError(f"unknown family: {family}")

    obj_coeffs = [Rat(rng.randint(-7, 7)) for _ in range(num_vars)]
    if all(c == 0 for c in obj_coeffs):
        obj_coeffs[0] = Rat(1)
    objective = rng.choice(["minimize", "maximize"])

    return Benchmark(
        bench_id=bench_id,
        family=family,
        names=names,
        witness=witness,
        assertions=assertions,
        objective=objective,
        obj_coeffs=obj_coeffs,
        seed=rng.randrange(2**31),
        description=desc,
    )


FAMILIES = ("piecewise", "nested", "imply", "mixed")


def validate_with_z3(z3_path: str, smt2: str, timeout_s: int = 30) -> Tuple[bool, str]:
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smt2", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(smt2)
            tmp_path = tmp.name
        proc = subprocess.run(
            [z3_path, f"-T:{timeout_s}", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s + 5,
        )
        Path(tmp_path).unlink(missing_ok=True)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, str(exc)
    out = proc.stdout.strip().splitlines()
    first = out[0].strip() if out else ""
    return first == "sat", first or proc.stderr.strip()


def write_manifest(output_dir: Path, benches: Sequence[Benchmark]) -> None:
    manifest = {
        "format": "QF_LRA-opt-benchmark-v1",
        "count": len(benches),
        "benchmarks": [
            {
                "file": f"{b.bench_id}.smt2",
                "id": b.bench_id,
                "family": b.family,
                "num_vars": len(b.names),
                "num_constraints": b.num_constraints,
                "objective": b.objective,
                "seed": b.seed,
                "description": b.description,
                "tags": [b.family, "qf_lra", "opt", "bool"],
            }
            for b in benches
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SAT-skewed LRA optimization benchmarks with Boolean structure."
    )
    parser.add_argument("--count", type=int, default=6, help="number of benchmarks")
    parser.add_argument("--seed", type=int, default=None, help="master RNG seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/lra_opt"),
        help="directory for .smt2 files and manifest.json",
    )
    parser.add_argument("--min-vars", type=int, default=3)
    parser.add_argument("--max-vars", type=int, default=8)
    parser.add_argument("--num-branches", type=int, default=5, help="piecewise OR branches")
    parser.add_argument("--bool-depth", type=int, default=3, help="nested bool depth")
    parser.add_argument("--num-implications", type=int, default=3)
    parser.add_argument("--num-halfspaces", type=int, default=4, help="extra linear half-spaces")
    parser.add_argument(
        "--family",
        choices=[*FAMILIES, "random"],
        default="random",
        help="benchmark family (random cycles families)",
    )
    parser.add_argument(
        "--validate-z3",
        metavar="Z3",
        default=None,
        help="if set, run z3 on each instance and require sat",
    )
    parser.add_argument(
        "--prefix",
        default="bool",
        help="filename prefix, e.g. bool -> bool_0001.smt2",
    )
    args = parser.parse_args(argv)

    if args.min_vars > args.max_vars:
        parser.error("--min-vars must be <= --max-vars")

    master_seed = args.seed if args.seed is not None else random.randrange(2**31)
    rng = random.Random(master_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benches: List[Benchmark] = []
    failures = 0

    for i in range(1, args.count + 1):
        bench_id = f"{args.prefix}_{i:04d}"
        num_vars = rng.randint(args.min_vars, args.max_vars)
        family = args.family if args.family != "random" else rng.choice(FAMILIES)
        bench_rng = random.Random(rng.randrange(2**31))

        bench = generate_benchmark(
            bench_rng,
            bench_id,
            family,
            num_vars,
            num_branches=args.num_branches,
            bool_depth=args.bool_depth,
            num_implications=args.num_implications,
            num_halfspaces=args.num_halfspaces,
        )
        smt2 = bench.to_smt2()
        out_path = args.output_dir / f"{bench_id}.smt2"
        out_path.write_text(smt2, encoding="utf-8")
        benches.append(bench)

        if args.validate_z3:
            ok, msg = validate_with_z3(args.validate_z3, smt2)
            status = "sat" if ok else f"FAIL({msg})"
            print(f"{bench_id}: {status}")
            if not ok:
                failures += 1
        else:
            print(f"wrote {out_path}  family={family}  vars={num_vars}")

    write_manifest(args.output_dir, benches)
    print(f"manifest -> {args.output_dir / 'manifest.json'}  (seed={master_seed})")

    if args.validate_z3 and failures:
        print(f"{failures}/{args.count} instances were not sat", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
