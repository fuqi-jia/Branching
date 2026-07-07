"""Spike: 用 z3 UserPropagator 的 add_decide/next_split 从外部**接管 z3 内部布尔分支决策**，
不改 z3。证明：不同策略 -> 不同决策序列 + 不同求解开销（=我们真的控制了内部分支）。
"""
import random
import z3


def rlimit(s):
    st = s.statistics()
    for k in st.keys():
        if k == "rlimit count":
            return st.get_key_value(k)
    return 0


class BranchProp(z3.UserPropagateBase):
    """接管 z3 决策：每次 z3 要 decide 时，改成按 policy 从未定原子里选一个 next_split。"""
    def __init__(self, s, atoms, policy):
        super().__init__(s)
        self.atoms = atoms
        self.policy = policy
        self._val = {}                 # name -> bool
        self.trail = []                 # 决策/传播分配的 name 轨迹
        self.trail_lim = []
        self.decisions = []             # 我们强制的决策序列（name）
        self.add_fixed(self._fixed)
        self.add_decide(self._decide)
        for a in atoms:
            self.add(a)                 # 注册原子，z3 才会在其上让我们决策
        self.name2atom = {str(a): a for a in atoms}

    def push(self):
        self.trail_lim.append(len(self.trail))

    def pop(self, n):
        for _ in range(n):
            lim = self.trail_lim.pop()
            while len(self.trail) > lim:
                self._val.pop(self.trail.pop(), None)

    def fresh(self, new_ctx):
        return BranchProp(new_ctx, self.atoms, self.policy)

    def _fixed(self, t, v):
        nm = str(t)
        if nm not in self._val:
            self._val[nm] = z3.is_true(v)
            self.trail.append(nm)

    def _decide(self, t, idx, phase):
        undecided = [a for a in self.atoms if str(a) not in self._val]
        if not undecided:
            return
        choice = self.policy(undecided)
        self.decisions.append(str(choice))
        self.next_split(choice, 0, 1)   # 强制在我们选的原子上分裂（相位=真）


def hard_3sat(n=24, ratio=4.2, seed=0):
    rng = random.Random(seed)
    xs = [z3.Bool(f"x{i}") for i in range(n)]
    clauses = []
    for _ in range(int(n * ratio)):
        lits = rng.sample(range(n), 3)
        clauses.append(z3.Or([xs[i] if rng.random() < 0.5 else z3.Not(xs[i]) for i in lits]))
    return xs, clauses


def solve_with(policy_name, policy, xs, clauses):
    s = z3.Solver()
    prop = BranchProp(s, xs, policy)     # 接管决策
    s.add(*clauses)
    res = s.check()
    return res, rlimit(s), prop.decisions


def solve_baseline(clauses):
    s = z3.Solver()
    s.add(*clauses)
    res = s.check()
    return res, rlimit(s)


if __name__ == "__main__":
    xs, clauses = hard_3sat(seed=3)
    # 基线：z3 自己的 VSIDS
    b_res, b_rl = solve_baseline(clauses)
    print(f"baseline(z3 VSIDS):  {b_res}  rlimit={b_rl}")

    # 策略 A：总选下标最小的未定原子
    a_res, a_rl, a_dec = solve_with(
        "asc", lambda und: min(und, key=lambda a: int(str(a)[1:])), xs, clauses)
    # 策略 B：总选下标最大的未定原子
    c_res, c_rl, c_dec = solve_with(
        "desc", lambda und: max(und, key=lambda a: int(str(a)[1:])), xs, clauses)

    print(f"policy ASC  (我们接管): {a_res}  rlimit={a_rl}  #decisions={len(a_dec)}  first5={a_dec[:5]}")
    print(f"policy DESC (我们接管): {c_res}  rlimit={c_rl}  #decisions={len(c_dec)}  first5={c_dec[:5]}")
    print()
    print("=== 证据 ===")
    print(f"1) decide 回调被触发、next_split 被接受: ASC 强制了 {len(a_dec)} 次决策，序列非空 = {bool(a_dec)}")
    print(f"2) 结果一致(正确性): baseline={b_res}, ASC={a_res}, DESC={c_res}, 一致 = {b_res==a_res==c_res}")
    print(f"3) 我们真的控制了分支: ASC vs DESC 决策序列不同 = {a_dec != c_dec}; rlimit 不同 = {a_rl != c_rl} ({a_rl} vs {c_rl})")
