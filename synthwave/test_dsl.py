import pytest
from synthwave.dsl import UOp, Ops, evaluate


def val(x):
    return UOp(Ops.Val, [x])


def var(x):
    return UOp(Ops.Var, [x])


def test_var_unbound():
    expr = var("x")
    with pytest.raises(NameError):
        evaluate(expr, env={})  # x not in env => error


def test_var_bound():
    expr = var("x")
    assert evaluate(expr, env={"x": val(99)}) == 99


def test_add():
    expr = UOp(Ops.Appl, [var("add"), val(10), val(32)])
    assert evaluate(expr) == 42


def test_mul():
    expr = UOp(Ops.Appl, [var("mul"), val(6), val(7)])
    assert evaluate(expr) == 42


def test_abstr_single():
    # λx. x + 1
    expr = UOp(Ops.Abstr, [
        "x",
        UOp(Ops.Appl, [var("add"), var("x"), val(1)])
    ])
    # Evaluate to a closure
    closure = evaluate(expr)
    assert isinstance(closure, UOp)
    assert closure.op == Ops.Closure
    assert closure.args[0] == ["x"]          # params
    assert closure.args[1] == expr.args[1]   # body
    assert closure.args[2] == {}             # captured env


def test_appl_single():
    # (λx. x + 1) 41 => 42
    func_expr = UOp(Ops.Abstr, [
        "x",
        UOp(Ops.Appl, [var("add"), var("x"), val(1)])
    ])
    expr = UOp(Ops.Appl, [func_expr, val(41)])
    assert evaluate(expr) == 42


def test_abstr_multi():
    # λx y. x + y
    expr = UOp(Ops.Abstr, [
        "x", "y",
        UOp(Ops.Appl, [var("add"), var("x"), var("y")])
    ])
    closure = evaluate(expr)
    assert isinstance(closure, UOp)
    assert closure.op == Ops.Closure
    assert closure.args[0] == ["x", "y"]


def test_appl_multi():
    # (λx y. x * y) 6 7 => 42
    func_expr = UOp(Ops.Abstr, [
        "x", "y",
        UOp(Ops.Appl, [var("mul"), var("x"), var("y")])
    ])
    expr = UOp(Ops.Appl, [func_expr, val(6), val(7)])
    assert evaluate(expr) == 42


def test_nested():
    outer = UOp(
        Ops.Abstr,
        [
            "x",
            "y",
            UOp(
                Ops.Appl,
                [
                    UOp( # λz. x + z
                        Ops.Abstr,
                        ["z", UOp(Ops.Appl, [var("add"), var("x"), var("z")])]
                    ),
                    UOp( # y * 2 => (mul y 2)
                        Ops.Appl,
                        [var("mul"), var("y"), val(2)]
                    ),
                ],
            ),
        ],
    )
    # Apply outer to 5, 10
    expr = UOp(Ops.Appl, [outer, val(5), val(10)])
    assert evaluate(expr) == 25


def test_environment_sharing():
    # let free = 100 in
    # (λx. x + free) 1 => 101
    abstraction_with_free = UOp(
        Ops.Abstr,
        ["x", UOp(Ops.Appl, [var("add"), var("x"), var("free")])]
    )
    expr = UOp(Ops.Appl, [abstraction_with_free, val(1)])
    assert evaluate(expr, env={"free": val(100)}) == 101

def test_partial_appl_simple():
    partial_expr = UOp(Ops.Appl, [var("add"), val(10)])
    partial_result = evaluate(partial_expr)
    assert isinstance(partial_result, UOp)
    assert partial_result.op == Ops.Closure
    expr = UOp(Ops.Appl, [partial_expr, val(32)])
    result = evaluate(expr)
    assert result == 42

def test_partial_appl_chain():
    # step1 = add 1         => closure
    # step2 = step1 2       => 3
    # step3 = add step2     => closure
    # step4 = step3 39      => 42
    step1 = UOp(Ops.Appl, [var("add"), val(1)])
    closure_step1 = evaluate(step1)
    assert isinstance(closure_step1, UOp)
    assert closure_step1.op == Ops.Closure
    step2 = UOp(Ops.Appl, [step1, val(2)])
    val_step2 = evaluate(step2)
    assert val_step2 == 3
    step3 = UOp(Ops.Appl, [var("add"), UOp(Ops.Val, [val_step2])])
    closure_step3 = evaluate(step3)
    assert isinstance(closure_step3, UOp)
    assert closure_step3.op == Ops.Closure
    step4 = UOp(Ops.Appl, [step3, val(39)])
    assert evaluate(step4) == 42
