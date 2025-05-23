import pytest
from synthwave.typing import infer
from synthwave.dsl import UOp, Ops
from synthwave.eval import evaluate
from synthwave.parser import Parser


def parse(s):
    from synthwave.eval import parse, KNOWN_VARS
    return parse(s, known=KNOWN_VARS)

def define(s: str) -> UOp:
    expr = evaluate(parse(s))
    if not isinstance(expr, UOp):
        expr = UOp(Ops.Val, [expr])
    return expr


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
    abstr = evaluate(expr)
    assert str(abstr) == "Abstr(x, Appl(add, x, 1))"

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
    abstr = evaluate(expr)
    assert str(abstr) == "Abstr(x, y, Appl(add, x, y))"


def test_appl_multi():
    # (λx y. x * y) 6 7 => 42
    func_expr = UOp(Ops.Abstr, [
        "x", "y",
        UOp(Ops.Appl, [var("mul"), var("x"), var("y")])
    ])
    expr = UOp(Ops.Appl, [func_expr, val(6), val(7)])
    assert evaluate(expr) == 42


def test_nested():
    outer = parse("λx y.(λz.+ x z) (mul y 2)")
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

def test_map_add5():
    # map [1,2,3] (add 5) => Int List
    expr = define("map [1,2,3] (add 5)")
    assert expr.args[0] == [6, 7, 8]

def test_sort_then_reverse():
    expr = define("reverse (sort [3,1,2])")
    assert expr.args[0] == [3, 2, 1]

def test_sort_reverse_map():
    expr = define("reverse (sort (map [1,5,10,30] (λx.add 3 (mul x 2))))")
    assert expr.args[0] == [63, 23, 13, 5]

def test_closure_call():
    inc = UOp(Ops.Closure, [
        UOp(Ops.Abstr, [
            "x",
            UOp(Ops.Appl, [
                var("add"),
                var("x"),
                val(1)
            ])
        ]),
        {}, # (no free variables)
    ])
    # The uop.__call__(5) will:
    # 1. Wrap 5 as UOp(Ops.Val, [5])
    # 2. Form Appl(inc_func, ...)
    # 3. Evaluate it
    assert inc(5) == 6

def test_shadowing():
    # We define an expression that uses shadowing:
    # (λx. ((λx. add(x 2)) mul(x 10))) 5
    outer_lambda = UOp(Ops.Abstr, [
        "x",
        UOp(Ops.Appl, [
            UOp(Ops.Abstr, [
                "x",
                UOp(Ops.Appl, [
                    var("add"),
                    var("x"),
                    val(2),
                ])
            ]),
            UOp(Ops.Appl, [
                var("mul"),
                var("x"),
                val(10)
            ])
        ])
    ])
    expr = UOp(Ops.Appl, [outer_lambda, val(5)])
    assert evaluate(expr) == 52

def test_infer_val():
    # Val(42) -> Int
    expr = val(42)
    ty = infer(expr)
    assert str(ty), "Int"

def test_infer_abstr_single():
    # λx. x + 1 => Int -> Int
    # Because x must be an Int to do (x + 1).
    expr = parse("(λx.add x 1)")
    ty = infer(expr)
    assert str(ty) == "Int -> Int"

def test_infer_appl_single():
    # (λx. x+1) 41 => Int
    func = UOp(Ops.Abstr, [
        "x",
        UOp(Ops.Appl, [
            var("add"),
            var("x"),
            val(1)
        ])
    ])
    expr = UOp(Ops.Appl, [func, val(41)])
    ty = infer(expr)
    assert str(ty) == "Int"

def test_infer_abstr_multi():
    # λx y. x + y => Int -> Int -> Int
    expr = UOp(Ops.Abstr, [
        "x",
        "y",
        UOp(Ops.Appl, [var("add"), var("x"), var("y")])
    ])
    ty = infer(expr)
    assert len(ty.params) == 2
    t1 = ty.params[0]
    assert str(ty) == f"{t1} -> {t1} -> {t1}"

def test_infer_appl_multi():
    # (λx y. x * y) 6 7 => Int
    func = UOp(Ops.Abstr, [
        "x",
        "y",
        UOp(Ops.Appl, [
            var("mul"),
            var("x"),
            var("y"),
        ])
    ])
    expr = UOp(Ops.Appl, [
        func,
        val(6),
        val(7),
    ])
    ty = infer(expr)
    assert str(ty) == "Int"

def test_infer_nested():
    # (λx y. (λz. x + z) (y * 2)) 5 10 => int
    # Inside:
    #    x y => body is (λz. x+z) y*2
    #    y*2 is int
    #    so we pass that to λz. x+z => x z are int => int
    # Overall => int
    outer = UOp(Ops.Abstr, [
        "x",
        "y",
        UOp(Ops.Appl, [
            UOp(Ops.Abstr, [
                "z",
                UOp(Ops.Appl, [
                    var("add"),
                    var("x"),
                    var("z"),
                ])
            ]),
            UOp(Ops.Appl, [
                var("mul"),
                var("y"),
                val(2),
            ])
        ])
    ])
    call = UOp(Ops.Appl, [outer, val(5), val(10)])
    ty = infer(call)
    assert str(ty) == "Int"

def test_infer_type_error():
    # A mismatch: Add(Val(5), λx. x+1)
    # => We unify left side as int, right side as ? => but it becomes a function type
    # => Should raise a unification TypeError
    expr = UOp(Ops.Appl, [
        var("add"),
        val(5),
        UOp(Ops.Abstr, [
            "x",
            UOp(Ops.Appl, [
                var("add"),
                var("x"),
                val(1),
            ])
        ])
    ])
    with pytest.raises(Exception) as excinfo:
        infer(expr)
    assert "unify" in str(excinfo.value) or "Cannot unify" in str(excinfo.value), \
        f"Expected unification error message, got {excinfo.value}"

def test_infer_map_add5():
    # map [1, 2, 3] (add 5) => Int List
    # Partially applying `add` with 5
    expr = parse("map [1,2,3] (add 5)")
    ty = infer(expr)
    assert str(ty) == "Int List"

def test_unbound_variable():
    # Unbound variable fails
    with pytest.raises(SyntaxError, match="Unbound variable"):
        parse("x")

def test_nested_lambda():
    # Test a nested lambda where inner ref is correctly bound
    ast = parse("λx.λy.x")
    assert ast.op == Ops.Abstr
    assert ast.args[0] == "x"
    inner_lambda = ast.args[1]
    assert inner_lambda.op == Ops.Abstr
    assert inner_lambda.args[0] == "y"
    inner_body = inner_lambda.args[1]
    assert inner_body.op == Ops.Var
    assert inner_body.args[0] == "x"

def test_unbound_in_nested_lambda():
    # Inner lambda refers to an unbound variable
    with pytest.raises(SyntaxError, match="Unbound variable"):
        parse("λx.(λy.z)")
