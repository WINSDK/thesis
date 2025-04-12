import pytest
from synthwave.dsl import UOp, Ops, TInt, TList, TArrow, evaluate, parse, infer


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
    assert closure.args[0] == "x"            # param 1
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
    assert closure.args[0] == "x"
    assert closure.args[1] == "y"


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
    assert evaluate(expr) == 42

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

def test_map_add5():
    # map (add 5) [1,2,3] => Int -> Int
    # Partially applying `add` with 5
    expr = UOp(Ops.Appl, [
        UOp(Ops.Appl, [
            var("map"),
            UOp(Ops.Appl, [
                var("add"),
                val(5),
            ])
        ]),
        val([1, 2, 3]),
    ])
    assert evaluate(expr) == [6, 7, 8]

def test_sort_then_reverse():
    expr = parse("reverse (sort [3, 1, 2])")
    assert evaluate(expr) == [3, 2, 1]

def test_sort_reverse_map():
    expr = parse("reverse (sort (map (Lx. add 3 (mul x 2)) [1, 5, 10, 30]))")
    assert evaluate(expr) == [63, 23, 13, 5]

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
    assert isinstance(ty, TInt), f"Expected TInt, got {ty}"

def test_infer_abstr_single():
    # λx. x + 1 => Int -> Int
    # Because x must be an Int to do (x + 1).
    expr = UOp(Ops.Abstr, [
        "x",
        UOp(Ops.Appl, [var("add"), var("x"), val(1)])
    ])
    ty = infer(expr)
    assert isinstance(ty, TArrow), f"Expected TArrow, got {ty}"
    assert isinstance(ty.param, TInt), f"Expected param TInt, got {ty.param}"
    assert isinstance(ty.result, TInt), f"Expected result TInt, got {ty.result}"

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
    assert isinstance(ty, TInt), f"Expected TInt, got {ty}"

def test_infer_abstr_multi():
    # λx y. x + y => Int -> Int -> Int
    expr = UOp(Ops.Abstr, [
        "x",
        "y",
        UOp(Ops.Appl, [var("add"), var("x"), var("y")])
    ])
    ty = infer(expr)
    assert isinstance(ty, TArrow), f"Expected TArrow, got {ty}"
    assert isinstance(ty.param, TInt), f"Expected first param TInt, got {ty.param}"
    rty = ty.result
    assert isinstance(rty, TArrow), f"Expected TArrow, got {rty}"
    assert isinstance(rty.param, TInt), f"Expected second param TInt, got {rty.param}"
    assert isinstance(rty.result, TInt), f"Expected result TInt, got {rty.result}"

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
    assert isinstance(ty, TInt), f"Expected TInt, got {ty}"

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
    assert isinstance(ty, TInt), f"Expected TInt, got {ty}"

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
    # map (add 5) [1,2,3] => Int List
    # Partially applying `add` with 5
    expr = UOp(Ops.Appl, [
        UOp(Ops.Appl, [
            var("map"),
            UOp(Ops.Appl, [
                var("add"),
                val(5),
            ])
        ]),
        val([1, 2, 3]),
    ])
    ty = infer(expr)
    assert isinstance(ty, TList), f"Expected list type, got {ty}"
    assert isinstance(ty.elems, TInt), f"Expected list of ints, got {ty.elems}"
