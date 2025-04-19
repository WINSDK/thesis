from collections.abc import Callable
from typing import Dict, Optional
from .dsl import UOp, Ops, ExternalError, reduce_redundant
from .helpers import fn_parameters
from .parser import parse, parse_poly_type

def eval_maybe_py(expr, env):
    # Specialized to lists and singular values
    if isinstance(expr, UOp):
        return _evaluate(expr, env)
    elif isinstance(expr, list):
        return [eval_maybe_py(x, env) for x in expr]
    return expr

def lookup(varname, env):
    if varname in env:
        var = env[varname]
    elif varname in BUILTINS:
        var = BUILTINS[varname]
    else:
        raise NameError(f"Unbound variable: {varname}")
    return var

def _evaluate(expr: UOp, env: Dict[str, UOp]):
    match expr.op:
        case Ops.Val:
            return eval_maybe_py(expr.args[0], env)
        case Ops.Var:
            expr = lookup(expr.args[0], env)
            return eval_maybe_py(expr, env)
        case Ops.Closure:
            return expr
        case Ops.Abstr | Ops.External:
            return UOp(Ops.Closure, [*expr.args, env])
        case Ops.Appl:
            func, *args = expr.args
            func = _evaluate(func, env) # Closure around either an abstr or external
            args = [_evaluate(a, env) for a in args]
            while args:
                if not (isinstance(func, UOp) and func.op == Ops.Closure):
                    raise TypeError(f"Cannot call a non-closure: {func}")
                *params, body, closure_env = func.args
                def apply(env, params, args):
                    env = env.copy()
                    env.update(dict(zip(params, args)))
                    return env
                if len(args) < len(params):
                    # Not enough arguments, so we do partial application
                    # which will require creating a new closure
                    not_applied = params[len(args):]
                    closure_env = apply(closure_env, params, args)
                    return UOp(Ops.Closure, [*not_applied, body, closure_env])
                # Full application or over-applied
                closure_env = apply(closure_env, params, args[:len(params)])
                if isinstance(body, UOp):
                    func = _evaluate(body, closure_env)
                    args = args[len(params):]
                    continue
                # Applying python function
                params = fn_parameters(body)
                pv = {p: eval_maybe_py(closure_env[p], env) for p in params}
                try:
                    result = body(**pv)
                except Exception as e:
                    params = ", ".join(map(str, pv.keys()))
                    raise ExternalError(str(e))
                args = args[len(params):]
                # It's a full application
                if not args:
                    return eval_maybe_py(result, env)
                # The builtin returned a new closure
                if isinstance(result, UOp) and result.op == Ops.Closure:
                    func = eval_maybe_py(result, env)
                    continue
                # The builtin returned another builtin
                if callable(result):
                    func = external_fn(result)
                    continue
                raise ExternalError(f"Overapplication: {result} is not callable.")
            return func

def substitute(expr, var: str, new_expr):
    if isinstance(expr, list):
        return [substitute(e, var, new_expr) for e in expr]
    if not isinstance(expr, UOp):
        return expr
    if expr.op == Ops.Var and expr.args[0] == var:
        return new_expr
    elif expr.op == Ops.Appl:
        body, *args = expr.args
        args = [substitute(e, var, new_expr) for e in args]
        return UOp(Ops.Appl, [body, *args])
    elif expr.op == Ops.Abstr:
        *params, body = expr.args
        # If the bound variable is the same as the substitution variable,
        # we leave it alone (or rename it to avoid capture).
        if not any(p == var for p in params):
            body = substitute(body, var, new_expr)
        return UOp(Ops.Abstr, [*params, body])
    else:
        return expr

def reify(expr):
    if isinstance(expr, list):
        return [reify(e) for e in expr]
    if not isinstance(expr, UOp):
        return expr
    if expr.op == Ops.Closure:
        *args, body, env = expr.args
        if not isinstance(body, UOp) and callable(body):
            params = fn_parameters(body)
            fn_name = body.__name__.removeprefix("builtin_")
            body = UOp(Ops.Var, [fn_name])
            val_args = [UOp(Ops.Val, [env[p]]) for p in params if p in env]
            inner_args = [UOp(Ops.Var, [a]) for a in args]
            inner = UOp(Ops.Appl, [body, *val_args, *inner_args])
            return UOp(Ops.Abstr, [*args, inner])
        # Reconstruct an abstraction
        for var, val in env.items():
            # Function that replaces free occurrences of `var` in `body` with the reified form.
            body = substitute(body, var, reify(val))
        return UOp(Ops.Abstr, [*args, reify(body)])
    elif expr.op == Ops.Appl:
        return UOp(Ops.Appl, [reify(e) for e in expr.args])
    elif expr.op == Ops.Abstr:
        *params, body = expr.args
        return UOp(Ops.Abstr, [*params, reify(body)])
    else:
        return expr

def back_substitution(expr, env: Dict[str, UOp]):
    env = env | BUILTINS
    def aux(expr):
        if isinstance(expr, list):
            return [aux(e) for e in expr]
        if not isinstance(expr, UOp):
            return expr
        expr = UOp(expr.op, [aux(a) for a in expr.args])
        for var, kexpr in env.items():
            if expr == kexpr:
                return UOp(Ops.Var, [var])
        return expr
    return aux(expr)

def evaluate(expr: UOp, env: Optional[Dict[str, UOp]]=None):
    if env is None:
        env = {}
    # Don't eval variables that hold abstr's and externals
    if expr.op == Ops.Var:
        inner_expr = lookup(expr.args[0], env)
        if inner_expr.op == Ops.Abstr or inner_expr.op == Ops.External:
            return expr
    if expr.op == Ops.Abstr or expr.op == Ops.External:
        # Don't evaluate abstr's or externals without application
        return expr
    rexpr = _evaluate(expr, env)
    rexpr = reify(rexpr)
    rexpr = reduce_redundant(rexpr)
    rexpr = back_substitution(rexpr, env)
    return rexpr

def external_fn(f: Callable):
    params = fn_parameters(f)
    return UOp(Ops.External, [*params, f])

### Arithmetic primitives

def builtin_add(x, y):
    return x + y

def builtin_mul(x, y):
    return x * y

def builtin_sub(x, y):
    return x - y

def builtin_div(x, y):
    return x // y

def builtin_mod(x, y):
    return x % y

def builtin_pow(x, y):
    return x ** y

### Comparison primitives

true_expr = parse("λt f. t")
false_expr = parse("λt f. f")

if_expr = parse("λc t e. c t e")

def church_bool(cond):
    return true_expr if cond else false_expr

def builtin_eq(x, y):
    if isinstance(x, UOp) and isinstance(y, UOp):
        return and_expr(x, y)
    return church_bool(x == y)

def builtin_neq(x, y):
    return not builtin_eq(x, y)

def builtin_gt(x, y):
    return church_bool(x > y)

def builtin_lt(x, y):
    return church_bool(x < y)

def builtin_geq(x, y):
    return church_bool(x <= y)  # note: swapped ordering if needed

def builtin_leq(x, y):
    return church_bool(x >= y)

### Boolean/logical operators

not_expr = parse("λp. p False True", known={"False", "True"})
and_expr = parse("λp q. p q False", known={"False"})
or_expr = parse("λp q. p True q", known={"True"})
xor_expr = parse("λa b. a (not b) b", known={"not"})

### List and collection utilities

nil_expr = parse("[]")
is_nil_expr = parse("λxs. eq xs nil", known={"eq", "nil"})

def builtin_lfold(xs, acc, f):
    for elem in xs:
        acc = f(acc, elem)
    return acc

def builtin_rfold(xs, f, acc):
    for elem in reversed(xs):
        acc = f(elem, acc)
    return acc

map_expr = parse(
    "λxs f. rfold xs (λx acc. cons (f x) acc) nil",
    known={"rfold", "cons", "nil"}
)
filter_expr = parse(
    "λxs p. rfold xs (λx acc. (p x) (cons x acc) acc) nil",
    known={"rfold", "cons", "nil"}
)

def builtin_zip(xs1, xs2):
    return list(map(list, zip(xs1, xs2)))

def builtin_length(x):
    return len(x)

def builtin_range(start, end):
    return list(range(start, end))

def builtin_cons(x, xs):
    return [x] + xs

def builtin_head(xs):
    return xs[0]

def builtin_tail(xs):
    return xs[1:]

def builtin_append(xs, x):
    return xs + [x]

def builtin_reverse(xs):
    return list(reversed(xs))

def builtin_sort(xs):
    return sorted(xs)

def builtin_flatten(xss):
    return [x for xs in xss for x in xs]

### String manipulation primitives

def split(s):
    return list(s)

def join(s):
    return "".join(s)

def builtin_str_concat(s1, s2):
    return s1 + s2

def builtin_substring(s, start, end):
    return s[start:end]

def builtin_split(s, sep=None):
    if sep is not None:
        sep = join(sep)
    return list(map(split, join(s).split(sep)))

def builtin_join(lst, sep):
    return split(join(sep).join(map(join, lst)))

### Conversion primitives

def builtin_read(s):
    return int(join(s))

def builtin_show(x):
    return split(str(x))

### Utility/functional primitives

def builtin_print(x):
    print(x)
    return x

id_expr = parse("λx. x")
compose_expr = parse("λf. λg. λx. f (g x)")

BUILTINS = {
    "True": true_expr,
    "False": false_expr,
    # arithmetic
    "add": external_fn(builtin_add),
    "+"  : external_fn(builtin_add),
    "mul": external_fn(builtin_mul),
    "*"  : external_fn(builtin_mul),
    "sub": external_fn(builtin_sub),
    "-"  : external_fn(builtin_sub),
    "div": external_fn(builtin_div),
    "/"  : external_fn(builtin_div),
    "mod": external_fn(builtin_mod),
    "%"  : external_fn(builtin_mod),
    "pow": external_fn(builtin_pow),
    "**" : external_fn(builtin_pow),
    # Comparisons
    "if": if_expr,
    "eq": external_fn(builtin_eq),
    "neq": external_fn(builtin_neq),
    "gt": external_fn(builtin_gt),
    "lt": external_fn(builtin_lt),
    "geq": external_fn(builtin_geq),
    "leq": external_fn(builtin_leq),
    # Boolean operators
    "not": not_expr,
    "and": and_expr,
    "or": or_expr,
    # List utilities
    "nil": nil_expr,
    "is_nil": is_nil_expr,
    "lfold": external_fn(builtin_lfold),
    "rfold": external_fn(builtin_rfold),
    "map": map_expr,
    "filter": filter_expr,
    "zip": external_fn(builtin_zip),
    "length": external_fn(builtin_length),
    "range": external_fn(builtin_range),
    "cons": external_fn(builtin_cons),
    "head": external_fn(builtin_head),
    "tail": external_fn(builtin_tail),
    "append": external_fn(builtin_append),
    "reverse": external_fn(builtin_reverse),
    "sort": external_fn(builtin_sort),
    "flatten": external_fn(builtin_flatten),
    # String manipulation
    "concat": external_fn(builtin_str_concat),
    "substr": external_fn(builtin_substring),
    "split": external_fn(builtin_split),
    "join": external_fn(builtin_join),
    # Conversion
    "show": external_fn(builtin_show),
    "read": external_fn(builtin_read),
    # Utility/functional
    "print": external_fn(builtin_print),
    "id": id_expr,
    "compose": compose_expr,
}

BUILTIN_SCHEMES = {
    # Arithmetic primitives
    "add":     parse_poly_type("T -> T -> T"),
    "+":       parse_poly_type("T -> T -> T"),
    "mul":     parse_poly_type("T -> T -> T"),
    "*":       parse_poly_type("T -> T -> T"),
    "sub":     parse_poly_type("T -> T -> T"),
    "-":       parse_poly_type("T -> T -> T"),
    "div":     parse_poly_type("T -> T -> T"),
    "/":       parse_poly_type("T -> T -> T"),
    "mod":     parse_poly_type("T -> T -> T"),
    "%":       parse_poly_type("T -> T -> T"),
    "pow":     parse_poly_type("T -> T -> T"),
    "**":      parse_poly_type("T -> T -> T"),
    # Comparisons
    "if":      parse_poly_type("T -> A -> B"),
    "eq":      parse_poly_type("T -> T -> Bool"),
    "neq":     parse_poly_type("T -> T -> Bool"),
    "gt":      parse_poly_type("T -> T -> Bool"),
    "lt":      parse_poly_type("T -> T -> Bool"),
    "geq":     parse_poly_type("T -> T -> Bool"),
    "leq":     parse_poly_type("T -> T -> Bool"),
    # Boolean operators
    "True":    parse_poly_type("Bool"),
    "False":   parse_poly_type("Bool"),
    "not":     parse_poly_type("Bool -> Bool"),
    "and":     parse_poly_type("Bool -> Bool -> Bool"),
    "or":      parse_poly_type("Bool -> Bool -> Bool"),
    # List utilities
    "nil":     parse_poly_type("List T"),
    "is_nil":  parse_poly_type("List T -> Bool"),
    "lfold":   parse_poly_type("List A -> B -> (B -> A -> A) -> B"),
    "rfold":   parse_poly_type("List A -> (A -> B -> B) -> B -> B"),
    "map":     parse_poly_type("List A -> (A -> B) -> List B"),
    "filter":  parse_poly_type("List A -> (A -> Bool) -> List A"),
    "zip":     parse_poly_type("List T -> List T -> List (List T)"),
    "length":  parse_poly_type("List A -> Int"),
    "range":   parse_poly_type("Int -> Int -> List Int"),
    "cons":    parse_poly_type("T -> List T -> List T"),
    "head":    parse_poly_type("List T -> T"),
    "tail":    parse_poly_type("List T -> List T"),
    "append":  parse_poly_type("List T -> T -> List T"),
    "reverse": parse_poly_type("List T -> List T"),
    "sort":    parse_poly_type("List T -> List T"),
    "flatten": parse_poly_type("List List T -> List T"),
    # String manipulation
    "concat":  parse_poly_type("String -> String -> String"),
    "substr":  parse_poly_type("String -> Int -> Int -> String"),
    "split":   parse_poly_type("String -> String -> List String"),
    "join":    parse_poly_type("List String -> String -> String"),
    # Conversion
    "show":    parse_poly_type("T -> String"),
    "read":    parse_poly_type("String -> Int"),
    # Utility/functional
    "print":   parse_poly_type("T -> T"),
    "id":      parse_poly_type("A -> A"),
    "compose": parse_poly_type("(B -> C) -> (A -> B) -> A -> C"),
}

KNOWN_VARS = set(BUILTINS.keys())

# Safeguard for missing schemes (type checker really doesn't like when that happens)
for k in KNOWN_VARS:
    if k not in BUILTIN_SCHEMES:
        print(f"{k} missing scheme")
        exit(1)
