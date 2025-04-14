from collections.abc import Callable
from typing import Dict, Optional
from .dsl import UOp, Ops, ExternalError
from .helpers import fn_parameters
from .parser import parse

def evaluate(expr: UOp, env: Optional[Dict[str, UOp]]=None):
    if env is None:
        env = {}
    def lookup(varname):
        if varname in env:
            var = env[varname]
        elif varname in BUILTINS:
            var = BUILTINS[varname]
        else:
            raise NameError(f"Unbound variable: {varname}")
        return var
    match expr.op:
        case Ops.Val:
            expr = expr.args[0]
            return evaluate(expr) if isinstance(expr, UOp) else expr
        case Ops.Var:
            expr = lookup(expr.args[0])
            return evaluate(expr) if isinstance(expr, UOp) else expr
        case Ops.Closure:
            return expr
        case Ops.Abstr | Ops.External:
            return UOp(Ops.Closure, [*expr.args, env])
        case Ops.Appl:
            func, *args = expr.args
            func = evaluate(func, env) # Closure around either an abstr or external
            args = [evaluate(a, env) for a in args]
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
                    func = evaluate(body, closure_env)
                    args = args[len(params):]
                    continue
                # Applying python function
                pv = {p: closure_env[p] for p in fn_parameters(body)}
                try:
                    result = body(**pv)
                except Exception as e:
                    params = ", ".join(map(str, pv.keys()))
                    raise ExternalError(str(e))
                args = args[len(params):]
                # It's a full application
                if not args:
                    return result
                # The builtin returned a new closure
                if isinstance(result, UOp) and result.op == Ops.Closure:
                    func = result
                    continue
                # The builtin returned another builtin
                if callable(result):
                    func = external_fn(result)
                    continue
                raise ExternalError(f"Overapplication: {result} is not callable.")
            return func

def external_fn(f: Callable):
    params = fn_parameters(f)
    return UOp(Ops.External, [*params, f])

def define(e):
    # Can't just `parse` sometimes. Some expressions require being evaluated
    # into closures: these can be abstractions or externals.
    return evaluate(parse(e))

y_combinator_expr = define("λf. (λx. f (x x)) (λx. f (x x))")

true_expr = define("True")
false_expr = define("False")

### Arithmetic primitives

def builtin_add(x: int, y: int) -> int:
    return x + y

def builtin_mul(x: int, y: int) -> int:
    return x * y

def builtin_sub(x: int, y: int) -> int:
    return x - y

def builtin_div(x: int, y: int) -> int:
    return x // y

def builtin_mod(x: int, y: int) -> int:
    return x % y

def builtin_pow(x: int, y: int) -> int:
    return x ** y

### Comparison primitives

if_expr = define("λc t e. c t e")

def church_bool(cond: bool) -> UOp:
    return true_expr if cond else false_expr

def builtin_eq(x, y) -> UOp:
    return church_bool(x == y)

def builtin_neq(x, y) -> UOp:
    return church_bool(x != y)

def builtin_gt(x, y) -> UOp:
    return church_bool(x > y)

def builtin_lt(x, y) -> UOp:
    return church_bool(x < y)

def builtin_geq(x, y) -> UOp:
    return church_bool(x <= y) # Swapped cause of ordering

def builtin_leq(x, y) -> UOp:
    return church_bool(x >= y)

### Boolean/logical operators

not_expr = define("λp. p False True")
and_expr = define("λp q. p q False")
or_expr = define("λp q. p True q")

### List and collection utilities

nil_expr = define("[]")
is_nil_expr = define("λxs. eq xs nil")

# 'a list -> 'acc -> ('acc -> 'a -> 'acc) -> 'acc
def builtin_lfold(xs, acc, f):
    for elem in xs:
        acc = f(acc, elem)
    return acc

# 'a list -> ('a -> 'acc -> 'acc) -> 'acc -> 'acc
def builtin_rfold(xs: list[int], f: Callable[[int, int], int], acc: int) -> int:
    for elem in reversed(xs):
        acc = f(elem, acc)
    return acc

map_expr = define("λxs f. rfold xs (λx acc. cons (f x) acc) nil")
filter_expr = define("λxs f. lfold xs nil (λacc x. (f x) (cons x acc) acc)")

def builtin_zip(xs1, xs2):
    return list(map(list, zip(xs1, xs2)))

def builtin_length(x) -> int:
    return len(x)

def builtin_range(start: int, end: int):
    return list(range(start, end))

def builtin_cons(x: int, xs: list[int]) -> list[int]:
    return [x] + xs

def builtin_head(xs):
    return xs[0]

def builtin_tail(xs):
    return xs[1:]

def builtin_append(xs, x):
    return xs + [x]

def builtin_reverse(xs: list[int]) -> list[int]:
    return list(reversed(xs))

def builtin_sort(xs: list[int]) -> list[int]:
    return sorted(xs)

### String manipulation primitives

String = list[str]

def split(s: str) -> String:
    return list(s)

def join(s: String) -> str:
    return "".join(s)

def builtin_str_concat(s1: String, s2: String) -> String:
    return s1 + s2

def builtin_substring(s: String, start: int, end: int) -> String:
    return s[start:end]

def builtin_split(s: String, sep: Optional[String] = None) -> list[String]:
    if sep is not None:
        sep: Optional[str] = join(sep)
    return list(map(split, join(s).split(sep)))

def builtin_join(lst: list[String], sep: String) -> String:
    return split(join(sep).join(map(join, lst)))

### Conversion primitives

def builtin_to_str(x) -> String:
    return split(str(x))

def builtin_to_int(s: String) -> int:
    return int(join(s))

### Utility/functional primitives

def builtin_print(x):
    # Print for side-effect and also return the value.
    print(x)
    return x

def builtin_identity(x):
    return x

def builtin_compose(f: Callable, g: Callable) -> Callable:
    return lambda x: f(g(x))

# Builtins dictionary: add both the original and new primitives.
BUILTINS = {
    "Y": y_combinator_expr,
    # arithmetic
    "add": external_fn(builtin_add),
    "mul": external_fn(builtin_mul),
    "sub": external_fn(builtin_sub),
    "div": external_fn(builtin_div),
    "mod": external_fn(builtin_mod),
    "pow": external_fn(builtin_pow),
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
    # String manipulation
    "concat": external_fn(builtin_str_concat),
    "substr": external_fn(builtin_substring),
    "split": external_fn(builtin_split),
    "join": external_fn(builtin_join),
    # Conversion
    "to_str": external_fn(builtin_to_str),
    "to_int": external_fn(builtin_to_int),
    # Utility/functional
    "print": external_fn(builtin_print),
    "id": external_fn(builtin_identity),
    "compose": external_fn(builtin_compose),
}
