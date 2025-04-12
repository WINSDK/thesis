from collections.abc import Callable
from typing import Dict, Optional
from .dsl import UOp, Ops, ExternalError
from .helpers import fn_parameters

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
        return evaluate(var, env) if isinstance(var, UOp) else var
    match expr.op:
        case Ops.Val:
            return expr.args[0]
        case Ops.Closure:
            return expr
        case Ops.Var:
            return lookup(expr.args[0])
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
                    params = ", ".join(map(str, pv.values()))
                    raise ExternalError(f"{body.__name__}({params}): {e}")
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

def external_fn(f):
    params = fn_parameters(f)
    return UOp(Ops.External, [*params, f])

def builtin_add(x: int, y: int) -> int:
    return x + y

def builtin_mul(x: int, y: int) -> int:
    return x * y

def builtin_map(f: Callable[[int], int], lst: list[int]) -> list[int]:
    return [f(x) for x in lst]

def builtin_cons(x: int, xs: list[int]) -> list[int]:
    return [x] + xs

def builtin_head(xs: list[int]) -> int:
    return xs[0]

def builtin_tail(xs: list[int]) -> list[int]:
    return xs[1:]

def builtin_append(xs: list[int], x: int) -> list[int]:
    return xs + [x]

def builtin_reverse(lst: list[int]) -> list[int]:
    return list(reversed(lst))

def builtin_sort(lst: list[int]) -> list[int]:
    return sorted(lst)

def builtin_leq(a: int, b: int) -> bool:
    return a <= b

BUILTINS = {
    "add": external_fn(builtin_add),
    "mul": external_fn(builtin_mul),
    "map": external_fn(builtin_map),
    "cons": external_fn(builtin_cons),
    "head": external_fn(builtin_head),
    "tail": external_fn(builtin_tail),
    "append": external_fn(builtin_append),
    "reverse": external_fn(builtin_reverse),
    "sort": external_fn(builtin_sort),
    "leq?": external_fn(builtin_leq),
}
