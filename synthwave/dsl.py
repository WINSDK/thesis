from typing import Any, Dict, Optional
from enum import auto, Enum, IntEnum
from dataclasses import dataclass
import inspect

class External(Exception):
    pass

class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)

class Ops(FastEnum):
    Val   = auto()
    Var   = auto()  
    # Application (M N): [param1, param2, .., body]
    Appl = auto()
    # Expression (λx.M): [func, arg1, arg2, ..]
    Abstr = auto() 
    # Context-specific functions: [params, body, env]
    Closure = auto()
    # Regular old python functions: [python_fn, arg1, arg2, ..]
    External = auto() 

@dataclass(eq=False, slots=True)
class UOp():
    op: Ops
    args: list[Any]
    name: str = ""

    def __repr__(self):
        op = self.name if len(self.name) > 0 else self.op.name
        if isinstance(self.args, list):
            args = ", ".join(map(str, self.args))
            return f"{op}({args})"
        else:
            return f"V({self.args})"
    
    def swap(self, **kwargs):
        new = (kwargs.pop("op", self.op), kwargs.pop("args", self.args), kwargs.pop("name", self.args))
        return UOp(*new)

def external_fn(f):
    fn_args = inspect.signature(f).parameters.keys()
    params = ["anon_" + p for p in fn_args]
    return UOp(
        Ops.Closure,
        [params, UOp(Ops.External, [f] + params), {}]
    )

BUILTINS = {
    "add": external_fn(lambda x, y: x + y),
    "mul": external_fn(lambda x, y: x * y),
}

def evaluate(uop: UOp, env: Optional[Dict[str, UOp]]=None, depth=0):
    assert isinstance(uop, UOp), f"Not an UOp: {uop}"
    if depth == 1000:
        raise RecursionError("Evaluate max_depth=1000")
    if env is None:
        env = {}
    def rec(x, env):
        return evaluate(x, env, depth + 1)
    def lookup(varname):
        if varname in env:
            var = env[varname]
        elif varname in BUILTINS:
            var = BUILTINS[varname]
        else:
            raise NameError(f"Unbound variable: {varname}")
        return rec(var, env) if isinstance(var, UOp) else var
    args = uop.args
    match uop.op:
        case Ops.Val:
            return args[0]
        case Ops.Closure:
            return uop
        case Ops.Var:
            return lookup(args[0])
        case Ops.Abstr:
            params, body = args[:-1], args[-1]
            return UOp(Ops.Closure, [params, body, env])
        case Ops.Appl:
            func, args = args[0], args[1:]
            func = rec(func, env)
            args = [rec(a, env) for a in args]
            while args:
                if not (isinstance(func, UOp) and func.op == Ops.Closure):
                    raise TypeError(f"Cannot call a non-closure: {func}")
                params, body, closure_env = func.args
                def apply(env, params, args):
                    env = env.copy()
                    for p, v in zip(params, args):
                        env[p] = v
                    return env
                if len(args) < len(params):
                    # Not enough arguments, so we do partial application
                    # which will require creating a new closure
                    not_applied = params[len(args):]
                    closure_env = apply(closure_env, params, args)
                    return UOp(Ops.Closure, [not_applied, body, closure_env])
                else:
                    # Full application or over-applied 
                    closure_env = apply(closure_env, params, args[:len(params)])
                    func = rec(body, closure_env)
                    args = args[len(params):]
            return func
        case Ops.External:
            params = {str.lstrip(p, "anon_"):lookup(p) for p in args[1:]}
            try:
                return (args[0])(**params)
            except Exception as e:
                args = ", ".join(map(str, params.values()))
                raise External(f"fn({args})") from e
