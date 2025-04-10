from typing import Union, Any, Dict, Optional
import typing
from enum import auto, IntEnum
from dataclasses import dataclass
import inspect

class ExternalCall(Exception):
    pass

class TypeErrorUnify(Exception):
    pass

class Ops(IntEnum):
    Val   = auto()
    Var   = auto()  
    # Application (M N): [body, param1, param2, ..]
    Appl = auto()
    # Expression (λx.M): [arg1, arg2, .., body]
    Abstr = auto() 
    # Context-specific functions: [args1, args2, .., body, env]
    Closure = auto()
    # Regular old python functions: [arg1, arg2, .., python_fn]
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
    assert callable(f), "Not a valid python function"
    fn_args = inspect.signature(f).parameters.keys()
    params = ["anon_" + p for p in fn_args]
    return UOp(
        Ops.Closure,
        [*params, UOp(Ops.External, params + [f]), {}]
    )

def builtin_add(x: int, y: int) -> int:
    return x + y

def builtin_mul(x: int, y: int) -> int:
    return x * y

BUILTINS = {
    "add": external_fn(builtin_add),
    "mul": external_fn(builtin_mul),
}

def evaluate(uop: UOp, env: Optional[Dict[str, UOp]]=None, depth=0):
    assert isinstance(uop, UOp), f"Not an expr: {uop}"
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
            return UOp(Ops.Closure, [*args, env])
        case Ops.Appl:
            func, *args = args
            func = rec(func, env)
            args = [rec(a, env) for a in args]
            while args:
                if not (isinstance(func, UOp) and func.op == Ops.Closure):
                    raise TypeError(f"Cannot call a non-closure: {func}")
                *params, body, closure_env = func.args
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
                    return UOp(Ops.Closure, [*not_applied, body, closure_env])
                else:
                    # Full application or over-applied 
                    closure_env = apply(closure_env, params, args[:len(params)])
                    func = rec(body, closure_env)
                    args = args[len(params):]
            return func
        case Ops.External:
            *params, body = args
            pv = {str.lstrip(p, "anon_"):lookup(p) for p in params}
            try:
                return body(**pv)
            except Exception as e:
                params = ", ".join(map(str, pv.values()))
                raise ExternalCall(f"fn({params})") from e

@dataclass
class TInt:
    def __repr__(self):
        return "Int"

@dataclass(frozen=True)
class TList:
    elems: "Type"

    def __repr__(self):
        return f"{str(self.elems)} List"

@dataclass(frozen=True)
class TArrow:
    param: "Type"
    result: "Type"

    def __repr__(self):
        return str(self.param) + " -> " + str(self.result)

@dataclass(frozen=True)
class TVar:
    name: str

    def __repr__(self):
        return self.name

Type = Union[TInt, TVar, TArrow]

def apply_subst(ty: Type, subst: Dict[TVar, Type]) -> Type:
    if isinstance(ty, TVar) and ty in subst:
        return apply_subst(subst[ty], subst)
    elif isinstance(ty, TArrow):
        return TArrow(
            apply_subst(ty.param, subst),
            apply_subst(ty.result, subst)
        )
    else:
        return ty

def compose_subst(s1: Dict[TVar, Type], s2: Dict[TVar, Type]) -> Dict[TVar, Type]:
    """Compose s2 with s1: apply s2 to all of s1's mappings"""
    # First, apply s2 to s1's range
    new_map = { tv: apply_subst(ty, s2) for tv, ty in s1.items() }
    # Then add s2's own bindings
    for tv, ty in s2.items():
        new_map[tv] = ty
    return new_map

def var_bind(tv: TVar, ty: Type, subst: Dict[TVar, Type]) -> Dict[TVar, Type]:
    """Bind the type var 'tv' to type 'ty' if allowed,"""
    if tv == ty:
        return subst
    # Cannot bind a type variable to a type that contains it
    if occurs_in_type(tv, ty, subst):
        raise TypeErrorUnify(f"Occurs check fails: {tv} in {ty}")
    # Override the subst
    new_subst = dict(subst)
    new_subst[tv] = ty
    return new_subst

def unify(t1: Type, t2: Type, subst: Dict[TVar, Type]) -> Dict[TVar, Type]:
    """Unify t1 and t2 under the existing substitution 'subst',"""
    # 1) Apply current subst so we see the "true" forms
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)
    # 2) Pattern-match on t1, t2
    if isinstance(t1, TVar):
        return var_bind(t1, t2, subst)
    elif isinstance(t2, TVar):
        return var_bind(t2, t1, subst)
    elif isinstance(t1, TInt) and isinstance(t2, TInt):
        # Already the same
        return subst
    elif isinstance(t1, TArrow) and isinstance(t2, TArrow):
        # unify param, then unify result
        subst2 = unify(t1.param, t2.param, subst)
        return unify(t1.result, t2.result, subst2)
    else:
        # e.g. TArrow != TInt or mismatch
        raise TypeErrorUnify(f"Cannot unify {t1} with {t2}")

def occurs_in_type(tv: TVar, ty: Type, subst: Dict[TVar, Type]) -> bool:
    ty = apply_subst(ty, subst)
    if ty == tv:
        return True
    elif isinstance(ty, TArrow):
        return occurs_in_type(tv, ty.param, subst) or \
               occurs_in_type(tv, ty.result, subst)
    else:
        return False


def fresh_type_var(prefix="a", counter=[0]) -> TVar:
    name = f"{prefix}{counter[0]}"
    counter[0] += 1
    return TVar(name)

def _infer(expr: UOp, env: Dict[str, Type], subst: Dict[TVar, Type]):
    assert isinstance(expr, UOp), f"Not an expr: {expr}"
    def var_ty(var):
        # This probably requires more checks.
        if var is int:
            return TInt()
        elif typing.get_origin(var) is list:
            return TList(elems=typing.get_args(var)[0])
        else:
            return TVar(var.__name__)
    def lookup(varname):
        if varname in env:
            # The type in env might be partially substituted
            return apply_subst(env[varname], subst)
        elif varname in BUILTINS:
            # Built-in functions are always external and therefore
            # don't require substitution
            return _infer(BUILTINS[varname], env, subst)
        else:
            raise NameError(f"Unbound variable: {varname}")
    match expr.op:
        case Ops.Val:
            py_ty = type(expr.args[0])
            return (var_ty(py_ty), subst)
        case Ops.Var:
            return (lookup(expr.args[0]), subst)
        case Ops.Closure:
            *_, body, _ = expr.args
            return _infer(body, env, subst)
        case Ops.Abstr:
            params, body = expr.args[:-1], expr.args[-1]
            # We create a fresh type variable for each param
            # or you might do something fancy if param type is annotated
            env = env.copy()
            param_types = []
            for p in params:
                tv = fresh_type_var()
                env[p] = tv
                param_types.append(tv)
            # Infer body with those param types in env
            (body_ty, s1) = _infer(body, env, subst)
            # Rfold for function types param1 -> param2 -> ... -> body 
            func_ty = body_ty
            for pt in reversed(param_types):
                func_ty = TArrow(pt, func_ty)
            return (func_ty, s1)
        case Ops.Appl:
            func, *args = expr.args
            # 1. Infer the type of func
            (func_ty, s1) = _infer(func, env, subst)
            # 2. For each argument, unify the function type with TArrow(...)
            #    and produce the resulting function type for next arg
            cur_subst = s1
            cur_func_ty = func_ty
            for arg in args:
                # The function must be an arrow: TArrow(paramType, resultType)
                param_ty = fresh_type_var()
                result_ty = fresh_type_var()
                cur_subst = unify(cur_func_ty, TArrow(param_ty, result_ty), cur_subst)
                # Infer arg type
                (arg_ty, cur_subst) = _infer(arg, env, cur_subst)
                # Unify arg type with param
                cur_subst = unify(arg_ty, param_ty, cur_subst)
                cur_func_ty = result_ty
            # after applying all arguments, cur_func_ty is the final type of the application
            return (apply_subst(cur_func_ty, cur_subst), cur_subst)
        case Ops.External:
            *params, body = expr.args
            param_types = body.__annotations__
            assert len(param_types) == len(params) + 1, \
                   "All external functions must fully typed, including their return type"
            func_ty = var_ty(param_types["return"])
            for pt_idx in reversed(range(len(params))):
                p = str.lstrip(params[pt_idx], "anon_")
                pt = var_ty(param_types[p])
                func_ty = TArrow(pt, func_ty)
            return func_ty

def infer(expr: UOp):
    ty, s = _infer(expr, {}, {})
    return apply_subst(ty, s)
