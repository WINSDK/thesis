from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
from enum import auto, IntEnum
import typing
from .helpers import fn_parameters
from .eval import BUILTINS
from .dsl import UOp, Ops, GENERICS


class T(IntEnum):
    Int = auto()
    Float = auto()
    Char = auto()
    Bool = auto()
    List = auto()
    Arrow = auto()

PRIMITIVES = [t.value for t in T if t.value not in [T.List, T.Arrow]]

class Type():
    t: T
    params: list["Type"] = []

    def __init__(self, t: T, params: Optional[list[Any]] = None):
        self.t = t
        if params is not None:
            self.params = params

    def __repr__(self):
        match self.t:
            case T.Int:
                return "Int"
            case T.Float:
                return "Float"
            case T.Char:
                return "Char"
            case T.Bool:
                return "Bool"
            case T.List:
                return f"{str(self.params[0])} List"
            case T.Arrow:
                return " -> ".join(map(str, self.params))

@dataclass(eq=True, frozen=True)
class TVar(Type):
    name: str
    generic: bool = False

    def __repr__(self):
        s = self.name
        if self.generic:
            # Strip numeric suffix
            while s[-1].isnumeric():
                s = s[:-1]
        return s

Subst = Dict[TVar, Type]

def arrow_of_list(params_ty: list[Type], body_ty: Type) -> Type:
    """Rfold for function types param1 -> param2 -> ... -> body"""
    func_ty = body_ty
    for pt in reversed(params_ty):
        func_ty = Type(T.Arrow, [pt, func_ty])
    return func_ty

def apply_subst(ty: Type, subst: Subst) -> Type:
    if isinstance(ty, TVar):
        if ty in subst:
            return apply_subst(subst[ty], subst)
        return ty
    elif len(ty.params) > 0:
        return Type(ty.t, [apply_subst(t, subst) for t in ty.params])
    else:
        return ty

def occurs_in_type(tv: TVar, ty: Type, subst: Subst) -> bool:
    ty = apply_subst(ty, subst)
    if ty == tv:
        return True
    if isinstance(ty, TVar):
        return False
    elif len(ty.params) > 0:
        return any(occurs_in_type(tv, t, subst) for t in ty.params)
    else:
        return False

def var_bind(tv: TVar, ty: Type, subst: Subst) -> Subst:
    """Bind the type var 'tv' to type 'ty' if allowed"""
    if tv == ty:
        return subst
    # Cannot bind a type variable to a type that contains it
    if occurs_in_type(tv, ty, subst):
        raise TypeError(f"Occurs check fails: {tv} in {ty}")
    # Override the subst
    subst = subst.copy()
    subst[tv] = ty
    return subst

def unify(t1: Type, t2: Type, subst: Subst) -> Subst:
    """Unify t1 and t2 under the existing subst"""
    # 1) Apply current subst so we see the "true" forms
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)
    # 2) Pattern-match on t1, t2
    if isinstance(t1, TVar):
        return var_bind(t1, t2, subst)
    elif isinstance(t2, TVar):
        return var_bind(t2, t1, subst)
    elif any(t1.t == t and t2.t == t for t in PRIMITIVES):
        return subst
    elif t1.t == t2.t:
        # This is fine, so long as types have fixed number of args
        for t1, t2 in zip(t1.params, t2.params):
            subst = unify(t1, t2, subst)
        return subst
    else:
        raise TypeError(f"Cannot unify {t1} with {t2}")

def fresh_type_var(prefix="a", counter=[0]) -> Type:
    name = f"{prefix}{counter[0]}"
    counter[0] += 1
    return TVar(name)

def infer_py_ty(ty, expr=None) -> Type:
    if ty is int:
        return Type(T.Int)
    if ty is bool:
        return Type(T.Bool)
    if ty is float:
        return Type(T.Float)
    if ty is str:
        return Type(T.Char)
    if ty is Any:
        return fresh_type_var()
    elif ty is list and expr is not None and len(expr) > 0:
        # Special case for list's, so that we can infer the type of
        # A value such as: Val([1, 2, 3]) => Int List
        return infer_py_ty(list[type(expr[0])])
    elif typing.get_origin(ty) is list:
        # Infer based on function annotations
        elems = typing.get_args(ty)[0]
        return Type(T.List, [infer_py_ty(elems)])
    elif typing.get_origin(ty) is Callable:
        # Infer based on function annotations
        params_ty, body_ty = typing.get_args(ty)
        params_ty = [infer_py_ty(p) for p in params_ty]
        return arrow_of_list(params_ty, infer_py_ty(body_ty))
    elif ty is list:
        return Type(T.List, [fresh_type_var()])
    elif ty is Callable:
        return Type(T.Arrow, [fresh_type_var(), fresh_type_var()])
    elif ty in GENERICS:
        return TVar(str(ty), generic=True)
    else:
        raise TypeError(f"Type {ty.__name__} isn't supported (yet)")

def _infer(expr: UOp, env: Subst, subst: Subst) -> Tuple[Type, Subst]:
    def lookup(varname):
        if varname in env:
            # The type in env might be partially substituted
            return (apply_subst(env[varname], subst), subst)
        elif varname in BUILTINS:
            # Built-in functions are always external and therefore
            # don't require substitution
            return _infer(BUILTINS[varname], env, subst)
        else:
            raise NameError(f"Unbound variable {varname}")
    match expr.op:
        case Ops.Val:
            ty = type(expr.args[0])
            return (infer_py_ty(ty, expr.args[0]), subst)
        case Ops.Var:
            return lookup(expr.args[0])
        case Ops.Closure:
            *args, body, closure = expr.args
            # Merge captured env into the type inference env
            env = env.copy()
            for k, v in closure.items():
                v_ty, subst = _infer(UOp(Ops.Val, [v]), env, subst)
                env[k] = v_ty
            if isinstance(body, UOp):
                body = UOp(Ops.Abstr, [*args, body])
                return _infer(body, env, subst)
            else:
                body = UOp(Ops.External, [*args, body])
                return _infer(body, env, subst)
        case Ops.Abstr:
            *params, body = expr.args
            # We create a fresh type variable for each param
            # or you might do something fancy if param type is annotated
            env = env.copy()
            params_ty = []
            for p in params:
                tv = fresh_type_var()
                env[p] = tv
                params_ty.append(tv)
            # Infer body with those param types in env
            (body_ty, subst) = _infer(body, env, subst)
            ty = arrow_of_list(params_ty, body_ty)
            return (ty, subst)
        case Ops.External:
            *params, body = expr.args
            params_ty = body.__annotations__
            if "return" not in params_ty:
                params_ty["return"] = fresh_type_var()
                # raise TypeError(f"Missing type information for: {body.__name__}")
            if len(fn_parameters(body)) + 1 != len(params_ty):
                for p in fn_parameters(body):
                    if p not in params_ty:
                        params_ty[p] = fresh_type_var()
                # raise TypeError(f"Missing type information for: {body.__name__}")
            body_ty = infer_py_ty(params_ty["return"])
            params_ty = [infer_py_ty(params_ty[p]) for p in params]
            ty = arrow_of_list(params_ty, body_ty)
            return (ty, subst)
        case Ops.Appl:
            func, *args = expr.args
            # 1. Infer the type of func
            (func_ty, subst) = _infer(func, env, subst)
            for arg in args:
                # The function must be an arrow: param_type -> result_type
                param_ty = fresh_type_var()
                result_ty = fresh_type_var()
                subst = unify(func_ty, Type(T.Arrow, [param_ty, result_ty]), subst)
                # Infer arg type
                (arg_ty, subst) = _infer(arg, env, subst)
                # Unify arg type with param
                subst = unify(arg_ty, param_ty, subst)
                func_ty = result_ty
            # After applying all arguments, func_ty is the final type of the application
            return (func_ty, subst)

def infer(expr) -> Type:
    if isinstance(expr, UOp):
        ty, s = _infer(expr, {}, {})
        return apply_subst(ty, s)
    else:
        return infer_py_ty(type(expr), expr)
