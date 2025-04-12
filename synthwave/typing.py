from dataclasses import dataclass
from collections.abc import Callable
from typing import Union, Dict, Tuple
import typing
from .helpers import fn_parameters
from .eval import BUILTINS
from .dsl import UOp, Ops


@dataclass
class TInt:
    def __repr__(self):
        return "Int"

@dataclass
class TBool:
    def __repr__(self):
        return "Bool"

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

    @staticmethod
    def of_list(params_ty: list["Type"], body_ty: "Type"):
        """Rfold for function types param1 -> param2 -> ... -> body"""
        func_ty = body_ty
        for pt in reversed(params_ty):
            func_ty = TArrow(pt, func_ty)
        return func_ty

@dataclass(frozen=True)
class TVar:
    name: str

    def __repr__(self):
        return self.name

Type = Union[TInt, TBool, TList, TVar, TArrow]
PRIMITIVES = (TInt, TBool, TList)

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

def occurs_in_type(tv: TVar, ty: Type, subst: Dict[TVar, Type]) -> bool:
    ty = apply_subst(ty, subst)
    if ty == tv:
        return True
    elif isinstance(ty, TArrow):
        return occurs_in_type(tv, ty.param, subst) or \
               occurs_in_type(tv, ty.result, subst)
    else:
        return False

def var_bind(tv: TVar, ty: Type, subst: Dict[TVar, Type]) -> Dict[TVar, Type]:
    """Bind the type var 'tv' to type 'ty' if allowed,"""
    if tv == ty:
        return subst
    # Cannot bind a type variable to a type that contains it
    if occurs_in_type(tv, ty, subst):
        raise TypeError(f"Occurs check fails: {tv} in {ty}")
    # Override the subst
    new_subst = subst.copy()
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
    # This must be manually modified for each new base type
    elif any(isinstance(t1, t) and isinstance(t2, t) for t in PRIMITIVES):
        return subst
    elif isinstance(t1, TArrow) and isinstance(t2, TArrow):
        # unify param, then unify result
        subst2 = unify(t1.param, t2.param, subst)
        return unify(t1.result, t2.result, subst2)
    else:
        raise TypeError(f"Cannot unify {t1} with {t2}")

def fresh_type_var(prefix="a", counter=[0]) -> TVar:
    name = f"{prefix}{counter[0]}"
    counter[0] += 1
    return TVar(name)

def infer_py_ty(ty, expr=None) -> Type:
    varname = ty.__name__
    if ty is int:
        return TInt()
    if ty is bool:
        return TBool()
    elif ty is list and expr is not None and len(expr) > 0:
        # Special case for list's, so that we can infer the type of
        # A value such as: Val([1, 2, 3]) => Int List
        return infer_py_ty(list[type(expr[0])])
    elif ty is list or ty is Callable:
        raise TypeError(f"Type {varname} is missing type spec")
    elif typing.get_origin(ty) is list:
        elems = typing.get_args(ty)[0]
        return TList(infer_py_ty(elems))
    elif typing.get_origin(ty) is Callable:
        params_ty, body_ty = typing.get_args(ty)
        params_ty = [infer_py_ty(p) for p in params_ty]
        return TArrow.of_list(params_ty, infer_py_ty(body_ty))
    else:
        raise TypeError(f"Type '{varname}' isn't supported (yet)")

def _infer(expr: UOp, env: Dict[str, Type], subst: Dict[TVar, Type]) -> Tuple[Type, Dict[TVar, Type]]:
    def lookup(varname):
        if varname in env:
            # The type in env might be partially substituted
            return (apply_subst(env[varname], subst), subst)
        elif varname in BUILTINS:
            # Built-in functions are always external and therefore
            # don't require substitution
            return _infer(BUILTINS[varname], env, subst)
        else:
            raise NameError(f"Unbound variable: {varname}")
    match expr.op:
        case Ops.Val:
            ty = type(expr.args[0])
            return (infer_py_ty(ty, expr.args[0]), subst)
        case Ops.Var:
            return lookup(expr.args[0])
        case Ops.Closure:
            *args, body, _ = expr.args
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
            (body_ty, s1) = _infer(body, env, subst)
            ty = TArrow.of_list(params_ty, body_ty)
            return (ty, s1)
        case Ops.Appl:
            func, *args = expr.args
            # 1. Infer the type of func
            (func_ty, s1) = _infer(func, env, subst)
            # 2. For each argument, unify the function type with TArrow(...)
            #    and produce the resulting function type for next arg
            cur_subst = s1
            cur_func_ty = func_ty
            for arg in args:
                # The function must be an arrow: param_type -> result_type
                param_ty = fresh_type_var()
                result_ty = fresh_type_var()
                cur_subst = unify(cur_func_ty, TArrow(param_ty, result_ty), cur_subst)
                # Infer arg type
                (arg_ty, cur_subst) = _infer(arg, env, cur_subst)
                # Unify arg type with param
                cur_subst = unify(arg_ty, param_ty, cur_subst)
                cur_func_ty = result_ty
            # After applying all arguments, cur_func_ty is the final type of the application
            return (apply_subst(cur_func_ty, cur_subst), cur_subst)
        case Ops.External:
            *params, body = expr.args
            params_ty = body.__annotations__
            assert "return" in params_ty, \
                  f"Function '{body.__name__}' missing return type annotation"
            assert len(fn_parameters(body)) + 1 == len(params_ty), \
                  f"Function '{body.__name__}' missing parameter annotations"
            body_ty = infer_py_ty(params_ty["return"])
            params_ty = [infer_py_ty(params_ty[p]) for p in params]
            ty = TArrow.of_list(params_ty, body_ty)
            return (ty, subst)

def infer(expr) -> Type:
    if isinstance(expr, UOp):
        ty, s = _infer(expr, {}, {})
        return apply_subst(ty, s)
    else:
        return infer_py_ty(type(expr), expr)
