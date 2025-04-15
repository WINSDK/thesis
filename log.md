# Thesis: Optimized Lambda calculus proof checker

## Terminology

Eta-reduction or η-reduction is the process of converting between λx.(f x)
whenever x is not free in f. That is f = λx.(f x).

### Point-free
Point-free programming is a programming paradigm in which a
function definition does not include information regarding its arguments, using
combinators and function composition `[...]` instead of variables.

```haskell
-- Instead of
sum (x:xs) = x + (sum xs)
sum [] = 0

-- We do
sum = foldr (+) 0
```

This is related to *eta-reduction* (along with some other techniques) in that
we can rewrite functions in a point-free form.

In functional languages based on lambda calculus, all functions take exactly one argument!

## Reference's

* [Drawing lambda terms.](https://cs.gmu.edu/~marks/463/slides/1.lambda_calculus/drawingtrees.html)
* [Fun example of an interactive calculus.](https://treecalcul.us)
* [Deepseek inference.](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)

## Eta reduction

It appears you can use type derivations to find the minimal expression
you can eta-reduce to.
                           η
Determining that λx.(f x) ---> f
```
x : A   f : A -> B
------------------
     f x : B
------------------
λx.(f x) : A → B
```

## Normal form

This terminology is not fully expressive. There's three kinds of normal form:
beta normal, beta-eta normal and head normal form. When talking about normal form
we usually refer to beta normal form but this is not the simplest possible form.
Beta-eta normal form.

An expression is in beta normal form if it is one of the following:
* A data constructor applied to arguments which are in normal form
* A lambda abstraction whose body is in normal form

Head normal form:
* A data constructor applied to some arguments (which may be any expressions at all)
* A lambda abstraction whose body is in head normal form (or not if weak-head normal form)

Read that weak-head normal form is the most commonly used in real-world compilers. 

# Index

### Naive notes

1. Phase 1 just get it to generate valid lambda syntax
2. Randomly (maybe not so random?) generate lambda expressions with input / output pairs
3. Maximize accuracy, minimize length

* https://arxiv.org/abs/2403.06988
* https://github.com/huggingface/trl
* https://arxiv.org/pdf/2403.03997
* https://huggingface.co/docs/trl/index
* https://faustocarcassi.com/arc-course/labs/lab3.html

### Plan

* Instead of single inference, try cons@64 (using beam search).
* Maybe second pass variable renaming for readability?
* Qwen2.5-14B-Instruct-bnb-4bit base seems like the SOTA in our case.
* https://huggingface.co/agentica-org/DeepCoder-14B-Preview
* Real SOTA might be R1 Qwen distilled but that requires CoT.
* TRY CHAIN OF THOUGHT MODEL.
* Generate hundred's of thousands of synthetic examples  
* What the hell, generate outwards the redex's?
* Neurosymbolic?
* How do I get into AI research

* Teach model what functions it has
* Make it generate new functions during training and use those too

* Add an index for the `readme.md` with a description of all the *.py files.

### What even is program synthesis

We have a library of primitives L = {f1 , f2 , . . .}, that
forms a DSL. We specifies tuples of input-output pairs that maps all inputs.
However many such solutions that will not necessarily generalize.

### Set of primitives types

* int
* float
* char (maybe)
* string
* bool
* tuple
* list
* array (maybe)
* unit

### RL

I imagine starting with shorter easier functions and incrementally
increasing the complexity as it learns.

Rewards:
* Type
* Syntax
* Valid Mapping
* Fully simplified (might not be the right approach)
* How close it is to the correct function? (probably a bad idea)
* Length (conditional on correctness)

### RAG

Get the model to output a <query-rag>function to reverse list</query-rag>
special token sequence. You can do this by augmenting the training data
for finetuning.

### Tokenizer

* Micro optimization: Capitalizing identifiers might help as these are more common 

### How do we even get this synthetic data?

Top-Down Synthesis for Library Learning (https://arxiv.org/abs/2211.16605):
*

#### Augmentation

We will create a large sample of strategies for augmenting the input
for a given function.

This will be done for:
* Providing more training data during finetuning
* Helping the model during inference by letting it try on a more diverse
  set of input's. 

# journal

### Day 1: Research into CFG-constrained output

Hugging Face Transformers-CFG seems like the simplest to use implementation.
**Guidance** is definitely the right tool for CFG restraining.

Approaches to look into and read:
* Bidirectional searching
* Constraint solving techniques.

### Day 2: Research into synthesis (initial)

Myth (https://github.com/silky/myth) (https://www.cis.upenn.edu/~stevez/papers/OZ15.pdf):
* An ML program synthesizer

Smyth (https://arxiv.org/pdf/1911.00583):
* All about doing program synthesis of partially written functions
* Has this notion of bidirectional evaluation
* Bidirectional evaluation

Scrybe (https://arxiv.org/pdf/2210.13873)
*

Guiding Enumerative Program Synthesis with Large Language Models

### Day 3: Types, specifically refinement types

Bidirectional synthesis (https://dl.acm.org/doi/pdf/10.1145/2737924.2738007):

Type-Driving program synthesis (https://people.csail.mit.edu/polikarn/publications/pldi16.pdf)
                               (https://github.com/nadia-polikarpova/synquid):
* Refinement types to constrain search space, these are also called liquid types?
* Bidirectional synthesis

Lambdabeam (https://arxiv.org/pdf/2306.02049):
* 

Abstractbeam (https://arxiv.org/pdf/2405.17514):
* Improvement on top of lambdabeam

MCTSr: mathematics as a blackbox

Winner of arc challenge (https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf):
* Augments (transforms) input several times
  - On input that is send to finetune the initial model (with info from public test data)
  - On input that is send to finetune the final model  (with info from private test data)
  - On output from inference (with info from private test data)
  - On all candidates before scoring
* Worked very well on (limit was 16GB inference + training):
  - Mistral-NeMo-Minitron-8B-Base
  - Uncensored Llama-3.2-3B-instruct

### Day 4: More random research

LILO (https://arxiv.org/pdf/2310.19791):
1. Uses a combination of LLM-guided search and Enumerative
2. Compresses (refactors) the output using stitch
3. Documents the now refactored intermediate functions
4 (pt1). Rewrite original (before compression) program using these functions
4 (pt2). Makes the model learn about these new functions?
* Important info
  * A.1 LLM SOLVER PROMPT
  * REGEX library

### Day 5

* Distill chatgpt - this is a good way to get the model going

#### Workflow (step by step)

Compile a set of primitive functions that are passed to `eval`.

Model A (S1):
```
<prompt>
  [3, 9, 1, 5] -> [9, 5, 3, 1]
  [1, 3] -> [3, 1]
  [30, 10, 20] -> [30, 20, 10]
<prompt/>
<response>
  λx. reverse(sort(x))
<response/>
```

* Execute eval(response, { here_goes_allowed_funcs })
* Exception: sort does not exist

Model B (S2):
```
<prompt>
  <context>
  λx. reverse(sort(x)) :: List A -> List A
  <context/>
  You are a lambda calculus wizard, living in the world of lambdas.
  We have an incomplete expression with a missing the defintion of `sort`.
  Generate an expression for the function `sort`:
<prompt/>
<response>
  sort = λl. (isNil l) l (insert (head l) (sort (tail l)))
<response/>
```

* Execute eval(response, { here_goes_allowed_funcs })
* Exception: insert does not exist

Model B (S3):
```
<prompt>
  <context>
  λl. (isNil l) l (insert (head l) (sort (tail l))) :: List A -> List A
  <context/>
  You are a lambda calculus wizard, living in the world of lambdas.
  We have an incomplete expression with a missing the defintion of `insert`.
  Generate an expression for the function `insert`:
<prompt/>
<response>
  insert = λn. λl. (isNil l)
                   (cons n nil)
                   ((leq? n (head l))
                     (cons n l)
                     (cons (head l) (insert n (tail l))))
<response/>
```
* Execute eval(response, { here_goes_allowed_funcs })
* Success (recurse back to previous step)

... Keep executing until we have a complete initial expression

An interesting issue: we might just end up *stuck*. This happens when we
encounter a cyclical dependency. The chain `odd -> Lx.not (even x) -> Lx.not (not (odd x))`
is an example of such a dependency. To prevent this, we need to maintain
a record of functions being generated during recursion and exclude these
from subsequent sampling.

We probably also want to annotate these functions with extra context. A simple
solution is generating docstrings for each generated sub-expression. A more
complicated but probably necessary addition relies on computing the type of each
expression.
