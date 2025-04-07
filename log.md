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

prereq:
* Find libraries that do constrained generation for cfg
* Find existing papers that do what we want to do
* Ask access for A100 server (find who to ask for the server too, straight away)

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
* Real SOTA might be R1 Qwen distilled but that requires CoT.
* Generate hundred's of thousands of synthetic examples  
* What the hell, generate outwards the redex's?
* Neurosymbolic?
* How do I get into AI research

* Teach model what functions it has
* Make it generate new functions during training and use those too

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
* How close it is to the correct function?

### RAG

Get the model to output a <query-rag>function to reverse list</query-rag>
special token sequence. You can do this by augmenting the training data
for finetuning.

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
* Not yet read

Guiding Enumerative Program Synthesis with Large Language Models

### Day 3: Types, specifically refinement types

Bidirectional synthesis (https://dl.acm.org/doi/pdf/10.1145/2737924.2738007):

Type-Driving program synthesis (https://people.csail.mit.edu/polikarn/publications/pldi16.pdf)
                               (https://github.com/nadia-polikarpova/synquid):
* Refinement types to constrain search space, these are also called liquid types?
* Bidirectional synthesis

Lambdabeam (https://arxiv.org/pdf/2306.02049):
* 

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

### Day 4: ???

LILO (https://arxiv.org/pdf/2310.19791):
1. Uses a combination of LLM-guided search and Enumerative
2. Compresses (refactors) the output using stitch
3. Documents the now refactored intermediate functions
4 (pt1). Rewrite original (before compression) program using these functions
4 (pt2). Makes the model learn about these new functions?
