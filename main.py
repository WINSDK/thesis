import regex
import os
from rewards import match_format_exactly

def parse_args():
    import argparse
    # Create args object with the same parameters as the CLI script
    defaults = argparse.Namespace(
        # Model options
        model_name="unsloth/Llama-3.2-3B-Instruct",
        # Training options
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        max_steps=1000,
        learning_rate=3e-4,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="linear",
        seed=3407,
        steps_per_save=40,
        home=os.getcwd()
    )
    parser = argparse.ArgumentParser()
    for k, v in vars(defaults).items():
        ty = type(v)
        if isinstance(v, bool):
            def s2b(v):
                if v.lower() == "true":
                    return True
                elif v.lower() == "false":
                    return False
                else:
                    raise argparse.ArgumentTypeError("Expected a boolean value (true/false)")
            ty = s2b
        parser.add_argument(f"--{k}", type=ty, nargs='?', default=v)
    parser.add_argument("--repl", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")
    return parser.parse_args()

PROMPT = """
# Constrains

* `String`'s are represented as `List Char`'s.
* Support for partial applition exists.
* You are allowed to provide answers in a point-free form.

# Functions availabe in the language

### Arithmetic primitives

add :: T -> T -> T
+ :: T -> T -> T
mul :: T -> T -> T
* :: T -> T -> T
sub :: T -> T -> T
- :: T -> T -> T
div :: T -> T -> T
/ :: T -> T -> T
mod :: T -> T -> T
% :: T -> T -> T
pow :: T -> T -> T
** :: T -> T -> T

### Comparisons

if :: T -> A -> B
eq :: T -> T -> Bool
neq :: T -> T -> Bool
gt :: T -> T -> Bool
lt :: T -> T -> Bool
geq :: T -> T -> Bool
leq :: T -> T -> Bool

### Boolean operators

True :: Bool
False :: Bool
not :: Bool -> Bool
and :: Bool -> Bool -> Bool
or :: Bool -> Bool -> Bool

### List utilities

nil :: List T
is_nil :: List T -> Bool
lfold :: List A -> B -> (B -> A -> A) -> B == lfold lst acc (lambda acc x. ...) 
rfold :: List A -> (A -> B -> B) -> B -> B == lfold lst (lambda x acc. ...) acc
map :: List A -> (A -> B) -> List B
filter :: List A -> (A -> Bool) -> List A
zip :: List T -> List T -> List (List T)
length :: List A -> Int
range :: Int -> Int -> List Int
cons :: T -> List T -> List T
head :: List T -> T
tail :: List T -> List T
append :: List T -> T -> List T
reverse :: List T -> List T
sort :: List T -> List T
flatten :: List List T -> List T

### String manipulation

concat :: String -> String -> String
substr :: String -> Int -> Int -> String
split :: String -> String -> List String
join :: List String -> String -> String

### Conversion

show :: T -> String
read :: String -> Int
ord  :: Char -> Int
chr  :: Int -> Char

### Utility/functional

id :: A -> A
compose :: (B -> C) -> (A -> B) -> A -> C

# Task

I will begin by giving you several examples of input-output pairs.
You will then be given a new input, and you must provide the corresponding output.

Input 1:
[1, 2, 3] -> [2, 4, 6]
[] -> []
[9, 9] -> [18, 18]
Output 1:
lambda xs.map xs (* 2)

Input 2:
[1, 2, 3, 6] -> [2, 6]
[9, 8] -> [8]
[1, 2, 3, 4, 5, 6] -> [2, 4, 6]
Output 2:
lambda xs.filter xs (lambda x.eq (mod x 2) 0)

Input 3:
{}
Output 3:
"""

# Start with the assumption that no tokens are allowed.
# 
def allowed_tokens(tokenizer, inputs_len):
    from synthwave.eval import KNOWN_VARS

    term_to_ids = {}
    v = tokenizer.get_vocab()
    def compute_allowed(terms):
        keep = set()
        for term in terms:
            if term in term_to_ids:
                ids = term_to_ids[term]
                keep.update(ids)
                continue
            if term.name == "$END":
                ids = [tokenizer.eos_token_id]
            elif term.name == "LAMBDA":
                ids = tokenizer.encode("lambda", add_special_tokens=False)
            elif term.name == "BOOL":
                ids = tokenizer.encode("TrueFalse", add_special_tokens=False)
            elif term.name == "WS":
                ids = tokenizer.encode(" ", add_special_tokens=False)
            elif term.name == "IDENT1": # Ident for atom's
                # Restrict identifiers to known variables
                ids = []
                for var in KNOWN_VARS:
                    tokens = tokenizer.encode(var, add_special_tokens=False)
                    ids.extend(tokens)
            else:
                # Compute all matching tokens
                regexpr = regex.compile(term.pattern.to_regexp())
                # 1. Take all the matches, for each
                # 2. Remove 1 char of the token's suffix
                # 3. Check if it still matches the regex
                # 4. If it does, remove original token from the vocab
                matching = {}
                for token, id in v.items():
                    match = regexpr.fullmatch(token, partial=True)
                    if not match:
                        continue
                    # Exact matches are preferred sometimes
                    if not match.partial and term.name not in ["SIGNED_INT"]:
                        matching[token] = id
                        continue
                    # Find maybe a shorter token shortest match
                    while len(token) > 1:
                        if regexpr.fullmatch(token[:-1], partial=True):
                            token = token[:-1]
                    # We might of already seen a matching token
                    if token in matching:
                        # Take the smallest id I guess
                        id = min(matching[token], id)
                    matching[token] = id
                ids = list(matching.values())
            term_to_ids[term] = ids
            keep.update(ids)
        return list(sorted(keep))
    def prefix_allowed_tokens_fn(_, input_ids):
        input = input_ids[inputs_len:]
        text = tokenizer.decode(input, skip_special_tokens=True)
        terms = parse_inc(text, known=KNOWN_VARS)
        x = compute_allowed(terms)
        if len(input) == 0:
            # Add begin <|begin_of_text|> token
            x.extend(tokenizer.encode(""))
        print(f"\rTokenizer restricted to {len(x):02} tokens: {text}", end='')
        print([t.name for t in terms], end='')
        return x
    return prefix_allowed_tokens_fn

def inference():
    print("Loading pretrained model. This may take a while...")
    model, tokenizer = synthwave.load_model(ARGS.model_name)
    print("Model loaded")

    # assert tokenizer.is_fast
    # tok_json = json.loads(tokenizer._tokenizer.to_str())
    # print(tok_json)
    # assert tok_json['model']['type'] == "BPE"
    # Keep all single_length tokens
    # indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    # print(indices)

    example = """\
[3, 9, 1, 5] -> [9, 5, 3, 1]
[1, 3] -> [3, 1]
[30, 10, 20] -> [30, 20, 10]
    """

    prompt = PROMPT.format(example) + tokenizer.eos_token

    print("Starting inference")
    with torch.no_grad():
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[-1]
        inputs.pop("token_type_ids", None)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            prefix_allowed_tokens_fn=allowed_tokens(tokenizer, input_len),
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            temperature = 1.0, top_p = 0.95, top_k = 64,
        )
        sequence = outputs['sequences'][0, input_len:]
        print("\nResponse:")
        print(tokenizer.decode(sequence))

def check_accuracy(x, indices, functions):
    function = functions[indices]
    accuracy = 0
    for inp, out in x['examples']:
        if out == function(inp):
            accuracy += 1/len(x['examples'])
    return {'accuracy': accuracy}

def train():
    from trl import GRPOConfig, GRPOTrainer
    from synthwave.torch import get_peft_model
    from synthwave import MODEL_DIR

    print("Loading pretrained model. This may take a while...")
    model, tokenizer = synthwave.load_model(ARGS.model_name)
    print("Model loaded")
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # Create a super basic test dataset with just a few examples
    basic_data = {
        "instruction": [
            "Summarize the following text",
            "Translate this to French",
            "Explain this concept",
            "Write a poem about",
            "List five advantages of",
            "Provide examples of",
        ],
        "input": [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world",
            "Machine learning is a subset of artificial intelligence",
            "autumn leaves falling",
            "renewable energy",
            "good leadership qualities",
        ],
        "output": [
            "A fox quickly jumps over a dog.",
            "Bonjour le monde",
            "Machine learning is an AI approach where systems learn patterns from data",
            "Golden leaves drift down\nDancing in the autumn breeze\nNature's last hurrah",
            "Renewable energy is sustainable, reduces pollution, creates jobs, promotes energy independence, and has lower operating costs.",
            "Good leaders demonstrate empathy, clear communication, decisiveness, integrity, and the ability to inspire others.",
        ],
    }

    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_dict(basic_data)
    print("Dataset initialized")

    # Use the same formatting function as before
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("Data is formatted and ready!")


    # Split into train/test with appropriate size for small dataset
    datasets = dataset.train_test_split(test_size=0.33)
    print(
        f"Training examples: {len(datasets['train'])}, Test examples: {len(datasets['test'])}"
    )

    # Train model
    print("Starting training")
    model = get_peft_model(
        model,
        load_in_4bit=True, 
        fast_inference=True, 
        r=256,
        lora_alpha=24,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        use_rslora=True,
        loftq_config=None,
        random_state=ARGS.seed,
    )
    FastLanguageModel.for_training(model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[],
        args=GRPOConfig(
            output_dir=str(MODEL_DIR / "finedtuned"),
            use_vllm = True, 
            learning_rate=ARGS.learning_rate,
            weight_decay=ARGS.weight_decay,
            warmup_ratio=0.25,
            warmup_steps=ARGS.warmup_steps,
            lr_scheduler_type=ARGS.lr_scheduler_type,
            optim=ARGS.optim,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=ARGS.per_device_train_batch_size,
            gradient_accumulation_steps=ARGS.gradient_accumulation_steps, 
            # Decrease if out of memory
            num_generations=8, 
            max_prompt_length=256,
            max_completion_length=64,
            # Set to 1 for a full training run
            num_train_epochs=1, 
            max_steps=ARGS.max_steps,
            save_steps=ARGS.steps_per_save,
            max_grad_norm=0.1,
            # Can use Weights & Biases
            report_to="none",
        ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()


def main():
    if ARGS.repl:
        from synthwave import parse, evaluate, infer, pretty_print
        from synthwave.eval import KNOWN_VARS
        while True:
            s = input("> ").strip()
            if len(s) == 0:
                continue
            try:
                result = evaluate(parse(s, known=KNOWN_VARS))
                result_ty = infer(result)
                print(f"  => {pretty_print(result)} :: {result_ty}")
            except Exception as e:
                print(f"\033[91m{repr(e)}\033[0m")
    elif ARGS.train:
        train()
    elif ARGS.inference:
        inference()


if __name__ == "__main__":
    ARGS = parse_args()
    os.chdir(ARGS.home)
    # Must come first
    if ARGS.train:
        from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)
    # Later
    from synthwave.parser import parse_inc
    import synthwave
    import torch
    main()
