import regex
import random
import os
import string

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
                    if not match.partial and term.name not in ["INTEGER"]:
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

    prompt = f"""\
[3, 9, 1, 5] -> [9, 5, 3, 1]
[1, 3] -> [3, 1]
[30, 10, 20] -> [30, 20, 10]\
{tokenizer.eos_token}\
"""

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

def train():
    from trl import GRPOConfig, GRPOTrainer
    from synthwave.torch import get_peft_model
    from synthwave import MODEL_DIR
    import datasetting
    import rewards

    print("Loading pretrained model. This may take a while...")
    model, tokenizer = synthwave.load_model(ARGS.model_name)
    print("Model loaded")

    dataset = datasetting.transform("handcrafted", datasetting.handcrafted)

    # Split into train/test with appropriate size for small dataset
    # datasets = dataset.train_test_split(test_size=0.33)
    # print(
    #     f"Training examples: {len(datasets['train'])}, Test examples: {len(datasets['test'])}"
    # )

    # Train model
    print("Starting training")
    model = get_peft_model(
        model,
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
        reward_funcs=[rewards.valid_syntax, rewards.diverse_output],
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
        train_dataset=dataset,
        # eval_dataset=datasets["test"],
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()


def weighted_choice(arr, alpha_weight=10):
    from synthwave.parser import IDENT, DIGIT, BOOLEAN, LAMBDA, DOT, WS, Modi
    non_terminal = random.choice(arr)
    if non_terminal.modi == Modi.Many:
        n = random.random() % 5 + 1
        print(non_terminal.chars)
        random.choices(non_terminal.chars, k=n)

    # priority = string.ascii_letters + string.digits + "(" + ")" + "]" + "[" + "lambda" + "." + " "
    # weights = [
    #     alpha_weight if item in priority else 1
    #     for item in arr
    # ]
    return random.choices(arr, weights=weights, k=1)[0]

def main():
    from synthwave.eval import KNOWN_VARS
    from synthwave.parser import parse_inc2

    # for _ in range(10):
    #     seq = ""
    #     while True:
    #         parsed, expect = parse_inc2(seq, known=KNOWN_VARS, strict=True)
    #         if parsed:
    #             if len(seq) < 10:
    #                 seq = ""
    #                 continue
    #             print(seq)
    #             print(parsed)
    #             break
    #         seq += weighted_choice(expect)

    # print(parse_inc2("", known=KNOWN_VARS))
    print(parse_inc2("add ", known=KNOWN_VARS))
    # print(parse_inc2("add 1 ", known=KNOWN_VARS))
    # print(parse_inc2("add 1 2 ", known=KNOWN_VARS))
    # print(parse_inc2("map [", known=KNOWN_VARS))
    # print(parse_inc2("if ", known=KNOWN_VARS))
    # print(parse_inc2("(lambda ABC x.1)", known=KNOWN_VARS))
    exit(1)

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
