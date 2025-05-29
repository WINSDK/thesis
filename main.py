import os

def parse_args():
    import argparse
    # Create args object with the same parameters as the CLI script
    defaults = argparse.Namespace(
        # Model options
        model_name="unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit",
        # Training options
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        max_steps=5000,
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
    from synthwave.parser import Modi

    inv_vocab = {tid: tok for tok, tid in tokenizer.get_vocab().items()}
    term_to_ids = {}
    trie = {}

    for tid, piece in inv_vocab.items():
        if piece.startswith("<"):                  # skip specials
            continue
        node = trie
        for ch in piece.lstrip(" \t\r\n").rstrip(" \t\r\n"):
            node = node.setdefault(ch, {})
            node.setdefault("_ids", []).append(tid)   # every prefix owns the id

    def ids_for_literal(literal: str) -> list[int]:
        if literal == ' ': # Edge case as we normally ignore whitespace
            return tokenizer.encode(" ", add_special_tokens=False)
        node = trie
        for ch in literal.lstrip(" \t\r\n").rstrip(" \t\r\n"):
            node = node.get(ch)
            if node is None:
                return []
        return node["_ids"]

    def compute_allowed(parsed, non_terms):
        ids = set()

        if parsed is not None:
            ids.add(tokenizer.eos_token_id)

        for term in non_terms:
            if term.name in term_to_ids:
                ids.update(term_to_ids[term.name])
                continue

            current = set()
            if term.modi == Modi.Single:
                for literal in term.tokens:
                    # The exact token(s).
                    current.update(tokenizer.encode(literal, add_special_tokens=False))
                    # Tokens that start with the literal and only add blanks.
                    # current.update(ids_for_literal(literal))
            elif term.modi == Modi.Many:
                for tid, piece in inv_vocab.items():
                    # Skip special tokens (they start with '<')
                    if piece.startswith("<"):
                        continue
                    if all(ch in term.tokens for ch in piece):
                        ids.add(tid)
            else:
                raise NotImplementedError()
            term_to_ids[term.name] = list(current)
            ids.update(current)

        return list(ids)

    def prefix_allowed_tokens_fn(_, input_ids):
        input = input_ids[inputs_len:]
        text = tokenizer.decode(input, skip_special_tokens=True)
        parsed, next = parse_inc2(text, known=KNOWN_VARS)
        x = compute_allowed(parsed, next)
        print(f"prefix: '{text}' restricted: {[t.name for t in next]}")
        print([ tokenizer.decode(y, skip_special_tokens=True) for y in x])
        print("===")
        return x
    return prefix_allowed_tokens_fn

def inference():
    from datasetting import transform_to_messages

    print("Loading pretrained model. This may take a while...")
    model, tokenizer = synthwave.load_model(ARGS.model_name)
    print("Model loaded")

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3",
    )

    # respond literally: lambda x. reverse (sort x)
    messages = transform_to_messages("""
        [3, 9, 1, 5] -> [9, 5, 3, 1]
        [1, 3] -> [3, 1]
        [30, 10, 20] -> [30, 20, 10]
    """.strip())
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, # Must add for generation
        enable_thinking=False,
    )

    print("Starting inference")
    with torch.no_grad():
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[-1]
        inputs.pop("token_type_ids", None)

        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            # prefix_allowed_tokens_fn=allowed_tokens(tokenizer, input_len),
            return_dict_in_generate=True,
            temperature = 1.0, top_p = 0.95, top_k = 64,
        )
        sequence = outputs['sequences'][0, input_len:]
        print("\nResponse:")
        print(tokenizer.decode(sequence, skip_special_tokens=True))

def train():
    from trl import GRPOConfig, GRPOTrainer
    from synthwave.torch import get_peft_model
    from synthwave import MODEL_DIR
    import datasetting
    import rewards

    print("Loading pretrained model. This may take a while...")
    model, tokenizer = synthwave.load_model(ARGS.model_name)
    print("Model loaded")

    train_dataset = datasetting.load_metaset3_dataset("train")
    test_dataset = datasetting.load_metaset3_dataset("test")

    # Train model
    print("Starting training")
    model = get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!
        r=24,
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
        reward_funcs=[
            rewards.match_format_exactly,
            rewards.match_format_approx,
            rewards.valid_lambda_expr,
            rewards.infers_type_correctly,
            rewards.correct,
        ],
        args=GRPOConfig(
            output_dir=str(MODEL_DIR / "finedtuned"),
            # use_vllm=True,
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
            # Logging configuration
            logging_dir=str(MODEL_DIR / "finedtuned" / "logs"),
            logging_strategy="steps",
            logging_steps=10,  # Log every 10 steps
            logging_first_step=True,
            log_completions=True,  # Log the actual completions during training
            report_to="tensorboard",  # Enable TensorBoard logging
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()


def main():
    from synthwave.eval import KNOWN_VARS

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
                raise e
    elif ARGS.train:
        train()
    elif ARGS.inference:
        inference()
    else:
        print("Unknown arg")
        exit(1)


if __name__ == "__main__":
    ARGS = parse_args()
    os.chdir(ARGS.home)

    # import datasetting
    # d = datasetting.load_metaset3_dataset("train")
    # print(d["tests"][100])
    # print(d["args_info"][100])
    # exit(0)

    # Must come first
    if ARGS.train:
        from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)
    # Later
    from unsloth.chat_templates import get_chat_template
    from synthwave.parser import parse_inc2
    import synthwave
    import torch
    main()
