from datasets import Dataset
import synthwave
from synthwave import IS_MACOS


def parse_args():
    import argparse
    # Create args object with the same parameters as the CLI script
    defaults = argparse.Namespace(
        # Model options
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        # LoRA options
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        # Training options
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        steps_per_save=10,
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
    return parser.parse_args()

def train_model():
    print("Loading pretrained model. This may take a while...")
    model, tokenizer = synthwave.load_model(ARGS.model_name)
    print("Model loaded")

    # Configure PEFT model
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
    {}

### Input:
    {}

### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
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
    if IS_MACOS:
        from synthwave import mlx
        from synthwave.mlx import Trainer, TrainingArgs

        model = mlx.get_peft_model(
            model=model,
            r=ARGS.r,
            lora_alpha=ARGS.lora_alpha,
            lora_dropout=ARGS.lora_dropout,
            random_state=ARGS.seed,
            use_rslora=ARGS.use_rslora,
        )
        mlx.train(Trainer(
            model=model,
            tokenizer=tokenizer,
            args=TrainingArgs(
                batch_size=ARGS.per_device_train_batch_size,
                iters=ARGS.max_steps,
                val_batches=25,
                steps_per_report=ARGS.logging_steps,
                steps_per_eval=200,
                steps_per_save=ARGS.steps_per_save,
                adapter_file="adapters.safetensors",
                max_seq_length=ARGS.max_seq_length,
                grad_checkpoint=ARGS.use_gradient_checkpointing,
            ),
            lr_scheduler_type=ARGS.lr_scheduler_type,
            learning_rate=ARGS.learning_rate,
            warmup_steps=ARGS.warmup_steps,
            max_steps=ARGS.max_steps,
            weight_decay=ARGS.weight_decay,
            optim=ARGS.optim,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
        ))
    else:
        raise NotImplementedError

def main():
    if ARGS.repl:
        from synthwave import parse, evaluate, infer
        while True:
            s = input("> ").strip()
            if len(s) == 0:
                continue
            try:
                expr = parse(s)
                print(f"  => {expr}")
                result = evaluate(expr)
                result_ty = infer(result)
                # Pretty print Char List
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], str):
                        result = '"' + "".join(result) + '"'
                print(f"    => {result} :: {result_ty}")
            except Exception as e:
                print(f"\033[91m{repr(e)}\033[0m")
    else:
        train_model()


if __name__ == "__main__":
    ARGS = parse_args()
    main()
