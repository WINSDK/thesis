from unsloth import FastLanguageModel

def get_peft_model(model: FastLanguageModel, **kwargs):
    # These options are fixed due to mlx impl limitations.
    kwargs.pop("target_modules", None) 
    kwargs.pop("bias", None) 
    kwargs.pop("loftq_config", None) 
    return FastLanguageModel.get_peft_model(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        **kwargs
    )
