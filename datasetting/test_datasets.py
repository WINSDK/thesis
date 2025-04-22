import datasetting
from datasetting import Prompt
from datasets import Dataset
from synthwave import parse, pretty_print
from synthwave.eval import KNOWN_VARS

def correctness_check(dataset: Dataset):
    for prompt, answer in zip(dataset["prompt"], dataset["answer"]):
        prompt = Prompt.of_dataset(prompt)
        func = parse(answer, known=KNOWN_VARS)
        for input, output in prompt.examples:
            input = parse(input)
            try:
                solution = pretty_print(func(input))
            except Exception as e:
                raise ValueError(answer) from e
            assert solution == output, f"{answer} does not map"

def test_handcrafted():
    correctness_check(datasetting.handcrafted)
