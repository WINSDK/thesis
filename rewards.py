import re
import synthwave

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

match_format.search(
    "<start_working_out>Let me think!<end_working_out>"\
    "<SOLUTION>2</SOLUTION>",
)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores

def valid_syntax(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        try:
            synthwave.parse(response)
            score = 3.0
        except Exception:
            score = 0.0
        scores.append(score)
    return scores

def infers_type_correctly(completions, **kwargs):
    raise NotImplementedError()
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        try:
            synthwave.parse(response)
            score += 3.0
        except Exception:
            pass
        scores.append(score)
    return scores
