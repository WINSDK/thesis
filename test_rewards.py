import rewards
from datasetting import REASONING_END, SOLUTION_START, SOLUTION_END, EOS


class TestSingleFunction:
    def test_single_function_basic(self):
        def dummy_func(response):
            return len(response)

        completions = [
            [{"content": "hello"}],
            [{"content": "world"}],
            [{"content": "test"}]
        ]

        result = rewards.single(dummy_func, completions)
        assert result == [5, 5, 4]


class TestMatchFormatExactly:
    def test_valid_format(self):
        valid_response = f"Some reasoning {REASONING_END} more text {SOLUTION_START}lambda x.x{SOLUTION_END}"
        completions = [[{"content": valid_response}]]

        result = rewards.match_format_exactly(completions)
        assert result == [1.5]

    def test_valid_format_with_eos(self):
        valid_response = f"Some reasoning {REASONING_END} more text {SOLUTION_START}lambda x.x{SOLUTION_END}{EOS}"
        completions = [[{"content": valid_response}]]

        result = rewards.match_format_exactly(completions)
        assert result == [1.5]

    def test_invalid_format(self):
        invalid_response = "Some text without proper format"
        completions = [[{"content": invalid_response}]]

        result = rewards.match_format_exactly(completions)
        assert result == [0.0]

    def test_missing_solution_tags(self):
        invalid_response = f"Some reasoning {REASONING_END} more text lambda x.x"
        completions = [[{"content": invalid_response}]]

        result = rewards.match_format_exactly(completions)
        assert result == [0.0]


class TestMatchFormatApprox:
    def test_perfect_format(self):
        response = f"text {REASONING_END} text {SOLUTION_START} code {SOLUTION_END} text"
        completions = [[{"content": response}]]

        result = rewards.match_format_approx(completions)
        assert result == [1.5]

    def test_missing_tags(self):
        response = "text without any tags"
        completions = [[{"content": response}]]

        result = rewards.match_format_approx(completions)
        assert result == [-3.0]

    def test_duplicate_tags(self):
        response = f"text {REASONING_END} text {REASONING_END} {SOLUTION_START} code {SOLUTION_END} text"
        completions = [[{"content": response}]]

        result = rewards.match_format_approx(completions)
        assert result == [0.0]  # -1.0 for duplicate REASONING_END, +0.5 each for others

    def test_partial_format(self):
        response = f"text {REASONING_END} text {SOLUTION_START} code"
        completions = [[{"content": response}]]

        result = rewards.match_format_approx(completions)
        assert result == [0.0]  # +0.5 + 0.5 - 1.0


class TestValidLambdaExpr:
    def test_valid_lambda_expr_success(self):
        valid_response = f"{REASONING_END} {SOLUTION_START}lambda x.x{SOLUTION_END}"
        completions = [[{"content": valid_response}]]

        result = rewards.valid_lambda_expr(completions)
        assert result == [3.0]

    def test_invalid_syntax_failure(self):
        invalid_response = f"{REASONING_END} {SOLUTION_START}invalid syntax here{SOLUTION_END}"
        completions = [[{"content": invalid_response}]]

        result = rewards.valid_lambda_expr(completions)
        assert result == [0.0]

    def test_prompt_example_1(self):
        valid_response = f"{REASONING_END} {SOLUTION_START}lambda xs.map xs (* 2){SOLUTION_END}"
        completions = [[{"content": valid_response}]]

        result = rewards.valid_lambda_expr(completions)
        assert result == [3.0]

    def test_prompt_example_2(self):
        valid_response = f"{REASONING_END} {SOLUTION_START}lambda xs.filter xs (lambda x.eq (mod x 2) 0){SOLUTION_END}"
        completions = [[{"content": valid_response}]]

        result = rewards.valid_lambda_expr(completions)
        assert result == [3.0]

    def test_prompt_example_3(self):
        valid_response = f"{REASONING_END} {SOLUTION_START}lambda xs.reverse (sort xs){SOLUTION_END}"
        completions = [[{"content": valid_response}]]

        result = rewards.valid_lambda_expr(completions)
        assert result == [3.0]

    def test_missing_solution_tags(self):
        invalid_response = f"{REASONING_END} lambda x.x"
        completions = [[{"content": invalid_response}]]

        result = rewards.valid_lambda_expr(completions)
        assert result == [0.0]

    def test_syntax_exception_handling(self):
        malformed_response = f"{REASONING_END} {SOLUTION_START}((({SOLUTION_END}"
        completions = [[{"content": malformed_response}]]
        result = rewards.valid_lambda_expr(completions)
        assert result == [0.0]


class TestRegexPattern:
    def test_solution_end_regex_pattern(self):
        import re

        # Test with just SOLUTION_END
        text1 = f"content{SOLUTION_END}"
        assert re.search(rewards.SOLUTION_END_REGEX, text1)

        # Test with SOLUTION_END + EOS
        text2 = f"content{SOLUTION_END}{EOS}"
        assert re.search(rewards.SOLUTION_END_REGEX, text2)

        # Test with SOLUTION_END + whitespace + EOS
        text3 = f"content{SOLUTION_END}   {EOS}"
        assert re.search(rewards.SOLUTION_END_REGEX, text3)

    def test_match_format_regex(self):
        valid_text = f"text {REASONING_END} more text {SOLUTION_START}solution content{SOLUTION_END}"
        match = rewards.MATCH_FORMAT.search(valid_text)
        assert match is not None
        assert match.group(1) == "solution content"

        # Invalid format - missing parts
        invalid_text = f"text {SOLUTION_START}solution{SOLUTION_END}"
        assert rewards.MATCH_FORMAT.search(invalid_text) is None
