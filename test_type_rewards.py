#!/usr/bin/env python3

from rewards import infers_type_correctly, correct
from datasetting import SOLUTION_START, SOLUTION_END

def test_type_reward():
    # Test case 1: Simple Int -> Int function (should match)
    completion_correct = [[{
        "content": f"""<start_working_out>
This should take an integer and return an integer by adding 1.
<end_working_out>

{SOLUTION_START}lambda x.+ x 1{SOLUTION_END}"""
    }]]

    # Test case 2: Wrong type function (should not match)
    completion_wrong = [[{
        "content": f"""<start_working_out>
This should return a list but we return an int.
<end_working_out>

{SOLUTION_START}lambda x.+ x 1{SOLUTION_END}"""
    }]]

    # Test case 3: Invalid syntax (should not match)
    completion_invalid = [[{
        "content": f"""<start_working_out>
Invalid lambda.
<end_working_out>

{SOLUTION_START}lambda x{SOLUTION_END}"""
    }]]

    # Sample prompt data for Int -> Int function
    prompt_data = {
        'args_info': {'a': 'int'},
        'return_type': 'int'
    }

    # Test correct type
    score_correct = infers_type_correctly(completion_correct, prompt_data=prompt_data)
    print(f"Correct type score: {score_correct}")

    # Test prompt data for List[Int] -> List[Int] function but wrong completion
    prompt_data_list = {
        'args_info': {'a': 'int[]'},
        'return_type': 'int[]'
    }

    score_wrong = infers_type_correctly(completion_wrong, prompt_data=prompt_data_list)
    print(f"Wrong type score: {score_wrong}")

    # Test invalid syntax
    score_invalid = infers_type_correctly(completion_invalid, prompt_data=prompt_data)
    print(f"Invalid syntax score: {score_invalid}")

    # Test with correct list function
    completion_list_correct = [[{
        "content": f"""<start_working_out>
This should map over a list and double each element.
<end_working_out>

{SOLUTION_START}lambda xs.map xs (* 2){SOLUTION_END}"""
    }]]

    score_list_correct = infers_type_correctly(completion_list_correct, prompt_data=prompt_data_list)
    print(f"Correct list type score: {score_list_correct}")

    # Test with multi-argument function: Int -> Int -> Int
    completion_multi_arg = [[{
        "content": f"""<start_working_out>
This should add two integers.
<end_working_out>

{SOLUTION_START}lambda x y.+ x y{SOLUTION_END}"""
    }]]

    prompt_data_multi = {
        'args_info': {'a': 'int', 'b': 'int'},
        'return_type': 'int'
    }

    score_multi_correct = infers_type_correctly(completion_multi_arg, prompt_data=prompt_data_multi)
    print(f"Correct multi-arg type score: {score_multi_correct}")

def test_correct_reward():
    # Test correct function with simple test case
    completion = [[{
        "content": f"""<start_working_out>
This should double each element in the list.
<end_working_out>

{SOLUTION_START}lambda xs.map xs (* 2){SOLUTION_END}"""
    }]]

    prompt_data = {
        'tests': '[{"input": {"a": [1, 2, 3]}, "output": [2, 4, 6]}, {"input": {"a": [5]}, "output": [10]}]',
        'args_info': '{"a": "int[]"}'
    }

    score = correct(completion, prompt_data=prompt_data)
    print(f"Correct function score: {score}")

if __name__ == "__main__":
    test_type_reward()
    test_correct_reward()