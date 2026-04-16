# Exercise 040: Streaming and Structured Outputs

## Overview

In this lab, you will implement an agentic loan calculator that uses Pydantic to ensure structured outputs and simulates token-by-token streaming for a better user experience.

## Objectives

- Define a Pydantic model for structured data validation.
- Implement a streaming simulation helper.
- Use `.format()` to inject variables into a system prompt.

## Instructions

1. Open `starter_code/e040-streaming_pydantic_lab.py`.
2. **Task 1**: Complete the `LoanCalculation` Pydantic model with fields for `monthly_payment` and `total_interest`.
3. **Task 2**: Implement the `simulate_streaming_response` function to print characters with a small delay.
4. **Task 3**: Use the LLM output to populate your Pydantic model.

## Validation

- Run your script.
- Ensure the output "streams" into the terminal.
- Verify that the final JSON object matches your Pydantic schema.
