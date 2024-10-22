

def write_report(results: list[dict], property_dicts: list[dict]):
    with open('soil_test_report.md', 'w', encoding='utf-8') as f:
        for result, property_dict in zip(results, property_dicts):
            # Write property name as a header
            f.write(f"# {property_dict['soil_test_property']['name']}\n\n")
            # Write the value
            f.write(f"{property_dict['value']}\n\n")
            # Write the answer
            f.write(f"{result['answer']}\n\n")
            # Write token count info
            f.write("# Token Count\n")
            f.write(f"Prompt Tokens: {result['token_info']['prompt_tokens']}\n")
            f.write(f"Completion Tokens: {result['token_info']['completion_tokens']}\n")
            f.write(f"Total Tokens: {result['token_info']['total_tokens']}\n\n")
            f.write("# LLM Info\n")
            f.write(f"Model: {result['other_info']['model']}\n")
            f.write(f"Total Duration: {result['other_info']['total_duration']}\n")
            f.write(f"Load Duration: {result['other_info']['load_duration']}\n")
            f.write(f"Eval Duration: {result['other_info']['eval_duration']}\n\n")

            # Add a separator between entries (optional)
            f.write("---\n\n")
