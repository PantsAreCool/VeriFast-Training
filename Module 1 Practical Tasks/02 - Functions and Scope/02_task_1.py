def format_prompt(template, values, defaults=None):
    if defaults is None:
        defaults = {}
        
    merged = defaults.copy()
    merged.update(values)

    result = template
    for key, val in merged.items():
        placeholder = "{{" + key + "}}"
        result = result.replace(placeholder, str(val))

    if "{{" in result and "}}" in result:
        print("Error: Missing required variables for this template.")
        return None
        
    return result

if __name__ == "__main__":
    template_str = "You are an expert in {{domain}}. Explain {{topic}} in a {{tone}} tone."
    
    user_vars = {"domain": "Physics", "topic": "Quantum Entanglement"}
    default_vars = {"tone": "educational"}
    
    formatted = format_prompt(template_str, user_vars, default_vars)
    print("FORMATTED PROMPT:")
    print(formatted)