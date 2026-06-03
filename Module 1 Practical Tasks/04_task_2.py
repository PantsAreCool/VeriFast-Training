class SchemaValidationError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")

def validate_response(data, schema):
    errors = []

    for field, rules in schema.items():
        if rules.get("required", False) and field not in data:
            errors.append(f"Missing required field: '{field}'")
            continue
            
        if field in data:
            val = data[field]
            expected_type = rules.get("type")

            if expected_type and not isinstance(val, expected_type):
                errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(val).__name__}")
                continue

            if "range" in rules and isinstance(val, (int, float)):
                min_val, max_val = rules["range"]
                if not (min_val <= val <= max_val):
                    errors.append(f"Field '{field}' value {val} is out of bounds ({min_val} to {max_val})")

            if "allowed" in rules and val not in rules["allowed"]:
                errors.append(f"Field '{field}' value '{val}' is not one of: {rules['allowed']}")

    if errors:
        raise SchemaValidationError(errors)
    return True

if __name__ == "__main__":
    ai_schema = {
        "confidence": {"type": float, "required": True, "range": (0.0, 1.0)},
        "sentiment": {"type": str, "required": True, "allowed": ["positive", "neutral", "negative"]},
        "tokens": {"type": int, "required": False}
    }

    bad_llm_output = {
        "confidence": 1.5,
        "sentiment": "happy",
        "tokens": "one-hundred"
    }

    print("VALIDATING LLM RESPONSE:")
    try:
        validate_response(bad_llm_output, ai_schema)
        print("Response is valid!")
    except SchemaValidationError as e:
        for error in e.errors:
            print(f" - Error: {error}")