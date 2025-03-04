def check_value(value, expected_type, parameter, expected_type_str) -> bool:
    if isinstance(value, expected_type):
        return True
    else:
        raise ValueError("The given value '{value}' is not supported for '{parameter}'. It must be of type '{expected_type_str}'.")