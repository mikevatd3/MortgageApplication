from importlib.resources import files


def load_prompt_template(name: str) -> str:
    return files("mortgageapplication").joinpath("prompts", name).read_text()


def continuation_joiner(prompt: str) -> str:
    """What goes between a prompt and the text the model would write next.

    A prompt ending in a newline means the text starts a new line, so no space: "approve", not " approve".
    The two are different tokens, and only the first is what the model emits there.
    """
    return "" if prompt[-1:].isspace() else " "



WHOLE_NUMBER_FIELDS = ("credit_score", "property_value", "loan_amount", "income")


def fill_application_prompt(template: str, row) -> str:
    """Fill the application prompt, showing scores and dollar amounts as whole numbers and the ratio to 3 places."""
    values = dict(row)
    for field in WHOLE_NUMBER_FIELDS:
        values[field] = int(round(values[field]))
    values["debt_to_income"] = round(values["debt_to_income"], 3)
    return template.format(**values)
