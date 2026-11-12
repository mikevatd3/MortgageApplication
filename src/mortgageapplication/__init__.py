from importlib.resources import files


def load_prompt_template(name: str) -> str:
    return files("mortgageapplication").joinpath("prompts", name).read_text()



WHOLE_NUMBER_FIELDS = ("credit_score", "property_value", "loan_amount", "income")


def fill_application_prompt(template: str, row) -> str:
    """Fill the application prompt, showing scores and dollar amounts as whole numbers and the ratio to 3 places."""
    values = dict(row)
    for field in WHOLE_NUMBER_FIELDS:
        values[field] = int(round(values[field]))
    values["debt_to_income"] = round(values["debt_to_income"], 3)
    return template.format(**values)
