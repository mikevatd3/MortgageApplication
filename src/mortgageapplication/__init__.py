from importlib.resources import files


def load_prompt_template(name: str) -> str:
    return files("mortgageapplication").joinpath("prompts", name).read_text()


def main() -> None:
    prompt = load_prompt_template("loan_prioritization.txt")
    print(prompt)

