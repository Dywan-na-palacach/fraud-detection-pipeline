from src.load_data import load_transactions
from src.rules import apply_simple_rules, eval_rules


def main():
    df = load_transactions()
    df = apply_simple_rules(df)
    eval_rules(df)


if __name__ == "__main__":
    main()
