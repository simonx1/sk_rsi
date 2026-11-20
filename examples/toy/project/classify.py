"""Label a support message with the first rule whose keyword it contains."""


def load_rules(path):
    rules = []
    with open(path) as handle:
        for line in handle:
            if "=>" in line:
                keyword, label = line.split("=>", 1)
                rules.append((keyword.strip().lower(), label.strip()))
    return rules


def classify(text, rules):
    lowered = text.lower()
    for keyword, label in rules:
        if keyword and keyword in lowered:
            return label
    return "other"
