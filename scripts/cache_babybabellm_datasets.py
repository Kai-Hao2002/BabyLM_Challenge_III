from datasets import load_dataset

DATASETS = [
    ("xnli", None),
    ("xcopa", None),
    ("CohereForAI/Global-MMLU", None),
    ("CohereLabs/Global-MMLU", None),
    ("fpadovani/xcomps-dataset", None),
    ("Junrui1202/zhoblimp", None),
]

for name, config in DATASETS:
    print("=" * 80)
    print(f"Trying to cache: {name} config={config}")
    print("=" * 80)

    try:
        if config is None:
            ds = load_dataset(name)
        else:
            ds = load_dataset(name, config)

        print(ds)

    except Exception as e:
        print(f"[FAILED] {name}")
        print(repr(e))
