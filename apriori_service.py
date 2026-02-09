import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def get_frequent_itemsets(transactions, min_support=0.001):
    all_items = sorted(set(item for trx in transactions for item in trx))

    oht = pd.DataFrame(
        [{item: (item in trx) for item in all_items} for trx in transactions]
    )

    frequent = apriori(oht, min_support=min_support, use_colnames=True)
    return frequent, oht


def get_association_rules(frequent_itemsets, min_confidence=0.2, min_lift=1.0):

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    # Filter supaya hanya 1 item + 1 item (2 item saja)
    rules = rules[
        (rules["antecedents"].apply(lambda x: len(x) == 1))
        & (rules["consequents"].apply(lambda x: len(x) == 1))
    ]

    # Filter lift minimal
    rules = rules[rules["lift"] >= min_lift]

    # urutkan berdasarkan lift
    rules = rules.sort_values("lift", ascending=False)

    return rules
