"""Run small, labeled evidence sets through the production SELECT path."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.eaes import EAESMixin
from common import config


def main():
    cases = [
        ("direct_fact", "When did Joanna bake the raspberry tart?", [
            ("date", "Joanna baked a raspberry tart on 5 October 2022."),
            ("distractor", "Nate baked a raspberry tart on 6 October 2022."),
        ], {"date"}),
        ("complementary_dates", "How long did Joanna take to write her novel Seabird?", [
            ("finish", "Joanna finished writing her novel Seabird on 1 October 2022."),
            ("start", "Joanna began writing her novel Seabird on 1 June 2022."),
            ("other_book", "Joanna began writing her novel Driftwood on 1 May 2022."),
        ], {"start", "finish"}),
        ("complementary_dates_reversed", "How long did Joanna take to write her novel Seabird?", [
            ("start", "Joanna began writing her novel Seabird on 1 June 2022."),
            ("finish", "Joanna finished writing her novel Seabird on 1 October 2022."),
            ("other_book", "Joanna began writing her novel Driftwood on 1 May 2022."),
        ], {"start", "finish"}),
        ("distinct_events", "How many times did Nate take his turtles for a walk?", [
            ("october", "Nate took his turtles for a walk on 25 October 2022."),
            ("march", "Nate took his turtles for a walk on 18 March 2022."),
            ("march_copy", "On 18 March 2022, Nate went walking with his turtles."),
        ], {"october", "march"}),
        ("parent_child_duplicate", "What made Nate's ice cream lactose-free?", [
            ("child", "Nate used coconut milk to make his ice cream lactose-free."),
            ("P1", "Nate made lactose-free ice cream using coconut milk."),
        ], {"child"}),
        ("partial_substitute", "What kind of cake does Joanna make for birthdays?", [
            ("generic", "Joanna makes chocolate cake for birthdays."),
            ("specific", "Joanna makes chocolate cake with raspberries for birthdays."),
        ], {"specific"}),
        ("wrong_event", "How many people attended Nate's June 2022 gaming party?", [
            ("july", "Seven people attended Nate's July 2022 gaming party."),
        ], set()),
    ]
    config.RETRIEVAL_ONLY = True
    selector = EAESMixin()
    print(f"utility model: {config.UTILITY_MODEL}", flush=True)
    passed = 0
    for name, question, facts, expected in cases:
        children, parents = [], []
        for rank, (node_id, fact) in enumerate(facts, 1):
            item = {"rewrite_content": fact, "score": 1 / rank}
            if node_id == "P1":
                parents.append({"parent_id": node_id, **item})
            else:
                children.append({"memory_id": node_id, **item})
        result = selector.select_eaes_evidence(question, {}, children, parents)
        decisions = result["selector_diagnostics"]["decisions"]
        kept = {row["id"] for row in decisions if row["decision"] == "KEEP"}
        passed += kept == expected
        print(json.dumps({
            "case": name, "expected": sorted(expected), "kept": sorted(kept),
            "passed": kept == expected, "decisions": decisions,
        }, ensure_ascii=False), flush=True)
    print(f"passed {passed}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()
