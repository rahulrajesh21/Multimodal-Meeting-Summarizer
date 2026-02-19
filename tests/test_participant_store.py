"""
Unit tests for ParticipantStore and role_hierarchy fallback.
Run from the project root: python tests/test_participant_store.py
"""

import os
import sys
import shutil
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.role_hierarchy import get_role_authority, get_fallback_weights
from src.participant_store import ParticipantStore

TEST_DIR = "test_participant_data"

def setup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def teardown():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def check(condition, label):
    status = "✅" if condition else "❌"
    print(f"  {status} {label}")
    return condition

def test_role_authority():
    print("\n── Role Authority Tests ──")
    all_pass = True
    all_pass &= check(get_role_authority("CEO") >= 0.9,          "CEO authority ≥ 0.90")
    all_pass &= check(get_role_authority("CTO") >= 0.9,          "CTO authority ≥ 0.90")
    all_pass &= check(get_role_authority("Developer") < 0.5,     "Developer authority < 0.50")
    all_pass &= check(get_role_authority("Client") >= 0.6,       "Client authority ≥ 0.60 (external)")
    all_pass &= check(get_role_authority("Investor") >= 0.65,    "Investor authority ≥ 0.65")
    all_pass &= check(get_role_authority("Senior External Stakeholder") != 0.5,
                                                                  "Keyword heuristic resolves 'Senior External Stakeholder'")
    all_pass &= check(get_role_authority("xyz_unknown_role_999") == 0.5,
                                                                  "Unknown role → neutral 0.50 fallback")
    return all_pass

def test_fallback_weights():
    print("\n── Fallback Weight Tests ──")
    all_pass = True

    ceo_w  = get_fallback_weights("CEO",       is_external=False)
    dev_w  = get_fallback_weights("Developer", is_external=False)
    cli_w  = get_fallback_weights("Client",    is_external=True)

    # Weights must sum to 1.0
    for label, w in [("CEO", ceo_w), ("Developer", dev_w), ("Client", cli_w)]:
        total = sum(w.values())
        all_pass &= check(abs(total - 1.0) < 0.01, f"{label} weights sum to 1.0 (got {total:.4f})")

    # CEO should have higher role_w than Developer
    all_pass &= check(ceo_w["role"] > dev_w["role"],
                      f"CEO role_w ({ceo_w['role']:.2f}) > Developer role_w ({dev_w['role']:.2f})")

    # Developer should have higher tonal_w than CEO
    all_pass &= check(dev_w["tonal"] > ceo_w["tonal"],
                      f"Developer tonal_w ({dev_w['tonal']:.2f}) > CEO tonal_w ({ceo_w['tonal']:.2f})")

    # External client should have lower temporal than CEO (no meeting history)
    all_pass &= check(cli_w["temporal"] < ceo_w["temporal"],
                      f"Client temporal_w ({cli_w['temporal']:.2f}) < CEO temporal_w ({ceo_w['temporal']:.2f})")
    return all_pass

def test_participant_store():
    print("\n── ParticipantStore Tests ──")
    all_pass = True
    setup()
    store = ParticipantStore(data_dir=TEST_DIR)

    # 1. add_participant
    profile = store.add_participant("Alice Johnson", "CTO", is_external=False)
    all_pass &= check(profile is not None,                "add_participant returns profile")
    all_pass &= check("weights" in profile,               "Profile contains 'weights'")
    all_pass &= check(abs(sum(profile["weights"].values()) - 1.0) < 0.01,
                                                          "Weights sum to 1.0")

    # 2. get_participant — exact
    found = store.get_participant("Alice Johnson")
    all_pass &= check(found is not None,                  "get_participant('Alice Johnson') found")

    # 3. get_participant — fuzzy (first name only)
    found_short = store.get_participant("alice")
    all_pass &= check(found_short is not None,            "Fuzzy lookup by first name works")

    # 4. get_participant — unknown
    missing = store.get_participant("Bob Unknown Person")
    all_pass &= check(missing is None,                    "Unknown participant returns None")

    # 5. get_weights_for_speaker — profile hit
    w = store.get_weights_for_speaker("Alice Johnson")
    all_pass &= check(w == profile["weights"],            "get_weights_for_speaker returns profile weights")

    # 6. get_weights_for_speaker — miss → neutral fallback
    w_miss = store.get_weights_for_speaker("Bob Smith")
    all_pass &= check(abs(sum(w_miss.values()) - 1.0) < 0.01,
                                                          "Unregistered speaker fallback weights sum to 1.0")

    # 7. update_weights
    new_w = {k: round(v * 0.9, 4) for k, v in profile["weights"].items()}
    total = sum(new_w.values())
    new_w = {k: round(v / total, 4) for k, v in new_w.items()}
    updated = store.update_weights("Alice Johnson", new_w)
    all_pass &= check(updated,                            "update_weights returns True for known participant")

    # Re-load to verify persistence
    store2 = ParticipantStore(data_dir=TEST_DIR)
    p2 = store2.get_participant("alice")
    all_pass &= check(p2 is not None,                    "Profile persists across store instances")

    # 8. register_custom_role
    store.register_custom_role("Field Sales Rep", 0.55)
    custom_path = os.path.join(TEST_DIR, "custom_roles.json")
    all_pass &= check(os.path.exists(custom_path),        "custom_roles.json created")
    with open(custom_path) as f:
        custom = json.load(f)
    all_pass &= check(custom.get("field sales rep") == 0.55,
                                                          "Custom role persisted correctly")

    # 9. get_ui_badge
    badge_hit  = store.get_ui_badge("Alice Johnson")
    badge_miss = store.get_ui_badge("Nobody Here")
    all_pass &= check("🟢" in badge_hit,  "UI badge shows 🟢 for known participant")
    all_pass &= check("🟡" in badge_miss, "UI badge shows 🟡 for unknown participant")

    teardown()
    return all_pass


if __name__ == "__main__":
    results = []
    results.append(("role_authority",    test_role_authority()))
    results.append(("fallback_weights",  test_fallback_weights()))
    results.append(("participant_store", test_participant_store()))

    print("\n══ Summary ══")
    all_ok = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        all_ok &= passed

    sys.exit(0 if all_ok else 1)
