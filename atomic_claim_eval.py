"""
Atomic Claim Verification Evaluation.

Decomposes reference and hypothesis summaries into atomic factual claims,
then uses an LLM judge to verify whether each claim is supported.

Metrics:
  - Claim Recall:    % of reference claims supported by hypothesis
  - Claim Precision: % of hypothesis claims supported by reference
  - Claim F1:        Harmonic mean of precision and recall

Requires LM Studio running locally with Qwen 3.5 (or similar).
"""

import os
import json
import time
import re
import sys
from typing import List, Dict, Tuple
from evaluate_ami_results import parse_ami_summary, load_system_summaries

# ── Config ───────────────────────────────────────────────────────────────────

LLM_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"
AMI_DIR = os.path.expanduser("~/Downloads/ami_public_manual_1.6.2/abstractive")

CONFIGS = {
    "Text-Only":        "data/jobs_text_only.json",
    "Text + Audio":     "data/jobs_text_audio.json",
    "Full RoME Fusion": "data/jobs_full.json",
}


# ── LLM Helper ───────────────────────────────────────────────────────────────

def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """Call local LM Studio endpoint."""
    import requests

    resp = requests.post(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2048,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # Strip <think>...</think> blocks if present (Qwen reasoning)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


# ── Step 1: Decompose into Atomic Claims ─────────────────────────────────────

DECOMPOSE_SYSTEM = """You are a precise fact extractor. Given a meeting summary, decompose it into a list of atomic factual claims. Each claim should be:
- A single, self-contained statement
- Verifiable as true or false
- As specific as possible (include names, numbers, decisions)

Output ONLY a JSON array of strings. No explanation."""

DECOMPOSE_USER = """Decompose this meeting summary into atomic claims:

\"\"\"
{summary}
\"\"\"

Output a JSON array of claim strings:"""


def decompose_claims(summary: str) -> List[str]:
    """Break a summary into atomic factual claims using LLM."""
    if not summary or not summary.strip():
        return []

    raw = llm_call(DECOMPOSE_SYSTEM, DECOMPOSE_USER.format(summary=summary[:3000]))

    # Extract JSON array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        # Fallback: split by newlines/bullets
        lines = [l.strip().lstrip("-•*0123456789.) ") for l in raw.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 10]

    try:
        claims = json.loads(match.group())
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if c and len(str(c).strip()) > 5]
    except json.JSONDecodeError:
        pass

    return []


# ── Step 2: Verify Claims ───────────────────────────────────────────────────

VERIFY_SYSTEM = """You are a strict fact checker. Given a CLAIM and a DOCUMENT, determine if the document SUPPORTS the claim.

Rules:
- "supported" = the document contains information that confirms or is consistent with the claim
- "not_supported" = the document contradicts the claim OR contains no relevant information
- Be strict: partial support counts as "not_supported"

Output ONLY a JSON object: {"verdict": "supported" or "not_supported"}"""

VERIFY_USER = """CLAIM: {claim}

DOCUMENT:
\"\"\"
{document}
\"\"\"

Verdict:"""


def verify_claim(claim: str, document: str) -> bool:
    """Check if a single claim is supported by the document."""
    raw = llm_call(VERIFY_SYSTEM, VERIFY_USER.format(claim=claim, document=document[:3000]))

    # Parse verdict
    match = re.search(r'"verdict"\s*:\s*"(supported|not_supported)"', raw, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "supported"

    # Fallback: check for keywords
    lower = raw.lower()
    if "supported" in lower and "not_supported" not in lower:
        return True
    return False


# ── Step 3: Compute Metrics ─────────────────────────────────────────────────

def evaluate_meeting(
    ref_summary: str, hyp_summary: str, meeting_id: str
) -> Dict:
    """Run atomic claim verification for one meeting."""

    # Decompose both summaries
    ref_claims = decompose_claims(ref_summary)
    hyp_claims = decompose_claims(hyp_summary)

    if not ref_claims or not hyp_claims:
        return {
            "meeting_id": meeting_id,
            "ref_claims": len(ref_claims),
            "hyp_claims": len(hyp_claims),
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "error": "empty claims",
        }

    # Recall: How many reference claims are supported by the hypothesis?
    ref_supported = 0
    for claim in ref_claims:
        if verify_claim(claim, hyp_summary):
            ref_supported += 1

    # Precision: How many hypothesis claims are supported by the reference?
    hyp_supported = 0
    for claim in hyp_claims:
        if verify_claim(claim, ref_summary):
            hyp_supported += 1

    recall = ref_supported / len(ref_claims)
    precision = hyp_supported / len(hyp_claims)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "meeting_id": meeting_id,
        "ref_claims": len(ref_claims),
        "hyp_claims": len(hyp_claims),
        "ref_supported": ref_supported,
        "hyp_supported": hyp_supported,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ATOMIC CLAIM VERIFICATION EVALUATION")
    print("=" * 70)

    # Check LM Studio is running
    try:
        import requests
        requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
    except Exception:
        print("ERROR: LM Studio not running at http://127.0.0.1:1234")
        print("Start LM Studio and load a model before running this script.")
        sys.exit(1)

    all_results = {}

    for config_name, file_path in CONFIGS.items():
        if not os.path.exists(file_path):
            print(f"\nSkipping {config_name}: {file_path} not found")
            continue

        print(f"\n{'─'*70}")
        print(f"Config: {config_name}")
        print(f"{'─'*70}")

        system_summaries = load_system_summaries(file_path)
        config_results = []

        for meeting_id, hyp_summary in system_summaries.items():
            xml_path = os.path.join(AMI_DIR, f"{meeting_id}.abssumm.xml")
            ref_summary = parse_ami_summary(xml_path)

            if not ref_summary or not hyp_summary or not hyp_summary.strip():
                continue

            print(f"  Evaluating {meeting_id}...", end=" ", flush=True)
            t0 = time.time()
            result = evaluate_meeting(ref_summary, hyp_summary, meeting_id)
            elapsed = time.time() - t0
            print(
                f"R={result['recall']:.2f} P={result['precision']:.2f} "
                f"F1={result['f1']:.2f} "
                f"(ref={result['ref_claims']} hyp={result['hyp_claims']}) "
                f"[{elapsed:.1f}s]"
            )
            config_results.append(result)

        if not config_results:
            continue

        # Aggregate
        avg_recall = sum(r["recall"] for r in config_results) / len(config_results)
        avg_precision = sum(r["precision"] for r in config_results) / len(config_results)
        avg_f1 = sum(r["f1"] for r in config_results) / len(config_results)

        total_ref = sum(r["ref_claims"] for r in config_results)
        total_hyp = sum(r["hyp_claims"] for r in config_results)
        total_ref_sup = sum(r.get("ref_supported", 0) for r in config_results)
        total_hyp_sup = sum(r.get("hyp_supported", 0) for r in config_results)

        micro_recall = total_ref_sup / total_ref if total_ref > 0 else 0
        micro_precision = total_hyp_sup / total_hyp if total_hyp > 0 else 0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0
        )

        print(f"\n  {'─'*50}")
        print(f"  {config_name} — MACRO AVERAGE ({len(config_results)} meetings)")
        print(f"  Claim Recall:    {avg_recall:.4f}")
        print(f"  Claim Precision: {avg_precision:.4f}")
        print(f"  Claim F1:        {avg_f1:.4f}")
        print(f"\n  MICRO AVERAGE (total claims)")
        print(f"  Total ref claims: {total_ref}  supported: {total_ref_sup}")
        print(f"  Total hyp claims: {total_hyp}  supported: {total_hyp_sup}")
        print(f"  Micro Recall:    {micro_recall:.4f}")
        print(f"  Micro Precision: {micro_precision:.4f}")
        print(f"  Micro F1:        {micro_f1:.4f}")

        all_results[config_name] = {
            "per_meeting": config_results,
            "macro": {"recall": avg_recall, "precision": avg_precision, "f1": avg_f1},
            "micro": {"recall": micro_recall, "precision": micro_precision, "f1": micro_f1},
        }

    # Save results
    out_path = "data/atomic_claim_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Final comparison table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Config':<22} {'Claim Recall':>13} {'Claim Prec':>12} {'Claim F1':>10}")
    print(f"{'─'*22} {'─'*13} {'─'*12} {'─'*10}")
    for name, res in all_results.items():
        m = res["macro"]
        print(f"{name:<22} {m['recall']:>13.4f} {m['precision']:>12.4f} {m['f1']:>10.4f}")


if __name__ == "__main__":
    main()
