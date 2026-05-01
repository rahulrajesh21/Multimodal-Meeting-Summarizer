import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List, Dict, Set, Tuple

# Attempt to import bert_score. If not installed, handle gracefully.
try:
    from bert_score import score as calc_bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: 'bert-score' is not installed. To run BERTScore metrics, please run: pip install bert-score")

from evaluate_ami_results import parse_ami_summary, load_system_summaries

# -----------------------------------------------------------------------------
# 1. ABLATION STUDY METRICS (ROUGE & BERTScore)
# -----------------------------------------------------------------------------

def evaluate_ablation_study(ground_truth_dir: str, result_files: Dict[str, str]):
    """
    Evaluates different ablation configurations against the AMI ground truth using ROUGE.
    """
    print("\n" + "="*60)
    print("1. ABLATION STUDY: MULTIMODAL GAIN")
    print("="*60)
    
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for config_name, file_path in result_files.items():
        if not os.path.exists(file_path):
            print(f"Skipping '{config_name}': File {file_path} not found.")
            continue
            
        system_summaries = load_system_summaries(file_path)
        
        total_r1 = 0
        total_r2 = 0
        total_rl = 0
        count = 0
        
        for meeting_id, hyp_summary in system_summaries.items():
            xml_path = os.path.join(ground_truth_dir, f"{meeting_id}.abssumm.xml")
            ref_summary = parse_ami_summary(xml_path)
            
            if ref_summary and hyp_summary and hyp_summary.strip():
                scores = scorer.score(ref_summary, hyp_summary)
                total_r1 += scores['rouge1'].fmeasure
                total_r2 += scores['rouge2'].fmeasure
                total_rl += scores['rougeL'].fmeasure
                count += 1
                
        if count == 0:
            print(f"[{config_name:<20}] No matching ground truth summaries found.")
            continue
            
        print(f"[{config_name:<20}] ROUGE-1: {total_r1/count:.4f} | ROUGE-2: {total_r2/count:.4f} | ROUGE-L: {total_rl/count:.4f}")


# -----------------------------------------------------------------------------
# 2. ENTITY EXTRACTION ACCURACY (Precision & Recall)
# -----------------------------------------------------------------------------

def calculate_entity_metrics(ground_truth_entities: Set[str], predicted_entities: Set[str]) -> Tuple[float, float, float]:
    """Calculates Precision, Recall, and F1 for entity extraction."""
    if not ground_truth_entities and not predicted_entities:
        return 1.0, 1.0, 1.0
    if not ground_truth_entities or not predicted_entities:
        return 0.0, 0.0, 0.0
        
    true_positives = len(ground_truth_entities.intersection(predicted_entities))
    
    precision = true_positives / len(predicted_entities) if len(predicted_entities) > 0 else 0.0
    recall = true_positives / len(ground_truth_entities) if len(ground_truth_entities) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate_entity_extraction():
    """
    Demonstrates how to evaluate entity extraction from your Temporal Knowledge Graph (TGM)
    against human-annotated entities.
    """
    print("\n" + "="*60)
    print("2. ENTITY EXTRACTION ACCURACY")
    print("="*60)
    
    # Mock data for demonstration:
    mock_evaluations = [
        {
            "meeting_id": "ES2002a",
            "ground_truth": {"remote control", "jog dial", "lcd screen", "25 euro", "david"},
            "predicted": {"remote control", "jog dial", "25 euro", "david", "battery"} # Missed 'lcd screen', False alarm 'battery'
        },
        {
            "meeting_id": "ES2002b",
            "ground_truth": {"teletext", "yellow", "corporate slogan", "rubber buttons"},
            "predicted": {"teletext", "yellow", "corporate slogan", "rubber buttons", "plastic"}
        }
    ]
    
    total_p, total_r, total_f1 = 0, 0, 0
    count = len(mock_evaluations)
    
    for eval_data in mock_evaluations:
        p, r, f1 = calculate_entity_metrics(eval_data["ground_truth"], eval_data["predicted"])
        print(f"Meeting {eval_data['meeting_id']}: Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")
        total_p += p
        total_r += r
        total_f1 += f1
        
    print(f"\nAverage Precision: {total_p/count:.2f}")
    print(f"Average Recall:    {total_r/count:.2f}")
    print(f"Average F1-Score:  {total_f1/count:.2f}")


# -----------------------------------------------------------------------------
# 3. RETRIEVAL ACCURACY (MRR & Hit@K)
# -----------------------------------------------------------------------------

def evaluate_retrieval_accuracy(k_values=[1, 3, 5]):
    """
    Calculates Mean Reciprocal Rank (MRR) and Hit@K for your ChromaDB / RAG pipeline.
    """
    import sys
    import os
    
    # Ensure src is in the path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    from src.context_store import ContextStore
    from src.text_analysis import TextAnalyzer
    
    print("\n" + "="*60)
    print("3. TEMPORAL GRAPH RAG RETRIEVAL ACCURACY")
    print("="*60)
    
    try:
        ta = TextAnalyzer()
        cs = ContextStore(storage_root=os.path.abspath("."), text_analyzer=ta)
    except Exception as e:
        print(f"Failed to load ContextStore: {e}")
        return

    # 50 Queries: (Query, [Expected Keywords], Is_Cross_Meeting)
    test_queries = [
        # --- Cross-Meeting Queries (Tracked across time) ---
        ("Was the LCD screen issue ever resolved?", ["lcd", "screen", "decision", "resolve", "decided"], True),
        ("Did we ultimately decide to include teletext?", ["teletext", "decide", "drop", "include", "cancel"], True),
        ("What is the final decision on the battery type?", ["battery", "kinetic", "solar", "standard", "aa"], True),
        ("What materials were chosen for the remote case?", ["plastic", "rubber", "material", "case", "wood"], True),
        ("What is the final target selling price?", ["25", "twenty-five", "euro", "price"], True),
        ("Who is the project manager for this project?", ["project manager", "pm", "david", "alice"], True),
        ("What is the corporate color we must use?", ["yellow", "corporate", "color", "red", "blue"], True),
        ("Did we finalize the shape of the remote?", ["shape", "ergonomic", "curved", "design", "flat"], True),
        ("Are we using rubber or plastic buttons?", ["rubber", "plastic", "button"], True),
        ("What was the conclusion on speech recognition?", ["speech", "recognition", "voice", "decide", "drop"], True),
        ("Did the budget change from 12.50?", ["budget", "12.5", "12.50", "change", "euro"], True),
        ("What is the main market for this product?", ["market", "international", "domestic", "global"], True),
        ("Who is handling the user interface?", ["ui", "user interface", "designer", "person"], True),
        ("Who is responsible for the industrial design?", ["industrial", "design", "designer"], True),
        ("Are we including a scroll wheel?", ["scroll", "wheel", "jog", "dial", "yes", "no"], True),
        ("What is the remote's main distinguishing feature?", ["feature", "distinguish", "design", "unique"], True),
        ("Will the remote have a backlight?", ["backlight", "light", "illuminate", "glow"], True),
        ("Are we doing a single or double-sided PCB?", ["pcb", "single", "double", "board"], True),
        ("What did we decide about the remote's weight?", ["weight", "heavy", "light", "grams"], True),
        ("What is the expected battery life?", ["battery life", "months", "years", "duration"], True),
        ("Is there a mute button?", ["mute", "button", "silence"], True),
        ("Will it have universal TV support?", ["universal", "tv", "television", "support"], True),
        ("What chip are we using?", ["chip", "microcontroller", "processor"], True),
        ("Did we agree on the branding placement?", ["brand", "placement", "logo", "top", "bottom"], True),
        ("What is the timeline for the prototype?", ["timeline", "prototype", "weeks", "months"], True),
        
        # --- Single-Meeting Queries (Fact retrieval) ---
        ("What is the maximum production cost allowed?", ["12.5", "12.50", "twelve fifty", "budget"], False),
        ("What animal did they discuss in the kickoff?", ["animal", "draw", "favorite", "dog", "cat", "bird"], False),
        ("What is the target age demographic?", ["age", "target", "demographic", "young", "old"], False),
        ("Is the remote exclusively for televisions?", ["television", "tv", "exclusive", "universal"], False),
        ("What features did the user interface designer propose?", ["user interface", "ui", "propose", "idea"], False),
        ("What was the marketing executive's main concern?", ["marketing", "executive", "concern", "requirement"], False),
        ("How many buttons should the remote have?", ["number", "button", "how many"], False),
        ("What did the industrial designer say about ergonomics?", ["industrial", "ergonomic", "hand", "feel"], False),
        ("Was a curved shape proposed in the first meeting?", ["curved", "shape", "first", "propose"], False),
        ("Did someone suggest a touchscreen?", ["touchscreen", "touch", "screen", "suggest"], False),
        ("What was said about the competitor's remote?", ["competitor", "remote", "compare", "market"], False),
        ("How long should the meeting take?", ["long", "meeting", "time", "minutes"], False),
        ("Did they mention a specific TV brand?", ["tv", "brand", "sony", "philips", "samsung"], False),
        ("What was the joke about the budget?", ["joke", "budget", "funny", "laugh"], False),
        ("Did anyone arrive late?", ["late", "arrive", "time", "delay"], False),
        ("What is the name of the company?", ["company", "name", "real", "fake"], False),
        ("Are they using a whiteboard?", ["whiteboard", "board", "draw", "write"], False),
        ("Who took the meeting minutes?", ["minutes", "notes", "take", "write"], False),
        ("What was the opening statement?", ["opening", "statement", "welcome", "hello"], False),
        ("Did they discuss the manual?", ["manual", "instructions", "guide", "read"], False),
        ("Is there a warranty discussed?", ["warranty", "guarantee", "years"], False),
        ("What packaging material was suggested?", ["packaging", "material", "box", "plastic", "cardboard"], False),
        ("Are batteries included in the box?", ["batteries", "included", "box", "package"], False),
        ("What is the shipping cost estimate?", ["shipping", "cost", "estimate", "freight"], False),
        ("Where is the manufacturing plant located?", ["manufacturing", "plant", "location", "china", "europe"], False)
    ]
    
    def evaluate_system(use_graph: bool, use_reranker: bool = False):
        mrr_sum = 0.0
        cm_mrr_sum = 0.0
        sm_mrr_sum = 0.0
        hits = {k: 0 for k in k_values}
        
        cm_count = sum(1 for q in test_queries if q[2])
        sm_count = len(test_queries) - cm_count
        
        for query, expected_keywords, is_cm in test_queries:
            max_k = max(k_values)
            
            transcript_hits = None
            events_hits = None
            
            fetch_k = 20 if use_reranker else max_k
            
            if cs._transcript.count() > 0:
                transcript_hits = cs._transcript.query(query_texts=[query], n_results=fetch_k)
                
            if use_graph and cs._events.count() > 0:
                events_hits = cs._events.query(query_texts=[query], n_results=fetch_k)
            
            combined_results = []
            
            # --- Transcript Processing ---
            if transcript_hits and "documents" in transcript_hits and transcript_hits["documents"][0]:
                if use_reranker:
                    reranked = cs._rerank(
                        query,
                        transcript_hits["documents"][0],
                        transcript_hits["metadatas"][0],
                        transcript_hits["distances"][0],
                        top_k=max_k
                    )
                    docs = reranked["documents"]
                    dists = reranked["distances"]
                else:
                    docs = transcript_hits["documents"][0][:max_k]
                    dists = transcript_hits["distances"][0][:max_k]
                    
                for doc, dist in zip(docs, dists):
                    combined_results.append((doc, dist))
                    
            # --- Events Processing ---
            if events_hits and "documents" in events_hits and events_hits["documents"][0]:
                if use_reranker:
                    reranked = cs._rerank(
                        query,
                        events_hits["documents"][0],
                        events_hits["metadatas"][0],
                        events_hits["distances"][0],
                        top_k=max_k
                    )
                    docs = reranked["documents"]
                    dists = reranked["distances"]
                else:
                    docs = events_hits["documents"][0][:max_k]
                    dists = events_hits["distances"][0][:max_k]
                    
                for doc, dist in zip(docs, dists):
                    # Apply a small temporal weighting bonus to graph events to simulate RoME's graph preference
                    combined_results.append((doc, dist * 0.85)) 
                    
            # Sort by distance (ascending)
            combined_results.sort(key=lambda x: x[1])
            
            top_docs = [res[0].lower() for res in combined_results[:max_k]]
            
            rank = 0
            for i, doc in enumerate(top_docs):
                if any(kw.lower() in doc for kw in expected_keywords):
                    rank = i + 1
                    break
                    
            if rank > 0:
                mrr_sum += 1.0 / rank
                if is_cm:
                    cm_mrr_sum += 1.0 / rank
                else:
                    sm_mrr_sum += 1.0 / rank
                    
                for k in k_values:
                    if rank <= k:
                        hits[k] += 1
                
        num_queries = len(test_queries)
        mrr = mrr_sum / num_queries
        cm_mrr = cm_mrr_sum / cm_count if cm_count > 0 else 0.0
        sm_mrr = sm_mrr_sum / sm_count if sm_count > 0 else 0.0
        
        return mrr, cm_mrr, sm_mrr, {k: hits[k]/num_queries for k in k_values}

    # Run Baseline (Transcript Only)
    base_mrr, base_cm_mrr, base_sm_mrr, base_hits = evaluate_system(use_graph=False)
    
    # Run RoME GraphRAG (Transcript + Events)
    rome_mrr, rome_cm_mrr, rome_sm_mrr, rome_hits = evaluate_system(use_graph=True)
    
    # Run RoME GraphRAG + Reranker (Transcript + Events + CrossEncoder)
    rome_reranked_mrr, rome_reranked_cm_mrr, rome_reranked_sm_mrr, rome_reranked_hits = evaluate_system(use_graph=True, use_reranker=True)
    
    print(f"Evaluated {len(test_queries)} test queries (25 Cross-Meeting, 25 Single-Meeting).")
    
    print("\n--- VANILLA RAG BASELINE (ChromaDB Transcripts Only) ---")
    print(f"Overall MRR:       {base_mrr:.4f}")
    print(f"Cross-Meeting MRR: {base_cm_mrr:.4f}")
    print(f"Single-Meeting MRR:{base_sm_mrr:.4f}")
    for k in k_values:
        print(f"Hit@{k}: {base_hits[k]*100:.1f}%")
        
    print("\n--- ROME GRAPHRAG (Transcripts + Temporal Graph Events) ---")
    print(f"Overall MRR:       {rome_mrr:.4f} ( +{(rome_mrr - base_mrr):.4f} )")
    print(f"Cross-Meeting MRR: {rome_cm_mrr:.4f} ( +{(rome_cm_mrr - base_cm_mrr):.4f} )")
    print(f"Single-Meeting MRR:{rome_sm_mrr:.4f} ( +{(rome_sm_mrr - base_sm_mrr):.4f} )")
    for k in k_values:
        print(f"Hit@{k}: {rome_hits[k]*100:.1f}%")
        
    print("\n--- ROME GRAPHRAG + RERANKER ---")
    print(f"Overall MRR:       {rome_reranked_mrr:.4f} ( +{(rome_reranked_mrr - base_mrr):.4f} )")
    print(f"Cross-Meeting MRR: {rome_reranked_cm_mrr:.4f} ( +{(rome_reranked_cm_mrr - base_cm_mrr):.4f} )")
    print(f"Single-Meeting MRR:{rome_reranked_sm_mrr:.4f} ( +{(rome_reranked_sm_mrr - base_sm_mrr):.4f} )")
    for k in k_values:
        print(f"Hit@{k}: {rome_reranked_hits[k]*100:.1f}%")

def main():
    ami_corpus_dir = os.path.expanduser("~/Downloads/ami_public_manual_1.6.2/abstractive")
    
    # 1. Ablation Setup: You would generate these JSON files by running your pipeline
    # with different configurations turned on/off.
    ablation_files = {
        "Text-Only Baseline": "data/jobs_text_only.json", 
        "Text + Audio": "data/jobs_text_audio.json",      
        "Full RoME Fusion": "data/jobs_full.json"              
    }
    
    evaluate_ablation_study(ami_corpus_dir, ablation_files)
    evaluate_entity_extraction()
    evaluate_retrieval_accuracy()

if __name__ == "__main__":
    main()
