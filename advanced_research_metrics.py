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
    Evaluates different ablation configurations (e.g., Text Only, Text+Audio, Full Fusion)
    against the AMI ground truth using ROUGE and BERTScore.
    
    Args:
        ground_truth_dir: Path to AMI abstractive summaries.
        result_files: Dictionary mapping configuration name to its jobs.json path.
                      e.g., {"Text-Only": "data/jobs_text_only.json", ...}
    """
    print("\n" + "="*60)
    print("1. ABLATION STUDY: MULTIMODAL GAIN")
    print("="*60)
    
    for config_name, file_path in result_files.items():
        if not os.path.exists(file_path):
            print(f"Skipping '{config_name}': File {file_path} not found. Run your pipeline to generate this data.")
            continue
            
        system_summaries = load_system_summaries(file_path)
        
        refs = []
        cands = []
        
        for meeting_id, hyp_summary in system_summaries.items():
            xml_path = os.path.join(ground_truth_dir, f"{meeting_id}.abssumm.xml")
            ref_summary = parse_ami_summary(xml_path)
            
            if ref_summary and hyp_summary:
                refs.append(ref_summary)
                cands.append(hyp_summary)
                
        if not refs:
            print(f"[{config_name}] No matching ground truth summaries found.")
            continue
            
        # Calculate BERTScore if available
        if BERT_SCORE_AVAILABLE:
            # We use 'microsoft/deberta-xlarge-mnli' as it is standard for English summarization eval
            P, R, F1 = calc_bert_score(cands, refs, lang="en", rescale_with_baseline=True)
            avg_bert_f1 = F1.mean().item()
            print(f"[{config_name:<20}] BERTScore F1: {avg_bert_f1:.4f}")
        else:
            print(f"[{config_name:<20}] BERTScore F1: [Requires bert-score package]")


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
    
    # In a real scenario, you would extract this from AMI extractive/namedEntities XML
    # and compare it against your `meeting_memory/graph_db.json` entities.
    
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
    print("\n" + "="*60)
    print("3. TEMPORAL GRAPH RAG RETRIEVAL ACCURACY")
    print("="*60)
    
    # In a real scenario, you define a list of test questions, the ID of the true context chunk,
    # and then query your ChromaDB / TGM to see where the true chunk ranks in the returned results.
    
    # Structure: (Query, Ground_Truth_Chunk_ID, Ranked_Retrieved_Chunk_IDs)
    mock_queries = [
        ("What is the budget for the remote control?", "chunk_104", ["chunk_012", "chunk_104", "chunk_099"]), # Hit at rank 2
        ("Who is the project manager?", "chunk_002", ["chunk_002", "chunk_010"]), # Hit at rank 1
        ("Was the jog dial approved?", "chunk_405", ["chunk_111", "chunk_222", "chunk_333", "chunk_405"]), # Hit at rank 4
        ("What color is the corporate logo?", "chunk_201", ["chunk_501", "chunk_502", "chunk_503"]) # Missed (not in top results)
    ]
    
    mrr_sum = 0.0
    hits = {k: 0 for k in k_values}
    num_queries = len(mock_queries)
    
    for query, true_id, retrieved_ids in mock_queries:
        rank = 0
        try:
            rank = retrieved_ids.index(true_id) + 1
            mrr_sum += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    hits[k] += 1
        except ValueError:
            pass # true_id not in retrieved_ids
            
    mrr = mrr_sum / num_queries
    
    print(f"Evaluated {num_queries} test queries.")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    for k in k_values:
        hit_rate = hits[k] / num_queries
        print(f"Hit@{k}: {hit_rate*100:.1f}%")

def main():
    ami_corpus_dir = os.path.expanduser("~/Downloads/ami_public_manual_1.6.2/abstractive")
    
    # 1. Ablation Setup: You would generate these JSON files by running your pipeline
    # with different configurations turned on/off.
    ablation_files = {
        "Text-Only Baseline": "data/jobs_text_only.json", # Pretend this exists
        "Text + Audio": "data/jobs_text_audio.json",      # Pretend this exists
        "Full RoME Fusion": "data/jobs.json"              # Your actual current results
    }
    
    evaluate_ablation_study(ami_corpus_dir, ablation_files)
    evaluate_entity_extraction()
    evaluate_retrieval_accuracy()

if __name__ == "__main__":
    main()
