import os
import json
import asyncio
import shutil
from src.llm_summarizer import LLMSummarizer
from src.fusion_layer import FusionLayer
from src.text_analysis import TextAnalyzer
from src.temporal_graph_memory import TemporalGraphMemory

async def run_fast_ablation():
    print("Running fast ablation study...")
    source_jobs = "data/jobs.json" # the base with raw transcripts
    
    with open(source_jobs, "r") as f:
        jobs = json.load(f)
        
    summarizer = LLMSummarizer()
    ta = TextAnalyzer()
    tm = TemporalGraphMemory()
    
    # 1. Full RoME
    print("\n--- Full RoME ---")
    fl_full = FusionLayer(text_analyzer=ta, temporal_memory=tm)
    fl_full.fusion_strategy = "weighted"
    fl_full.weights = {
        'semantic': 0.70,
        'tonal': 0.30,
        'role': 0.0,
        'temporal': 0.10,
        'recurrence': 0.05,
        'unresolved': 0.05
    }
    jobs_full = json.loads(json.dumps(jobs))
    for job_id, job in jobs_full.items():
        if "AMI Meeting" not in job.get("title", ""): continue
        transcript = job.get("transcript", "")
        if not transcript: continue
        seg_dicts = []
        for i, line in enumerate(transcript.splitlines()):
            if line.strip():
                parts = line.split(":", 1)
                seg_dicts.append({"speaker": parts[0].strip() if len(parts) > 1 else "Unknown", "text": parts[-1].strip(), "start": i * 5.0, "end": (i + 1) * 5.0})
        scored = fl_full.score_segments(seg_dicts, role="Attendee")
        top_segs = sorted(scored, key=lambda s: s.fused_score, reverse=True)[:30]
        top_segs = sorted(top_segs, key=lambda s: s.start_time)
        text_block = "\n".join(f"[{s.start_time:.1f}s] {getattr(s, 'speaker', 'Unknown')}: {s.text}" for s in top_segs)
        new_summary = summarizer.summarize(text=text_block, role="General", focus="Main topics and decisions", tm=tm)
        job["summaries"] = {"General": new_summary}
    with open("data/jobs_full.json", "w") as f:
        json.dump(jobs_full, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    asyncio.run(run_fast_ablation())

