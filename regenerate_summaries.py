import os
import json
import asyncio
from src.llm_summarizer import LLMSummarizer
from src.fusion_layer import FusionLayer
from src.text_analysis import TextAnalyzer
from src.temporal_graph_memory import TemporalGraphMemory

async def regenerate_summaries_async():
    print("Regenerating summaries with modified prompt...")
    jobs_path = "data/jobs.json"
    
    with open(jobs_path, "r") as f:
        jobs = json.load(f)
        
    summarizer = LLMSummarizer()
    if not summarizer.is_ready:
        print("LM Studio is not available. Please start LM Studio and try again.")
        return

    print("Initializing TextAnalyzer and FusionLayer...")
    ta = TextAnalyzer()
    tm = TemporalGraphMemory()
    fl = FusionLayer(text_analyzer=ta, temporal_memory=tm)

    for job_id, job in jobs.items():
        title = job.get("title", "")
        if "AMI Meeting" not in title:
            continue
            
        print(f"Processing {title}...")
        transcript = job.get("transcript", "")
        if not transcript:
            print(f"  No transcript found for {title}")
            continue
            
        # Reconstruct segments
        seg_dicts = []
        for i, line in enumerate(transcript.splitlines()):
            if line.strip():
                parts = line.split(":", 1)
                seg_dicts.append({
                    "speaker": parts[0].strip() if len(parts) > 1 else "Unknown",
                    "text": parts[-1].strip(),
                    "start": i * 5.0,
                    "end": (i + 1) * 5.0,
                })

        # Process like in the main app
        scored = fl.score_segments_contextual(seg_dicts, role="Attendee", use_ml=False)
        
        # Take the top 30 highest-scored segments to summarize (to keep context small)
        top_segs = sorted(scored, key=lambda s: s.fused_score, reverse=True)[:30]
        top_segs = sorted(top_segs, key=lambda s: s.start_time)
        
        text_block = "\n".join(
            f"[{s.start_time:.1f}s] {getattr(s, 'speaker', 'Unknown')}: {s.text}" for s in top_segs
        )
        
        try:
            new_summary = summarizer.summarize(text=text_block, role="General", focus="Main topics and decisions", tm=tm)
            job["summaries"] = {
                "General": new_summary
            }
            print(f"  Successfully generated new summary for {title}")
        except Exception as e:
            print(f"  Failed to summarize {title}: {e}")

    # Save back to jobs.json
    with open(jobs_path, "w") as f:
        json.dump(jobs, f, indent=2)
        
    print("Done regenerating summaries.")

def regenerate_summaries():
    asyncio.run(regenerate_summaries_async())

if __name__ == "__main__":
    regenerate_summaries()