import sys
import os
import json
import asyncio
from pathlib import Path
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from api.main import process_meeting_job, process_teams_job, _jobs, _save_jobs
import src.fusion_layer as fl

def reset_jobs():
    for job_id in _jobs:
        _jobs[job_id]["status"] = "pending"
        _jobs[job_id]["progress"] = 0
        _jobs[job_id]["stage"] = "Pending"
    _save_jobs()

async def run_jobs(output_file):
    reset_jobs()
    for job_id in list(_jobs.keys()):
        if "AMI Meeting" in _jobs[job_id].get("title", ""):
            print(f"Processing {job_id}...")
            if "teams_meeting_id" in _jobs[job_id]:
                await process_teams_job(job_id)
            else:
                await process_meeting_job(job_id)
            
    # Then generate summaries
    import regenerate_summaries
    regenerate_summaries.regenerate_summaries()

    # Move jobs.json to output_file
    shutil.copy("data/jobs.json", output_file)
    print(f"Saved {output_file}")

async def main():
    original_fuse = fl.FusionLayer._fuse_with_weights
    
    # --- TEXT ONLY RUN ---
    print("=== RUNNING TEXT-ONLY ABLATION ===")
    def text_only_fuse(self, semantic_score, tonal_score, role_relevance, temporal_score, recurrence_score, unresolved_score, weights):
        # Force tonal to 0, use text + role + temporal
        tonal_score = 0.0
        new_weights = weights.copy()
        new_weights['tonal'] = 0.0
        # Rebalance
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        return original_fuse(self, semantic_score, tonal_score, role_relevance, temporal_score, recurrence_score, unresolved_score, new_weights)
    
    fl.FusionLayer._fuse_with_weights = text_only_fuse
    await run_jobs("data/jobs_text_only.json")
    
    # --- TEXT + AUDIO RUN ---
    print("=== RUNNING TEXT+AUDIO ABLATION ===")
    fl.FusionLayer._fuse_with_weights = original_fuse
    # We will assume original includes audio, but since we disabled visual (which does nothing for score), this IS Text+Audio.
    await run_jobs("data/jobs_text_audio.json")

if __name__ == "__main__":
    asyncio.run(main())
