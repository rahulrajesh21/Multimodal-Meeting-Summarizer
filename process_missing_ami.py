import sys
import os
import json
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from api.main import process_meeting_job, process_teams_job, _jobs, _save_jobs

async def process_specific_jobs():
    target_meetings = ["ES2009d", "ES2010a", "ES2010b", "ES2010c", "ES2010d"]
    
    # reset their status
    job_ids_to_process = []
    for job_id, job_data in _jobs.items():
        title = job_data.get("title", "")
        if any(m in title for m in target_meetings):
            print(f"Resetting {job_id} ({title})")
            _jobs[job_id]["status"] = "pending"
            _jobs[job_id]["progress"] = 0
            _jobs[job_id]["stage"] = "Pending"
            job_ids_to_process.append(job_id)
            
    _save_jobs()
    
    for job_id in job_ids_to_process:
        title = _jobs[job_id].get("title", "")
        print(f"Processing {job_id} ({title})...")
        if "teams_meeting_id" in _jobs[job_id]:
            await process_teams_job(job_id)
        else:
            await process_meeting_job(job_id)
            
    # Then generate summaries
    import regenerate_summaries
    print("Regenerating summaries...")
    regenerate_summaries.regenerate_summaries()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(process_specific_jobs())
