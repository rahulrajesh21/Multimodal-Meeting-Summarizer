import requests
import json
import time

TEAMS_SERVER = "http://localhost:8001"
ROME_API = "http://localhost:8000"

def import_all_teams_meetings():
    print("Fetching meetings from Teams Media Server...")
    response = requests.get(f"{TEAMS_SERVER}/v1.0/me/onlineMeetings")
    if response.status_code != 200:
        print("Failed to fetch from Teams server")
        return
    
    meetings = response.json().get("value", [])
    print(f"Found {len(meetings)} meetings on Teams server.")
    
    # Get currently queued/processed jobs in RoME
    try:
        rome_jobs = requests.get(f"{ROME_API}/api/meetings").json()
        existing_teams_ids = [j.get("teams_meeting_id") for j in rome_jobs if "teams_meeting_id" in j]
    except Exception as e:
        print(f"Failed to fetch RoME jobs: {e}. Is the RoME API running?")
        return

    for m in meetings:
        mid = m["id"]
        subject = m.get("subject", "")
        
        if mid in existing_teams_ids:
            print(f"Skipping {subject} (already imported)")
            continue
            
        print(f"Triggering import for {subject} ({mid})...")
        payload = {
            "teams_meeting_id": mid
        }
        res = requests.post(f"{ROME_API}/api/meetings/process-teams", json=payload)
        if res.status_code == 200:
            print(f"  Successfully queued.")
        else:
            print(f"  Failed to queue: {res.text}")
            
if __name__ == "__main__":
    import_all_teams_meetings()
