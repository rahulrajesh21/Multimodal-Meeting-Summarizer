import os
import json
import xml.etree.ElementTree as ET
from calculate_metrics import simple_rouge_1, simple_rouge_l

def parse_ami_summary(xml_path):
    """Parses the AMI abstractive summary XML to extract the text."""
    if not os.path.exists(xml_path):
        return ""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        sentences = []
        for elem in root.findall(".//sentence"):
            if elem.text:
                sentences.append(elem.text.strip())
        return " ".join(sentences)
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return ""

def load_system_summaries(jobs_json_path):
    """Loads system generated summaries from jobs.json"""
    with open(jobs_json_path, "r") as f:
        jobs = json.load(f)
    
    meeting_summaries = {}
    for job_id, job_data in jobs.items():
        title = job_data.get("title", "")
        # e.g., "AMI Meeting ES2002a"
        if "AMI Meeting" in title:
            meeting_id = title.split()[-1]
            summaries_dict = job_data.get("summaries", {})
            # Combine all role summaries into one big summary for the meeting
            combined_summary = "\n".join(summaries_dict.values())
            meeting_summaries[meeting_id] = combined_summary
    return meeting_summaries

def main():
    print("Running AMI Evaluation against Real System Outputs...")
    
    ami_corpus_dir = os.path.expanduser("~/Downloads/ami_public_manual_1.6.2/abstractive")
    jobs_json_path = "data/jobs.json"
    
    system_summaries = load_system_summaries(jobs_json_path)
    
    if not system_summaries:
        print("No AMI meeting results found in data/jobs.json")
        return
    
    total_rouge_1 = 0
    total_rouge_l = 0
    count = 0
    
    for meeting_id, hyp_summary in system_summaries.items():
        xml_path = os.path.join(ami_corpus_dir, f"{meeting_id}.abssumm.xml")
        ref_summary = parse_ami_summary(xml_path)
        
        if not ref_summary:
            print(f"Warning: Ground truth not found for {meeting_id} at {xml_path}")
            continue
            
        r1 = simple_rouge_1(ref_summary, hyp_summary)
        rl = simple_rouge_l(ref_summary, hyp_summary)
        
        print(f"\n--- Meeting {meeting_id} ---")
        print(f"ROUGE-1: {r1:.2f}%")
        print(f"ROUGE-L: {rl:.2f}%")
        
        total_rouge_1 += r1
        total_rouge_l += rl
        count += 1
        
    if count > 0:
        print("\n=== FINAL AVERAGE SCORES ===")
        print(f"Average ROUGE-1: {total_rouge_1 / count:.2f}%")
        print(f"Average ROUGE-L: {total_rouge_l / count:.2f}%")

if __name__ == "__main__":
    main()
