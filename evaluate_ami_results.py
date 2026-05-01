import os
import json
import re
import xml.etree.ElementTree as ET
from rouge_score import rouge_scorer

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
        return clean_summary_text(" ".join(sentences))
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return ""

def clean_summary_text(text):
    """Removes highly structured formatting and role-specific markers to prevent artificial metric penalization."""
    # Remove markers like [Attendee], [Project Manager], etc.
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove bold markers like **Summary (Attendee Perspective):** or **Action Item:**
    text = re.sub(r'\*\*(.*?)\*\*', '', text)
    
    # Remove timestamps like (AMI Meeting ES2002a, Speaker_B, 0:55)
    text = re.sub(r'\(AMI Meeting.*?\)', '', text)
    
    # Remove emojis and specific text like ✅ DECIDED or ⚠️ UNRESOLVED
    text = re.sub(r'[✅⚠️]\s*(DECIDED|UNRESOLVED)', '', text)
    
    # Remove bullet points and dashes
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

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
            seen = set()
            sentences = []
            for role_text in summaries_dict.values():
                cleaned = clean_summary_text(role_text)
                for sent in cleaned.split('.'):
                    sent = sent.strip()
                    if sent and sent not in seen:
                        seen.add(sent)
                        sentences.append(sent)
            combined_summary = '. '.join(sentences)
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
    total_rouge_2 = 0
    total_rouge_l = 0
    count = 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                       use_stemmer=True)
    
    for meeting_id, hyp_summary in system_summaries.items():
        xml_path = os.path.join(ami_corpus_dir, f"{meeting_id}.abssumm.xml")
        ref_summary = parse_ami_summary(xml_path)
        
        if not ref_summary:
            print(f"Warning: Ground truth not found for {meeting_id} at {xml_path}")
            continue
            
        if not hyp_summary.strip():
            print(f"Warning: Empty generated summary for {meeting_id}, skipping.")
            continue
        scores = scorer.score(ref_summary, hyp_summary)
        
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure
        print(f"\n--- Meeting {meeting_id} ---")
        print(f"  ROUGE-1 F1: {r1:.4f}")
        print(f"  ROUGE-2 F1: {r2:.4f}")
        print(f"  ROUGE-L F1: {rl:.4f}")
        
        # DEBUG: print first 200 chars of each to verify alignment
        print(f"  [REF]:  {ref_summary[:200]}")
        print(f"  [HYP]:  {hyp_summary[:200]}")
        total_rouge_1 += r1
        total_rouge_2 += r2
        total_rouge_l += rl
        count += 1
        
    if count > 0:
        print("\n=== FINAL AVERAGE SCORES ===")
        print(f"Meetings evaluated: {count}")
        print(f"Average ROUGE-1 F1: {total_rouge_1 / count:.4f}")
        print(f"Average ROUGE-2 F1: {total_rouge_2 / count:.4f}")
        print(f"Average ROUGE-L F1: {total_rouge_l / count:.4f}")

if __name__ == "__main__":
    main()