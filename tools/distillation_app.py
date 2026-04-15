import streamlit as st
import pandas as pd
import json
import requests
import os
import time
import torch
import shutil
from typing import List

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "topic_classifier")
DATA_DIR = os.path.join(PROJECT_ROOT, "tools", "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "distillation_dataset.csv")

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
LM_STUDIO_ENDPOINT = "http://localhost:1234/v1/chat/completions"

CLASSES = ["decision", "discussion", "idea", "problem", "risk", "update"]

st.set_page_config(page_title="Topic Classifier Distillation", page_icon="🧠", layout="wide")


# --- STEP 1: DATA GENERATION ---

def generate_samples(topic_class: str, count: int, backend: str, model_name: str) -> List[str]:
    """Ask Ollama or LM Studio to generate extremely varied, realistic synthetic meeting utterances."""
    prompt = f"""
You are an expert at creating highly realistic simulated data for AI training.
Generate exactly {count} distinct sentences or short paragraphs that represent a "{topic_class}" occurring during a business meeting.

Topic definition:
- decision: Formally agreeing to a path forward, voting, or setting a final choice.
- discussion: General back-and-forth conversation, brainstorming without a finalized idea.
- idea: Proposing a brand new concept, feature, or solution.
- problem: Highlighting a bug, an issue, a blocker, or something that is going wrong.
- risk: Highlighting a potential future issue, dependency failure, or compliance concern.
- update: Giving a status report, sprint update, or informational summary of past work.

Rules:
1. Make them sound like real people talking in a Zoom meeting (use filler words occasionally, informal tone, corporate jargon).
2. Vary the length from 1 sentence to 3 sentences.
3. Output ONLY a valid JSON array of strings, nothing else. Example: ["sentence 1", "sentence 2"]
"""
    try:
        text = ""
        if backend == "Ollama":
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.8}
            }
            res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=None)
            res.raise_for_status()
            text = res.json().get("response", "[]").strip()
        else:
            # LM Studio (OpenAI Compatible)
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "stream": False
            }
            res = requests.post(LM_STUDIO_ENDPOINT, json=payload, timeout=None)
            res.raise_for_status()
            text = res.json().get("choices", [{}])[0].get("message", {}).get("content", "[]").strip()
        
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            raw_array = match.group(0)
            try:
                data = json.loads(raw_array)
                if isinstance(data, dict) and "samples" in data:
                    return data["samples"]
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError as e:
                st.warning(f"Failed to parse JSON for {topic_class}: {e}. Raw text: {raw_array}")
        else:
            st.warning(f"No JSON array brackets found for {topic_class}. Raw text: {text}")
            
        return []
    except Exception as e:
        st.error(f"Failed to generate for {topic_class}: {e}")
        return []


# --- UI ---

st.title("🧠 Topic Classifier Distillation")
st.markdown("""
Train a lightning-fast **Small BERT** model to replace the LLM for topic classification!
Pipeline: `LLM (Ollama) -> Synthetic Data -> Fine-tune DistilBERT -> Fast Inference`
""")

tab1, tab2, tab3 = st.tabs(["1. Generate Dataset", "2. Train DistilBERT", "3. Test Classifier"])

with tab1:
    st.header("Step 1: Generate Synthetic Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        samples_per_class = st.number_input("Samples per class", min_value=10, max_value=600, value=30)
    with col2:
        backend_choice = st.selectbox("LLM Backend", ["LM Studio", "Ollama"])
    with col3:
        if backend_choice == "Ollama":
            user_model = st.text_input("Model Name", "qwen3:4b")
        else:
            user_model = st.text_input("Model Name", "local-model")
    
    if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 0:
        try:
            df_existing = pd.read_csv(DATASET_PATH)
            st.success(f"Found existing dataset with {len(df_existing)} rows.")
            st.dataframe(df_existing.head())
        except pd.errors.EmptyDataError:
            pass
        
    if st.button("Generate Dataset via " + backend_choice, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_data = []
        for i, c in enumerate(CLASSES):
            status_text.text(f"Generating {samples_per_class} samples for '{c}' via {backend_choice}...")
            samples = generate_samples(c, samples_per_class, backend_choice, user_model)
            for s in samples:
                all_data.append({"text": s, "label": c})
            progress_bar.progress((i + 1) / len(CLASSES))
            
        df = pd.DataFrame(all_data)
        df.to_csv(DATASET_PATH, index=False)
        status_text.text("Generation complete!")
        st.success(f"Saved {len(df)} total samples to {DATASET_PATH}")
        st.dataframe(df)

with tab2:
    st.header("Step 2: Train DistilBERT")
    st.markdown("Fine-tunes `distilbert-base-uncased` on your generated dataset using your Mac's **MPS GPU**.")
    
    epochs = st.slider("Epochs", 1, 10, 3)
    
    if st.button("Start Fine-Tuning", type="primary"):
        if not os.path.exists(DATASET_PATH):
            st.error("Please generate the dataset first in Tab 1!")
        else:
            with st.spinner("Initializing Training Loop... (this will output to terminal)"):
                try:
                    import transformers
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
                    from datasets import Dataset
                    import numpy as np
                    import evaluate
                    
                    # 1. Load Data
                    df = pd.read_csv(DATASET_PATH)
                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                    
                    # Clean: remove any duplicate header rows or unmapped labels
                    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
                    
                    label2id = {c: i for i, c in enumerate(CLASSES)}
                    id2label = {i: c for i, c in enumerate(CLASSES)}
                    df["label_id"] = df["label"].map(label2id).astype(int)
                    
                    hg_dataset = Dataset.from_pandas(df)
                    hg_dataset = hg_dataset.train_test_split(test_size=0.2, seed=42)
                    
                    # 2. Tokenize
                    model_id = "distilbert-base-uncased"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    
                    def tokenize_function(examples):
                        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
                    
                    tokenized_datasets = hg_dataset.map(tokenize_function, batched=True)
                    # Rename label_id to labels for Trainer
                    tokenized_datasets = tokenized_datasets.rename_column("label_id", "labels")
                    tokenized_datasets = tokenized_datasets.remove_columns(["text", "label"])
                    tokenized_datasets.set_format("torch")
                    
                    # 3. Model
                    device = "mps" if torch.backends.mps.is_available() else "cpu"
                    st.info(f"Using device: `{device}`")
                    
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_id, 
                        num_labels=len(CLASSES),
                        id2label=id2label,
                        label2id=label2id
                    )
                    
                    metric = evaluate.load("accuracy")
                    def compute_metrics(eval_pred):
                        logits, labels = eval_pred
                        predictions = np.argmax(logits, axis=-1)
                        return metric.compute(predictions=predictions, references=labels)
                    
                    # 4. Training Args
                    training_args = TrainingArguments(
                        output_dir=os.path.join(DATA_DIR, "results"),
                        eval_strategy="epoch",
                        save_strategy="epoch",
                        learning_rate=2e-5,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=16,
                        num_train_epochs=epochs,
                        weight_decay=0.01,
                        load_best_model_at_end=True,
                        logging_steps=10,
                        skip_memory_metrics=True,
                    )
                    
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_datasets["train"],
                        eval_dataset=tokenized_datasets["test"],
                        processing_class=tokenizer,
                        compute_metrics=compute_metrics,
                    )
                    
                    # Train
                    st.text("Training started! Check the terminal for progress bars.")
                    trainer.train()
                    
                    # Save
                    st.text("Training complete. Saving model...")
                    trainer.save_model(MODELS_DIR)
                    tokenizer.save_pretrained(MODELS_DIR)
                    
                    # Clean up
                    shutil.rmtree(os.path.join(DATA_DIR, "results"), ignore_errors=True)
                    
                    eval_results = trainer.evaluate()
                    st.success(f"Model saved to `{MODELS_DIR}`!")
                    st.json(eval_results)
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with tab3:
    st.header("Step 3: Interactive Evaluation")
    st.markdown("Test the fine-tuned model live.")
    
    test_text = st.text_area("Enter a meeting transcript line:", "I think we should migrate the database to Postgres next sprint.")
    
    if st.button("Predict Topic", type="primary"):
        if not os.path.exists(os.path.join(MODELS_DIR, "config.json")):
            st.error("No trained model found! Please train it in Tab 2 first.")
        else:
            with st.spinner("Loading model and computing..."):
                from transformers import pipeline
                device_id = 0 if torch.backends.mps.is_available() else -1
                if torch.backends.mps.is_available():
                     device_id = "mps"
                
                # Load pipeline
                classifier = pipeline("text-classification", model=MODELS_DIR, tokenizer=MODELS_DIR, device=device_id)
                
                t0 = time.time()
                result = classifier(test_text, top_k=6)
                t1 = time.time()
                
                top_pred = result[0]  # List of dicts, sorted by score
                
                st.write(f"**Predicted Topic:** `{top_pred['label']}`")
                st.write(f"**Confidence:** `{top_pred['score']*100:.1f}%`")
                st.write(f"**Latency:** `{(t1-t0)*1000:.1f} ms`")
                
                st.json(result)
