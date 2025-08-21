from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch
import pandas as pd
from tqdm import tqdm
import evaluate
import json
import os
from pathlib import Path
from textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously.
        This is a utility function that will return a list of batches.
        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric(self, dataset, metric, model, tokenizer):
        dialogue_batches = list(self.generate_batch_sized_chunks(dataset["dialogue"], batch_size=16))
        summary_batches = list(self.generate_batch_sized_chunks(dataset["summary"], batch_size=16))
        for dialogue_batch, summary_batch in tqdm(
            zip(dialogue_batches, summary_batches), total=len(dialogue_batches)
        ):
            inputs = tokenizer(dialogue_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt"
            )
            summaries = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128, num_beams=8, length_penalty=2.0, early_stopping=True)
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
            metric.add_batch(predictions=decoded_summaries, references=summary_batch)
        score = metric.compute()
        return score
    
    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer - try from trained model first, fallback to original model
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        except:
            print(f"Could not load tokenizer from {self.config.tokenizer_path}, using original model")
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        
        # Load model - try from trained model first, fallback to original model
        try:
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        except:
            print(f"Could not load model from {self.config.model_path}, using original model")
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail").to(device)
        
        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # Use test split for evaluation
        test_dataset = dataset_samsum_pt["test"]
        
        # Load ROUGE metric using evaluate library
        rouge_metric = evaluate.load("rouge")
        
        # Calculate metrics
        score = self.calculate_metric(test_dataset, rouge_metric, model_pegasus, tokenizer)
        
        # Extract ROUGE scores
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, score[rn]) for rn in rouge_names)
        
        # Save results
        df = pd.DataFrame(rouge_dict, index=[0])
        metric_file_path = os.path.join(self.config.root_dir, self.config.metric_file_name)
        df.to_csv(metric_file_path, index=False)
        
        # Also save as JSON for easier reading
        json_file_path = os.path.join(self.config.root_dir, "rouge_scores.json")
        with open(json_file_path, 'w') as f:
            json.dump(rouge_dict, f, indent=2)
        
        print(f"Evaluation completed. Results saved to {metric_file_path} and {json_file_path}")
        print("ROUGE Scores:")
        for key, value in rouge_dict.items():
            print(f"{key}: {value:.4f}")
        
        return rouge_dict
