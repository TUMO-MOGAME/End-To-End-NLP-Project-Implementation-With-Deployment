from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, load_dataset
import torch
from textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        #loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, # output directory
            num_train_epochs=self.config.num_train_epochs, # total number of training epochs
            warmup_steps=self.config.warmup_steps, # number of warmup steps for learning rate scheduler
            per_device_train_batch_size=self.config.per_device_train_batch_size, # batch size for training
            weight_decay=self.config.weight_decay, # strength of weight decay
            logging_steps=self.config.logging_steps, # number of steps for logging
            eval_strategy=self.config.evaluation_strategy, # evaluation strategy to adopt during training
            eval_steps=self.config.eval_steps, # number of steps for evaluation
            save_steps=self.config.save_steps, # number of steps for saving the model
            gradient_accumulation_steps=self.config.gradient_accumulation_steps, # number of steps for gradient accumulation
            per_device_eval_batch_size=self.config.per_device_eval_batch_size, # batch size for evaluation
            load_best_model_at_end=self.config.load_best_model_at_end, # whether to load the best model at the end of training
            greater_is_better=self.config.greater_is_better, # whether a greater metric value is better
            metric_for_best_model=self.config.metric_for_best_model, # metric to use for best model selection
            save_total_limit=self.config.save_total_limit, # maximum number of checkpoints to save
            report_to=self.config.report_to, # reporting destination
            push_to_hub=self.config.push_to_hub, # whether to push the model to the hub
            hub_model_id=self.config.hub_model_id if self.config.push_to_hub else None # model id for the hub
        )
        trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["train"], 
                  eval_dataset=dataset_samsum_pt["validation"])
        trainer.train()
        trainer.evaluate(dataset_samsum_pt["validation"])
        trainer.save_model(self.config.root_dir)
        trainer.push_to_hub()
        tokenizer.save_pretrained(self.config.root_dir)
        model_pegasus.save_pretrained(self.config.root_dir)
