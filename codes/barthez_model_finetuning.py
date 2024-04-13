from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, RobertaTokenizerFast, EncoderDecoderModel, T5Tokenizer, T5ForConditionalGeneration, AutoConfig, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from evaluate import load

from peft import LoraConfig, TaskType, get_peft_model

import numpy as np
import torch
import pandas as pd

# we use the specific AutoModel class suited for sequence to sequence tasks
# we use the Barthez model (French specific)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = "moussaKam/barthez-orangesum-title"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, num_beams = 2) #since early_stopping = True, we need to set num_beans > 1 

# we take the tokenizer corresponding to our pre trained model (otherwise compatibility issues)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# in a same HuggingFace dataset object, we store both our train and our validation file
dataset = load_dataset("csv", data_files={"train": "../data/train.csv", "valid": "../data/validation.csv"})


prefix = "" # always empty except for the t5 model that requires prefix = "summarize :" since it is a fully text to text model (nothing else than text)

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

    labels = tokenizer(text_target=examples["titles"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

rouge = load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# since it is a sequence to sequence task, we cannot use only TrainingArguments but we need to use Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="my__model_barthez",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
my_checkpoint = "my_model_no_summarization_barthez"
trainer.save_model(my_checkpoint)