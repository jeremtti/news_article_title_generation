from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, RobertaTokenizerFast, EncoderDecoderModel, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, Dataset
from evaluate import load

from peft import LoraConfig, TaskType, get_peft_model

import numpy as np
import torch
import pandas as pd


# Load the fine tuned trained model and its associated tokenizer
my_checkpoint = "my_model_no_summarization_barthez"
tokenizer = AutoTokenizer.from_pretrained(my_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(my_checkpoint)


def generate_summary(text):
    input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_new_tokens=200)
    generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_summary

test_df = pd.read_csv('../data/test_text.csv')

summaries = []
for idx, text in enumerate(test_df['text']):
    if idx % 10 == 0:
        print(f'Processing {idx}...')
    summaries.append([idx,generate_summary(text)])

submission_df = pd.DataFrame(summaries, columns=['ID', 'titles'])
submission_df.to_csv('fine_tuned_submission.csv', index=False)