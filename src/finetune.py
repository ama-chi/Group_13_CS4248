import torch
import numpy as np
import argparse

from datasets import load_dataset

from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer

SEED = 42

torch.manual_seed(42)
np.random.seed(42)

def train(args):
    model = AutoModelForQuestionAnswering.from_pretrained(args.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME)

    squad = load_dataset("squad")

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    train_dataset = squad["train"].map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
    test_dataset = squad["validation"].map(preprocess_function, batched=True, remove_columns=squad["validation"].column_names)

    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir=f"finetune_{args.MODEL_NAME}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        num_train_epochs=3,
        seed=SEED,
        weight_decay=0.01,
        push_to_hub=False,
        save_total_limit = 1,
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MODEL_NAME",
        type=str,
        required=True,
        help="Model name to use for training. Default is 'distilbert-base-uncased'."
    )

    args = parser.parse_args()

    print(f"***Training with \"{args.MODEL_NAME}\" ***")

    train(args)