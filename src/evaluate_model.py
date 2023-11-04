import os
import argparse
import json

import torch
from transformers import pipeline

def parse_args():
    def check_valid_path(value):
        if not os.path.exists(value):
            raise argparse.ArgumentTypeError(f"The path {value} does not exist")
        return value
    
    def check_json_extension(value):
        check_valid_path(os.path.dirname(value))
        base, ext = os.path.splitext(os.path.basename(value))
        if ext.lower() != '.json':
            raise argparse.ArgumentTypeError("Output file must have a .json extension")
        return value

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run QA model inference.")

    # Add arguments to the parser
    parser.add_argument(
        "--dev_json_path",
        type=check_json_extension,
        required=True,
        help="Path to the evaluation dataset JSON file. Default is '../scripts/evaluation/dev-v1.1.json'."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for processing. Default is 64."
    )
    parser.add_argument(
        "--output_path",
        type=check_json_extension,
        required=True,
        help="Path where predictions should be saved."
    )
    parser.add_argument(
        "--model_path",
        type=check_valid_path,
        required=True,
        help="Path to the local model directory."
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

def load_evaluation_set(dev_json_path):
    question_id = []
    question_context_pair = []

    with open(dev_json_path) as f:
        dataset = json.load(f)['data']
        
        for article in dataset:
            for p in article['paragraphs']:
                for qa in p['qas']:
                    question_id.append(qa['id'])
                    question_context_pair.append({"question": qa['question'], "context": p["context"]})
    return question_id, question_context_pair

def save_pred_json(question_id, predictions, pred_json_path):
    pred_json = {}

    for qid, pred in zip(question_id, predictions):
        pred_json[qid] = pred['answer']

    with open(pred_json_path, 'w') as f:
        json.dump(pred_json, f)


def evaluate(args):
    device = 0 if torch.cuda.is_available() else -1

    question_answerer = pipeline(task="question-answering", device=device, model=args.model_path)

    print("Loading evaluation set from {}...".format(args.dev_json_path))
    question_id, question_context_pair = load_evaluation_set(args.dev_json_path)

    print("Evaluating with batch size {}...".format(args.batch_size))
    predictions = question_answerer(question_context_pair, batch_size=args.batch_size)

    print("Done evaluating. Saving predictions to {}...".format(args.output_path))
    save_pred_json(question_id, predictions, args.output_path)

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)