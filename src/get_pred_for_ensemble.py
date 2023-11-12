# Example usage
# python example_pred_for_ensemble.py \
# --dev_json_path ../scripts/evaluation/dev-v1.1.json \
# --model_path "diffuserconfuser/squad-albert-base-v2" 

import os
import argparse
import json
from collections import OrderedDict

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
        "--model_path",
        type=str,
        required=True,
        help="Path to the local model directory."
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

def load_test_dict(dev_json_path) -> OrderedDict:
    test_dict = OrderedDict()

    with open(dev_json_path) as f:
        dataset = json.load(f)['data']
        
        for article in dataset:
            for p in article['paragraphs']:
                for qa in p['qas']:
                    test_dict[qa['id']] = {"question": qa['question'], "context": p["context"]}
    
    return test_dict 

def save_pred_json(test_dict, predictions, pred_json_path):
    for qid, pred in zip(test_dict.keys(), predictions):
        test_dict[qid].update(pred)

    with open(pred_json_path, 'w') as f:
        json.dump(test_dict, f)


def evaluate(args):
    device = 0 if torch.cuda.is_available() else -1

    # For registered models on HuggingFace
    question_answerer = pipeline(task="question-answering", device=device, model=args.model_path)
    
    # For custom model
    # from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    # from custom_models.bert import BertCustomConfig, BertCustomDense

    # AutoModelForQuestionAnswering.register(BertCustomConfig, BertCustomDense)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # model = BertCustomDense.from_pretrained(args.model_path)
    # question_answerer = pipeline(task="question-answering", device=device, model=model, tokenizer=tokenizer)


    print("Loading evaluation set from {}...".format(args.dev_json_path))
    test_dict = load_test_dict(args.dev_json_path)

    print("Evaluating with batch size {}...".format(args.batch_size))
    predictions = question_answerer(test_dict.values(), batch_size=args.batch_size)

    output_path = f"{os.path.basename(args.model_path)}_preds.json"
    print("Done evaluating. Saving predictions to {}...".format(output_path))
    save_pred_json(test_dict, predictions, output_path)

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
