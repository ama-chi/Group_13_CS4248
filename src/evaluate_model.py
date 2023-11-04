import json
import torch
from transformers import pipeline

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


batch_size = 64
file_path = "../scripts/evaluation/dev-v1.1.json"
prediction_path = "../scripts/evaluation/pred.json"

device = 0 if torch.cuda.is_available() else -1

question_answerer = pipeline(task="question-answering", device=device, model="./local_model")
# question_answerer = pipeline(task="question-answering", device=device)

print("Loading evaluation set from {}...".format(file_path))
question_id, question_context_pair = load_evaluation_set(file_path)

print("Evaluating with batch size {}...".format(batch_size))
predictions = question_answerer(question_context_pair, batch_size=batch_size)

print("Done evaluating. Saving predictions to {}...".format(prediction_path))
save_pred_json(question_id, predictions, prediction_path)