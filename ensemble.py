# TODO
from datasets import load_dataset
import csv
import random
import argparse
import json
import os
from datasets import load_metric

def parse_arguments():
    """
    Usage: python ensemble.py -input_csv model_data.csv (-p) -output_path results.json
    """
    def check_json_extension(value):
        base, ext = os.path.splitext(os.path.basename(value))
        if ext.lower() != '.json':
            raise argparse.ArgumentTypeError("Output file must have a .json extension")
        return value
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_csv', required=True, help="Path to the csv file which includes the values of models in google sheets")
    parser.add_argument('-p', action="store_true", help="Choose model predictions based on probability")
    parser.add_argument('-output_path', required=True, type=check_json_extension ,help="Path to the output json file for results")
    return parser.parse_args()

# Define a function to categorize questions
def naive_categorize_question(example):
    question = example['question'].lower()
    if 'date' in question:
        return 'date'
    if 'during' in question:
        return 'during'
    if 'how are' in question:
        return 'how are'
    if 'how big' in question or 'how large' in question:
        return 'how big/size'
    if 'how many' in question or 'how much' in question:
        return 'how m/m'
    if 'how old' in question:
        return 'how old'
    if 'what time' in question:
        return 'what time'
    if 'what' in question or 'which' in question:
        return 'what'
    if 'when' in question:
        return 'when'
    if 'where' in question:
        return 'where'
    if 'who' in question:
        return 'who'
    if 'whom' in question:
        return 'whom'
    if 'why' in question:
        return 'why'
    else:
        return 'undefined'
    

def ensemble(input_csv, is_probabilisitc, output_json):

    # Load respective json files into their dictionaries
    with open('preds_for_ensemble/finetune/squad-albert-base-v2_preds.json', 'r') as f:
        squad_albert_base_v2_data_dict = json.load(f)

    with open('preds_for_ensemble/finetune/squad-bert-base-uncased_preds.json', 'r') as f:
        squad_bert_base_uncased_data_dict = json.load(f)

    with open('preds_for_ensemble/finetune/squad-distilbert-base-uncased_preds.json', 'r') as f:
        squad_distilbert_base_uncased_data_dict = json.load(f)

    with open('preds_for_ensemble/finetune/squad-roberta-base.json', 'r') as f:
        squad_roberta_base_data_dict = json.load(f)

    with open('preds_for_ensemble/finetune/squad-xlm-roberta-base_preds.json', 'r') as f:
        squad_xlm_roberta_base_data_dict = json.load(f)

    with open('preds_for_ensemble/finetune/squad-xlnet-base-cased_preds.json', 'r') as f:
        squad_xlnet_base_cased_data_dict = json.load(f)

    with open('preds_for_ensemble/pretrained/distilbert-base-cased-distilled-squad_preds.json', 'r') as f:
        pretrained_distilbert_base_cased = json.load(f)

    with open('preds_for_ensemble/pretrained/distilbert-base-uncased-distilled-squad_preds.json', 'r') as f:
        pretrained_distilbert_base_uncased = json.load(f)

    with open('preds_for_ensemble/pretrained/quangb1910128_bert-finetuned-squad_preds.json', 'r') as f:
        pretrained_quang_bert = json.load(f)

    with open('preds_for_ensemble/finetune_custom/frozen-bert-custom_preds.json', 'r') as f:
        custom_frozen_bert = json.load(f)

    with open('preds_for_ensemble/finetune_custom/frozen-roberta-custom_preds.json', 'r') as f:
        custom_frozen_roberta = json.load(f) 

    # split csv file accoring to their columns
    cols = {'what': 1, 'where': 2, 'who': 3, 'undefined': 4, 'how m/m': 5, 'during': 6, 'when': 7, 'date': 8, 'how old': 9, 'why': 10, 'how big/size': 11, 'what time': 12, 'how are': 13}

    # compile a list of all the predictions from every model
    # NOTE: the order of the models should be same as that of the csv file / google sheets
    list_of_data_dict = [pretrained_distilbert_base_cased, 
                        pretrained_distilbert_base_uncased, 
                        pretrained_quang_bert, 
                        squad_distilbert_base_uncased_data_dict, 
                        squad_xlm_roberta_base_data_dict, 
                        squad_roberta_base_data_dict, 
                        squad_bert_base_uncased_data_dict, 
                        squad_xlnet_base_cased_data_dict, 
                        squad_albert_base_v2_data_dict, 
                        custom_frozen_roberta, 
                        custom_frozen_bert]
    
    results = None
    if is_probabilisitc:
        results = ensemble_with_probability(input_csv, cols, list_of_data_dict)
    else:
        results = ensemble_with_best_model(input_csv, cols, list_of_data_dict)

    f1_score = f"{results['f1']:.2f}"
    exact_match = f"{results['exact_match']:.2f}"
    results_dict = {"F1 score": f1_score, "Exact match": exact_match}
    with open(output_json, 'w') as f:
        json.dump(results_dict, f)
    
    

def get_values_of_col(index, input_csv):
    """
    Gets the values of the particular column of a csv file
    """
    result_list = []
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data_row = row[0]
            data_split = data_row.split()
            data = data_split[index]
            result_list.append(data)
    return result_list

def random_choice(model_indices):
    """
    Randomly chooses a model based on the scores of the top 3 models.
    The best model will be chosen 95% of the time, the second best model will be chosen 3% of the time, and the third best model will be chosen 2% of the time.
    """
    random_number = random.random()
    if random_number >= 0 and random_number < 0.95:
        return model_indices[0]
    if random_number >= 0.95 and random_number < 0.98:
        return model_indices[1]
    else:
        return model_indices[2]
    
    
def ensemble_with_probability(input_csv, cols, list_of_data_dict):
    # load squad metric
    metric = load_metric("squad")

    # load squad dataset
    dataset = load_dataset('squad', split='validation')

    predictions = []

    curr = 0

    for entry in dataset:
        # identify the id of the question
        question_id = entry['id']
        # identify the category of the question
        category = naive_categorize_question(entry)
        index_csv = cols.get(category)
        scores_str = get_values_of_col(index_csv, input_csv)
        # get scores of models performing in that particular category
        scores = list(map(float, scores_str))
        # sort the scores from highest to lowest and retrieve the top 3 scores
        top_3_scores = sorted(scores, reverse=True)[:3]
        # obtain indices of top 3 models
        # note that model_indices are already sorted from largest to smallest
        model_indices = []
        for score in top_3_scores:
            model_index = scores.index(score)
            model_indices.append(model_index)
        # get chosen model in string
        chosen_model_index = random_choice(model_indices)
        # obtain predictions of the chosen model
        chosen_model_preds = list_of_data_dict[chosen_model_index]
        # obtain question entry by choosing question id
        question = chosen_model_preds.get(question_id)
        pred = {'id': question_id, 'prediction_text': question['answer']}
        predictions.append(pred)
        print("curr iter: ", curr)
        curr += 1

    print("prediction done")        

    ref_ds = dataset.select_columns(['id', 'answers'])
    references = []
    for entry in ref_ds:
        answer = {'id': entry['id'], 'answers': entry['answers']}
        references.append(answer)

    results = metric.compute(predictions=predictions, references=references)
    return results


def ensemble_with_best_model(input_csv, cols, list_of_data_dict):
    # load squad metric
    metric = load_metric("squad")

    # load squad dataset
    dataset = load_dataset('squad', split='validation')
    predictions = []

    curr = 0

    for entry in dataset:
        # identify the id of the question
        question_id = entry['id']
        # identify the category of the question
        category = naive_categorize_question(entry)
        index_csv = cols.get(category)
        scores_str = get_values_of_col(index_csv, input_csv)
        # get scores of models performing in that particular category
        scores = list(map(float, scores_str))
        # choose the best performing model
        chosen_model_index = scores.index(sorted(scores, reverse=True)[0])
        # obtain predictions of the chosen model
        chosen_model_preds = list_of_data_dict[chosen_model_index]
        # obtain question entry by choosing question id
        question = chosen_model_preds.get(question_id)
        pred = {'id': question_id, 'prediction_text': question['answer']}
        predictions.append(pred)
        print("curr iter: ", curr)
        curr += 1

    print("prediction done")        

    ref_ds = dataset.select_columns(['id', 'answers'])
    references = []
    for entry in ref_ds:
        answer = {'id': entry['id'], 'answers': entry['answers']}
        references.append(answer)

    results = metric.compute(predictions=predictions, references=references)
    return results

if __name__ == '__main__':
    args = parse_arguments()
    if args.p:
        ensemble(args.input_csv, True, args.output_path)
    else:
        ensemble(args.input_csv, False, args.output_path)

