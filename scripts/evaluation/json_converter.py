import os
import json
import argparse

def parse_args():
    def check_json_extension(value):
        base, ext = os.path.splitext(os.path.basename(value))
        if ext.lower() != '.json':
            raise argparse.ArgumentTypeError("Output file must have a .json extension")
        return value

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run QA model inference.")

    # Add arguments to the parser
    parser.add_argument(
        "--full_json",
        type=check_json_extension,
        required=True,
        help="Path to full prediction json file to be used by ensemble logic"
    )

    parser.add_argument(
        "--lite_json",
        type=check_json_extension,
        required=True,
        help="Path to lite prediction json file to be used by prof evaluation script"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args
    
def main(full_json_path, lite_json_path):
    lite_pred = {}
    with open(full_json_path) as f:
        full_pred = json.load(f)
        
        for qid, pred in full_pred.items():
            lite_pred[qid] = pred['answer']
    
    with open(lite_json_path, 'w') as f:
        json.dump(lite_pred, f)


if __name__ == "__main__":
    args = parse_args()
    print(f"Converting {args.full_json} to {args.lite_json}... to run prof evaluation")
    main(args.full_json, args.lite_json)
    print("Done!")
        