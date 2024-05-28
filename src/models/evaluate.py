# src/models/evaluate.py
# Script for evaluating the model

import os
import json

def evaluate_model(model, test_dataset, args):
    loss, accuracy = model.evaluate(test_dataset)
    
    # Create the output file name
    outfile_prefix = os.path.splitext(os.path.basename(args.pretrain_path))[0]
    outfile = os.path.join(args.eval_res_dir, f'test_results_{outfile_prefix}.json')
    
    # Print the results
    print(f'Test loss: {loss}')
    print(f"Test accuracy: {accuracy}")
    
    # Save the results as JSON
    os.makedirs(args.eval_res_dir, exist_ok=True)
    result_dict = {'test_loss': loss, 'test_accuracy': accuracy}
    with open(outfile, 'w') as json_file:
        json.dump(result_dict, json_file)
    
    return loss, accuracy