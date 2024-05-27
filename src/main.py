import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from data.data_loader import load_data, preprocess
from models.lenet import LeNet
from models.train import train_model
from models.evaluate import evaluate_model
from visualization.visualize import plot_training_curves
import numpy as np
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='LeNet Experiment')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'fine-tune'],
                        help='Mode: train, evaluate, fine-tune')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'caltech101', 'caltech256'],
                        help='Dataset name')
    parser.add_argument('--raw-dir', type='str', default=None, 
                        help='Directory to raw dataset for Caltech 101 and 256. E.g: path/to/101_ObjectCategories')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose logs')
    parser.add_argument('--pretrain-path', type=str,
                        help='Path to pre-trained model weights for fine-tuning')
    parser.add_argument('--log-dir', type=str, default='../logs', help='where to save training-progress, model, etc')
    parser.add_argument('--eval-res-dir', type=str, default='../logs/evaluation', help='path to evaluation result directory')
    args = parser.parse_args()
    
    if args.mode == 'evaluate' or args.mode == 'fine-tune':
        if not args.pretrain_path:
                parser.error(f"--mode {args.mode} requires --pretrain-path.")
                
    return args
    
def main():
    args = parse_arguments()
    
    # Load dataset and parameters
    dataset = load_data(args.dataset)
    train_ds, validation_ds, test_ds = dataset['train'], dataset['valid'], dataset['test']
    
    train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    validation_ds = validation_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    
    # input_shape = dataset['input_shape']
    num_classes = dataset['num_classes']
    
    # Train the model
    if args.mode == 'train':
        model = LeNet(num_classes=num_classes)
        
        # Define the checkpoint callback
        checkpoint_dir = args.log_dir + '/ckpt'
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_lenet_{args.dataset}.keras')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        
        history = train_model(model, train_ds=train_ds, validation_ds=validation_ds, learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose, callbacks=[checkpoint_callback])
        
        # Save history for analysis
        np.savez(os.path.join(args.log_dir, 'tb_logs', f'history_lenet_{args.dataset}.npz'), loss=history.history['loss'], accuracy=history.history['accuracy'], val_loss=history.history['val_loss'], val_accuracy=history.history['val_accuracy'])
    elif args.mode == 'evaluate':
        model = load_model(args.pretrain_path)
        
        loss, accuracy = evaluate_model(model, test_ds)
        
        outfile_prefix = os.path.splitext(os.path.basename(args.pretrain_path))[0]
        outfile = os.path.join(args.eval_res_dir, f'test_results_{outfile_prefix}.json')
        
        print(f'Test loss: {loss}')
        print(f"Test accuracy: {accuracy}")
        
         # Save the results as JSON
        result_dict = {'test_loss': loss, 'test_accuracy': accuracy}
        with open(outfile, 'w') as json_file:
            json.dump(result_dict, json_file)
    elif args.mode == 'fine-tune':
        # model = load_model(args.pretrain_path)
        print('Fine tune mode')
        pass


if __name__ == "__main__":
    main()