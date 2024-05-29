import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, AveragePooling2D, Dense

import argparse
from data.data_loader import load_data
from data.preprocess import adapt_input, preprocess
from models.creating_model import LeNetModel, ANN
from models.train import train_model
from models.evaluate import evaluate_model
from visualization.visualize import plot_training_curves
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='LeNet Experiment')

    parser.add_argument('--mode', type=str, default='train', required=True, choices=['train', 'evaluate', 'fine-tune'],
                        help='Mode: train, evaluate, fine-tune')
    parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'ANN'],
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='mnist', required=True, choices=['mnist', 'fmnist', 'caltech101', 'caltech256'],
                        help='Dataset name')
    parser.add_argument('--raw-dir', type=str, default=None, 
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
    """
    Main function that loads the arguments, loads the dataset and parameters, 
    and trains the model based on the given mode.
    """
    args = parse_arguments()
    
    # Load dataset and parameters
    dataset = load_data(args.dataset, args.raw_dir)
    train_ds, validation_ds, test_ds = dataset['train'], dataset['valid'], dataset['test']
    
    input_shape = dataset['input_shape']
    num_classes = dataset['num_classes']
    
    # Train the model
    if args.mode == 'train':
        # Preprocess and batch the datasets
        train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        
        if args.model == 'lenet':
            model = LeNetModel(num_classes=num_classes, input_shape=input_shape)
        elif args.model == 'ANN':
            model = ANN(num_classes=num_classes, input_shape=input_shape)
        
        # Define the checkpoint callback
        checkpoint_dir = args.log_dir + '/ckpt'
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{args.model}_{args.dataset}.keras')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        
        # Train the model
        history = train_model(model, train_ds=train_ds, validation_ds=validation_ds, learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose, callbacks=[checkpoint_callback])
        
        # Save history for analysis
        os.makedirs(os.path.join(args.log_dir, 'tb_logs'), exist_ok=True)
        np.savez(os.path.join(args.log_dir, 'tb_logs', f'history_{args.model}_{args.dataset}.npz'), loss=history.history['loss'], accuracy=history.history['accuracy'], val_loss=history.history['val_loss'], val_accuracy=history.history['val_accuracy'])
    elif args.mode == 'evaluate':
        if 'lenet' in args.pretrain_path:
            args.model = 'lenet'
        elif 'ANN' in args.pretrain_path:
            args.model = 'ANN'
            
        model = load_model(args.pretrain_path)
        
        test_ds = test_ds.map(lambda x, y: (adapt_input(x, model.input_shape[1:]), y)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        # test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        
        # Load the pre-trained model
        
        # Evaluate the model
        loss, accuracy = evaluate_model(model, test_ds, args)
    elif args.mode == 'fine-tune':
        if 'lenet' in args.pretrain_path:
            args.model = 'lenet'
        elif 'ANN' in args.pretrain_path:
            args.model = 'ANN'
        
        # Load the pre-trained model
        base_model = load_model(args.pretrain_path)
        
        # Convert dataset to appropriate format
        train_ds = train_ds.map(lambda x, y: (adapt_input(x, base_model.input_shape[1:]), y)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.map(lambda x, y: (adapt_input(x, base_model.input_shape[1:]), y)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

        # Define the model
        outputs = Dense(units=num_classes, activation='softmax', name='output_layer')(base_model.layers[-2].output)
        model = Model(inputs=base_model.input, outputs=outputs)
        
        # Freeze layers for transfer learning
        base_model.trainable = False
        history = train_model(model, train_ds=train_ds, validation_ds=validation_ds, learning_rate=0.001, epochs=10, batch_size=args.batch_size, verbose=args.verbose, callbacks=None)
        
        # Unfreeze layers for fine-tuning
        base_model.trainable = True
        
        checkpoint_dir = args.log_dir + '/ckpt'
        ds_name = os.path.basename(args.pretrain_path).split('_')[-1].split('.')[0]
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{args.model}_{ds_name}finetune_{args.dataset}.keras')
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
        os.makedirs(os.path.join(args.log_dir, 'tb_logs'), exist_ok=True)
        np.savez(os.path.join(args.log_dir, 'tb_logs', f'history_{args.model}_{ds_name}finetune_{args.dataset}.npz'), loss=history.history['loss'], accuracy=history.history['accuracy'], val_loss=history.history['val_loss'], val_accuracy=history.history['val_accuracy'])

if __name__ == "__main__":
    main()