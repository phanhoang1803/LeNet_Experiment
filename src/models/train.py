# src/models/train.py
# Script for training the model

import keras

def train_model(model, train_ds, validation_ds, learning_rate = 0.001, epochs=10, batch_size=32, verbose=True, callbacks=None):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=validation_ds,
                        batch_size=batch_size,
                        verbose=verbose,
                        callbacks=callbacks
                        )
    
    return history