import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.metrics import RootMeanSquaredError


def save(model, test_dataset, name):
    name = name.replace(' ', '_').lower()
    model.save('./models/' + name)
    
    labels_df = pd.read_csv('./dataset/training_solutions_rev1.csv')
    
    predictions = model.predict(test_dataset)
    dir = './dataset/images_test_rev1/'
    image_paths = os.listdir(dir)
    ids = np.array([x.split('.')[0] for x in image_paths]).reshape(-1, 1)
    submission_df = pd.DataFrame(np.hstack((ids, predictions)), columns=labels_df.columns)
    submission_df = submission_df.sort_values(by=['GalaxyID'])
    submission_name = './submissions/' + name + '.csv'
    submission_df.to_csv(submission_name, index=False)
    
def plot_history(history, name):
    acc = history.history['root_mean_squared_error']
    val_acc = history.history['val_root_mean_squared_error']


    plt.figure(figsize=(6, 4))
    plt.plot(acc, label='Training RMSE')
    plt.plot(val_acc, label='Validation RMSE')
    plt.legend()
    plt.grid()
    plt.ylabel('RMSE')
    plt.ylim([min(plt.ylim()), max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(name)

    if name:
        plt.savefig(f"./images/{name.replace(' ', '_').lower()}.png")
    plt.show()
    
def compile_and_fit(model, train_dataset, val_dataset, epochs, optimizer='adamax'):
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', RootMeanSquaredError()])
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    return history