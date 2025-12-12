from os import listdir
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch


def data_load_and_process(dataset, feature_reduction='resize256', classes=[0,1]):
  
    if dataset == 'COVID19':
        data_dir = '/Users/jungguchoi/Library/Mobile Documents/com~apple~CloudDocs/1_Post_doc(Cleveland_clinic:2024.10~2025.09)/1_Research_project/3_quantum_embedding_comparison_sequence(2024.09 ~ XXXX.XX)/2_exp/58_Dr_Park_Comments_JULY2225/2_COVID_19_dataset/0_dataset/'
        
        dataset_train = pd.read_csv(data_dir+"/train_preprocessed.csv")
        dataset_test = pd.read_csv(data_dir+"/extest_preprocessed.csv")
        dataset_merged = pd.concat([dataset_train, dataset_test], axis=0)
        print("merged dataset:", dataset_merged.shape)

        dataset_value = dataset_merged.iloc[:,1:-1]
        dataset_label = dataset_merged.iloc[:,-1]

        x_train, x_test, y_train, y_test = train_test_split(dataset_value, dataset_label, test_size=0.2, shuffle=True,
                                                            stratify=dataset_label, random_state=10)

        x_train, x_test, y_train, y_test =\
            x_train.values.tolist(), x_test.values.tolist(), y_train.values.tolist(), y_test.values.tolist()
        
        y_train = np.array([1 if y == 1 else 0 for y in y_train])
        y_test = np.array([1 if y == 1 else 0 for y in y_test])
        print("[1] X_train:",len(x_train),"X_test:",len(x_test),"Y_train:",len(y_train),"Y_test:",len(y_test))

        x_train, x_test = np.array(x_train), np.array(x_test)
        
    
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'kmnist':
       
        # Path to training images and corresponding labels provided as numpy arrays
        kmnist_train_images_path = "/home/tak/Github/QEmbedding/archive/kmnist-train-imgs.npz"
        kmnist_train_labels_path = "/home/tak/Github/QEmbedding/archive/kmnist-train-labels.npz"

        # Path to the test images and corresponding labels
        kmnist_test_images_path = "/home/tak/Github/QEmbedding/archive/kmnist-test-imgs.npz"
        kmnist_test_labels_path = "/home/tak/Github/QEmbedding/archive/kmnist-test-labels.npz"

        x_train = np.load(kmnist_train_images_path)['arr_0']
        y_train = np.load(kmnist_train_labels_path)['arr_0']

        # Load the test data from the corresponding npz files
        x_test = np.load(kmnist_test_images_path)['arr_0']
        y_test = np.load(kmnist_test_labels_path)['arr_0']
    
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    if len(classes) == 2:
        train_filter_tf = np.where((y_train == classes[0] ) | (y_train == classes[1] ))
        test_filter_tf = np.where((y_test == classes[0] ) | (y_test == classes[1] ))


    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]
    print("[2] x_train:",len(x_train),"x_test:",len(x_test),"y_train:",len(y_train),"y_test:",len(y_test))
        
    if feature_reduction == False:
        print("[3] x_train:",len(x_train),"x_test:",len(x_test),"y_train:",len(y_train),"y_test:",len(y_test))
        return x_train, x_test, y_train, y_test


    if feature_reduction in ['PCA12', 'PCA8', 'PCA4', 'PCA2', 'PCA1']:
        x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
        x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
        x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()
        if feature_reduction == 'PCA12':
            dim_reduct = 12
        if feature_reduction == 'PCA8':
            dim_reduct = 8
        if feature_reduction == 'PCA4':
            dim_reduct = 4
        if feature_reduction == 'PCA2':
            dim_reduct = 2
        if feature_reduction == 'PCA1':
            dim_reduct = 1
            
        print("before PCA:", x_train.shape, x_test.shape)
        X_train = PCA(dim_reduct).fit_transform(x_train)
        X_test = PCA(dim_reduct).fit_transform(x_test)
        print("after PCA:", X_train.shape, X_test.shape)
        x_train, x_test = [], []
        for x in X_train:
            #x = (x - x.min()) * (np.pi / (x.max() - x.min()))
            x_train.append(x / 2)
        for x in X_test:
            #x = (x - x.min()) * (np.pi / (x.max() - x.min()))
            x_test.append(x / 2)
        return x_train, x_test, y_train, y_test