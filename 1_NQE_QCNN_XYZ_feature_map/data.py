from os import listdir
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import torch


def data_load_and_process(dataset, feature_reduction='resize256', classes=[0,1]):
  
    if dataset == 'protein':
        data_dir = "/Users/jungguchoi/Library/Mobile Documents/com~apple~CloudDocs/1_Post_doc(Cleveland_clinic:2024.10~2025.09)/1_Research_project/3_quantum_embedding_comparison_sequence(2024.09 ~ XXXX.XX)/0_data/0_other/0_preprocessed/0_full_data_dfs/"
        dataset_df = pd.read_csv(data_dir+"/PKM2_df_full_data.csv")

        dataset_df_class_0 = dataset_df.loc[dataset_df['class'] == 0]
        dataset_df_class_1 = dataset_df.loc[dataset_df['class'] == 1]
        print("class 0:",dataset_df_class_0.shape,"/class 1:",dataset_df_class_1.shape)

        minimum_row = min(len(dataset_df_class_0), len(dataset_df_class_1))
        dataset_df_class_0_selected = dataset_df_class_0.iloc[:int(minimum_row)*6,:]
        dataset_df_class_1_selected = dataset_df_class_1.iloc[:int(minimum_row),:]
        print("class 0(selected):",dataset_df_class_0_selected.shape,"/class 1(selected):",dataset_df_class_1_selected.shape)

        selected_concat_df = pd.concat([dataset_df_class_0_selected, dataset_df_class_1_selected], axis=0)
        selected_concat_df = selected_concat_df.sample(frac=1)
        print("concat selected df:", selected_concat_df.shape)

        dataset_value = selected_concat_df.iloc[:,1:-2]
        dataset_label = selected_concat_df['class']
        print(dataset_value.head())
        print(dataset_label.head())

        scaler = MinMaxScaler()
        dataset_value = pd.DataFrame(scaler.fit_transform(dataset_value))

        x_train, x_test, y_train, y_test = train_test_split(dataset_value, dataset_label, test_size=0.2, shuffle=True,
                                                            stratify=dataset_label, random_state=10)

        x_train, x_test, y_train, y_test =\
            x_train.values.tolist(), x_test.values.tolist(), y_train.values.tolist(), y_test.values.tolist()
        
        y_train = np.array([1 if y == 1 else 0 for y in y_train])
        y_test = np.array([1 if y == 1 else 0 for y in y_test])
        print("[1] X_train:",len(x_train),"X_test:",len(x_test),"Y_train:",len(y_train),"Y_test:",len(y_test))

        x_train, x_test = np.array(x_train) / 69.0, np.array(x_test) / 69.0
        x_train, x_test = x_train, x_test

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
        #x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
        #x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
        #x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()
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
        print(pd.DataFrame(x_train).head())
        print(pd.DataFrame(x_test).head())
        
        
        X_train = PCA(dim_reduct).fit_transform(pd.DataFrame(x_train))
        X_test = PCA(dim_reduct).fit_transform(pd.DataFrame(x_test))
        print("after PCA:", X_train.shape, X_test.shape)
        #x_train, x_test = [], []
        #for x in X_train:
            #x = (x - x.min()) * (np.pi / (x.max() - x.min()))
        #    x_train.append(x / 2)
        #for x in X_test:
            #x = (x - x.min()) * (np.pi / (x.max() - x.min()))
        #    x_test.append(x / 2)
        return x_train, x_test, y_train, y_test