from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from datasets import Audio, Dataset
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(path:str, labels_to_ids, label_column_name='usable_dialect', audio_column_name='full_audio_file', samp_rate=16000):
    df = pd.read_pickle(path)
    df.reset_index(drop=True, inplace=True)
    df['class_label'] = df[label_column_name].apply(labels_to_ids)
    df = df.astype({audio_column_name: "string", "class_label": "int64"})
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column(audio_column_name, Audio(sampling_rate=samp_rate))
    ds = ds.rename_column(audio_column_name, "audio")
    return ds

def get_predicted_label(item, model, feature_extractor):
    inputs = feature_extractor(item['audio']['array'], sampling_rate=item['audio']['sampling_rate'], return_tensors="pt").to('cuda')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits).item()
    predicted_label = model.config.id2label[predicted_class_id]
    return predicted_label, predicted_class_id

def print_test_results(args):
    test_dataset_path = args.test_dataset
    model_path = args.model_path
    save_path = args.save_path 

    print('loading model from:', model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path).to('cuda')
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    def change_labels_to_ids(label):
        return model.config.label2id[label]
    labels = model.config.label2id.keys()
    
    id2label = model.config.id2label
    labels = [id2label[i] for i in sorted(id2label.keys(), key=int)]


    print('loading test dataset from:', test_dataset_path)
    test_dataset = load_dataset(test_dataset_path, change_labels_to_ids, label_column_name=args.label_column_name, audio_column_name=args.audio_column_name)

    print('True label columns name: {}\nAudio column name: {}'.format(args.label_column_name, args.audio_column_name))

    true_label_ids = []
    pred_label_ids = []
    pred_labels = []
    for item in tqdm(test_dataset, desc="Predicting"):
        predicted_label, predicted_class_id = get_predicted_label(item, model, feature_extractor)
        pred_label_ids.append(predicted_class_id)
        true_label_ids.append(item["class_label"])
        pred_labels.append(predicted_label)

    print("Accuracy: ", accuracy_score(true_label_ids, pred_label_ids))
    print("MSE: ", mean_squared_error(true_label_ids, pred_label_ids))

    if save_path:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # save off the dataframe with the predicted labels
        test_df = test_dataset.to_pandas()
        test_df['predicted_label'] = pred_labels
        test_df.to_pickle(
            os.path.join(save_path, 'dataset_{}.pkl'.format(
                os.path.basename(model_path)
            ))
        )
        with open(os.path.join(save_path, 'accuracy_{}.txt'.format(
            os.path.basename(model_path)
        )), 'w') as open_f:
            open_f.write("Accuracy: {}\nMSE: {}".format(
                accuracy_score(true_label_ids, pred_label_ids),
                mean_squared_error(true_label_ids, pred_label_ids)
            )
        )

    try:
        cm = confusion_matrix(test_dataset[args.label_column_name], pred_labels, labels=labels)
    except ValueError:
        # if it failed to predict an entire class then we can't pass the labels parameter
        cm = confusion_matrix(test_dataset[args.label_column_name], pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, vmin=0, vmax=1, linecolor="white",
                annot=True, fmt=".2f", linewidth=0.5, cmap="Blues",
                xticklabels=labels, 
                yticklabels=labels)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(
            os.path.join(
                save_path, 
                'conf_matrix_{}.png'.format(
                    os.path.basename(model_path)
                )
            )
        )
    else:
        plt.show()


if __name__ == '__main__':
    # === ARGUMENTS & INITIALIZATION ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/idatro/repo/did_prosody_whisper-main/model_output/unmodified/checkpoint-2073")
    parser.add_argument("--test_dataset", type=str, default="/home/idatro/repo/did_prosody_whisper-main/datasets/ssc_data/test_data.pkl")
    parser.add_argument("--save_path", type=str, default="/home/idatro/repo/did_prosody_whisper-main/model_output/unmodified/checkpoint-2073/eval_results")
    parser.add_argument('--label_column_name', type=str, default='dialect_region')
    parser.add_argument('--audio_column_name', type=str, default='full_audio_file')
    
    args = parser.parse_args()

    print_test_results(
        args
    )

    print('Done with evaluation!')