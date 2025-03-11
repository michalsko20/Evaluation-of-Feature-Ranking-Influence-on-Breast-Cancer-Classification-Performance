import os
import pandas as pd

def collect_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..", "data","processed","out")
    save_dir = os.path.join(script_dir, "..", "..", "data","csv_files","out")
    all_data = []

    g1_path = os.path.join(base_dir, "G1_larger")
    for root, dirs, files in os.walk(g1_path):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                df = pd.read_csv(file_path)
                df_last = df.tail(1).copy()
                df_last["target"] = "G1"
                all_data.append(df_last)

    g2_path = os.path.join(base_dir, "G2_larger")
    for root, dirs, files in os.walk(g2_path):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                df = pd.read_csv(file_path)
                df_last = df.tail(1).copy()
                df_last["target"] = "G2"
                all_data.append(df_last)

    g3_path = os.path.join(base_dir, "G3_larger")
    for root, dirs, files in os.walk(g3_path):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                df = pd.read_csv(file_path)
                df_last = df.tail(1).copy()
                df_last["target"] = "G3"
                all_data.append(df_last)

    data = pd.concat(all_data, ignore_index=True)
    print(data.head())
    print(data.shape)
    data.to_csv(os.path.join(save_dir,"all_data.csv"), index=False)

def load_data(csv_name = "all_data.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data","csv_files",csv_name)
    data = pd.read_csv(data_path)
    return data

def save_data(csv_name, data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "..", "..", "data","csv_files","out")
    data.to_csv(os.path.join(save_dir,csv_name), index=False)

def combine_data(data, df_features):
    print("data shape:", data.shape)
    print("df_features shape:", df_features.shape)

    # Check if the number of rows matches
    assert data.shape[0] == df_features.shape[0], "The number of rows in 'data' and 'df_features' does not match!"

    # Extract classical features from 'data' (excluding 'target' column)
    classical_features = data.drop(columns=['target'])

    # Extract CNN features from 'df_features' (excluding 'label' column)
    cnn_features = df_features.drop(columns=['label'])

    # Combine horizontally (axis=1)
    combined_features = pd.concat([classical_features, cnn_features], axis=1)

    # Target labels (e.g., from 'data')
    combined_label = data['target']

    # Now we have:
    # combined_features -> all features
    # combined_label    -> labels
    print("combined_features shape:", combined_features.shape)

    # You can also combine everything into a single DataFrame, e.g.:
    combined_df = combined_features.copy()
    combined_df['target'] = combined_label

    # Optionally, save to a CSV file
    save_data("combined_data.csv", combined_df)