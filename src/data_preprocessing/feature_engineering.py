from preprocess import load_important_features, remove_features, file_load


def main():

    # load cleaned data
    df = file_load(
        file_path="/home/smebellis/ece579/final_project/network_anomaly_detection/data/processed/combined_df.pkl"
    )
    # load features list
    important_features = load_important_features(
        "/home/smebellis/ece579/final_project/network_anomaly_detection/src/data_preprocessing/important_features.json"
    )

    # Keep the Label column as an important feature
    label_column = " Label"
    if label_column not in important_features:
        important_features.append(label_column)

    # remove features from original datafram
    cleaned_df = df[important_features]

    # output datafram
    print(cleaned_df)

    # save dataframe
    cleaned_df.to_pickle(
        "/home/smebellis/ece579/final_project/network_anomaly_detection/data/processed/cleaned_df.pkl"
    )

    print("Cleaned DataFrame saved sucessfully")


if __name__ == "__main__":
    main()
