from src.data_preprocessing import load_data, clean_data, encode_features
from src.model import split_data, train_model, save_model
from src.evaluate import evaluate_model  

def main():
    # Step 1: Load data
    df = load_data('data/telco.csv')

    # Step 2: Clean data
    df_clean = clean_data(df)

    # Step 3: Encode features
    df_encoded = encode_features(df_clean)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # Step 5: Train model
    model = train_model(X_train, y_train)

    # Step 6: Evaluate model
    evaluate_model(model, X_test, y_test)

    # Step 7: Save model
    save_model(model)

if __name__ == '__main__':
    main()
