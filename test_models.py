import backend
import time
import pandas as pd

def run_tests():
    # Create a dummy user for prediction
    new_id = backend.add_new_ratings(['ML0101EN', 'DV0101EN'])
    user_ids = [new_id]

    # Default parameters for each model
    params = {
        "Course Similarity": {'top_courses': 10, 'sim_threshold': 50},
        "User Profile": {'user_sim_threshold': 50},
        "Clustering": {'cluster_no': 20},
        "Clustering with PCA": {'cluster_no': 20, 'pca_n_components': 5},
        "KNN": {'k_neighbors': 5},
        "NMF": {'n_components': 5},
        "Neural Network": {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'learning_rate': 0.01},
        "Regression with Embedding Features": {'reg_model': 'Linear Regression'},
        "Classification with Embedding Features": {'clf_model': 'Logistic Regression'}
    }

    results = []

    for model_name in backend.models:
        print(f"--- Testing model: {model_name} ---")
        
        # Training
        training_time = -1
        training_status = "Not Implemented"
        try:
            start_time = time.time()
            backend.train(model_name, params.get(model_name, {}))
            training_time = time.time() - start_time
            training_status = "Success"
        except Exception as e:
            training_status = f"Failure: {e}"
        
        print(f"Training status: {training_status}")
        if training_time != -1:
            print(f"Training time: {training_time:.4f} seconds")

        # Prediction
        prediction_time = -1
        prediction_status = "Not Implemented"
        try:
            start_time = time.time()
            backend.predict(model_name, user_ids, params.get(model_name, {}))
            prediction_time = time.time() - start_time
            prediction_status = "Success"
        except Exception as e:
            prediction_status = f"Failure: {e}"

        print(f"Prediction status: {prediction_status}")
        if prediction_time != -1:
            print(f"Prediction time: {prediction_time:.4f} seconds")
            
        results.append({
            "model": model_name,
            "training_status": training_status,
            "training_time": training_time,
            "prediction_status": prediction_status,
            "prediction_time": prediction_time
        })
        print("-" * (len(model_name) + 20))
        print()

    # Print summary
    print("\n--- Test Summary ---")
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == '__main__':
    run_tests()
