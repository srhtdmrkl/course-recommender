# Course Recommender System

This is a Streamlit application that provides personalized course recommendations to users. Users can select courses they have audited or completed, and the application will suggest new courses based on the selected recommendation model.

## Features

*   **Interactive Course Selection**: Users can select courses from an interactive table.
*   **Multiple Recommendation Models**: The application supports a variety of recommendation models, including:
    *   Course Similarity
    *   User Profile
    *   Clustering
    *   Clustering with PCA
    *   K-Nearest Neighbors (KNN)
    *   Non-negative Matrix Factorization (NMF)
    *   Neural Network
    *   Regression with Embedding Features
    *   Classification with Embedding Features
*   **Tunable Hyper-parameters**: Each model comes with a set of hyper-parameters that can be tuned through the UI to customize the recommendations.
*   **In-app Model Training**: Users can train the models directly from the application's sidebar.

## How to Run

1.  **Clone the repository**

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up the data directory**:
    *   Create a `data` directory in the root of the project.
    *   Place the following data files inside the `data` directory:
        *   `course_processed.csv`
        *   `courses_bows.csv`
        *   `ratings.csv`
        *   `sim.csv`

4.  **Run the application**:
    ```bash
    streamlit run recommender_app.py
    ```

## File Structure

```
.
├── data/                  # Directory for all data files
│   ├── course_processed.csv
│   ├── courses_bows.csv
│   ├── ratings.csv
│   └── sim.csv
├── recommender_app.py     # Main Streamlit application file
├── backend.py             # Core logic for recommendation models
├── requirements.txt       # Python dependencies
├── test_models.py         # Script for testing the models
└── README.md              # This file
```
