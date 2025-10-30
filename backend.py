import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("data/ratings.csv")


def load_course_sims():
    return pd.read_csv("data/sim.csv")


def load_courses():
    df = pd.read_csv("data/course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("data/courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("data/ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


# Model training
def train(model_name, params):
    if model_name == models[1]: # User Profile
        # No training needed for user profile model as it's created on the fly at prediction time
        pass
    elif model_name == models[2]:  # Clustering
        # Load course BoW features
        bow_df = load_bow()
        # Pivot the table to create a course-word matrix
        course_word_matrix = bow_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)
        # Get the number of clusters from params
        n_clusters = params.get('cluster_no', 20)
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(course_word_matrix)
        with open('data/cluster_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
    elif model_name == models[3]:  # Clustering with PCA
        # Load course BoW features
        bow_df = load_bow()
        # Pivot the table to create a course-word matrix
        course_word_matrix = bow_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)
        # Get the number of PCA components from params
        n_components = params.get('pca_n_components', 5)
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(course_word_matrix)
        # Get the number of clusters from params
        n_clusters = params.get('cluster_no', 20)
        # Perform K-Means clustering on PCA features
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_features)
        # Save the models
        with open('data/pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)
        with open('data/cluster_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
    elif model_name == models[4]: # KNN
        # Load course BoW features
        bow_df = load_bow()
        course_word_matrix = bow_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)
        
        # Get the number of neighbors from params
        k = params.get('k_neighbors', 5)
        
        # Create and train the KNN model
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(course_word_matrix)
        
        # Save the model
        with open('data/knn_model.pkl', 'wb') as f:
            pickle.dump(knn, f)
        with open('data/knn_course_word_matrix.pkl', 'wb') as f:
            pickle.dump(course_word_matrix, f)
    elif model_name == models[5]:  # NMF
        # Load ratings
        ratings_df = load_ratings()
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating', fill_value=0)
        # Get the number of components from params
        n_components = params.get('n_components', 5)
        # Perform NMF
        nmf = NMF(n_components=n_components, init='random', random_state=0)
        nmf.fit(user_item_matrix)
        # Save the model and the columns
        with open('data/nmf_model.pkl', 'wb') as f:
            pickle.dump(nmf, f)
        with open('data/nmf_model_columns.pkl', 'wb') as f:
            pickle.dump(user_item_matrix.columns, f)
    elif model_name == models[6]: # Neural Network
        # Load data
        ratings_df = load_ratings()
        bow_df = load_bow()
        course_word_matrix = bow_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)

        # Create user profiles
        user_profiles = {}
        for user_id in ratings_df['user'].unique():
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            if enrolled_course_ids:
                # Filter out courses that are not in the course_word_matrix
                enrolled_course_ids = [c for c in enrolled_course_ids if c in course_word_matrix.index]
                if enrolled_course_ids:
                    user_profiles[user_id] = course_word_matrix.loc[enrolled_course_ids].mean().values
        
        # Create training data
        X = []
        y = []
        for index, row in ratings_df.iterrows():
            user_id = row['user']
            course_id = row['item']
            rating = row['rating']
            if user_id in user_profiles and course_id in course_word_matrix.index:
                user_profile = user_profiles[user_id]
                course_vector = course_word_matrix.loc[course_id].values
                feature_vector = list(user_profile) + list(course_vector)
                X.append(feature_vector)
                y.append(rating)

        # Get hyper-parameters
        hidden_layer_sizes = params.get('hidden_layer_sizes', (100, 50))
        activation = params.get('activation', 'relu')
        learning_rate_init = params.get('learning_rate', 0.01)

        # Train the model
        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate_init=learning_rate_init, random_state=0)
        mlp.fit(X, y)

        # Save the model and data
        with open('data/nn_model.pkl', 'wb') as f:
            pickle.dump(mlp, f)
        with open('data/nn_user_profiles.pkl', 'wb') as f:
            pickle.dump(user_profiles, f)
        with open('data/nn_course_word_matrix.pkl', 'wb') as f:
            pickle.dump(course_word_matrix, f)
    elif model_name == models[7]: # Regression with Embedding Features
        # Train NMF to get embeddings
        ratings_df = load_ratings()
        user_item_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating', fill_value=0)
        
        n_components = params.get('n_components', 5) # Use n_components from NMF params
        nmf = NMF(n_components=n_components, init='random', random_state=0)
        W = nmf.fit_transform(user_item_matrix)
        H = nmf.components_

        user_embeddings = pd.DataFrame(W, index=user_item_matrix.index)
        course_embeddings = pd.DataFrame(H.T, index=user_item_matrix.columns)

        # Create training data
        X = []
        y = []
        for index, row in ratings_df.iterrows():
            user_id = row['user']
            course_id = row['item']
            rating = row['rating']
            if user_id in user_embeddings.index and course_id in course_embeddings.index:
                user_vec = user_embeddings.loc[user_id].values
                course_vec = course_embeddings.loc[course_id].values
                feature_vector = list(user_vec) + list(course_vec)
                X.append(feature_vector)
                y.append(rating)
        
        # Get regression model from params
        reg_model_name = params.get('reg_model', 'Linear Regression')
        if reg_model_name == 'Ridge':
            reg_model = Ridge()
        elif reg_model_name == 'Lasso':
            reg_model = Lasso()
        else:
            reg_model = LinearRegression()
            
        # Train the regression model
        reg_model.fit(X, y)

        # Save the models
        with open('data/reg_nmf_model.pkl', 'wb') as f:
            pickle.dump(nmf, f)
        with open('data/reg_model.pkl', 'wb') as f:
            pickle.dump(reg_model, f)
        with open('data/reg_model_columns.pkl', 'wb') as f:
            pickle.dump(user_item_matrix.columns, f)
    elif model_name == models[8]: # Classification with Embedding Features
        # Train NMF to get embeddings
        ratings_df = load_ratings()
        user_item_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating', fill_value=0)
        
        n_components = params.get('n_components', 5) # Use n_components from NMF params
        nmf = NMF(n_components=n_components, init='random', random_state=0)
        W = nmf.fit_transform(user_item_matrix)
        H = nmf.components_

        user_embeddings = pd.DataFrame(W, index=user_item_matrix.index)
        course_embeddings = pd.DataFrame(H.T, index=user_item_matrix.columns)

        # Create training data
        X = []
        y = []
        for index, row in ratings_df.iterrows():
            user_id = row['user']
            course_id = row['item']
            rating = row['rating']
            if user_id in user_embeddings.index and course_id in course_embeddings.index:
                user_vec = user_embeddings.loc[user_id].values
                course_vec = course_embeddings.loc[course_id].values
                feature_vector = list(user_vec) + list(course_vec)
                X.append(feature_vector)
                y.append(1 if rating >= 3.0 else 0) # Convert rating to binary class
        
        # Get classification model from params
        clf_model_name = params.get('clf_model', 'Logistic Regression')
        if clf_model_name == 'SVM':
            clf_model = SVC(probability=True)
        elif clf_model_name == 'Random Forest':
            clf_model = RandomForestClassifier()
        else:
            clf_model = LogisticRegression()
            
        # Train the classification model
        clf_model.fit(X, y)

        # Save the models
        with open('data/clf_nmf_model.pkl', 'wb') as f:
            pickle.dump(nmf, f)
        with open('data/clf_model.pkl', 'wb') as f:
            pickle.dump(clf_model, f)
        with open('data/clf_model_columns.pkl', 'wb') as f:
            pickle.dump(user_item_matrix.columns, f)
    else:
        # For other models, do nothing for now
        pass


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            sim_matrix = load_course_sims().to_numpy()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        elif model_name == models[1]: # User Profile
            # Get user's enrolled courses
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            # Load course BoW features
            bow_df = load_bow()
            course_word_matrix = bow_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)

            # Create user profile
            user_profile = course_word_matrix.loc[enrolled_course_ids].mean().values.reshape(1, -1)

            # Calculate similarity between user profile and all courses
            cosine_sim = cosine_similarity(user_profile, course_word_matrix)
            
            # Create a series with the similarity scores
            sim_scores = pd.Series(cosine_sim.flatten(), index=course_word_matrix.index)

            # Filter out enrolled courses
            sim_scores = sim_scores.drop(enrolled_course_ids, errors='ignore')

            # Get similarity threshold
            sim_threshold = params.get('user_sim_threshold', 50) / 100.0

            # Filter courses based on threshold
            recommended_courses = sim_scores[sim_scores >= sim_threshold]

            # Sort the courses by similarity score
            top_courses = recommended_courses.sort_values(ascending=False)

            # Get top N courses
            top_n = params.get('top_courses', 10)

            for course_id, score in top_courses.head(top_n).items():
                users.append(user_id)
                courses.append(course_id)
                scores.append(score)
        
        elif model_name == models[2]: # Clustering
            with open('data/cluster_model.pkl', 'rb') as f:
                kmeans = pickle.load(f)
            
            bow_df = load_bow()
            course_word_matrix = bow_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)
            
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            
            # Find the cluster for each enrolled course
            enrolled_course_indices = [course_word_matrix.index.get_loc(c) for c in enrolled_course_ids if c in course_word_matrix.index]
            enrolled_course_clusters = kmeans.predict(course_word_matrix.iloc[enrolled_course_indices])
            
            # Find all courses in the same clusters
            all_course_clusters = kmeans.labels_
            
            recommended_courses = []
            for cluster_id in enrolled_course_clusters:
                courses_in_cluster = course_word_matrix.index[all_course_clusters == cluster_id]
                recommended_courses.extend(courses_in_cluster)
            
            # Remove already enrolled courses
            recommended_courses = [c for c in recommended_courses if c not in enrolled_course_ids]
            
            # Remove duplicates
            recommended_courses = list(dict.fromkeys(recommended_courses))

            for course_id in recommended_courses:
                users.append(user_id)
                courses.append(course_id)
                scores.append(1.0) # Constant score for now

        elif model_name == models[3]: # Clustering with PCA
            with open('data/pca_model.pkl', 'rb') as f:
                pca = pickle.load(f)
            with open('data/cluster_model.pkl', 'rb') as f:
                kmeans = pickle.load(f)

            bow_df = load_bow()
            course_word_matrix = bow_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)
            
            pca_features = pca.transform(course_word_matrix)

            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            
            # Find the cluster for each enrolled course
            enrolled_course_indices = [course_word_matrix.index.get_loc(c) for c in enrolled_course_ids if c in course_word_matrix.index]
            enrolled_course_clusters = kmeans.predict(pca_features[enrolled_course_indices])

            # Find all courses in the same clusters
            all_course_clusters = kmeans.labels_

            recommended_courses = []
            for cluster_id in enrolled_course_clusters:
                courses_in_cluster = course_word_matrix.index[all_course_clusters == cluster_id]
                recommended_courses.extend(courses_in_cluster)

            # Remove already enrolled courses
            recommended_courses = [c for c in recommended_courses if c not in enrolled_course_ids]
            
            # Remove duplicates
            recommended_courses = list(dict.fromkeys(recommended_courses))

            for course_id in recommended_courses:
                users.append(user_id)
                courses.append(course_id)
                scores.append(1.0) # Constant score for now
        elif model_name == models[4]: # KNN
            with open('data/knn_model.pkl', 'rb') as f:
                knn = pickle.load(f)
            with open('data/knn_course_word_matrix.pkl', 'rb') as f:
                course_word_matrix = pickle.load(f)

            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            recommended_courses = set()
            for course_id in enrolled_course_ids:
                if course_id in course_word_matrix.index:
                    course_index = course_word_matrix.index.get_loc(course_id)
                    distances, indices = knn.kneighbors(course_word_matrix.iloc[course_index].values.reshape(1, -1))
                    
                    for i in range(1, len(indices.flatten())):
                        neighbor_index = indices.flatten()[i]
                        neighbor_course_id = course_word_matrix.index[neighbor_index]
                        recommended_courses.add(neighbor_course_id)

            # Remove already enrolled courses
            recommended_courses = recommended_courses - set(enrolled_course_ids)

            for course_id in recommended_courses:
                users.append(user_id)
                courses.append(course_id)
                scores.append(1.0) # Constant score for now
        
        elif model_name == models[5]: # NMF
            with open('data/nmf_model.pkl', 'rb') as f:
                nmf = pickle.load(f)
            with open('data/nmf_model_columns.pkl', 'rb') as f:
                model_columns = pickle.load(f)
            
            ratings_df = load_ratings()
            user_item_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating', fill_value=0)
            user_item_matrix = user_item_matrix.reindex(columns=model_columns, fill_value=0)
            
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            user_index = user_item_matrix.index.get_loc(user_id)
            user_vector = user_item_matrix.iloc[user_index].values.reshape(1, -1)
            
            # Transform the user vector into the lower-dimensional space
            user_topic_dist = nmf.transform(user_vector)
            
            # Reconstruct the full ratings vector
            predicted_ratings = user_topic_dist @ nmf.components_
            
            # Create a series with the predicted ratings
            predicted_ratings_series = pd.Series(predicted_ratings.flatten(), index=user_item_matrix.columns)
            
            # Filter out courses the user has already rated
            unrated_courses = predicted_ratings_series.drop(enrolled_course_ids, errors='ignore')
            
            # Sort the courses by predicted rating
            top_courses = unrated_courses.sort_values(ascending=False)
            
            # Get top N courses
            top_n = params.get('top_courses', 10)
            
            for course_id, score in top_courses.head(top_n).items():
                users.append(user_id)
                courses.append(course_id)
                scores.append(score)
        elif model_name == models[6]: # Neural Network
            with open('data/nn_model.pkl', 'rb') as f:
                mlp = pickle.load(f)
            with open('data/nn_user_profiles.pkl', 'rb') as f:
                user_profiles = pickle.load(f)
            with open('data/nn_course_word_matrix.pkl', 'rb') as f:
                course_word_matrix = pickle.load(f)

            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            # Get user profile
            if user_id in user_profiles:
                user_profile = user_profiles[user_id]
            else: # New user
                enrolled_course_ids = [c for c in enrolled_course_ids if c in course_word_matrix.index]
                if not enrolled_course_ids:
                    return pd.DataFrame({'USER': [], 'COURSE_ID': [], 'SCORE': []})
                user_profile = course_word_matrix.loc[enrolled_course_ids].mean().values

            # Predict ratings for unrated courses
            unrated_courses = course_word_matrix.index.difference(enrolled_course_ids)
            
            pred_X = []
            for course_id in unrated_courses:
                course_vector = course_word_matrix.loc[course_id].values
                feature_vector = list(user_profile) + list(course_vector)
                pred_X.append(feature_vector)

            if pred_X:
                predicted_ratings = mlp.predict(pred_X)
                
                # Create a series with the predicted ratings
                predicted_ratings_series = pd.Series(predicted_ratings, index=unrated_courses)

                # Sort the courses by predicted rating
                top_courses = predicted_ratings_series.sort_values(ascending=False)

                # Get top N courses
                top_n = params.get('top_courses', 10)

                for course_id, score in top_courses.head(top_n).items():
                    users.append(user_id)
                    courses.append(course_id)
                    scores.append(score)
        elif model_name == models[7]: # Regression with Embedding Features
            with open('data/reg_nmf_model.pkl', 'rb') as f:
                nmf = pickle.load(f)
            with open('data/reg_model.pkl', 'rb') as f:
                reg_model = pickle.load(f)
            with open('data/reg_model_columns.pkl', 'rb') as f:
                model_columns = pickle.load(f)

            ratings_df = load_ratings()
            user_item_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating', fill_value=0)
            user_item_matrix = user_item_matrix.reindex(columns=model_columns, fill_value=0)
            
            W = nmf.transform(user_item_matrix)
            H = nmf.components_
            user_embeddings = pd.DataFrame(W, index=user_item_matrix.index)
            course_embeddings = pd.DataFrame(H.T, index=user_item_matrix.columns)

            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            # Get user embedding
            if user_id in user_embeddings.index:
                user_vec = user_embeddings.loc[user_id].values
            else: # New user - can't predict
                return pd.DataFrame({'USER': [], 'COURSE_ID': [], 'SCORE': []})

            # Predict ratings for unrated courses
            unrated_courses = course_embeddings.index.difference(enrolled_course_ids)
            
            pred_X = []
            for course_id in unrated_courses:
                if course_id in course_embeddings.index:
                    course_vec = course_embeddings.loc[course_id].values
                    feature_vector = list(user_vec) + list(course_vec)
                    pred_X.append(feature_vector)

            if pred_X:
                predicted_ratings = reg_model.predict(pred_X)
                
                predicted_ratings_series = pd.Series(predicted_ratings, index=unrated_courses)

                top_courses = predicted_ratings_series.sort_values(ascending=False)

                top_n = params.get('top_courses', 10)

                for course_id, score in top_courses.head(top_n).items():
                    users.append(user_id)
                    courses.append(course_id)
                    scores.append(score)
        elif model_name == models[8]: # Classification with Embedding Features
            with open('data/clf_nmf_model.pkl', 'rb') as f:
                nmf = pickle.load(f)
            with open('data/clf_model.pkl', 'rb') as f:
                clf_model = pickle.load(f)
            with open('data/clf_model_columns.pkl', 'rb') as f:
                model_columns = pickle.load(f)

            ratings_df = load_ratings()
            user_item_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating', fill_value=0)
            user_item_matrix = user_item_matrix.reindex(columns=model_columns, fill_value=0)
            
            W = nmf.transform(user_item_matrix)
            H = nmf.components_
            user_embeddings = pd.DataFrame(W, index=user_item_matrix.index)
            course_embeddings = pd.DataFrame(H.T, index=user_item_matrix.columns)

            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            # Get user embedding
            if user_id in user_embeddings.index:
                user_vec = user_embeddings.loc[user_id].values
            else: # New user - can't predict
                return pd.DataFrame({'USER': [], 'COURSE_ID': [], 'SCORE': []})

            # Predict probabilities for unrated courses
            unrated_courses = course_embeddings.index.difference(enrolled_course_ids)
            
            pred_X = []
            # Create a list to store the course ids for which we are making predictions
            pred_course_ids = []
            for course_id in unrated_courses:
                if course_id in course_embeddings.index:
                    course_vec = course_embeddings.loc[course_id].values
                    feature_vector = list(user_vec) + list(course_vec)
                    pred_X.append(feature_vector)
                    pred_course_ids.append(course_id)

            if pred_X:
                predicted_probs = clf_model.predict_proba(pred_X)[:, 1] # Get probability of class 1
                
                predicted_probs_series = pd.Series(predicted_probs, index=pred_course_ids)

                top_courses = predicted_probs_series.sort_values(ascending=False)

                top_n = params.get('top_courses', 10)

                for course_id, score in top_courses.head(top_n).items():
                    users.append(user_id)
                    courses.append(course_id)
                    scores.append(score)

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
