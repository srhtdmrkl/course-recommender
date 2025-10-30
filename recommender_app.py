import streamlit as st
import pandas as pd
import time
import os
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()


def check_model_exists(model_name):
    if model_name == backend.models[0]: # Course Similarity
        return True # No model to train
    elif model_name == backend.models[1]: # User Profile
        return True # No model to train
    elif model_name == backend.models[2]: # Clustering
        return os.path.exists('cluster_model.pkl')
    elif model_name == backend.models[3]: # Clustering with PCA
        return os.path.exists('pca_model.pkl') and os.path.exists('cluster_model.pkl')
    elif model_name == backend.models[4]: # KNN
        return os.path.exists('knn_model.pkl') and os.path.exists('knn_course_word_matrix.pkl')
    elif model_name == backend.models[5]: # NMF
        return os.path.exists('nmf_model.pkl') and os.path.exists('nmf_model_columns.pkl')
    elif model_name == backend.models[6]: # Neural Network
        return os.path.exists('nn_model.pkl') and os.path.exists('nn_user_profiles.pkl') and os.path.exists('nn_course_word_matrix.pkl')
    elif model_name == backend.models[7]: # Regression with Embedding Features
        return os.path.exists('reg_nmf_model.pkl') and os.path.exists('reg_model.pkl') and os.path.exists('reg_model_columns.pkl')
    elif model_name == backend.models[8]: # Classification with Embedding Features
        return os.path.exists('clf_nmf_model.pkl') and os.path.exists('clf_model.pkl') and os.path.exists('clf_model_columns.pkl')
    else:
        return False


# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_on=['selectionChanged'],
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


@st.cache_data
def cached_train(model_name, params_tuple):
    """
    A cached version of the training function.
    Note: params are passed as a tuple of items to be hashable for caching.
    """
    params = dict(params_tuple)
    backend.train(model_name, params)
    return True


def predict(model_name, user_ids, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# TODO: Add hyper-parameters for other models
# User profile model
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)
    params['user_sim_threshold'] = profile_sim_threshold
# Clustering model
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
    params['cluster_no'] = cluster_no
# Clustering with PCA
elif model_selection == backend.models[3]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
    params['cluster_no'] = cluster_no
    pca_n_components = st.sidebar.slider('Number of PCA components',
                                          min_value=2, max_value=20,
                                          value=5, step=1)
    params['pca_n_components'] = pca_n_components
# KNN
elif model_selection == backend.models[4]:
    k_neighbors = st.sidebar.slider('Number of neighbors (k)',
                                     min_value=1, max_value=20,
                                     value=5, step=1)
    params['k_neighbors'] = k_neighbors
# NMF
elif model_selection == backend.models[5]:
    n_components = st.sidebar.slider('Number of components',
                                     min_value=2, max_value=20,
                                     value=5, step=1)
    params['n_components'] = n_components
# Neural Network
elif model_selection == backend.models[6]:
    hidden_layer_sizes = st.sidebar.text_input('Hidden layer sizes (comma-separated)', '100,50')
    params['hidden_layer_sizes'] = tuple(map(int, hidden_layer_sizes.split(',')))
    activation = st.sidebar.selectbox('Activation function', ['relu', 'tanh', 'logistic'])
    params['activation'] = activation
    learning_rate = st.sidebar.slider('Learning rate',
                                      min_value=0.001, max_value=0.1,
                                      value=0.01, step=0.001, format="%.3f")
    params['learning_rate'] = learning_rate
# Regression with Embedding Features
elif model_selection == backend.models[7]:
    reg_model = st.sidebar.selectbox('Regression model', ['Linear Regression', 'Ridge', 'Lasso'])
    params['reg_model'] = reg_model
# Classification with Embedding Features
elif model_selection == backend.models[8]:
    clf_model = st.sidebar.selectbox('Classification model', ['Logistic Regression', 'SVM', 'Random Forest'])
    params['clf_model'] = clf_model
else:
    pass


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    # Convert params dict to a hashable type for caching
    params_tuple = tuple(sorted(params.items()))
    with st.spinner('Training...'):
        cached_train(model_selection, params_tuple)
    st.success('Done!')


# Prediction
st.sidebar.subheader('4. Prediction')
# Check if model exists
model_exists = check_model_exists(model_selection)
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses", disabled=not model_exists)
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    
    st.subheader("Recommended courses: ")
    gb_rec = GridOptionsBuilder.from_dataframe(res_df)
    gb_rec.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb_rec.configure_side_bar()
    grid_options_rec = gb_rec.build()

    AgGrid(
        res_df,
        gridOptions=grid_options_rec,
        enable_enterprise_modules=True,
        fit_columns_on_grid_load=True,
    )
