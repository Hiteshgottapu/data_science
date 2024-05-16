import streamlit as st
import pandas as pd
import plotly.express as px
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import re

# Function to display the artifact description
def artifact_description():
    st.header("Artifact Description")
    st.subheader("Quality Rating System")
    st.write("A scoring system that evaluates and ranks code quality.")
    
    st.subheader("Detailed Issue Reporting")
    st.write("Reports detailing specific issues and suggesting fixes.")
    
    st.subheader("Adaptability to Languages")
    st.write("Capability to analyze multiple programming languages.")
    
    st.subheader("User Dashboard")
    st.write("A user-friendly dashboard for managing projects, viewing reports, and tracking code quality over time.")

# Function to display the aim and objectives of the project
def aim_and_objectives():
    st.header("Aim & Objectives of the Project")
    st.subheader("Primary Aim")
    st.write("To develop machine-learning-based software that can be widely used to automatically assess the quality of code.")
    
    st.subheader("Objectives")
    st.write("- To identify and employ the best machine learning techniques for code analysis.")
    st.write("- To design a software architecture composed of separate modules that can be easily expanded with new analysis models and programming environments.")
    st.write("- To test the tool's effectiveness against well-known benchmarks and real-world code bases.")
    st.write("- To enhance user interaction by providing clear, easy-to-understand reports and actionable insights for improving code quality.")

# Function to load and preprocess the PROMISE dataset
@st.cache_data
def load_dataset():
    # Load the PROMISE dataset
    data = pd.read_csv("promise_dataset.csv")
    
    # Preprocess the dataset
    data.dropna(inplace=True)
    
    return data

# Function to preprocess the dataset for model training
def preprocess_data(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Function to train a TensorFlow model
# Function to train a TensorFlow model
def train_model(X_train, X_test, y_train, y_test, epochs, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, history, accuracy, precision, recall, f1


# Function to display model training results
def display_training_results(history, accuracy, precision, recall, f1):
    st.header("Model Training Results")
    st.write("Training Loss:", history.history['loss'][-1])
    st.write("Training Accuracy:", history.history['accuracy'][-1])
    st.write("Training Precision:", history.history['precision'][-1])
    st.write("Training Recall:", history.history['recall'][-1])
    st.write("Validation Loss:", history.history['val_loss'][-1])
    st.write("Validation Accuracy:", history.history['val_accuracy'][-1])
    st.write("Validation Precision:", history.history['val_precision'][-1])
    st.write("Validation Recall:", history.history['val_recall'][-1])
    st.write("Test Accuracy:", accuracy)
    st.write("Test Precision:", precision)
    st.write("Test Recall:", recall)
    st.write("Test F1-Score:", f1)
    
    fig_acc = px.line(x=range(len(history.history['accuracy'])), y=[history.history['accuracy'], history.history['val_accuracy']],
                      labels={'x': 'Epoch', 'value': 'Accuracy'}, title='Model Accuracy')
    fig_acc.update_layout(yaxis=dict(title='Accuracy'), xaxis=dict(title='Epoch'))
    fig_acc.update_traces(name='Train', line=dict(color='blue'), selector=dict(type='scatter', mode='lines', showlegend=True))
    fig_acc.update_traces(name='Validation', line=dict(color='red'), selector=dict(type='scatter', mode='lines', showlegend=True))
    st.plotly_chart(fig_acc)
    
    fig_loss = px.line(x=range(len(history.history['loss'])), y=[history.history['loss'], history.history['val_loss']],
                       labels={'x': 'Epoch', 'value': 'Loss'}, title='Model Loss')
    fig_loss.update_layout(yaxis=dict(title='Loss'), xaxis=dict(title='Epoch'))
    fig_loss.update_traces(name='Train', line=dict(color='blue'), selector=dict(type='scatter', mode='lines', showlegend=True))
    fig_loss.update_traces(name='Validation', line=dict(color='red'), selector=dict(type='scatter', mode='lines', showlegend=True))
    st.plotly_chart(fig_loss)

# Function to perform Exploratory Data Analysis
def perform_eda(data):
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Dataset Overview")
    st.write(data.head())
    
    st.subheader("Dataset Statistics")
    st.write(data.describe())
    
    st.subheader("Correlation Heatmap")
    fig = px.imshow(data.corr(), text_auto=True, aspect="auto", color_continuous_scale='Viridis')
    st.plotly_chart(fig)

# Function to create a dashboard with different graphs
def dashboard(data):
    st.header("Interactive Dashboard")
    st.write("Explore the dataset using the interactive visualizations below. Select the desired variables from the dropdowns, and the plots will update automatically.")

    # Create a sidebar for selecting variables
    st.sidebar.header("Select Variables")
    selected_variables = st.sidebar.multiselect("Choose variables to display", data.columns, default=list(data.columns[:3]))

    if selected_variables:
        # Scatter Plot
        st.subheader("Scatter Plot")
        scatter_x = st.selectbox("Select X-axis for Scatter Plot", selected_variables, key="scatter_x")
        scatter_y = st.selectbox("Select Y-axis for Scatter Plot", [var for var in selected_variables if var != scatter_x], key="scatter_y")
        fig_scatter = px.scatter(data, x=scatter_x, y=scatter_y, title=f"{scatter_x} vs {scatter_y} (Scatter Plot)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Histogram
        st.subheader("Histogram")
        hist_var = st.selectbox("Select variable for Histogram", selected_variables, key="hist_var")
        fig_hist = px.histogram(data, x=hist_var, nbins=30, title=f"Histogram of {hist_var}")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Box Plot
        st.subheader("Box Plot")
        box_var = st.selectbox("Select variable for Box Plot", selected_variables, key="box_var")
        fig_box = px.box(data, y=box_var, title=f"Box Plot of {box_var}")
        st.plotly_chart(fig_box, use_container_width=True)

        # Pair Plot
        st.subheader("Pair Plot")
        fig_pair = px.scatter_matrix(data[selected_variables], title='Pair Plot')
        st.plotly_chart(fig_pair, use_container_width=True)

        # Line Plot
        st.subheader("Line Plot")
        line_x = st.selectbox("Select X-axis for Line Plot", selected_variables, key="line_x")
        line_y = st.selectbox("Select Y-axis for Line Plot", [var for var in selected_variables if var != line_x], key="line_y")
        fig_line = px.line(data, x=line_x, y=line_y, title=f"{line_x} vs {line_y} (Line Plot)")
        st.plotly_chart(fig_line, use_container_width=True)

        # Heatmap
        st.subheader("Correlation Heatmap")
        fig_heatmap = px.imshow(data[selected_variables].corr(), text_auto=True, aspect="auto", color_continuous_scale='Viridis')
        st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        st.warning("Please select at least one variable to display the visualizations.")

# Function to analyze uploaded code and rate its quality
def analyze_code(uploaded_file):
    st.header("Code Analysis Results")
    
    # Read uploaded file
    content = uploaded_file.read().decode("utf-8")
    lines = content.split('\n')
    
    num_lines = len(lines)
    num_functions = sum(1 for line in lines if line.strip().startswith('def '))
    num_classes = sum(1 for line in lines if line.strip().startswith('class '))
    num_comments = sum(1 for line in lines if line.strip().startswith('#'))
    num_empty_lines = sum(1 for line in lines if line.strip() == '')
    
    # Basic code quality metrics
    comment_density = num_comments / num_lines if num_lines > 0 else 0
    function_density = num_functions / num_lines if num_lines > 0 else 0
    class_density = num_classes / num_lines if num_lines > 0 else 0
    empty_line_ratio = num_empty_lines / num_lines if num_lines > 0 else 0
    
    # Code quality rating (out of 10)
    rating = (comment_density * 2 + function_density + class_density + (1 - empty_line_ratio)) * 2.5
    rating = min(rating, 10)  # Ensure rating does not exceed 10
    
    st.write("Total Lines of Code:", num_lines)
    st.write("Number of Functions:", num_functions)
    st.write("Number of Classes:", num_classes)
    st.write("Number of Comments:", num_comments)
    st.write("Number of Empty Lines:", num_empty_lines)
    
    st.write("Comment Density:", comment_density)
    st.write("Function Density:", function_density)
    st.write("Class Density:", class_density)
    st.write("Empty Line Ratio:", empty_line_ratio)
    
    st.subheader(f"Code Quality Rating: {rating:.2f} / 10")
    
    # Display code content
    st.subheader("Uploaded Code")
    st.code(content, language='python')

# Main function to run the Streamlit app
def main():
    st.set_page_config(layout="wide")  # Set the layout to wide to fit the dashboard
    st.title("Agile Code Quality Assessment Tool")
    
    st.sidebar.title("Sections")
    selected_section = st.sidebar.selectbox("Select a section", 
                                            options=["Artifact Description", "Aim & Objectives", "EDA", "Dashboard", "Model Training", "Code Analysis"])
    
    if selected_section == "Artifact Description":
        artifact_description()
    elif selected_section == "Aim & Objectives":
        aim_and_objectives()
    elif selected_section == "EDA":
        data = load_dataset()
        perform_eda(data)
    elif selected_section == "Dashboard":
        data = load_dataset()
        dashboard(data)
        
    elif selected_section == "Model Training":
        data = load_dataset()
        epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=10, step=1)
        batch_size = st.slider("Batch Size", min_value=1, max_value=128, value=32, step=1)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        st.write("Dataset loaded and preprocessed.")
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                time.sleep(1)  # simulate some delay
                model, history, accuracy, precision, recall, f1 = train_model(X_train, X_test, y_train, y_test, epochs, batch_size)
                st.write("Model trained successfully.")
                display_training_results(history, accuracy, precision, recall, f1)
    elif selected_section == "Code Analysis":
        uploaded_file = st.file_uploader("Upload your code file", type=["py", "java", "js", "cpp", "c", "cs", "rb", "php"])
        if uploaded_file is not None:
            analyze_code(uploaded_file)

if __name__ == "__main__":
    main()
