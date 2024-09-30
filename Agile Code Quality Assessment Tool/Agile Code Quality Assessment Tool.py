import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample, class_weight
import joblib
import time
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
from xgboost import XGBClassifier
from langdetect import detect, DetectorFactory
from sklearn.metrics import roc_curve, auc
import re
import warnings
warnings.filterwarnings('ignore')

# Set seed for langdetect to get consistent results
DetectorFactory.seed = 0

# Set page configuration
st.set_page_config(page_title="Code Quality Assessment with Animations", layout="wide")

# Enhanced CSS for improved UI/UX with animations, micro-interactions, and dark mode support
css_code = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--background-color, #f4f6f9);
            color: var(--text-color, #2c3e50);
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-bottom: 20px;
            transition: all 0.3s ease-in-out;
        }
        .sidebar .sidebar-content {
            background-color: #f9f9f9;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button {
            color: white;
            background: linear-gradient(45deg, #007bff, #0056b3);
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            transition: background 0.3s ease-in-out, transform 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #0056b3, #004085);
            transform: translateY(-3px);
        }
        .card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }
        .card:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(45deg, #007bff, #0056b3);
            color: white;
            margin-bottom: 2rem;
            border-radius: 10px;
            transition: background 0.3s ease-in-out;
        }
        .header:hover {
            background: linear-gradient(45deg, #0056b3, #004085);
        }
        .header h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            transition: color 0.3s ease-in-out;
        }
        .header:hover h1 {
            color: #cce5ff;
        }
        .header p {
            font-size: 1.25rem;
            transition: color 0.3s ease-in-out;
        }
        .header:hover p {
            color: #cce5ff;
        }
        .footer {
            text-align: center;
            padding: 1rem;
            background-color: #2c3e50;
            color: white;
            margin-top: 2rem;
            border-radius: 10px;
            transition: background 0.3s ease-in-out;
        }
        .footer:hover {
            background-color: #1a252f;
        }
        .dark-mode {
            --background-color: #2c3e50;
            --text-color: #f4f6f9;
        }
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }
            .header p {
                font-size: 1rem;
            }
        }
    </style>
"""

# Apply CSS styles
st.markdown(css_code, unsafe_allow_html=True)

def header():
    """Display a custom header with hover effects."""
    st.markdown("""
        <div class="header">
            <h1>Code Quality Assessment Dashboard</h1>
            <p>Analyze and improve your code quality using advanced machine learning techniques</p>
        </div>
    """, unsafe_allow_html=True)

def footer():
    """Display a custom footer with dark mode toggle."""
    st.markdown("""
        <div class="footer">
            <p>&copy; 2024 Code Quality Assessment | Built with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

# Define the main content of the app with updated UI elements
class CodeQualityApp:
    def __init__(self):
        self.rf_model = None
    
    def card_component(self, title, content, icon):
        """Reusable card component with hover effects."""
        st.markdown(f"""
        <div class="card">
            <h1>{icon}</h1>
            <h3>{title}</h3>
            <p>{content}</p>
        </div>
        """, unsafe_allow_html=True)

    def artifact_description(self):
        st.markdown("<div class='custom-header'>Artifact Description</div>", unsafe_allow_html=True)
        self.card_component("Quality Rating System", "A scoring system that evaluates and ranks code quality.", "📊")
        self.card_component("Detailed Issue Reporting", "Reports detailing specific issues and suggesting fixes.", "🔍")
        self.card_component("Adaptability to Languages", "Capability to analyze multiple programming languages.", "🌐")
        self.card_component("User Dashboard", "A user-friendly dashboard for managing projects, viewing reports, and tracking code quality over time.", "📋")

    def aim_and_objectives(self):
        st.markdown("<div class='custom-header'>Aim & Objectives of the Project</div>", unsafe_allow_html=True)
        self.card_component("Primary Aim", "To develop machine-learning-based software that can be widely used to automatically assess the quality of code.", "🎯")
        objectives = [
            "To identify and employ the best machine learning techniques for code analysis.",
            "To design a software architecture composed of separate modules that can be easily expanded with new analysis models and programming environments.",
            "To test the tool's effectiveness against well-known benchmarks and real-world code bases.",
            "To enhance user interaction by providing clear, easy-to-understand reports and actionable insights for improving code quality."
        ]
        st.write("- " + "\n- ".join(objectives))

    def preprocess_data(self, data):
        """Preprocess the dataset and return train-test splits."""
        try:
            X = data.iloc[:, :-1].values.astype(np.float32)
            y = data.iloc[:, -1].values.astype(np.int32)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            st.error(f"An error occurred during data preprocessing: {e}")
            return None, None, None, None
    def display_roc_curve(self, y_test, y_prob):
        """Display the ROC curve and calculate AUC."""
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        st.markdown("<div class='custom-header'>ROC Curve</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    def display_class_distribution(self, y_train, y_train_upsampled):
        """Display the class distribution before and after resampling."""
        st.markdown("<div class='custom-header'>Class Distribution</div>", unsafe_allow_html=True)
        original_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
        upsampled_dist = pd.Series(y_train_upsampled).value_counts(normalize=True).sort_index()

        st.write("**Original Class Distribution**")
        st.bar_chart(original_dist)

        st.write("**Upsampled Class Distribution**")
        st.bar_chart(upsampled_dist)

    def perform_meta_learning_kfold(self, X_train, y_train):
        """Perform Meta-learning with KFold Cross-validation and Hyperparameter Tuning."""
        
        # Define base models and hyperparameters
        rf = RandomForestClassifier(random_state=42)
        gb = GradientBoostingClassifier(random_state=42)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
        param_grid_gb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }

        # Limit parallel jobs and number of iterations for memory efficiency
        rf_random_search = RandomizedSearchCV(rf, param_distributions=param_grid_rf, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=1)
        gb_random_search = RandomizedSearchCV(gb, param_distributions=param_grid_gb, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=1)
        xgb_random_search = RandomizedSearchCV(xgb, param_distributions=param_grid_xgb, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=1)

        # Use a smaller sample of the data for tuning
        X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

        # Fit the models on the training sample
        rf_random_search.fit(X_train_sample, y_train_sample)
        gb_random_search.fit(X_train_sample, y_train_sample)
        xgb_random_search.fit(X_train_sample, y_train_sample)

        # Get the best models
        best_rf = rf_random_search.best_estimator_
        best_gb = gb_random_search.best_estimator_
        best_xgb = xgb_random_search.best_estimator_

        # Cache the best models
        joblib.dump(best_rf, 'best_rf_model.pkl')
        joblib.dump(best_gb, 'best_gb_model.pkl')
        joblib.dump(best_xgb, 'best_xgb_model.pkl')

        # Create a VotingClassifier for meta-learning
        ensemble_model = VotingClassifier(estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('xgb', best_xgb)
        ], voting='soft')

        # Perform K-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_scores = cross_val_score(ensemble_model, X_train, y_train, cv=kf, scoring='accuracy')

        # Fit the final model
        ensemble_model.fit(X_train, y_train)

        return ensemble_model, np.mean(accuracy_scores)

    def perform_eda(self, data):
        """Perform exploratory data analysis on the dataset."""
        try:
            st.markdown("<div class='custom-header'>Exploratory Data Analysis (EDA)</div>", unsafe_allow_html=True)
            st.subheader("Dataset Overview")
            st.write(data.head())

            st.subheader("Dataset Statistics")
            st.write(data.describe())

            st.subheader("Correlation Heatmap")
            fig = px.imshow(data.corr(), text_auto=True, aspect="auto", color_continuous_scale='Viridis')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred during EDA: {e}")

    def customize_plot(self, fig, title):
        """Apply consistent customizations to the plots."""
        fig.update_layout(
            title=title,
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=0),
            title_x=0.5,
            font=dict(family="Roboto, sans-serif")
        )
        return fig

    def generate_animated_data(self):
        """Generate sample data for an animated plot."""
        np.random.seed(42)
        n_points = 500
        time_steps = 50
        data = pd.DataFrame({
            "x": np.random.randn(n_points),
            "y": np.random.randn(n_points),
            "time": np.repeat(np.arange(time_steps), n_points // time_steps),
            "category": np.random.choice(['A', 'B', 'C'], size=n_points)
        })
        return data

    def plot_animated_scatter(self, data):
        """Plot an animated scatter plot."""
        fig = px.scatter(
            data,
            x="x",
            y="y",
            animation_frame="time",
            animation_group="category",
            color="category",
            hover_name="category",
            size_max=55,
            range_x=[-3, 3],
            range_y=[-3, 3],
            title="Animated Scatter Plot"
        )

        fig = self.customize_plot(fig, "Animated Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)
    def lime_explanation(self, model, X_test, feature_names):
        """Explain model predictions using LIME."""
        try:
            st.markdown("<div class='custom-header'>LIME Explanation</div>", unsafe_allow_html=True)
            
            # Create a LIME explainer
            explainer = LimeTabularExplainer(X_test, feature_names=feature_names, class_names=['Low', 'High'], mode='classification')
            
            # Select a random instance from the test set for explanation
            i = np.random.randint(0, X_test.shape[0])
            instance = X_test[i].reshape(1, -1)
            
            # Get the explanation for this instance
            explanation = explainer.explain_instance(instance[0], model.predict_proba, num_features=5)
            
            # Display the explanation
            st.write(f"Instance {i}:")
            
            # Plot the LIME explanation
            fig = explanation.as_pyplot_figure()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred during LIME explanation: {e}")

    def dashboard(self, data):
        """Create an interactive dashboard with various plots."""
        try:
            st.markdown("<div class='custom-header'>Interactive Dashboard</div>", unsafe_allow_html=True)
            st.write("Explore the dataset using the interactive visualizations below. Select the desired variables from the dropdowns, and the plots will update automatically.")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Scatter Plot")
                scatter_x = st.selectbox("Select X-axis for Scatter Plot", data.columns.tolist(), index=0)
                scatter_y = st.selectbox("Select Y-axis for Scatter Plot", [col for col in data.columns if col != scatter_x], index=1)
                
                if scatter_x and scatter_y:
                    fig_scatter = px.scatter(data, x=scatter_x, y=scatter_y, 
                                             title=f"{scatter_x} vs {scatter_y}",
                                             color=scatter_y, 
                                             trendline="ols",
                                             hover_data=[scatter_x, scatter_y])
                    fig_scatter = self.customize_plot(fig_scatter, f"{scatter_x} vs {scatter_y}")
                    st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                st.subheader("Histogram")
                hist_var = st.selectbox("Select variable for Histogram", data.columns.tolist())
                bins = st.slider("Number of Bins", min_value=5, max_value=100, value=30)
                
                if hist_var:
                    fig_hist = px.histogram(data, x=hist_var, nbins=bins, title=f"Histogram of {hist_var}",
                                            color=hist_var, marginal="box")
                    fig_hist = self.customize_plot(fig_hist, f"Histogram of {hist_var}")
                    st.plotly_chart(fig_hist, use_container_width=True)

            with st.expander("More Visualizations", expanded=False):
                st.markdown("<div class='expander-header'>Additional Plots</div>", unsafe_allow_html=True)
                st.subheader("Box Plot")
                box_var = st.selectbox("Select variable for Box Plot", data.columns.tolist())
                
                if box_var:
                    fig_box = px.box(data, y=box_var, title=f"Box Plot of {box_var}")
                    fig_box = self.customize_plot(fig_box, f"Box Plot of {box_var}")
                    st.plotly_chart(fig_box, use_container_width=True)

                st.subheader("Line Plot")
                line_x = st.selectbox("Select X-axis for Line Plot", data.columns.tolist())
                line_y = st.selectbox("Select Y-axis for Line Plot", [col for col in data.columns if col != line_x])
                
                if line_x and line_y:
                    fig_line = px.line(data, x=line_x, y=line_y, title=f"{line_x} vs {line_y}")
                    fig_line = self.customize_plot(fig_line, f"{line_x} vs {line_y}")
                    st.plotly_chart(fig_line, use_container_width=True)

                st.subheader("Correlation Heatmap")
                selected_vars = st.multiselect("Select Variables for Heatmap", data.columns.tolist(), default=data.columns.tolist()[:3])

                if len(selected_vars) > 0:
                    fig_heatmap = px.imshow(data[selected_vars].corr(), 
                                            text_auto=True, 
                                            aspect="auto", 
                                            color_continuous_scale='Viridis',
                                            labels=dict(color="Correlation"))
                    fig_heatmap = self.customize_plot(fig_heatmap, "Correlation Heatmap")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.warning("Please select at least one variable for the heatmap.")
        except Exception as e:
            st.error(f"An error occurred in the dashboard: {e}")

    def extract_code_features(self, content, language=None):
        """Extract features from the uploaded code based on the language."""
        try:
            # Split the code content into lines
            lines = content.split('\n')

            # Feature 1-5: Basic counts
            num_lines = len(lines)
            num_functions = sum(1 for line in lines if re.match(r'^\s*(def |function )', line))  # Python, JS functions
            num_classes = sum(1 for line in lines if re.match(r'^\s*class ', line))  # Class definitions
            num_comments = sum(1 for line in lines if re.match(r'^\s*#|//|/\*|\*', line))  # Comments in various languages
            num_empty_lines = sum(1 for line in lines if line.strip() == '')

            # Feature 6: Code complexity (based on language-specific constructs)
            complexity = self.calculate_complexity(lines, language)

            # Additional features
            # Feature 7: Number of import statements
            num_imports = sum(1 for line in lines if re.match(r'^\s*import |from ', line))

            # Feature 8: Number of loops (for, while)
            num_loops = sum(1 for line in lines if re.match(r'^\s*(for |while )', line))

            # Feature 9: Number of conditional statements (if, elif, else)
            num_conditionals = sum(1 for line in lines if re.match(r'^\s*(if |else if |elif )', line))

            # Feature 10: Average line length (to capture code density)
            avg_line_length = np.mean([len(line) for line in lines if line.strip()])

            # Feature 11-12: Derived metrics
            avg_function_length = num_lines / (num_functions if num_functions > 0 else 1)
            comment_to_code_ratio = num_comments / (num_lines if num_lines > 0 else 1)

            # Additional derived or specific language features
            # Feature 13: Number of method calls (method_name())
            num_method_calls = sum(1 for line in lines if re.search(r'\w+\(.*\)', line))

            # Feature 14: Number of return statements
            num_return_statements = sum(1 for line in lines if re.match(r'^\s*return ', line))

            # Feature 15: Number of variable assignments
            num_assignments = sum(1 for line in lines if re.match(r'^\s*\w+\s*=', line))

            # Feature 16: Number of string literals
            num_string_literals = sum(1 for line in lines if re.search(r'".*?"|\'.*?\'', line))

            # Feature 17: Number of numeric literals
            num_numeric_literals = sum(1 for line in lines if re.search(r'\b\d+\b', line))

            # Feature 18: Number of boolean literals (True/False or true/false)
            num_boolean_literals = sum(1 for line in lines if re.search(r'\b(True|False|true|false)\b', line))

            # Feature 19: Max indentation level (number of leading spaces or tabs)
            max_indentation_level = max([len(line) - len(line.lstrip(' ')) for line in lines if line.strip()])

            # Feature 20: Number of function definitions starting with "test" (for test functions)
            num_test_functions = sum(1 for line in lines if re.match(r'^\s*def test', line))

            # Feature 21: Number of nested loops (based on indentation)
            num_nested_loops = self.count_nested_loops(lines)

            # Collect all features into a numpy array (21 features)
            features = np.array([num_lines, num_functions, num_classes, num_comments, num_empty_lines, complexity,
                                num_imports, num_loops, num_conditionals, avg_line_length, avg_function_length,
                                comment_to_code_ratio, num_method_calls, num_return_statements, num_assignments,
                                num_string_literals, num_numeric_literals, num_boolean_literals,
                                max_indentation_level, num_test_functions, num_nested_loops], dtype=np.float32)

            return features.reshape(1, -1)

        except Exception as e:
            st.error(f"An error occurred while extracting code features: {e}")
            return None

    # Helper method to count nested loops based on indentation
    def count_nested_loops(self, lines):
        """Count nested loops based on indentation levels."""
        indentation_stack = []
        nested_loops = 0

        for line in lines:
            stripped_line = line.strip()
            if re.match(r'^\s*(for |while )', stripped_line):
                current_indent = len(line) - len(line.lstrip())
                while indentation_stack and indentation_stack[-1] >= current_indent:
                    indentation_stack.pop()
                indentation_stack.append(current_indent)
                if len(indentation_stack) > 1:  # More than one loop on the stack means nesting
                    nested_loops += 1

        return nested_loops



    def calculate_complexity(self, lines, language):
        """Calculate the complexity of the code."""
        complexity = 0
        if language in ['Python', 'JavaScript']:
            complexity += sum(1 for line in lines if 'if ' in line or 'else ' in line or 'elif ' in line)
        elif language in ['Java', 'C++', 'C#']:
            complexity += sum(1 for line in lines if re.search(r'\b(if|else|switch|case|for|while|do)\b', line))
        elif language in ['Go', 'Rust']:
            complexity += sum(1 for line in lines if re.search(r'\b(if|else|switch|for)\b', line))

        return complexity


    def detect_language(self, content):
        """Detect the language of the provided code content."""
        try:
            return detect(content)
        except Exception as e:
            st.error(f"An error occurred during language detection: {e}")
            return "Unknown"

    def predict_code_quality(self, content):
        """Predict code quality using the trained model and return 'Yes' or 'No'."""
        try:
            # Detect the programming language of the uploaded code
            language = self.detect_language(content)
            st.markdown("<div class='custom-header'>Predicting Code Quality</div>", unsafe_allow_html=True)
            st.subheader(f"Uploaded Code (Detected Language: {language})")
            st.code(content, language=language.lower())

            # Extract features from the code content
            features = self.extract_code_features(content, language)
            if features is None:
                st.error("Failed to extract code features.")
                return

            # Display the extracted features for debugging
            st.write("Extracted Features:", features)

            # Load the trained model
            self.rf_model = joblib.load('ensemble_model.pkl')

            # Predict the probability of the code being "Good" (Yes) or "Bad" (No)
            predicted_probability = self.rf_model.predict_proba(features)
            
            # Assuming the model predicts two classes [0: 'No', 1: 'Yes'], we can choose a threshold
            # For binary classification, let's assume 'Yes' is assigned if probability of class 1 is >= 0.5
            predicted_class = "Yes" if predicted_probability[0][1] >= 0.5 else "No"

            # Display the predicted class ("Yes" or "No") based on the probability
            st.subheader(f"Predicted Code Quality: {predicted_class}")
            st.write("Predicted Probability Distribution:", predicted_probability)

        except FileNotFoundError:
            st.error("Model file not found. Please ensure the model has been trained and saved correctly.")
        except Exception as e:
            st.error(f"An error occurred during code quality prediction: {e}")

    def main(self):
        header()
        
        st.sidebar.title("Sections")
        selected_section = st.sidebar.selectbox("Select a section", 
                                                options=["Artifact Description", "Aim & Objectives", "EDA", "Dashboard", "Model Training", "Code Analysis"])

        if selected_section == "Artifact Description":
            self.artifact_description()

        elif selected_section == "Aim & Objectives":
            self.aim_and_objectives()

        elif selected_section == "EDA":
            data = load_dataset()
            if data is not None:
                self.perform_eda(data)

        elif selected_section == "Dashboard":
            data = load_dataset()
            if data is not None:
                self.dashboard(data)

                # Add animated plot to the dashboard
                st.subheader("Animated Plot")
                animated_data = self.generate_animated_data()
                self.plot_animated_scatter(animated_data)

        if selected_section == "Model Training":
            st.markdown("<div class='custom-header'>Ensemble Model Training with Meta-Learning</div>", unsafe_allow_html=True)

            data = load_dataset()
            if data is not None:
                st.subheader("Data Preprocessing")
                X_train, X_test, y_train, y_test = self.preprocess_data(data)

                if st.button("Train Ensemble Model with Meta-Learning"):
                    with st.spinner("Training the Ensemble model..."):
                        model, accuracy = self.perform_meta_learning_kfold(X_train, y_train)
                        self.card_component("Accuracy", f"{accuracy * 100:.2f}%", "✅")
                        y_prob = model.predict_proba(X_test)[:, 1]
                        y_pred = (y_prob >= 0.5).astype(int)
                        self.card_component("Test Accuracy:" ,f"{accuracy_score(y_test, y_pred):.2f}%","✅")

                        self.display_roc_curve(y_test, y_prob)

                        # LIME Explanation
                        self.lime_explanation(model, X_test, feature_names=data.columns[:-1])
                    
        elif selected_section == "Code Analysis":
            st.markdown("<div class='custom-header'>Code Analysis and Quality Prediction</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload your code file", type=["py", "java", "js", "cpp", "cs", "go", "rs"])

            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8")
                self.predict_code_quality(content)

        footer()

# Load dataset function
@st.cache_resource
def load_dataset():
    """Load and preprocess the PROMISE dataset."""
    try:
        data = pd.read_csv("promise_dataset.csv")
        data.dropna(inplace=True)
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please make sure 'promise_dataset.csv' is in the working directory.")
        return None
    except pd.errors.EmptyDataError:
        st.error("The dataset file is empty. Please provide a valid CSV file.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the dataset: {e}")
        return None

# Run the app
if __name__ == "__main__":
    app = CodeQualityApp()
    app.main()
