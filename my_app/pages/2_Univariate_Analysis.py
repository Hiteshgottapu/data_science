import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore

# Use the new caching mechanism with st.cache_data
@st.cache_data
def load_data():
    # Load dataset
    train = pd.read_csv('train.csv')
    return train

# Load the data

train = load_data()

# Title of the app
st.title("Term Deposit Classification Model")


# Data Overview section

st.header("Data Overview")
st.write("Summary and initial observations of the dataset.")
# Display basic information about the dataset
st.write(train.describe())
st.write(train.info())
st.write("First few rows of the dataset:")
st.write(train.head())

# Univariate Analysis section

st.header("Univariate Analysis")

    # Target variable distribution
st.subheader("Target Variable Distribution: Subscribed")
subscribed_counts = train['subscribed'].value_counts()
fig = px.bar(subscribed_counts, x=subscribed_counts.index, y=subscribed_counts.values,
                labels={'index': 'Subscribed', 'y': 'Count'},
                title="Distribution of Subscription Status")
st.plotly_chart(fig)

    # Age distribution
st.subheader("Age Distribution")
fig = px.histogram(train, x='age', nbins=40, title="Age Distribution",
                       labels={'age': 'Age', 'count': 'Count'})
st.plotly_chart(fig)

    # Job distribution
st.subheader("Job Distribution")
job_counts = train['job'].value_counts()
fig = px.bar(job_counts, x=job_counts.index, y=job_counts.values,
                labels={'index': 'Job', 'y': 'Count'},
                title="Distribution of Jobs")
st.plotly_chart(fig)

# Default distribution
st.subheader("Default Distribution")
default_counts = train['default'].value_counts()
fig = px.bar(default_counts, x=default_counts.index, y=default_counts.values,
                labels={'index': 'Default', 'y': 'Count'},
                title="Distribution of Default Status")
st.plotly_chart(fig)

