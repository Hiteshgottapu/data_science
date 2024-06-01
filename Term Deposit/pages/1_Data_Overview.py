import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = load_data()

st.title('Data Overview')

# Display the first few rows of training and test data
st.write('### Training Data')
st.write(train.head())

st.write('### Test Data')
st.write(test.head())

# Display data shapes
st.write('### Data Shapes')
st.write('Training Data Shape:', train.shape)
st.write('Test Data Shape:', test.shape)

# Display data types
st.write('### Data Types')
st.write(train.dtypes)

# Display summary statistics
st.write('### Summary Statistics')
st.write(train.describe())

# Data Visualization
st.write('### Data Visualization')

# 3D Scatter plot
st.write('#### 3D Scatter plot for Numerical Columns')
fig = px.scatter_3d(train, x='age', y='balance', z='duration', color='subscribed')
fig.update_layout(width=800, height=600)  # Adjust size here
st.plotly_chart(fig)

# 3D Surface plot for correlation
st.write('#### 3D Surface plot for Correlation')
correlation_matrix = train.corr()
fig = go.Figure(data=[go.Surface(z=correlation_matrix.values)])
fig.update_layout(title='Correlation Surface Plot', width=800, height=600)  # Adjust size here
st.plotly_chart(fig)
