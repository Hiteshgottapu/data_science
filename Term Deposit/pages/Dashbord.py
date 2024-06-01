import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Function to load data
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')  # Make sure train.csv exists
    test = pd.read_csv('test.csv')    # Make sure test.csv exists
    return train, test

# Load data
train, test = load_data()

# Title
st.title('Interactive Dashboard')

# Sidebar
st.sidebar.title('Dashboard Controls')

# Sidebar options
numerical_columns = train.select_dtypes(include='number').columns.tolist()
x_column = st.sidebar.selectbox('Select X Column', numerical_columns)
y_column = st.sidebar.selectbox('Select Y Column', numerical_columns)
z_column = st.sidebar.selectbox('Select Z Column', numerical_columns)

# Data Visualization
st.write('### Data Visualization')

# Create two columns
col1, col2 = st.columns(2)

# 3D Scatter plot
st.write('#### 3D Scatter plot')
fig = px.scatter_3d(train, x=x_column, y=y_column, z=z_column, color='subscribed')
fig.update_layout(width=600, height=500)
st.plotly_chart(fig)

# Checkbox for Correlation Surface Plot
show_correlation_surface = st.sidebar.checkbox('Show Correlation Surface Plot')

if show_correlation_surface:
    st.write('#### 3D Surface plot for Correlation')
    correlation_matrix = train.corr()
    fig = go.Figure(data=[go.Surface(z=correlation_matrix.values)])
    fig.update_layout(title='Correlation Surface Plot', width=600, height=500)
    st.plotly_chart(fig)

# Additional Comparisons
st.sidebar.header('Additional Comparisons')
comparison_type = st.sidebar.radio('Select Comparison Type', ['Histogram', 'Box Plot'])

if comparison_type == 'Histogram':
    selected_column_hist = st.sidebar.selectbox('Select a Column for Histogram', numerical_columns)
    st.write('### Histogram')
    fig = px.histogram(train, x=selected_column_hist, color='subscribed', marginal='rug')
    st.plotly_chart(fig)

elif comparison_type == 'Box Plot':
    selected_column_box = st.sidebar.selectbox('Select a Column for Box Plot', numerical_columns)
    st.write('### Box Plot')
    fig = px.box(train, x='subscribed', y=selected_column_box, points='all')
    st.plotly_chart(fig)
    