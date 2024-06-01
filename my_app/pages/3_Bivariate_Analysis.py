import streamlit as st # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore

# Function to load data with caching
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    return train

train = load_data()

st.title('Bivariate Analysis')

# Crosstab between job and subscribed
st.subheader('Subscription Rate by Job')
job_crosstab = pd.crosstab(train['job'], train['subscribed'])
st.write("Job crosstab:", job_crosstab)  # Debug: Check the crosstab output

# Check if 'no' and 'yes' columns are present
if 'no' in job_crosstab.columns and 'yes' in job_crosstab.columns:
    job_crosstab_norm = job_crosstab.div(job_crosstab.sum(1), axis=0).reset_index()
    job_crosstab_norm = job_crosstab_norm.melt(id_vars='job', value_vars=['no', 'yes'], var_name='subscribed', value_name='percentage')
    fig = px.bar(job_crosstab_norm, x='job', y='percentage', color='subscribed', labels={'percentage': 'Percentage', 'job': 'Job'}, barmode='stack')
    st.plotly_chart(fig)
else:
    st.write("Expected columns 'no' and 'yes' not found in job crosstab.")

# Crosstab between default and subscribed
st.subheader('Subscription Rate by Default Status')
default_crosstab = pd.crosstab(train['default'], train['subscribed'])
st.write("Default crosstab:", default_crosstab)  # Debug: Check the crosstab output

# Check if 'no' and 'yes' columns are present
if 'no' in default_crosstab.columns and 'yes' in default_crosstab.columns:
    default_crosstab_norm = default_crosstab.div(default_crosstab.sum(1), axis=0).reset_index()
    default_crosstab_norm = default_crosstab_norm.melt(id_vars='default', value_vars=['no', 'yes'], var_name='subscribed', value_name='percentage')
    fig = px.bar(default_crosstab_norm, x='default', y='percentage', color='subscribed', labels={'percentage': 'Percentage', 'default': 'Default'}, barmode='stack')
    st.plotly_chart(fig)
else:
    st.write("Expected columns 'no' and 'yes' not found in default crosstab.")

# Correlation matrix
st.subheader('Correlation Matrix')
train['subscribed'] = train['subscribed'].replace({'no': 0, 'yes': 1})
numeric_train = train.select_dtypes(include=['float64', 'int64'])
corr = numeric_train.corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='YlGnBu')
st.plotly_chart(fig)
