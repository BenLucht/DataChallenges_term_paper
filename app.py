# pylint: disable=E0611
# pylint: disable=F0401
# disables pylint errors for no name in module and unable to import
import itertools

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import trimap
import pacmap
from sklearn.cluster import DBSCAN # for building a clustering model
from sklearn.preprocessing import MinMaxScaler # for feature scaling
from sklearn import metrics # for calculating Silhouette score
from cluster.selfrepresentation import ElasticNetSubspaceClustering
from biclustlib.algorithms import BiCorrelationClusteringAlgorithm



st.write('Data Challenges')

@st.cache
def load_data():
    df_read = pd.read_csv('data/physionet_data_compressed.csv', compression='zip')
    df_read = df_read.sort_values('id')
    return df_read

@st.cache
def load_data_median():
    df_read = pd.read_csv('data/physionet_data_imputed_median_all_compressed.csv', compression='zip')
    df_read = df_read.sort_values('id')
    return df_read

@st.cache
def load_data_gender_median():
    df_read = pd.read_csv('data/physionet_data_imputed_median_age60_compressed.csv', compression='zip')
    df_read = df_read.sort_values('id')
    return df_read

def print_properties(input_df):
    describe = input_df.describe()
    properties = []
    for col in input_df.columns:
        properties.append([col, describe[col]["min"], describe[col]["max"], (1-(describe[col]["count"]/input_df.shape[0]))*100])
    describe = pd.DataFrame(properties, columns = ['property', 'min', 'max', '% missing'])
    return describe

def print_hist(input_df):
    male = input_df.query("Gender==1")
    female = input_df.query("Gender==0")
    col1, col2 = st.columns(2)
    with col1:
        f = px.histogram(male, x="Age", nbins = 25, title="Male Patients by Age")
        f.update_yaxes(range=[0, 1000])
        st.plotly_chart(f)
    with col2:
        f = px.histogram(female, x="Age", nbins = 25, title="Female Patients by Age")
        f.update_yaxes(range=[0, 1000])
        st.plotly_chart(f)

@st.cache
def get_dbscan_model(df, X_scaled, e, n):
    model = DBSCAN(eps=e, # default=0.5, The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                   min_samples=n, # default=5, The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
                   metric='euclidean', # default='euclidean'. The metric to use when calculating distance between instances in a feature array.
                   metric_params=None, # default=None, Additional keyword arguments for the metric function.
                   algorithm='auto', # {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’, The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.
                   leaf_size=30, # default=30, Leaf size passed to BallTree or cKDTree.
                   p=None, # default=None, The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2
                   n_jobs=None, # default=None, The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
                   )
    result = model.fit(X_scaled)
    df['label'] = result.labels_
    df = df.sort_values(by=['label'])

    # Create a 3D scatter plot
    fig = px.scatter_3d(df, x=df['Age'], y=df['Gender'], z=df['SepsisLabel'],
                        opacity=1, color=df['label'].astype(str),
                        color_discrete_sequence=['black']+px.colors.qualitative.Plotly,
                        hover_data=df.columns,
                        width=900, height=900
                        )
    return fig

def get_ensc_model(df, X_scaled):
    model = ElasticNetSubspaceClustering(n_clusters=4,algorithm='lasso_lars',gamma=5).fit(X_scaled)
    df['label'] = model.labels_
    df = df.sort_values(by=['label'])
    # Create a 3D scatter plot
    fig = px.scatter_3d(df, x=df['HCO3'], y=df['HR'], z=df['DBP'],
                        opacity=1, color=df['label'].astype(str),
                        color_discrete_sequence=['black']+px.colors.qualitative.Plotly,
                        hover_data=df.columns,
                        width=900, height=900
                        )
    return fig, model


@st.cache
def reduce_trimap(df):
    return trimap.TRIMAP().fit_transform(df.values)

@st.cache
def reduce_pacmap(df):
    return pacmap.PaCMAP().fit_transform(df.values)


st.subheader('1A')
st.write('Complete raw data set ENSC')
df = load_data()
df = df[df.columns[2:]]
print(df.columns)
df = df.fillna(value=0)
df = df.head(10000)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
fig, model = get_ensc_model(df, X_scaled)
st.plotly_chart(fig)


st.write('1B')
st.write('Silhouette Score')
st.write(metrics.silhouette_score(X_scaled, model.labels_, metric='euclidean'))
# # Create empty lists
# S=[] # this is to store Silhouette scores
# comb=[] # this is to store combinations of epsilon / min_samples
# cluster_range=range(3,6)
# gamma_range=range(3,6)
#
# for n in cluster_range:
#     for g in gamma_range:
#         # Set the model and its parameters
#         model = ElasticNetSubspaceClustering(n_clusters=n,algorithm='lasso_lars',gamma=g).fit(X_scaled)
#         # Fit the model
#         clm = model.fit(X_scaled)
#         # Calculate Silhoutte Score and append to a list
#         S.append(metrics.silhouette_score(X_scaled, clm.labels_, metric='euclidean'))
#         comb.append(str(n)+"|"+str(g)) # axis values for the graph
# plt.figure(figsize=(16,8), dpi=300)
# plt.plot(comb, S, 'bo-', color='black')
# plt.xlabel('Clusters | Gamma')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score based on different combination of Hyperparameters')
# st.pyplot(plt)

st.write('2B')
st.write('Gender Median Imputed')
df = load_data_gender_median()
df = df[df.columns[2:]]
df = df.fillna(value=0)
df = df.head(10000)
fig, _ = get_ensc_model(df, X_scaled)
st.plotly_chart(fig)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

st.write('Median Imputed')
df = load_data_median()
df = df[df.columns[2:]]
df = df.fillna(value=0)
df = df.head(10000)
fig, _ = get_ensc_model(df, X_scaled)
st.plotly_chart(fig)
#
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(df)


#------------------

# Exercise 3
# st.subheader('1A')
# st.write('Complete raw data set displayed after PaCMAP 2D conversion')
# st.image('images/12.png')
#
# st.write('Clustering before transformation did not yield meaningful result. Clusters that can be visually '
#          'distinguished might be influenced by blood pressure values')
# st.image('images/11.png')
#
# st.write("DBSCAN with random epsilon and minimum samples looks pretty bad")
#
# df = load_data()
# df = df[df.columns[2:]]
# print(df.columns)
# df = df.fillna(value=0)
# df = df.head(10000)
#
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(df)
# fig = get_dbscan_model(df, X_scaled, 0.08, 3)
# st.plotly_chart(fig)
#
# st.write('1B')
# st.write('We use the Silhouette Score')
# sh_explanation = """
# 1: Means clusters are well apart from each other and clearly distinguished.
# 0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
# -1: Means clusters are assigned in the wrong way.
#
# Silhouette Score = (b-a)/max(a,b)
# where
# a= average intra-cluster distance i.e the average distance between each point within a cluster.
# b= average inter-cluster distance i.e the average distance between all clusters."""
# st.write(sh_explanation)
#
# # Create empty lists
# S=[] # this is to store Silhouette scores
# comb=[] # this is to store combinations of epsilon / min_samples
#
# # Define ranges to explore
# eps_range=range(12,30) # note, we will scale this down by 100 as we want to explore 0.06 - 0.11 range
# minpts_range=range(3,5)
#
# for k in eps_range:
#     for j in minpts_range:
#         # Set the model and its parameters
#         model = DBSCAN(eps=k/100, min_samples=j)
#         # Fit the model
#         clm = model.fit(X_scaled)
#         # Calculate Silhoutte Score and append to a list
#         S.append(metrics.silhouette_score(X_scaled, clm.labels_, metric='euclidean'))
#         comb.append(str(k)+"|"+str(j)) # axis values for the graph
#
# # Plot the resulting Silhouette scores on a graph
# plt.figure(figsize=(16,8), dpi=300)
# plt.plot(comb, S, 'bo-', color='black')
# plt.xlabel('Epsilon/100 | MinPts')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score based on different combination of Hyperparameters')
# st.pyplot(plt)
#
# fig = get_dbscan_model(df, X_scaled, 0.19, 3)
# st.plotly_chart(fig)
#
# subset = df.copy()
# subset = subset[['Gender','Age','SepsisLabel']]
# scaler = MinMaxScaler()
# subset_scaled = scaler.fit_transform(subset)
# fig = get_dbscan_model(subset, subset_scaled, 0.19, 3)
# st.write('1C')
# st.write('Subset of features: Gender, Age, HR, O2Sat, Temp, SBP, MAP, DBP, Resp using only the features with'
#          ' low proportion of missing data does not improve the results whatsoever.')
# st.image('images/17.png')
#
# st.write('Now with DBScan')
# st.write('Subset of features: Gender, Age, SepsisLabel')
# st.plotly_chart(fig)
#
# subset = df.copy()
# subset = subset[['Gender','Age','SepsisLabel', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']]
# scaler = MinMaxScaler()
# subset_scaled = scaler.fit_transform(subset)
# fig = get_dbscan_model(subset, subset_scaled, 0.19, 3)
# st.write('Subset of features: Gender, Age, SepsisLabel, HR, O2Sat, Temp, SBP, MAP, DBP, Resp')
# st.plotly_chart(fig)
#
#
# st.subheader('2')
# st.write("Using imputed data yields a different looking result. Clustering still cannot capture meaningful separation before transformation")
# st.image('images/13.png')
#
# st.write("but visually distinct groups are formed")
# st.image('images/14.png')
#
#
# st.write("Still main differentiation appears to be in blood pressure. Using only a subset is, again, not helpful")
# st.image('images/18.png')
#
# st.write("DB Scan Imputed also results in no improvements")
# df = load_data_gender_median()
# df = df[df.columns[2:]]
# df = df.fillna(value=0)
# df = df.head(10000)
#
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(df)
#
# fig = get_dbscan_model(df, X_scaled, 0.19, 3)
# st.plotly_chart(fig)


# Excercise 2:

# st.write(print_properties(df))
# df = df.head(10000)
# print_hist(df)
#
# df = df.fillna(value=0)
# st.subheader('2A')
# # st.write("Trimap visualization")
# # trimap = reduce_trimap(df)
#
# st.write("Pacmap visualization")
# pacmap_df = pd.DataFrame(reduce_pacmap(df), columns=['x','y'])
# f = px.scatter(pacmap_df, x='x', y='y')
# st.plotly_chart(f)
#
# st.subheader('2B')
# col1, col2 = st.columns(2)
# with col1:
#     st.write("Pacmap Male visualization")
#     pacmap_df_male = pd.DataFrame(reduce_pacmap(df.query("Gender==1")), columns=['x','y'])
#     f = px.scatter(pacmap_df_male, x='x', y='y')
#     st.plotly_chart(f)
#
# with col2:
#     st.write("Pacmap Female visualization")
#     pacmap_df_female = pd.DataFrame(reduce_pacmap(df.query("Gender==0")), columns=['x','y'])
#     f = px.scatter(pacmap_df_female, x='x', y='y')
#     st.plotly_chart(f)
#
# st.subheader('3A')
# st.write('HR, 02Sat, TEMP, SBP, MAP, DBP, RESP')
#
# st.subheader('3B')
# st.write('Yes the imputations change when applied to a subgroup')
#
# st.subheader('3C')
# col1, col2 = st.columns(2)
#
# df = load_data_median()
# df = df.head(10000)
# df = df.fillna(value=0)
# with col1:
#     st.write("Pacmap Male with all Medians")
#     pacmap_df_male = pd.DataFrame(reduce_pacmap(df.query("Gender==1")), columns=['x','y'])
#     f = px.scatter(pacmap_df_male, x='x', y='y')
#     st.plotly_chart(f)
#
# with col2:
#     st.write("Pacmap Female with all Medians")
#     pacmap_df_female = pd.DataFrame(reduce_pacmap(df.query("Gender==0")), columns=['x','y'])
#     f = px.scatter(pacmap_df_female, x='x', y='y')
#     st.plotly_chart(f)
#
# df = load_data_gender_median()
# df = df.head(10000)
# df = df.fillna(value=0)
# with col1:
#     st.write("Pacmap Male with Gender Median")
#     pacmap_df_male = pd.DataFrame(reduce_pacmap(df.query("Gender==1")), columns=['x','y'])
#     f = px.scatter(pacmap_df_male, x='x', y='y')
#     st.plotly_chart(f)
#
# with col2:
#     st.write("Pacmap Female with Gender Median")
#     pacmap_df_female = pd.DataFrame(reduce_pacmap(df.query("Gender==0")), columns=['x','y'])
#     f = px.scatter(pacmap_df_female, x='x', y='y')
#     st.plotly_chart(f)