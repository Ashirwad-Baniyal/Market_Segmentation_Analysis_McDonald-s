import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import resample
# Load the data
mcdonalds = pd.read_csv("mcdonalds.csv")
# Inspect basic features
print("Variable names:")
print(mcdonalds.columns)

print("\nSample size:")
print(len(mcdonalds))

print("\nFirst three rows of the data:")
print(mcdonalds.head(3))

MD_x = mcdonalds.iloc[:, :11].applymap(lambda x: 1 if x == "Yes" else 0)

# Calculating average values of transformed segmentation variables
average_values = np.round(MD_x.mean(), 2)
print(average_values)

# Perform principal component analysis
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

import matplotlib.pyplot as plt

# Plot consumers projected onto PC space
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey')

# Add arrows for original variables
for i, var in enumerate(mcdonalds.columns[:11]):  # Assuming the first 11 columns are the segmentation variables
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5)
    plt.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1, var, color='g')
plt.show()


np.random.seed(1234)

# Define the range of clusters
cluster_range = range(2, 9)

# Store results
kmeans_models = []
inertia_scores = []

# Run KMeans for each number of clusters
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
    kmeans.fit(MD_x)
    kmeans_models.append(kmeans)
    inertia_scores.append(kmeans.inertia_)

# Select the best model based on inertia (lower is better)
best_kmeans = kmeans_models[np.argmin(inertia_scores)]
# Print the best number of clusters
print("Best number of clusters:", best_kmeans.n_clusters)

# Plot the inertia scores to create a scree plot using a bar graph
plt.figure(figsize=(10, 6))
plt.bar(cluster_range, inertia_scores)
plt.xlabel('Number of Segments')
plt.ylabel('Sum of Within-Cluster Distances')
plt.title('Scree Plot for KMeans Clustering')
plt.xticks(cluster_range)
plt.grid(axis='y')  # grid on y-axis only
plt.show()


# Define the range of clusters
cluster_range = range(2, 9)
n_bootstraps = 100
n_reps = 10

# Function to compute ARI for each bootstrap sample
def compute_ari_for_bootstrap(X, cluster_range, n_reps, n_bootstraps):
    ari_results = {n_clusters: [] for n_clusters in cluster_range}
    
    for n_clusters in tqdm(cluster_range, desc="Number of Clusters"):
        for _ in range(n_bootstraps):
            # Bootstrap sample
            bootstrap_sample = X.sample(frac=1, replace=True).values
            labels_list = []
            
            for _ in range(n_reps):
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=None)
                kmeans.fit(bootstrap_sample)
                labels_list.append(kmeans.labels_)
                
            # Compute pairwise ARI
            base_labels = labels_list[0]
            for labels in labels_list[1:]:
                ari = adjusted_rand_score(base_labels, labels)
                ari_results[n_clusters].append(ari)
                
    return ari_results
#  Compute ARI for each bootstrap sample
ari_results = compute_ari_for_bootstrap(MD_x, cluster_range, n_reps, n_bootstraps)
# Plot the results
plt.figure(figsize=(10, 6))
data_to_plot = [ari_results[n] for n in cluster_range]
plt.boxplot(data_to_plot, labels=cluster_range)
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.title('Global Stability Boxplot')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
kmeans.fit(MD_x)

# Extracting the cluster labels
cluster_labels = kmeans.labels_

# Function to compute similarity within a cluster
# Placeholder function: replace with actual similarity calculation
def compute_similarity_within_cluster(cluster_data):
    # Example: similarity as the inverse of Euclidean distances within cluster (not a typical similarity measure)
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(cluster_data, metric='euclidean')
    similarities = 1 - squareform(distances) / distances.max()  # Normalize to range [0, 1]
    return similarities[np.triu_indices_from(similarities, k=1)]  # Upper triangle without diagonal

# Plotting gorge plots for each cluster
# plt.figure(figsize=(12, 8))
# for cluster in range(4):
#     cluster_data = MD_x[cluster_labels == cluster]
#     similarity = compute_similarity_within_cluster(cluster_data)
    
#     plt.subplot(2, 2, cluster + 1)
#     sns.histplot(similarity, bins=10, kde=False)
#     plt.xlim(0, 1)
#     plt.xlabel("Similarity")
#     plt.ylabel("Count")
#     plt.title(f"Cluster {cluster + 1}")

# plt.tight_layout()
# plt.show()


# SLSA
k_values = range(2, 9)
cluster_results = {}

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD_x)
    cluster_results[k] = kmeans.labels_

# Step 2: Prepare the data for the SLSA plot
transition_matrix = []

for k in k_values:
    transition_matrix.append(cluster_results[k])

transition_matrix = np.array(transition_matrix)

# Step 3: Create a plot for segment level stability across solutions
fig, ax = plt.subplots(figsize=(14, 8))

for i in range(transition_matrix.shape[1]):  # For each data point
    ax.plot(k_values, transition_matrix[:, i], marker='o', linestyle='-', alpha=0.5)

# Customize the plot
ax.set_xlabel("Number of segments")
ax.set_ylabel("Segment membership")
ax.set_title("Segment Level Stability Across Solutions (SLSA) Plot")
ax.set_xticks(k_values)
ax.set_xticklabels(k_values)

plt.show()
# SLSW
def compute_slsw(MD_x, labels, n_bootstraps=100):
    n_clusters = len(np.unique(labels))
    stability = np.zeros(n_clusters)
    
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        ari_scores = []
        
        for _ in range(n_bootstraps):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(cluster_indices, size=len(cluster_indices), replace=True)
            bootstrap_sample = MD_x.iloc[bootstrap_indices]
            
            # KMeans on bootstrap sample
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=None)
            kmeans.fit(bootstrap_sample)
            bootstrap_labels = kmeans.labels_
            
            # Match labels with original
            original_cluster_indices = np.where(labels[bootstrap_indices] == cluster)[0]
            bootstrap_cluster_indices = np.where(bootstrap_labels == mode(bootstrap_labels[original_cluster_indices])[0])[0]
            
            # Compute ARI
            ari = adjusted_rand_score(labels[bootstrap_indices][original_cluster_indices], bootstrap_labels[bootstrap_cluster_indices])
            ari_scores.append(ari)
        
        # Average ARI for this cluster
        stability[cluster] = np.mean(ari_scores)
    
    return stability

# Perform KMeans clustering for 4 segments
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
kmeans.fit(MD_x)
labels_4_segments = kmeans.labels_

# Calculate segment-level stability within solutions (SLSW)
slsw_stability = compute_slsw(MD_x, labels_4_segments)

# Plot the result with customized labels and y-axis limits
plt.figure(figsize=(8, 6))
plt.bar(range(1, 5), slsw_stability, color='skyblue')
plt.ylim(0, 1)
plt.xticks(range(1, 5), labels=range(1, 5))
plt.yticks(np.arange(0.0, 1.1, 0.2))
plt.xlabel("Segment Number")
plt.ylabel("Segment Stability")
plt.title("Segment Level Stability Within Solutions (SLSW)")
plt.show()

# Using mixutres of distribuitions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

segment_range = range(2, 9)

# Store models and information criteria values
models = []
aic_values = []
bic_values = []
icl_values = []

# Perform Gaussian Mixture Model for each number of segments
for n_segments in segment_range:
    gmm = GaussianMixture(n_components=n_segments, n_init=10, random_state=1234)
    gmm.fit(MD_x)
    models.append(gmm)
    aic_values.append(gmm.aic(MD_x))
    bic_values.append(gmm.bic(MD_x))
    icl_values.append(gmm.lower_bound_)

# Plot the information criteria values
plt.figure(figsize=(10, 6))
plt.plot(segment_range, aic_values, label='AIC')
plt.plot(segment_range, bic_values, label='BIC')
plt.plot(segment_range, icl_values, label='ICL')
plt.xlabel('Number of Segments')
plt.ylabel('Value of Information Criteria')
plt.title('Information Criteria for Latent Class Analysis')
plt.legend()
plt.grid(True)
plt.show()



# Regression model

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset
mcdonalds = pd.read_csv('mcdonalds.csv')
mcdonalds=mcdonalds.iloc[:, :-3]
print(mcdonalds.head())
# Assume 'Like' is the target variable and the rest are features
# Replace 'Like' with the actual name of your target variable if different
target_variable = 'Like'

# Define the mapping from ordinal to numeric
like_mapping = {
    "I love it!+5": 1,
    "+4": 2,
    "+3": 3,
    "+2": 4,
    "+1": 5,
    "0": 6,
    "-1": 7,
    "-2": 8,
    "-3": 9,
    "-4": 10,
    "I hate it!-5": 11
}

# Map the 'Like' column to numeric values
mcdonalds[target_variable + '_numeric'] = mcdonalds[target_variable].map(like_mapping)

# Compute the new numeric values for the target variable
mcdonalds[target_variable + '_numeric_transformed'] = 6 - mcdonalds[target_variable + '_numeric']

# Define features (excluding the original and transformed 'Like' columns)
features = mcdonalds.drop(columns=[target_variable, target_variable + '_numeric', target_variable + '_numeric_transformed'])

# Convert categorical variables to numeric (if any)
features = pd.get_dummies(features, drop_first=True)

# Define the target variable
target = mcdonalds[target_variable + '_numeric_transformed']

# Fit a Gaussian Mixture Model to the features to segment the data into two clusters
gmm = GaussianMixture(n_components=2, random_state=42)
segments = gmm.fit_predict(features)

# Fit separate linear regression models to each segment using statsmodels for significance testing
segment_models = []
coefficients = []
p_values = []
for segment in np.unique(segments):
    X_segment = features[segments == segment]
    y_segment = target[segments == segment]
    
    # Add a constant for the intercept
    X_segment = sm.add_constant(X_segment)
    
    model = sm.OLS(y_segment, X_segment).fit()
    segment_models.append(model)
    
    coefficients.append(model.params)
    p_values.append(model.pvalues)

# Convert to DataFrame for easier plotting
coefficients_df = pd.DataFrame(coefficients, columns=['Intercept'] + list(features.columns))
p_values_df = pd.DataFrame(p_values, columns=['Intercept'] + list(features.columns))

# Plot the regression coefficients with significance levels
fig, ax = plt.subplots(figsize=(12, 8))
index = np.arange(coefficients_df.shape[1])
bar_width = 0.35

for i in range(len(segment_models)):
    bars = ax.bar(index + i * bar_width, coefficients_df.iloc[i], bar_width, label=f'Segment {i+1}')
    for j, bar in enumerate(bars):
        p_val = p_values_df.iloc[i, j]
        if p_val < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), '*', ha='center', va='bottom', color='red')

ax.set_xlabel('Features')
ax.set_ylabel('Coefficients')
ax.set_title('Regression Coefficients of Two-Segment Mixture of Linear Regression Models with Significance')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(coefficients_df.columns, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()


#mosiac plot 
# from sklearn.preprocessing import LabelEncoder
# def encoding(x):
#     MD_x[x] = LabelEncoder().fit_transform(MD_x[x])
#     return MD_x

# category = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
#        'tasty', 'expensive', 'healthy', 'disgusting']

# for i in category:
#     encoding(i)

# data = MD_x.loc[:, category]

# kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(MD_x)
# MD_x['cluster_num'] = kmeans.labels_ 
# print (kmeans.labels_) 
# print (kmeans.inertia_) 
# print(kmeans.n_iter_) 
# print(kmeans.cluster_centers_)

# from statsmodels.graphics.mosaicplot import mosaic
# from itertools import product

# crosstab =pd.crosstab(MD_x['cluster_num'],MD_x['Like'])
# #Reordering cols
# crosstab = crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4','+5']]
# print(crosstab)
# mosaic(crosstab.stack())
# plt.rcParams['figure.figsize'] = (10, 5)
# plt.rcParams['font.size'] = 5
# plt.show()

# crosstab =pd.crosstab(MD_x['cluster_num'],MD_x['Gender'])
# #Reordering cols
# crosstab = crosstab[['Female', 'Male']]
# mosaic(crosstab.stack())
# plt.show()