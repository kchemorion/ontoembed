import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Load the vector store
data = np.load('vector_store.npz', allow_pickle=True)
entities = data['entities']
vectors = data['vectors']

print(f"Number of entities: {len(entities)}")
print(f"Vector dimensions: {vectors.shape[1]}")

# Create a mapping of entities to vectors (optional)
vector_store = {entity: vector for entity, vector in zip(entities, vectors)}

# ---------------------------------------------------------
# 1. t-SNE Visualization
# ---------------------------------------------------------
print("Generating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42)
vectors_tsne = tsne.fit_transform(vectors)

plt.figure(figsize=(12, 8))
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], alpha=0.7, s=30, cmap='viridis')
plt.title('t-SNE Visualization of Ontology Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Optionally, annotate some points
for i, entity in enumerate(entities):
    if i % 50 == 0:  # Adjust for fewer/more labels
        plt.annotate(entity, (vectors_tsne[i, 0], vectors_tsne[i, 1]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.savefig('tsne_embeddings.png', dpi=300, bbox_inches='tight')
plt.close()
print("t-SNE visualization saved as 'tsne_embeddings.png'.")

# ---------------------------------------------------------
# 2. PCA Visualization
# ---------------------------------------------------------
print("Generating PCA visualization...")
pca = PCA(n_components=2)
vectors_pca = pca.fit_transform(vectors)

plt.figure(figsize=(12, 8))
plt.scatter(vectors_pca[:, 0], vectors_pca[:, 1], alpha=0.7, s=30, cmap='plasma')
plt.title('PCA Visualization of Ontology Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Optionally, annotate some points
for i, entity in enumerate(entities):
    if i % 50 == 0:
        plt.annotate(entity, (vectors_pca[i, 0], vectors_pca[i, 1]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.savefig('pca_embeddings.png', dpi=300, bbox_inches='tight')
plt.close()
print("PCA visualization saved as 'pca_embeddings.png'.")

# ---------------------------------------------------------
# 3. UMAP Visualization
# ---------------------------------------------------------
print("Generating UMAP visualization...")
umap_model = umap.UMAP(n_components=2, random_state=42)
vectors_umap = umap_model.fit_transform(vectors)

plt.figure(figsize=(12, 8))
plt.scatter(vectors_umap[:, 0], vectors_umap[:, 1], alpha=0.7, s=30, cmap='Spectral')
plt.title('UMAP Visualization of Ontology Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')

# Optionally, annotate some points
for i, entity in enumerate(entities):
    if i % 50 == 0:
        plt.annotate(entity, (vectors_umap[i, 0], vectors_umap[i, 1]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.savefig('umap_embeddings.png', dpi=300, bbox_inches='tight')
plt.close()
print("UMAP visualization saved as 'umap_embeddings.png'.")

# ---------------------------------------------------------
# 4. Heatmap of Pairwise Cosine Similarities
# ---------------------------------------------------------
print("Generating heatmap of pairwise similarities...")
# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(vectors)

# For large ontologies, plotting the full matrix may be impractical.
# We'll select a subset of entities for demonstration.
subset_size = 50  # Adjust as needed
subset_indices = np.random.choice(len(entities), size=subset_size, replace=False)
subset_entities = entities[subset_indices]
subset_similarity_matrix = similarity_matrix[np.ix_(subset_indices, subset_indices)]

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(subset_similarity_matrix, xticklabels=subset_entities, yticklabels=subset_entities, cmap='viridis')
plt.title('Heatmap of Pairwise Cosine Similarities')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('similarity_heatmap.png', dpi=300)
plt.close()
print("Heatmap saved as 'similarity_heatmap.png'.")

# ---------------------------------------------------------
# 5. Dendrogram (Hierarchical Clustering)
# ---------------------------------------------------------
print("Generating dendrogram...")
from scipy.cluster.hierarchy import dendrogram, linkage

# For large datasets, use a subset
vectors_subset = vectors[subset_indices]

# Perform hierarchical clustering
linked = linkage(vectors_subset, 'ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked,
           labels=subset_entities,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram of Ontology Embeddings')
plt.xlabel('Entities')
plt.ylabel('Distance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300)
plt.close()
print("Dendrogram saved as 'dendrogram.png'.")


from mpl_toolkits.mplot3d import Axes3D

# Reduce dimensions to 3D using PCA (or t-SNE/UMAP)
pca_3d = PCA(n_components=3)
vectors_3d = pca_3d.fit_transform(vectors)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2], alpha=0.7, s=30)

ax.set_title('3D PCA Visualization of Ontology Embeddings')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('pca_embeddings_3d.png', dpi=300, bbox_inches='tight')
plt.close()
print("3D PCA visualization saved as 'pca_embeddings_3d.png'.")


from wordcloud import WordCloud

# Generate a word cloud of entities
text = ' '.join(entities)
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(text)

plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('entities_wordcloud.png', dpi=300, bbox_inches='tight')
plt.close()
print("Word cloud saved as 'entities_wordcloud.png'.")

print("All visualizations generated.")
