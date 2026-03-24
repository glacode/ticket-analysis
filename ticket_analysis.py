import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import umap
import plotly.express as px
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords (required only for the first run)
nltk.download('stopwords')

def clean_text(text):
    # Removes numbers and punctuation for the keyword analysis phase
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    return text

def run_semantic_analysis_pro(csv_file, n_clusters=20):
    print("--- 1. Data Loading ---")
    df = pd.read_csv(csv_file, encoding='latin1', sep=None, engine='python').fillna('')
    
    # Combine fields to get more semantic context
    documents = (df['short_description'] + " " + df['description']).tolist()

    print("--- 2. Generating Embeddings (Multilingual Model) ---")
    # Using 'paraphrase-multilingual-MiniLM-L12-v2' which converts text into a 
    # 384-dimensional vector. These vectors represent the semantic meaning of the text.
    # Because it's a multilingual model, similar concepts in different languages 
    # (e.g., 'password reset' and 'ripristino password') will have very similar coordinates.
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(documents, show_progress_bar=True)

    print("--- 3. Dimensionality Reduction (UMAP) ---")
    # Reducing the 384-dimensional embeddings down to 5 dimensions using UMAP.
    # This step is crucial because high-dimensional spaces (384D) suffer from the 
    # 'Curse of Dimensionality', making it hard for K-Means to find clear groups.
    # UMAP preserves the 'local structure', meaning tickets with similar meanings 
    # stay very close to each other in the new 5D space.
    reducer = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
    embeddings_reduced = reducer.fit_transform(embeddings)

    print(f"--- 4. Clustering into {n_clusters} groups ---")
    # K-Means groups the 5D points into 'n_clusters' based on their spatial proximity.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(embeddings_reduced)

    print("--- 5. Human-readable Label Generation (Medoid Method) ---")
    # Finding the 'medoid' (the real ticket closest to the cluster center) to act 
    # as a representative human-readable label for the entire group.
    centroids = kmeans.cluster_centers_
    closest_idx, _ = pairwise_distances_argmin_min(centroids, embeddings_reduced)
    
    cluster_map = {}
    for i in range(n_clusters):
        human_label = df.iloc[closest_idx[i]]['short_description']
        human_label = (human_label[:57] + '..').upper()
        cluster_map[i] = human_label

    df['cluster_label'] = df['cluster_id'].map(cluster_map)

    # 6. Save the CSV file
    output_file = 'semantic_clustered_tickets.csv'
    df.to_csv(output_file, index=False, encoding='utf-16')
    print(f"--- CSV created: {output_file} ---")

    # 7. Sunburst Visualization
    available_columns = [c for c in ['short_description', 'assigned_to', 'state'] if c in df.columns]
    fig = px.sunburst(
        df, 
        path=['cluster_label', 'number'], 
        hover_data=available_columns,
        title=f"Semantic Analysis of {len(df)} Tickets",
        height=900
    )
    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    fig.write_html("semantic_report_pro.html")

    # --- NEW PART: Generating Detailed HTML Cluster Report ---
    print("--- 8. Generating detailed HTML report ---")
    
    # Calculate percentages
    counts = df['cluster_label'].value_counts(normalize=True) * 100
    
    html_content = """
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: auto; padding: 20px; }
            details { background: #f9f9f9; border: 1px solid #aaa; border-radius: 4px; padding: .5em .5em 0; margin-bottom: 10px; }
            summary { font-weight: bold; padding: .5em; cursor: pointer; border-bottom: 1px solid #ddd; }
            details[open] { padding: .5em; }
            details[open] summary { border-bottom: 2px solid #007bff; margin-bottom: 10px; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
            th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            th { background-color: #007bff; color: white; }
            tr:hover { background-color: #f1f1f1; }
            .pct { float: right; color: #666; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>Ticket Cluster Details</h1>
    """

    for label in counts.index:
        percentage = f"{counts[label]:.2f}%"
        html_content += f"<details>\n<summary>{label} <span class='pct'>({percentage})</span></summary>\n"
        
        # Filter tickets belonging to this cluster
        cluster_data = df[df['cluster_label'] == label][['number', 'short_description', 'state']]
        
        # Convert the subset to HTML table
        html_content += cluster_data.to_html(index=False, classes='table')
        html_content += "\n</details>\n"

    html_content += "</body>\n</html>"

    with open("cluster_details.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("--- Completed! Open semantic_report_pro.html and cluster_details.html ---")

if __name__ == "__main__":
    run_semantic_analysis_pro('servicenow_tickets.csv')