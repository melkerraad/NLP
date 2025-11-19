import json
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter, defaultdict
from pathlib import Path

# Load data
with open('data/20newsgroups_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
documents = data['data']

# Limit dataset to ~200k words after preprocessing
max_words = 210000
stop_words = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'not', 'there', 'one', 'all', 'who', 'people', 'what', 'about', 'your', 'some',
    'any', 'which', 'know', 'get', 'just', 'more', 'don', 'like', 'think', 'when', 'their',
    'out', 'how', 'time', 'than', 'where', 'also', 'because', 'only', 'such', 'make', 'then',
    'his', 'very', 'use', 'why', 'into', 'way', 'our', 'good', 'other'
])

# Remove common header/meta tokens specific to 20 Newsgroups
stop_words.update({
    'edu', 'com', 'subject', 'lines', 'organization', 'writes', 'article', 'nntp',
    'posting', 'host', 'reply', 'distribution', 'university', 'internet', 'usenet',
    'message', 'references', 'sender'
})

# Tokenize and remove stop words (first pass to count words)
tokenized_docs = []
for doc in documents:
    tokens = re.findall(r'\b[a-z]+\b', doc.lower())
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokenized_docs.append(tokens)

# Remove rare words (appearing < 10 times)
word_counts = Counter(word for doc in tokenized_docs for word in doc)
filtered_docs = [[word for word in doc if word_counts[word] >= 10] for doc in tokenized_docs]

# Limit to max_words by taking documents until we reach the limit
limited_docs = []
total_words = 0
for doc in filtered_docs:
    if total_words + len(doc) > max_words:
        break
    limited_docs.append(doc)
    total_words += len(doc)
filtered_docs = limited_docs

# Build vocabulary
vocab = sorted(set(word for doc in filtered_docs for word in doc))
word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = {idx: word for word, idx in word_to_id.items()}

# Convert documents to integer IDs
documents_int = [[word_to_id[word] for word in doc] for doc in filtered_docs]

print(f"Processed {len(filtered_docs)} documents")
print(f"Total words: {sum(len(doc) for doc in filtered_docs):,}")
print(f"Vocabulary size: {len(vocab)}")

V = len(vocab)  # Vocabulary size
D = len(documents_int)  # Number of documents

# Create output directory
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

def compute_umass_coherence(top_words, documents_int, word_to_id, top_n=20):
    """Compute UMass coherence score for a topic given its top words."""
    coherence = 0.0
    word_ids = [word_to_id.get(word, -1) for word in top_words[:top_n]]
    word_ids = [w for w in word_ids if w != -1]
    
    if len(word_ids) < 2:
        return 0.0
    
    # Count document co-occurrences
    doc_word_counts = defaultdict(set)  # word_id -> set of doc indices containing it
    for d, doc in enumerate(documents_int):
        doc_word_set = set(doc)
        for word_id in word_ids:
            if word_id in doc_word_set:
                doc_word_counts[word_id].add(d)
    
    # Compute coherence: sum over pairs (w_i, w_j) where j < i
    for i in range(1, len(word_ids)):
        w_i = word_ids[i]
        for j in range(i):
            w_j = word_ids[j]
            D_wj = len(doc_word_counts[w_j])
            if D_wj == 0:
                continue
            D_wi_wj = len(doc_word_counts[w_i] & doc_word_counts[w_j])
            coherence += np.log((D_wi_wj + 1) / D_wj)
    
    return coherence

def run_lda(documents_int, vocab, word_to_id, id_to_word, K, alpha, beta, num_iterations=100):
    """Run LDA with given hyperparameters and return results."""
    V = len(vocab)
    D = len(documents_int)
    
    # Initialize topic assignments randomly
    topic_assignments = []
    for doc in documents_int:
        doc_topics = [random.randint(0, K-1) for _ in doc]
        topic_assignments.append(doc_topics)
    
    # Initialize count matrices
    n_wt = np.zeros((V, K), dtype=int)
    n_dt = np.zeros((D, K), dtype=int)
    n_t = np.zeros(K, dtype=int)
    
    # Populate count matrices from initial assignments
    for d, doc in enumerate(documents_int):
        for i, word_id in enumerate(doc):
            t = topic_assignments[d][i]
            n_wt[word_id, t] += 1
            n_dt[d, t] += 1
            n_t[t] += 1
    
    # Collapsed Gibbs sampling
    for it in range(num_iterations):
        for d, doc in enumerate(documents_int):
            if not doc:
                continue
            for i, word_id in enumerate(doc):
                current_topic = topic_assignments[d][i]
                
                # Remove current assignment
                n_wt[word_id, current_topic] -= 1
                n_dt[d, current_topic] -= 1
                n_t[current_topic] -= 1
                
                # Compute conditional distribution
                left = (n_wt[word_id, :] + beta) / (n_t + V * beta)
                right = (n_dt[d, :] + alpha)
                probs = left * right
                probs /= probs.sum()
                
                # Sample new topic
                new_topic = np.random.choice(K, p=probs)
                topic_assignments[d][i] = new_topic
                
                # Update counts
                n_wt[word_id, new_topic] += 1
                n_dt[d, new_topic] += 1
                n_t[new_topic] += 1
        
        if (it + 1) % 20 == 0:
            print(f"  Iteration {it + 1}/{num_iterations} completed")
    
    # Compute topic-word distributions and top words
    topic_word_dist = (n_wt + beta) / (n_t[np.newaxis, :] + V * beta)
    top_n = 20
    top_words_per_topic = []
    for topic_idx in range(K):
        top_word_ids = np.argsort(topic_word_dist[:, topic_idx])[-top_n:][::-1]
        top_words = [id_to_word[word_id] for word_id in top_word_ids]
        top_words_per_topic.append(top_words)
    
    # Compute coherence scores
    coherence_scores = []
    for topic_idx in range(K):
        coherence = compute_umass_coherence(top_words_per_topic[topic_idx], documents_int, word_to_id, top_n)
        coherence_scores.append(coherence)
    
    return {
        'n_wt': n_wt,
        'n_dt': n_dt,
        'n_t': n_t,
        'top_words': top_words_per_topic,
        'coherence_scores': coherence_scores,
        'topic_word_dist': topic_word_dist
    }

# Test different hyperparameter combinations
hyperparams = [
    {'K': 10, 'alpha': 0.1, 'beta': 0.1},
    {'K': 10, 'alpha': 0.01, 'beta': 0.01},
    {'K': 50, 'alpha': 0.1, 'beta': 0.1},
    {'K': 50, 'alpha': 0.01, 'beta': 0.01},
]

results = []
all_coherence_scores = []

for params in hyperparams:
    K = params['K']
    alpha = params['alpha']
    beta = params['beta']
    
    print(f"\n{'='*60}")
    print(f"Running LDA with K={K}, alpha={alpha}, beta={beta}")
    print(f"{'='*60}")
    
    result = run_lda(documents_int, vocab, word_to_id, id_to_word, K, alpha, beta, num_iterations=100)
    result['params'] = params
    results.append(result)
    all_coherence_scores.append({
        'K': K,
        'alpha': alpha,
        'beta': beta,
        'avg_coherence': np.mean(result['coherence_scores']),
        'coherence_scores': result['coherence_scores']
    })
    
    # Save top words table for this configuration as LaTeX
    config_name = f"K{K}_alpha{alpha}_beta{beta}"
    tex_file = output_dir / f"top_words_{config_name}.tex"
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write("% Auto-generated LaTeX table for top words per topic\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{lp{11cm}r}\n")
        f.write("\\toprule\n")
        f.write("Topic & Top 20 Words & Coherence \\\\\n")
        f.write("\\midrule\n")
        for topic_idx in range(K):
            words = result['top_words'][topic_idx]
            words_str = ', '.join(words).replace('_', '\\_')
            coherence = result['coherence_scores'][topic_idx]
            f.write(f"Topic~{topic_idx} & {words_str} & {coherence:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write(f"\\end{{tabular}}\n\\caption{{Top words for configuration $K={K}$, $\\alpha=\\beta={alpha}$}}\n")
        f.write("\\end{table}\n")
    print(f"Saved top words LaTeX table to {tex_file}")

# Create summary table
summary_file = output_dir / "summary.csv"
with open(summary_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['K', 'Alpha', 'Beta', 'Avg Coherence', 'Min Coherence', 'Max Coherence'])
    for score_data in all_coherence_scores:
        coherences = score_data['coherence_scores']
        writer.writerow([
            score_data['K'],
            score_data['alpha'],
            score_data['beta'],
            f"{score_data['avg_coherence']:.4f}",
            f"{min(coherences):.4f}",
            f"{max(coherences):.4f}"
        ])
print(f"\nSaved summary table to {summary_file}")

# Create coherence comparison plots (separate figures)

# Plot 1: Average coherence by configuration
fig1, ax1 = plt.subplots(figsize=(8, 5))
configs = [f"K={d['K']}\nα=β={d['alpha']}" for d in all_coherence_scores]
avg_coherences = [d['avg_coherence'] for d in all_coherence_scores]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

ax1.bar(configs, avg_coherences, color=colors)
ax1.set_ylabel('Average UMass Coherence')
ax1.set_title('Average Coherence by Configuration')
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

plt.tight_layout()
plot_file1 = output_dir / "coherence_average.png"
plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"Saved average coherence plot to {plot_file1}")

# Plot 2: Coherence distribution by K
k10_scores = [d['coherence_scores'] for d in all_coherence_scores if d['K'] == 10]
k50_scores = [d['coherence_scores'] for d in all_coherence_scores if d['K'] == 50]

if k10_scores and k50_scores:
    k10_all = [s for scores in k10_scores for s in scores]
    k50_all = [s for scores in k50_scores for s in scores]
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.boxplot([k10_all, k50_all], labels=['K=10', 'K=50'])
    ax2.set_ylabel('UMass Coherence')
    ax2.set_title('Coherence Distribution by Number of Topics')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file2 = output_dir / "coherence_distribution.png"
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved coherence distribution plot to {plot_file2}")

print(f"\n{'='*60}")
print("All results saved to", output_dir)
print(f"{'='*60}")
