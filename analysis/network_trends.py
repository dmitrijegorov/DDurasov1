import pandas as pd
import networkx as nx
import nltk
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import re
import spacy
import numpy as np

spacy.cli.download("ru_core_news_sm")

TARGET_YEAR = 2025
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['это', 'весь', 'который', 'свой', 'еще', 'раз', 'здравствуйте',
                          'такой', 'какой', 'например', 'каждый', 'очень', 'просто',
                          'однако', 'также', 'именно', 'конечно', 'коллега', 'спасибо',
                          'пожалуйста', 'добрый', 'день', 'вечер', 'утро'])

ENGLISH_TO_RUSSIAN_TERMS = {
    'revit': 'ревит',
    'enscape': 'энскейп',
    'autocad': 'автокад',
    'ai': 'ии',
    'api': 'апи',
    'product': 'продукт',
    'urban': 'городской',
    'planner': 'планировщик',
    'hr': 'эйчар',
    'admin': 'админ',
    'customer': 'клиент',
    'service': 'сервис',
    'design': 'дизайн',
    'development': 'разработка',
    'software': 'софт',
    'it': 'айти',
    'digital': 'цифровой',
    'marketing': 'маркетинг',
    'sale': 'продажа',
    'communication': 'коммуникация',
    'big data': 'большие данные',
    'machine learning': 'машинное обучение'
}
nlp = spacy.load('ru_core_news_sm')

def preprocess_text(text, stop_words, translation_dict):
    if not isinstance(text, str):
        return []
    text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'\1\2', text) # Bold/Underscore
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text) # Links [text](url) -> text

    text = text.lower()

    # Sort keys by length descending to match longer phrases first
    sorted_eng_terms = sorted(translation_dict.keys(), key=len, reverse=True)
    for eng, rus in [(term, translation_dict[term]) for term in sorted_eng_terms]:
        text = re.sub(r'\b' + re.escape(eng) + r'\b', rus, text)

    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^\w\s-]', '', text) # \w includes letters, numbers, underscore
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace

    # Tokenize
    tokens = word_tokenize(text, language='russian')

    # Lemmatize
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words and len(token.text) > 2]

    # Remove stopwords and short tokens
    processed_tokens = [
        token for token in tokens
        if token not in stop_words and len(token) > 2 and not token.isdigit()
    ]
    return processed_tokens

file_paths_map = {
    2: "FINAL_Gk_february.parquet",
    3: "FINAL_Gk_march.parquet",
    4: "FINAL_Gk_april.parquet",
    5: "FINAL_Gk_may.parquet"
}

all_dfs = []
for month_num, path in file_paths_map.items():
    try:
        print(f"Loading: {path}")
        temp_df = pd.read_parquet(path)
        all_dfs.append(temp_df)
    except FileNotFoundError:
        print(f"Error: File {path} not found. Please ensure it exists.")

if not all_dfs:
    raise ValueError("No data files were loaded. Stopping script.")

df_combined = pd.concat(all_dfs, ignore_index=True)

# Date column is datetime
df_combined['Date'] = pd.to_datetime(df_combined['Date'])
df_combined['Year'] = df_combined['Date'].dt.year
df_combined['Month'] = df_combined['Date'].dt.month

monthly_graphs = []
all_monthly_metrics_dfs = []
monthly_frequencies = {} # To store keyword/n-gram frequencies

target_months_map = {
    2: "February",
    3: "March",
    4: "April",
    5: "May"
}
ordered_month_names = [target_months_map[m_num] for m_num in sorted(target_months_map.keys())]

for month_num, month_name in target_months_map.items():
    print(f"\n--- Processing data for {month_name} ({TARGET_YEAR}) ---")

    # Filter data for the specific month and year
    # Using 'Content' column for processing. If 'clean_content' is guaranteed to be better, use that.
    month_df = df_combined[
        (df_combined['Year'] == TARGET_YEAR) & (df_combined['Month'] == month_num)
    ]

    if month_df.empty:
        print(f"No data found for {month_name} {TARGET_YEAR}. Skipping.")
        # Add an empty graph and empty metrics to maintain structure if needed
        monthly_graphs.append(nx.Graph())
        empty_metrics_df = pd.DataFrame(columns=['Node', f'In_Degree_{month_name}',
                                                 f'Katz_{month_name}', f'PageRank_{month_name}'])
        all_monthly_metrics_dfs.append(empty_metrics_df)
        monthly_frequencies[month_name] = {
            'keywords': Counter(),
            'bigrams': Counter(),
            'trigrams': Counter()
        }
        continue

    G = nx.Graph()
    current_month_keywords = Counter()
    current_month_bigrams = Counter()
    current_month_trigrams = Counter()

    print(f"Processing {len(month_df)} messages for {month_name}...")
    for index, row in month_df.iterrows():
        # Using 'Content'. If 'clean_content' is more suitable and consistently available, use row['clean_content']
        text_to_process = row['Content']

        processed_tokens = preprocess_text(text_to_process, russian_stopwords, ENGLISH_TO_RUSSIAN_TERMS)

        if not processed_tokens:
            continue

        # Update frequencies
        current_month_keywords.update(processed_tokens)

        if len(processed_tokens) >= 2:
            current_month_bigrams.update([" ".join(ng) for ng in ngrams(processed_tokens, 2)])
        if len(processed_tokens) >= 3:
            current_month_trigrams.update([" ".join(ng) for ng in ngrams(processed_tokens, 3)])

        # Add nodes and edges to the graph
        unique_tokens_in_message = sorted(list(set(processed_tokens))) # Sorted for consistent edge pairing

        if len(unique_tokens_in_message) > 1:
            for token in unique_tokens_in_message:
                if not G.has_node(token):
                    G.add_node(token)

            # Add edges between all pairs of unique tokens in this message
            from itertools import combinations
            for token1, token2 in combinations(unique_tokens_in_message, 2):
                if G.has_edge(token1, token2):
                    G[token1][token2].setdefault('weight', 0)
                    G[token1][token2]['weight'] += 1
                else:
                    G.add_edge(token1, token2, weight=1)
        elif len(unique_tokens_in_message) == 1:
             if not G.has_node(unique_tokens_in_message[0]):
                G.add_node(unique_tokens_in_message[0])


    monthly_graphs.append(G)
    monthly_frequencies[month_name] = {
        'keywords': current_month_keywords,
        'bigrams': current_month_bigrams,
        'trigrams': current_month_trigrams
    }

    print(f"Graph for {month_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Top 5 keywords: {current_month_keywords.most_common(5)}")
    print(f"Top 5 bigrams: {current_month_bigrams.most_common(5)}")
    print(f"Top 5 trigrams: {current_month_trigrams.most_common(5)}")

    # Calculate graph metrics
    if G.number_of_nodes() > 0:
        in_degree = dict(G.degree())

        # katz_centrality_numpy is often more stable if available
        try:
            # Alpha should be less than 1/spectral_radius(A)
            if G.number_of_edges() > 0 :
                 eigenvalues = nx.adjacency_spectrum(G)
                 largest_eigenvalue = max(abs(e.real) for e in eigenvalues) if eigenvalues.size > 0 else 1
                 alpha_katz = min(0.01, 0.9 / largest_eigenvalue if largest_eigenvalue > 0 else 0.01) # Ensure alpha is safe
                 katz = nx.katz_centrality_numpy(G, alpha=alpha_katz)
            else: # No edges, katz is not well-defined, set to 0
                katz = {node: 0 for node in G.nodes()}
        except Exception as e:
            print(f"Could not calculate Katz centrality for {month_name}: {e}. Setting to 0.")
            katz = {node: 0 for node in G.nodes()}

        pagerank = nx.pagerank(G, weight='weight' if G.number_of_edges() > 0 else None)

        month_metrics_df = pd.DataFrame({
            'Node': list(G.nodes()),
            f'In_Degree_{month_name}': [in_degree.get(node, 0) for node in G.nodes()],
            f'Katz_{month_name}': [katz.get(node, 0) for node in G.nodes()],
            f'PageRank_{month_name}': [pagerank.get(node, 0) for node in G.nodes()]
        })
    else: # Empty graph
        month_metrics_df = pd.DataFrame(columns=['Node', f'In_Degree_{month_name}',
                                                 f'Katz_{month_name}', f'PageRank_{month_name}'])

    all_monthly_metrics_dfs.append(month_metrics_df)

if not all_monthly_metrics_dfs:
    print("No metrics were calculated to combine.")
    final_metrics_df = pd.DataFrame()
else:
    if not all_monthly_metrics_dfs[0].empty:
        final_metrics_df = all_monthly_metrics_dfs[0]
    else:
        final_metrics_df = next((df for df in all_monthly_metrics_dfs if not df.empty),
                                pd.DataFrame(columns=['Node']))
        # If initial df was empty, ensure 'Node' column exists for merging
        if 'Node' not in final_metrics_df.columns and not final_metrics_df.empty:
             final_metrics_df = pd.DataFrame(columns=['Node'])


    for i in range(len(all_monthly_metrics_dfs)):
        # If this is the df we started with, skip it
        if all_monthly_metrics_dfs[i] is final_metrics_df and not final_metrics_df.empty :
            continue
        if not all_monthly_metrics_dfs[i].empty:
            if final_metrics_df.empty:
                 final_metrics_df = all_monthly_metrics_dfs[i]
            else:
                 final_metrics_df = pd.merge(final_metrics_df, all_monthly_metrics_dfs[i], on='Node', how='outer')


    # Fill NaN values for metrics (e.g., if a node didn't appear in a month) with 0
    metric_cols = [col for col in final_metrics_df.columns if col != 'Node']
    final_metrics_df[metric_cols] = final_metrics_df[metric_cols].fillna(0)

    # Sort by PageRank (descending)
    april_pagerank_col = f'PageRank_{target_months_map[4]}' # PageRank_April
    if april_pagerank_col in final_metrics_df.columns:
        final_metrics_df = final_metrics_df.sort_values(by=april_pagerank_col, ascending=False)
    else:
        print(f"Warning: Column '{april_pagerank_col}' not found for sorting. April data might be missing or empty.")

print("\n\n--- Final Results ---")
print("\nList of Monthly Graphs (first graph's info):")
if monthly_graphs:
    G_example = monthly_graphs[0]
    print(f"Graph for {target_months_map[2]}: {G_example.number_of_nodes()} nodes, {G_example.number_of_edges()} edges.")
    if G_example.number_of_nodes() > 0:
        print(f"Nodes (first 5): {list(G_example.nodes)[:5]}")
else:
    print("No graphs were generated.")


print("\nFrequencies (example for March):")
current_month = "March"
if current_month in monthly_frequencies and monthly_frequencies[current_month]['keywords']:
    print(f"Top 5 keywords for March: {monthly_frequencies[current_month]['keywords'].most_common(5)}")
    print(f"Top 5 bigrams for March: {monthly_frequencies[current_month]['bigrams'].most_common(5)}")
    print(f"Top 5 trigrams for March: {monthly_frequencies[current_month]['trigrams'].most_common(5)}")
else:
    print("No frequency data available for March to show as example.")

print("\nFrequencies (example for April):")
current_month = "April"
if current_month in monthly_frequencies and monthly_frequencies[current_month]['keywords']:
    print(f"Top 5 keywords for March: {monthly_frequencies[current_month]['keywords'].most_common(5)}")
    print(f"Top 5 bigrams for March: {monthly_frequencies[current_month]['bigrams'].most_common(5)}")
    print(f"Top 5 trigrams for March: {monthly_frequencies[current_month]['trigrams'].most_common(5)}")
else:
    print("No frequency data available for March to show as example.")

print("\nCombined Metrics DataFrame (Top 10 rows):")
if not final_metrics_df.empty:
    print(final_metrics_df.head(10))
else:
    print("Final metrics DataFrame is empty.")

print(f"\nTotal {len(monthly_graphs)} graphs created for months: {', '.join(target_months_map.values())}")


# <--- stemmer : SnowballStemmer --- >
def preprocess_text_stemming(text, stemmer, stop_words, translation_dict):
    if not isinstance(text, str): return []
    text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'\1\2', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = text.lower()
    sorted_eng_terms = sorted(translation_dict.keys(), key=len, reverse=True)
    for eng, rus in [(term, translation_dict[term]) for term in sorted_eng_terms]:
        text = re.sub(r'\b' + re.escape(eng) + r'\b', rus, text)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text, language='russian')

    stemmed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2 and not token.isdigit():
            if token == '-' or re.fullmatch(r'-?\d+(\.\d+)?-?', token):
                 continue
            stem = stemmer.stem(token)
            stemmed_tokens.append(stem)

    # Remove stopwords and short tokens
    processed_tokens = [
        stem for stem in stemmed_tokens
        if stem not in stop_words and len(stem) > 2 and not stem.isdigit()
    ]
    return processed_tokens

russian_stemmer = SnowballStemmer("russian")

# Emerging Terms, requires monthly_frequencies
def find_emerging_terms(monthly_frequencies, ordered_month_names, term_type='keywords',
                        early_months_count=2, late_months_count=2,
                        low_freq_threshold=1, high_freq_threshold=5, min_ratio_increase=5):
    emerging_terms = {}
    if len(ordered_month_names) < early_months_count + late_months_count:
        print("Not enough periods for comparison")
        return emerging_terms

    all_terms = set()
    for month_name in ordered_month_names:
        if month_name in monthly_frequencies and term_type in monthly_frequencies[month_name]:
            all_terms.update(monthly_frequencies[month_name][term_type].keys())

    early_period_names = ordered_month_names[:early_months_count]
    late_period_names = ordered_month_names[-late_months_count:]

    for term in all_terms:
        avg_freq_early = np.mean([monthly_frequencies[m][term_type].get(term, 0) for m in early_period_names if m in monthly_frequencies])
        avg_freq_late = np.mean([monthly_frequencies[m][term_type].get(term, 0) for m in late_period_names if m in monthly_frequencies])

        is_low_early = all(monthly_frequencies[m][term_type].get(term, 0) <= low_freq_threshold for m in early_period_names if m in monthly_frequencies)
        is_high_late = any(monthly_frequencies[m][term_type].get(term, 0) >= high_freq_threshold for m in late_period_names if m in monthly_frequencies)
        ratio_increase_met = (avg_freq_late / (avg_freq_early + 1e-6)) >= min_ratio_increase

        if is_low_early and is_high_late and ratio_increase_met:
            emerging_terms[term] = {
                'avg_freq_early': round(avg_freq_early, 2),
                'avg_freq_late': round(avg_freq_late, 2),
                'freq_trend': [monthly_frequencies[m][term_type].get(term,0) for m in ordered_month_names if m in monthly_frequencies]
            }
    sorted_emerging_terms = dict(sorted(emerging_terms.items(), key=lambda item: item[1]['avg_freq_late'], reverse=True))
    return sorted_emerging_terms


# Burst Detection, requires monthly_frequencies
def detect_bursts(monthly_frequencies, ordered_month_names, term_type='keywords', top_n_terms_to_check=50):
    try:
        from burstdetection import GMMBurstsDetector as BD
    except ImportError:
        print("burstdetection not found.")
        return {}

    bursty_terms = {}
    total_term_counts = Counter()
    for month_name in ordered_month_names:
        if month_name in monthly_frequencies and term_type in monthly_frequencies[month_name]:
            total_term_counts.update(monthly_frequencies[month_name][term_type])
    if not total_term_counts: return {}

    for term, _ in total_term_counts.most_common(top_n_terms_to_check):
        frequencies = [monthly_frequencies[m][term_type].get(term, 0) for m in ordered_month_names if m in monthly_frequencies]
        if not frequencies or sum(frequencies) < 3: continue
        data_for_burst = [(i, freq) for i, freq in enumerate(frequencies) if freq > 0]
        if len(data_for_burst) < 2 : continue
        try:
            bd = BD(n_states=2, transition_cost=1.0, min_burst_length=1, observation_model='poisson')
            bursts = bd.fit_predict(data_for_burst)
            burst_periods = []
            is_bursting = False
            start_burst = -1
            for i, (time_idx, freq) in enumerate(data_for_burst):
                state = bursts[i]
                month_original_idx = time_idx
                if state > 0 and not is_bursting:
                    is_bursting = True
                    start_burst = month_original_idx
                elif state == 0 and is_bursting:
                    is_bursting = False
                    burst_periods.append((ordered_month_names[start_burst], ordered_month_names[month_original_idx-1 if month_original_idx > start_burst else start_burst]))
                    start_burst = -1
            if is_bursting and start_burst != -1:
                 burst_periods.append((ordered_month_names[start_burst], ordered_month_names[data_for_burst[-1][0]]))
            if burst_periods:
                bursty_terms[term] = {'frequencies': frequencies, 'burst_periods': burst_periods}
        except Exception: pass
    return bursty_terms

# Deviations in PageRank and Katz centrality indices, requires final_metrics_df
def find_centrality_growth(final_metrics_df, ordered_month_names,
                           metric_prefix='PageRank', min_growth_factor=1.5, min_final_value=0.01):
    growing_nodes = {}
    if final_metrics_df.empty or len(ordered_month_names) < 2: return growing_nodes
    first_month_col = f"{metric_prefix}_{ordered_month_names[0]}"
    last_month_col = f"{metric_prefix}_{ordered_month_names[-1]}"
    if not (first_month_col in final_metrics_df.columns and last_month_col in final_metrics_df.columns):
        print(f"Columns {metric_prefix} not found.")
        return growing_nodes

    for index, row in final_metrics_df.iterrows():
        node = row['Node']
        initial_value = row[first_month_col]
        final_value = row[last_month_col]
        growth_achieved = False
        growth_factor_val = float('inf')
        if initial_value < 1e-6 and final_value >= min_final_value :
            growth_achieved = True
        elif initial_value > 1e-6:
            growth_factor_val = final_value / initial_value
            if growth_factor_val >= min_growth_factor and final_value >= min_final_value:
                growth_achieved = True
        if growth_achieved:
            all_values = [row.get(f"{metric_prefix}_{m}", 0) for m in ordered_month_names]
            growing_nodes[node] = {
                'initial_value': round(initial_value, 4), 'final_value': round(final_value, 4),
                'growth_factor': round(growth_factor_val,2) if growth_factor_val != float('inf') else 'inf',
                'metric_trend': [round(v, 4) for v in all_values]}
    sorted_growing_nodes = dict(sorted(growing_nodes.items(), key=lambda item: item[1]['final_value'], reverse=True))
    return sorted_growing_nodes

# Graph Differencing, requires G1, G2 : Graph
def compare_graphs(G1, G2, G1_name="G1", G2_name="G2"):
    diff = {}
    nodes1, nodes2 = set(G1.nodes()), set(G2.nodes())
    canonical_edges1 = set(tuple(sorted(e)) for e in G1.edges())
    canonical_edges2 = set(tuple(sorted(e)) for e in G2.edges())

    diff['new_nodes_in_G2'] = list(nodes2 - nodes1)
    diff['removed_nodes_from_G1'] = list(nodes1 - nodes2)
    diff['new_edges_in_G2'] = [tuple(e) for e in canonical_edges2 - canonical_edges1]
    diff['removed_edges_from_G1'] = [tuple(e) for e in canonical_edges1 - canonical_edges2]

    common_edges = canonical_edges1.intersection(canonical_edges2)
    edge_weight_changes = {}
    for u_orig, v_orig in common_edges: # u_orig, v_orig are sorted
        # G1
        weight1 = G1.get_edge_data(u_orig, v_orig, default={}).get('weight')
        if weight1 is None and G1.has_edge(v_orig, u_orig): # Check reverse for safety, though sorted tuple should match
             weight1 = G1.get_edge_data(v_orig, u_orig, default={}).get('weight')
        weight1 = weight1 if weight1 is not None else 1 # Default weight if not specified

        # G2
        weight2 = G2.get_edge_data(u_orig, v_orig, default={}).get('weight')
        if weight2 is None and G2.has_edge(v_orig, u_orig):
             weight2 = G2.get_edge_data(v_orig, u_orig, default={}).get('weight')
        weight2 = weight2 if weight2 is not None else 1

        if weight1 != weight2:
             edge_weight_changes[tuple(sorted((u_orig,v_orig)))] = {'from': weight1, 'to': weight2, 'change': weight2 - weight1}

    diff['edge_weight_changes'] = edge_weight_changes

    print(f"\n--- Comparing graphs {G1_name} and {G2_name} ---")
    print(f"New vertices {G2_name}: {diff['new_nodes_in_G2'][:10]}" + ("..." if len(diff['new_nodes_in_G2']) > 10 else ""))
    return diff


# LDA and NMF for months
# <--- preprocess_fn requires `russian_stemmer` --- >
def get_topics_per_month(df_combined, target_months_map, TARGET_YEAR,
                         preprocess_fn, stemmer_or_lemmatizer, stop_words, translation_dict, # stemmer вместо mystem_analyzer
                         n_topics=5, n_top_words=7, model_type='lda'):
    monthly_topics_data = {}
    for month_num, month_name in target_months_map.items():
        print(f"\n--- Topic model ({model_type.upper()}) for: {month_name} ---")
        month_df = df_combined[
            (df_combined['Date'].dt.year == TARGET_YEAR) &
            (df_combined['Date'].dt.month == month_num)
        ]
        if month_df.empty or 'Content' not in month_df.columns:
            print(f"No 'Content' for {month_name}.")
            monthly_topics_data[month_name] = {}
            continue

        corpus = month_df['Content'].apply(
            lambda x: " ".join(preprocess_fn(x, stemmer_or_lemmatizer, stop_words, translation_dict)) # передаем stemmer
        ).tolist()
        corpus = [doc for doc in corpus if doc.strip()]

        if not corpus or len(corpus) < n_topics:
            print(f"Skipping {month_name} with size of ({len(corpus)}).")
            monthly_topics_data[month_name] = {}
            continue

        if model_type == 'lda':
            vectorizer = CountVectorizer(max_df=0.95, min_df=2)
        else:
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

        try:
            X = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError as e:
             print(f"Vectorizer error for {month_name}: {e}. Skipping.")
             monthly_topics_data[month_name] = {}
             continue
        if X.shape[0] == 0 or X.shape[1] == 0:
            print(f"No features for {month_name}. Skipping.")
            monthly_topics_data[month_name] = {}
            continue

        current_n_topics = n_topics
        if X.shape[1] < n_topics :
            print(f"Number of features ({X.shape[1]}) is less than requested topics ({n_topics}). Decreasing n_topics down to {X.shape[1]}.")
            current_n_topics = X.shape[1]

        if current_n_topics == 0:
            print(f"Zero topics for {month_name}. Skipping.")
            monthly_topics_data[month_name] = {}
            continue


        if model_type == 'lda':
            model = LatentDirichletAllocation(n_components=current_n_topics, random_state=42, max_iter=10)
        else:
            model = NMF(n_components=current_n_topics, random_state=42, max_iter=200, l1_ratio=0.0)

        try:
            model.fit(X)
        except ValueError as e:
            print(f"Model training error for {model_type.upper()} for {month_name} (even with {current_n_topics} topics): {e}")
            monthly_topics_data[month_name] = {}
            continue

        month_topics = {}
        for topic_idx, topic_work_dist in enumerate(model.components_):
            top_word_indices = topic_work_dist.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_word_indices]
            month_topics[f"Topic {topic_idx+1}"] = top_words
        monthly_topics_data[month_name] = month_topics

        print(f"Topics for {month_name} ({model_type.upper()}):")
        for topic_name, words in month_topics.items():
            print(f"  {topic_name}: {', '.join(words)}")
    return monthly_topics_data


def trend_analysis(term_type='keywords'):
    print(f"\n======= 1. Emerging terms ({term_type}) =======")
    # monthly_frequencies are required
    emerging_kw = find_emerging_terms(monthly_frequencies, ordered_month_names, term_type=term_type,
                                      low_freq_threshold=2, high_freq_threshold=10, min_ratio_increase=3)
    for term, data in list(emerging_kw.items())[:5]:
        print(f"Term: {term}, Early avg. frequency: {data['avg_freq_early']}, Late avg. frequency: {data['avg_freq_late']}, Trend: {data['freq_trend']}")

    print(f"\n======= 2. Burst Detection ({term_type}) =======")
    bursty_kw = detect_bursts(monthly_frequencies, ordered_month_names, term_type=term_type, top_n_terms_to_check=20)
    for term, data in list(bursty_kw.items())[:5]:
        print(f"Term: {term}, Freq: {data['frequencies']}, Bursts (from, to): {data['burst_periods']}")

    print("\n======= 3. PageRank =======")
    # final_metrics_df также должен быть построен на стеммированных/лемматизированных узлах
    growing_pagerank_nodes = find_centrality_growth(final_metrics_df, ordered_month_names, metric_prefix='PageRank',
                                                    min_growth_factor=2.0, min_final_value=0.05)
    for node, data in list(growing_pagerank_nodes.items())[:5]:
        print(f"Узел: {node}, PageRank Initial: {data['initial_value']}, Final: {data['final_value']}, Increase: {data['growth_factor']}, Trend: {data['metric_trend']}")

    print("\n======= 4. Graph Differencing =======")
    if len(monthly_graphs) >= 2:
        # monthly_graphs должны содержать стеммированные узлы
        idx_mar = ordered_month_names.index("March") if "March" in ordered_month_names else -1
        idx_apr = ordered_month_names.index("April") if "April" in ordered_month_names else -1
        if idx_mar != -1 and idx_apr != -1 and idx_mar < idx_apr:
            _ = compare_graphs(monthly_graphs[idx_mar], monthly_graphs[idx_apr], G1_name=ordered_month_names[idx_mar], G2_name=ordered_month_names[idx_apr])
        else: print("No graphs for March and April for comparison.")
    else: print("No graphs.")

    print("\n======= 5. Topic modeling (LDA) =======")
    # russian_stemmer
    lda_topics_by_month = get_topics_per_month(
        df_combined, target_months_map, TARGET_YEAR,
        preprocess_text_stemming, russian_stemmer, russian_stopwords, ENGLISH_TO_RUSSIAN_TERMS,
        n_topics=3, n_top_words=5, model_type='lda'
    )

    print("\n======= 6. Topic modeling (NMF) =======")
    # preprocess_text_stemming и russian_stemmer
    nmf_topics_by_month = get_topics_per_month(
        df_combined, target_months_map, TARGET_YEAR,
        preprocess_text_stemming, russian_stemmer, russian_stopwords, ENGLISH_TO_RUSSIAN_TERMS,
        n_topics=3, n_top_words=5, model_type='nmf'
    )

# Demo
trend_analysis()
trend_analysis("bigrams")