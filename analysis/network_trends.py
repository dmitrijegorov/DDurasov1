import pandas as pd
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from collections import Counter
import re
import spacy

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


print("\nFrequencies (example for March if available):")
if "March" in monthly_frequencies and monthly_frequencies["March"]['keywords']:
    print(f"Top 5 keywords for March: {monthly_frequencies['March']['keywords'].most_common(5)}")
    print(f"Top 5 bigrams for March: {monthly_frequencies['March']['bigrams'].most_common(5)}")
    print(f"Top 5 trigrams for March: {monthly_frequencies['March']['trigrams'].most_common(5)}")
else:
    print("No frequency data available for March to show as example.")

print("\nCombined Metrics DataFrame (Top 10 rows):")
if not final_metrics_df.empty:
    print(final_metrics_df.head(10))
else:
    print("Final metrics DataFrame is empty.")

print(f"\nTotal {len(monthly_graphs)} graphs created for months: {', '.join(target_months_map.values())}")