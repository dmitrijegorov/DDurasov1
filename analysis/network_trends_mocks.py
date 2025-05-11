import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import numpy as np
import re
import spacy
from network_trends import trend_analysis

spacy.cli.download("ru_core_news_sm")

TARGET_YEAR = 2025
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['это', 'весь', 'который', 'свой', 'еще', 'раз', 'здравствуйте',
                          'такой', 'какой', 'например', 'каждый', 'очень', 'просто',
                          'однако', 'также', 'именно', 'конечно', 'коллега', 'спасибо',
                          'пожалуйста', 'добрый', 'день', 'вечер', 'утро', 'мой', 'твой',
                          'ваш', 'наш', 'говорить', 'сказать', 'делать', 'сделать'])

ENGLISH_TO_RUSSIAN_TERMS = {
    'testfit': 'тестфит', 'revit': 'ревит', 'enscape': 'энскейп',
    'autocad': 'автокад', 'ai': 'ии', 'product': 'продукт', 'design': 'дизайн',
    'development': 'разработка', 'hr': 'эйчар', 'marketing': 'маркетинг',
    'software': 'софт', 'it': 'айти', 'digital': 'цифровой'
}

months_list = ["February", "March", "April", "May"]
monthly_frequencies = {
    month: {
        'keywords': Counter({f'слово{i}_{month.lower()}': np.random.randint(1, 20) for i in range(5)}),
        'bigrams': Counter({f'биграмма{i}_{month.lower()}': np.random.randint(1, 10) for i in range(3)})
    } for month in months_list
}
monthly_frequencies["February"]['keywords']['внезапн_термин'] = 0
monthly_frequencies["March"]['keywords']['внезапн_термин'] = 1
monthly_frequencies["April"]['keywords']['внезапн_термин'] = 15
monthly_frequencies["May"]['keywords']['внезапн_термин'] = 25

monthly_graphs_nodes = {
    "February": ['слов1_february', 'слов2_february', 'общ1'],
    "March": ['слов1_march', 'слов3_march', 'общ1', 'общ2'],
    "April": ['слов1_april', 'слов4_april', 'общ1', 'общ2', 'нов_узел_apr'],
    "May": ['слов1_may', 'слов5_may', 'общ1', 'нов_узел_may', 'нов_узел_apr']
}
monthly_graphs = []
for month_name in months_list:
    G = nx.Graph()
    if monthly_graphs_nodes.get(month_name):
        nodes_for_graph = monthly_graphs_nodes[month_name]
        G.add_nodes_from(nodes_for_graph)
        if len(nodes_for_graph) > 1:
            for i in range(len(nodes_for_graph) // 2):
                u, v = np.random.choice(nodes_for_graph, 2, replace=False)
                G.add_edge(u,v, weight=np.random.randint(1,5))
    monthly_graphs.append(G)

nodes_all = set()
for m_nodes in monthly_graphs_nodes.values(): nodes_all.update(m_nodes)
nodes_all = list(nodes_all)
final_metrics_data = {'Node': nodes_all}
for month_name in months_list:
    final_metrics_data[f'PageRank_{month_name}'] = [np.random.rand() * 0.1 for _ in nodes_all]
    final_metrics_data[f'Katz_{month_name}'] = [np.random.rand() * 0.01 for _ in nodes_all]
    if 'общ1' in nodes_all: # Имитация роста для 'общ1'
        idx = nodes_all.index('общ1')
        if month_name == "February": final_metrics_data[f'PageRank_{month_name}'][idx] = 0.01
        if month_name == "March": final_metrics_data[f'PageRank_{month_name}'][idx] = 0.03
        if month_name == "April": final_metrics_data[f'PageRank_{month_name}'][idx] = 0.08
        if month_name == "May": final_metrics_data[f'PageRank_{month_name}'][idx] = 0.12
final_metrics_df = pd.DataFrame(final_metrics_data)

sample_data_combined = []
for i, month_name in enumerate(months_list):
    month_num = i + 2
    sample_data_combined.extend([
        {'Date': pd.Timestamp(f'{TARGET_YEAR}-{month_num:02d}-01'), 'Content': f'Текст про {month_name.lower()} и ии разработка', 'clean_content': ''},
        {'Date': pd.Timestamp(f'{TARGET_YEAR}-{month_num:02d}-15'), 'Content': f'Другой текст про {month_name.lower()} и софт и дизайн', 'clean_content': ''}
    ])
df_combined = pd.DataFrame(sample_data_combined)

target_months_map = {i+2: month_name for i, month_name in enumerate(months_list)}
ordered_month_names = [target_months_map[m_num] for m_num in sorted(target_months_map.keys())]

trend_analysis()
