from collections import Counter
import numpy as np

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