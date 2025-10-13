"""
@author:Libo(tongjilibo@163.com)
@description:
Pure numpy backend implementation of similarity functions
"""

import heapq
import queue
from typing import Union
import numpy as np


def normalize_embeddings(embeddings: np.ndarray):
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    return embeddings / norms


def cos_sim(a: Union[np.ndarray], b: Union[np.ndarray]):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return np.dot(a_norm, b_norm.T)


def dot_score(a: Union[np.ndarray], b: Union[np.ndarray]):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return np.dot(a, b.T)


def pairwise_dot_score(a: Union[np.ndarray], b: Union[np.ndarray]):
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i])
    :return: Vector with res[i] = dot_prod(a[i], b[i])
    """
    return np.sum(a * b, axis=-1)


def pairwise_cos_sim(a: Union[np.ndarray], b: Union[np.ndarray]):
    """
   Computes the pairwise cossim cos_sim(a[i], b[i])
   :return: Vector with res[i] = cos_sim(a[i], b[i])
   """
    a_norm = normalize_embeddings(np.atleast_2d(a))
    b_norm = normalize_embeddings(np.atleast_2d(b))
    return pairwise_dot_score(a_norm, b_norm)


def semantic_search(
        query_embeddings: Union[np.ndarray],
        corpus_embeddings: Union[np.ndarray],
        query_chunk_size: int = 100,
        corpus_chunk_size: int = 500000,
        top_k: int = 10,
        score_function=cos_sim
):
    """
    This function performs a cosine similarity search between a list of query embeddings and corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2-dimensional array with the query embeddings.
    :param corpus_embeddings: A 2-dimensional array with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but
        requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed,
        but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Funtion for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys
        'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """
    query_embeddings = np.atleast_2d(query_embeddings)
    corpus_embeddings = np.atleast_2d(corpus_embeddings)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarity
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                        corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

            # Get top-k scores for each query in chunk
            for query_itr in range(cos_scores.shape[0]):
                query_scores = cos_scores[query_itr]
                # Use argpartition for efficiency if top_k << len(corpus_batch)
                if top_k < len(query_scores):
                    top_k_indices = np.argpartition(query_scores, -top_k)[-top_k:]
                    top_k_scores = query_scores[top_k_indices]
                else:
                    top_k_indices = np.arange(len(query_scores))
                    top_k_scores = query_scores
                
                query_id = query_start_idx + query_itr
                for sub_corpus_id, score in zip(top_k_indices, top_k_scores):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(queries_result_list[query_id], (
                            score, corpus_id))  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {'corpus_id': corpus_id, 'score': score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x['score'],
                                               reverse=True)

    return queries_result_list


def paraphrase_mining_embeddings(
        embeddings: np.ndarray,
        query_chunk_size: int = 5000,
        corpus_chunk_size: int = 100000,
        max_pairs: int = 500000,
        top_k: int = 100,
        score_function=cos_sim
):
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param embeddings: An array with the embeddings
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list of triplets with the format [score, id1, id2]
    """

    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0
    n = len(embeddings)

    for corpus_start_idx in range(0, n, corpus_chunk_size):
        for query_start_idx in range(0, n, query_chunk_size):
            scores = score_function(embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                    embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

            # Get top-k scores for each query in chunk
            for query_itr in range(scores.shape[0]):
                query_scores = scores[query_itr]
                
                if top_k < len(query_scores):
                    top_k_indices = np.argpartition(query_scores, -top_k)[-top_k:]
                    top_k_scores = query_scores[top_k_indices]
                else:
                    top_k_indices = np.arange(len(query_scores))
                    top_k_scores = query_scores
                
                i = query_start_idx + query_itr
                for j_idx, j_score in zip(top_k_indices, top_k_scores):
                    j = corpus_start_idx + j_idx

                    if i != j and j_score > min_score:
                        pairs.put((j_score, i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, i, j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def community_detection(embeddings, threshold=0.75, min_community_size=10, batch_size=10000):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """
    embeddings = np.atleast_2d(embeddings)
    n = len(embeddings)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, n)
    sort_max_size = min(max(2 * min_community_size, 50), n)

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        # Compute cosine similarity scores
        cos_scores = cos_sim(embeddings[start_idx:end_idx], embeddings)

        # Minimum size for a community
        # For each row in chunk, find top_k values
        for i in range(end_idx - start_idx):
            row_scores = cos_scores[i]
            top_k_indices = np.argpartition(row_scores, -min_community_size)[-min_community_size:]
            top_k_values = row_scores[top_k_indices]
            
            if top_k_values.min() >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                sort_indices = np.argpartition(row_scores, -sort_max_size)[-sort_max_size:]
                sort_scores = row_scores[sort_indices]
                # Sort by score descending
                sorted_idx = np.argsort(sort_scores)[::-1]
                sort_indices = sort_indices[sorted_idx]
                sort_scores = sort_scores[sorted_idx]

                for idx, val in zip(sort_indices, sort_scores):
                    if val < threshold:
                        break
                    global_idx = idx
                    new_cluster.append(global_idx)

                extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        community = sorted(community)
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities

