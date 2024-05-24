import copy
import json
from tqdm import tqdm
from pyserini.search import LuceneSearcher, get_topics, get_qrels

import tempfile
import shutil
from rank_gpt import write_eval_file
from trec_eval import EvalFunction

import copy
import transformers

transformers.logging.set_verbosity_error()

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rank_gpt import permutation_pipeline

import pickle

THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',
}

datasets = ['dl19', 'dl20', 'news', 'touche', 'signal', 'covid']

def run_retriever(data, index, topic, k=100):
  searcher = LuceneSearcher.from_prebuilt_index(index)
  topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
  qrels = get_qrels(topic)

  ranks = []
  #ranks_raw = []

  for qid in tqdm(topics):
      if qid in qrels:
          query = topics[qid]['title']
          hits_raw = searcher.search(query, k=k)

          #ranks_raw.append(hits_raw)
          ranks.append({'query': query, 'hits':[]})

          rank = 0
          for hit in hits_raw:
              rank += 1
              content = json.loads(searcher.doc(hit.docid).raw())
              if 'title' in content:
                  content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
              else:
                  content = content['contents']
              content = ' '.join(content.split())
              ranks[-1]['hits'].append({
                  'content': content,
                  'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})

  return ranks#, ranks_raw

def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = remove_duplicate(permutation)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item

def stage1_reranker(ranks, ranks_raw, reranker):
  return ranks  

def stage2_reranker(stage1_ranks, model, device, tokenizer):
  stage2_ranks = []

  post_prompt = "Extract three sentences from the passage above which best answers the query. Only respond with the sentences, do not say any words or explain. If there is no sentence which adequately answers the query, simply write None."


  for rank in tqdm(stage1_ranks):
      query = rank['query']
      hits = rank['hits']
      titles = []

      for hit in hits:
          passage = hit['content']

          query_prompt = f"Query: {query}.\n"
          passage_prompt = f"Passage: {passage}.\n"

          prompt = query_prompt + passage_prompt + post_prompt

          inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
          outputs = model.generate(inputs['input_ids'], max_length=60, num_beams=5, early_stopping=True)
          title = tokenizer.decode(outputs[0], skip_special_tokens=True)

          titles.append(title)

      modified_hits = copy.deepcopy(hits)
      for i, title in enumerate(titles):
          modified_hits[i]['content'] = title


      #print(titles[0:5])

      modified_rank = copy.deepcopy(rank)
      modified_rank['hits'] = modified_hits

      stage2_ranks.append(modified_rank)

  return stage2_ranks

def permute_stage2_ranker(stage2_ranks):
    new_ranks = []
    
    for rank in tqdm(stage2_ranks):
        query = rank['query']
        hits = rank['hits']
        
        some_list = []
        none_list = []

        for i in range(100):
            passage = hits[i]['content']

            if passage == 'None':
               none_list.append(i)
            else:
               some_list.append(i)
        
        some_list.extend(none_list)

        new_rank = receive_permutation(rank, some_list)
        new_ranks.append(new_rank)

    return new_ranks

def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo',
                    api_key=None):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name, api_key=api_key)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item

def stage3_reranker(stage2_ranks, openai_key, window_size=20):
  stage3_ranks = []

  for rank in tqdm(stage2_ranks):
    new_rank = sliding_windows(rank, rank_start=0, rank_end=100, window_size=window_size, step=10,
                                    model_name='gpt-3.5-turbo', api_key=openai_key)
    stage3_ranks.append(new_rank)

  return stage3_ranks

def evaluate(ranks, topic, filename):
  output_file = tempfile.NamedTemporaryFile(delete=False).name
  write_eval_file(ranks, output_file)
  EvalFunction.eval(['-c', '-m', 'ndcg_cut.1', topic, output_file])
  EvalFunction.eval(['-c', '-m', 'ndcg_cut.5', topic, output_file])
  EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', topic, output_file])

  shutil.move(output_file, filename)

def pickle_ranks(ranks, filename):
  with open(filename, 'wb') as handle:
    pickle.dump(ranks, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pipeline(datasets):
  for data in datasets:
    index = THE_INDEX[data]
    topic = THE_TOPICS[data]

    print("#"*40)
    print(f'Stage 0: Retrieving {data} dataset...')
    print("#"*40)

    ranks = run_retriever(data, index, topic)
    #evaluate(ranks, topic, f'stage0_eval_{data}.txt')
    #pickle_ranks(ranks, f'stage0_ranks_{data}.pickle')

    #stage 2 ranking pipeline
    model_name = "google/flan-t5-xl"
    mn = "flant5xl"

    openai_key = ""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)

    device = "cuda" #if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("#"*40)
    print (f'Stage 2: Compressing {data} passages...')
    print("#"*40)

    #stage2_ranks = ranks
    #stage2_ranks = stage2_reranker(stage2_ranks, model, device, tokenizer)
    #pickle_ranks(stage2_ranks, f'stage2_ranks_{data}_{mn}_{method}.pickle')

    methods = ["extract_sentence_1", "extract_sentence_2", "extract_sentence_3"]
    window = 100

    for method in methods:
        with open(f'stage2_ranks_{data}_{mn}_{method}.pickle', 'rb') as pickle_file:
            stage2_ranks = pickle.load(pickle_file)
        
        print("#"*40)
        print (f'Stage 3: Permuting {data} passages...')
        print("#"*40)

        stage2_ranks_permuted = permute_stage2_ranker(stage2_ranks)

        #print([hit['content'] for hit in stage2_ranks[0]['hits']])

        print("#"*40)
        print (f'Stage 3: Re-Ranking {data} compressed passages...')
        print("#"*40)

        stage3_ranks = stage3_reranker(stage2_ranks, openai_key, window_size=window)
        #eval(stage3_ranks, topic,f'stage3_eval_{data}_{mn}.txt')
        pickle_ranks(stage3_ranks, f'stage3_ranks_{data}_{mn}_{method}_{window}_without_sort.pickle')

        stage3_ranks = stage3_reranker(stage2_ranks_permuted, openai_key, window_size=window)
        pickle_ranks(stage3_ranks, f'stage3_ranks_{data}_{mn}_{method}_{window}_with_sort.pickle')


pipeline(['dl19']) #datasets)
