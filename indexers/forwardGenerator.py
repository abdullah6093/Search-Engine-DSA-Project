import glob
import os
import csv
from collections import Counter
from preprocessor import preprocess_files, read_file, clean_and_lemmatize
import time

def build_lexicon(file_paths):
    # Get all unique lemmas from files
    lemmas = set()
    doc_words = []  # list of (doc_id, [words]) for all docs
    
    for doc_id, filepath in enumerate(file_paths):
        words_in_doc = read_file(filepath)  # set of words from your preprocessor
        lemmas.update(words_in_doc)
        doc_words.append((doc_id, words_in_doc))
        
    lemmas = clean_and_lemmatize(lemmas)  # lemmatize the union of all words
    
    # Map lemma to word_id
    lemma_to_id = {word: idx+1 for idx, word in enumerate(sorted(lemmas))}
    
    return lemma_to_id, doc_words

def forward_index(file_paths, output_path="./processed_data/forward_index.csv"):
    lemma_to_id, doc_words = build_lexicon(file_paths)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['doc_id', 'word_id', 'frequency'])
        
        for doc_id, words in doc_words:
            # Convert words to lemmas and count frequency per doc
            lemmas_in_doc = [word for word in words if word in lemma_to_id]
            freq = Counter(lemmas_in_doc)
            for lemma, count in freq.items():
                writer.writerow([doc_id, lemma_to_id[lemma], count])

if __name__ == "__main__":
    folders = ["./dataset/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/"]
    file_paths = []
    for f in folders:
        file_paths.extend(glob.glob(os.path.join(f + "*.json")))
    
    start = time.time()
    forward_index(file_paths)
    print(f"Forward index created in {time.time() - start:.2f} seconds")
