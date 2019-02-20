#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from feml import *
import nltk
from config import *

def find_ngrams(token_list, n):
	#print(token_list)
	s = []
	
	"""
	whitespace_embedder = ""
	for i in range(n-1):
		whitespace_embedder += " "
	
	for i in range(len(token_list)):
		token_list[i] = whitespace_embedder + token_list[i] + whitespace_embedder
	"""
	
	for token in token_list:
		for i in range(len(token)- n + 1):
			s.append(token[i:n+i])
	
	return s

def find_ngrams_tokens(token_list, n):
	s = []
	
	
	for i in range(len(token_list)- n + 1):
		s.append(token_list[i] + " " + token_list[i+1])
	"""
	for i in range(len(token_list)):
		if i == 0:
			s.append(token_list[i] + " ")
		elif i == len(token_list) - 1:
			s.append(" " + token_list[i])
		else:
			s.append(" " + token_list[i])
			#s.append(token_list[i] + " ")
	"""
	
	return s

def get_corpus(ids, conn, args, n_grams = False, n_grams_tokens = False):
	
	""" This function queries the names of all the pois in the dataset
		and creates a corpus from the words in them"""
	
	#nltk.download()
	
	# get all poi details
	if args['pois_tbl_name'] is not None:
		sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
		df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	else:
		df = pd.read_csv(args['pois_csv_name'])
		
	#is_in_ids = df['poi_id'] in ids
	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	all_ids = list(all_ids)
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
	
	corpus = []
	
	# for every poi name
	for index, row in df.iterrows():
		# perform stemming based on the language it's written in
		stemmed_word = perform_stemming(row[config.initialConfig.name], lang_detect=True)
		# break it in tokens
		not_stopwords, stopwords = normalize_str(row[config.initialConfig.name])
		not_stopwords = list(not_stopwords)
		
		if n_grams_tokens:
			not_stopwords = find_ngrams_tokens(not_stopwords, config.initialConfig.term_n_gram_size)
		elif n_grams:
			not_stopwords = find_ngrams(not_stopwords, config.initialConfig.character_n_gram_size)		
		corpus.append(not_stopwords)
		
	corpus = [elem for sublist in corpus for elem in sublist]
	
	if not n_grams and not n_grams_tokens:
		corpus = [elem for elem in corpus if len(elem) > 2]
		
	return corpus
	
def get_top_k_features(corpus, args, k):
	word_counter = {}
	
	for word in corpus:
		if word in word_counter:
			word_counter[word] += 1
		else:
			word_counter[word] = 1
			
	popular_words = sorted(word_counter, key = word_counter.get, reverse = True)	
	
	"""
	import csv
	
	
	with open('word_features_ranked.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		for word in popular_words:
			list_to_csv = [word, word_counter[word]]
				
			writer.writerow(list_to_csv)

	csvFile.close()
	"""

	#print(popular_words)
	k_new = int(k * len(popular_words))
	top_k = popular_words[:k_new]
	
	return top_k, k_new

def get_poi_top_k_features(ids, conn, top_k_features, args, k):
	# get all poi details
	
	if args['pois_tbl_name'] is not None:
		sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
		df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	else:
		df = pd.read_csv(args['pois_csv_name'])
		
	#is_in_ids = df['poi_id'] in ids
	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	all_ids = list(all_ids)
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
	
	poi_id_to_boolean_top_k_features_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	for poi_id in poi_id_to_boolean_top_k_features_dict:
		poi_id_to_boolean_top_k_features_dict[poi_id] = [0 for _ in range(0, int(k))]
	
	for index, row in df.iterrows():
		 for i in range(len(top_k_features)):
			 if top_k_features[i] in row[config.initialConfig.name].lower():
				 poi_id_to_boolean_top_k_features_dict[row[config.initialConfig.poi_id]][i] = 1
				 
	return poi_id_to_boolean_top_k_features_dict

def get_features_top_k(ids, conn, args, k, test_ids = None):
	""" This function extracts frequent terms from the whole corpus of POI names. 
		During this process, it optionally uses stemming. Selects the top-K most 
		frequent terms and creates feature positions for each of these terms."""
		
	# get corpus
	corpus = get_corpus(ids, conn, args)
	
	# find top k features
	top_k_features, k_feat = get_top_k_features(corpus, args, k)
		
	# get boolean values dictating whether pois have or haven't any of the top features in their names
	if test_ids == None:
		return get_poi_top_k_features(ids, conn, top_k_features, args, k_feat)
	else:
		return get_poi_top_k_features(test_ids, conn, top_k_features, args, k_feat)
	
def get_features_top_k_ngrams(ids, conn, args, k, test_ids = None):
	""" This function extracts frequent n-grams (n is specified) from the whole 
	corpus of POI names. It selects the top-K most frequent n-gram tokens and creates
	feature positions for each of these terms."""
	
	# get corpus
	corpus = get_corpus(ids, conn, args, n_grams = True)
	#print("Length of corpus is {0}".format(len(corpus)))
	
	# find top k features
	top_k_features, k_feat = get_top_k_features(corpus, args, k)
				
	# get boolean values dictating whether pois have or haven't any of the top features in their names
	if test_ids == None:
		return get_poi_top_k_features(ids, conn, top_k_features, args, k_feat)
	else:
		return get_poi_top_k_features(test_ids, conn, top_k_features, args, k_feat)
	
	
def get_features_top_k_ngrams_tokens(ids, conn, args, k, test_ids = None):
	""" This function extracts frequent n-grams (n is specified) from the whole 
	corpus of POI names. It selects the top-K most frequent n-gram tokens and creates
	feature positions for each of these terms."""
	
	# get corpus
	corpus = get_corpus(ids, conn, args, n_grams_tokens = True)
	
	# find top k features
	top_k_features, k_feat = get_top_k_features(corpus, args, k)
	
	# get boolean values dictating whether pois have or haven't any of the top features in their names
	if test_ids == None:
		return get_poi_top_k_features(ids, conn, top_k_features, args, k_feat)
	else:
		return get_poi_top_k_features(test_ids, conn, top_k_features, args, k_feat)
	
def get_poi_id_to_class_centroid_similarities(ids, poi_id_to_encoded_labels_dict, encoded_labels_set, conn, args, encoded_labels_corpus_dict, test = False):
	
	if args['pois_tbl_name'] is not None:
		sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
		df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	else:
		df = pd.read_csv(args['pois_csv_name'])
		
	#is_in_ids = df['poi_id'] in ids
	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	all_ids = list(all_ids)
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
	
	if not test:
		encoded_labels_corpus_dict = dict.fromkeys(encoded_labels_set)
		for key in encoded_labels_corpus_dict:
			encoded_labels_corpus_dict[key] = []
		
		for poi_id, name in zip(df[config.initialConfig.poi_id], df[config.initialConfig.name]):
			# perform stemming based on the language it's written in
			stemmed_word = perform_stemming(name, lang_detect=True)
			# break it in tokens
			not_stopwords, stopwords = normalize_str(name)
			not_stopwords = list(not_stopwords)
			encoded_labels_corpus_dict[poi_id_to_encoded_labels_dict[poi_id][0][0]].append(not_stopwords)
	
		for key in encoded_labels_corpus_dict:
			encoded_labels_corpus_dict[key] = [item for sublist in encoded_labels_corpus_dict[key] for item in sublist]
		
	poi_id_to_similarity_per_label = dict.fromkeys(ids)
	for poi_id in poi_id_to_similarity_per_label:
		poi_id_to_similarity_per_label[poi_id] = []
	
	count = 0
	
	for poi_id, name in zip(df[config.initialConfig.poi_id], df[config.initialConfig.name]):
		# perform stemming based on the language it's written in
		stemmed_word = perform_stemming(name, lang_detect=True)
		# break it in tokens
		not_stopwords, stopwords = normalize_str(name)
		not_stopwords = list(not_stopwords)
		
		for label in encoded_labels_corpus_dict:
			for token in not_stopwords:
				if token in encoded_labels_corpus_dict[label]:
					count += encoded_labels_corpus_dict[label].count(token)
			
			if len(encoded_labels_corpus_dict[label]) == 0:
				corpus_length = 1
			else:
				corpus_length = len(encoded_labels_corpus_dict[label])
			similarity = float(count) / float(corpus_length)
			poi_id_to_similarity_per_label[poi_id].append(similarity)
			count = 0
			
	return poi_id_to_similarity_per_label, encoded_labels_corpus_dict
		
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=True,
		help="name of table containing pois information")
	ap.add_argument("-k", "--k", required=True,
		help="the number of the desired top-k most frequent tokens")
	ap.add_argument("-n", "--n", required=True,
		help="the n-gram size")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	features_top_k_words = get_features_top_k(conn, args)
	
	#print(features_top_k_words)
	
	#features_top_k_ngrams = get_features_top_k_ngrams(conn, args)
	#print(features_top_k_ngrams)
	
	"""
	import csv
	
	with open('data.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		for id in features_top_k_ngrams:
			writer.writerow(features_top_k_ngrams[id])

	csvFile.close()
	"""
if __name__ == "__main__":
   main()
