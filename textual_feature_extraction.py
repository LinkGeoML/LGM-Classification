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
import glob
from config import *

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.analysis import NgramWordAnalyzer
from whoosh import index as windex
from whoosh import qparser
from whoosh import scoring
from whoosh.reading import IndexReader

def find_ngrams(token_list, n):
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	s = []
	
	for token in token_list:
		for i in range(len(token)- n + 1):
			s.append(token[i:n+i])
	
	return s

def find_ngrams_tokens(token_list, n):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	s = []
	
	
	for i in range(len(token_list)- n + 1):
		s.append(token_list[i] + " " + token_list[i+1])
	
	return s

def get_corpus(ids, conn, args, n_grams = False, n_grams_tokens = False):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	""" This function queries the names of all the pois in the dataset
		and creates a corpus from the words in them"""
	
	#nltk.download()
	
	# get all poi details
	if args['pois_tbl_name'] is not None:
		sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
		df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	else:
		df = pd.read_csv(args['pois_csv_name'])
		
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
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	word_counter = {}
	
	for word in corpus:
		if word in word_counter:
			word_counter[word] += 1
		else:
			word_counter[word] = 1
			
	popular_words = sorted(word_counter, key = word_counter.get, reverse = True)	
	
	k_new = int(k * len(popular_words))
	top_k = popular_words[:k_new]
	
	return top_k, k_new

def get_poi_top_k_features(ids, conn, top_k_features, args, k, feature_type):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	# get all poi details
		
	if args['pois_tbl_name'] is not None:
		sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
		df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	else:
		df = pd.read_csv(args['pois_csv_name'])
		
	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	all_ids = list(all_ids)
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
			
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			if args['step'] == 1 or args['step'] == 2:
				index_folderpath = latest_experiment_folder + '/' + feature_type + '_index_' + str(args['fold_number'])
			else:
				index_folderpath = latest_experiment_folder + '/' + feature_type + '_index_train'
			exists = os.path.exists(index_folderpath)
			if not exists:
				os.mkdir(index_folderpath)
				schema = Schema(path=ID(stored=True), content=TEXT)
				ix = create_in(index_folderpath, schema)
				writer = ix.writer()
				
				if args['step'] == 1 or args['step'] == 2:
					descriptor_filepath = latest_experiment_folder + '/' + feature_type + '_index_' + str(args['fold_number']) + '/k.txt'
				else:
					descriptor_filepath = latest_experiment_folder + '/' + feature_type + '_index_train' + '/k.txt'
				with open(descriptor_filepath, 'w') as f:
					f.write('%d' % k)
				
				for i in range(k):
					writer.add_document(path=str(i), content=top_k_features[i])
				writer.commit()
			else:
				ix = windex.open_dir(index_folderpath)
				if args['step'] == 1 or args['step'] == 2:
					descriptor_filepath = latest_experiment_folder + '/' + feature_type + '_index_' + str(args['fold_number']) + '/k.txt'
				else:
					descriptor_filepath = latest_experiment_folder + '/' + feature_type + '_index_train' + '/k.txt'
				with open(descriptor_filepath, 'r') as f:
					k = f.read()
	else:
		if args['step'] == 1 or args['step'] == 2:
			index_folderpath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + feature_type + '_index_' + str(args['fold_number']) 
		else:
			index_folderpath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + feature_type + '_index_train'
		ix = windex.open_dir(index_folderpath)
		if args['step'] == 1 or args['step'] == 2:
			descriptor_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + feature_type + '_index_' + str(args['fold_number']) + '/k.txt'
		else:
			descriptor_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + feature_type + '_index_train' + '/k.txt'
		with open(descriptor_filepath, 'r') as f:
			k = f.read()
	
	poi_id_to_boolean_top_k_features_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	for poi_id in poi_id_to_boolean_top_k_features_dict:
		poi_id_to_boolean_top_k_features_dict[poi_id] = [0 for _ in range(0, int(k))]
	
	for index, row in df.iterrows():
			with ix.searcher() as searcher:
				if feature_type == "char_ngrams_index":
					ngramAnalyzer = NgramWordAnalyzer(minsize=config.initialConfig.character_n_gram_size, maxsize=config.initialConfig.character_n_gram_size)
					tokens = [token.text for token in ngramAnalyzer(row[config.initialConfig.name].lower())]
					for token in tokens:
						if token in top_k_features:
							query = QueryParser("content", ix.schema).parse(token)
							results = searcher.search(query)
							if len(results) > 0:
								poi_id_to_boolean_top_k_features_dict[row[config.initialConfig.poi_id]][int(results[0]['path'])] = 1
				else:
					query = QueryParser("content", ix.schema).parse(row[config.initialConfig.name].lower())
					results = searcher.search(query)
					if len(results) > 0:
						poi_id_to_boolean_top_k_features_dict[row[config.initialConfig.poi_id]][int(results[0]['path'])] = 1
	return poi_id_to_boolean_top_k_features_dict

def get_features_top_k(ids, conn, args, k, test_ids = None):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	""" This function extracts frequent terms from the whole corpus of POI names. 
		During this process, it optionally uses stemming. Selects the top-K most 
		frequent terms and creates feature positions for each of these terms."""
		
	# get corpus
	corpus = get_corpus(ids, conn, args)
	
	# find top k features
	top_k_features, k_feat = get_top_k_features(corpus, args, k)
		
	# get boolean values dictating whether pois have or haven't any of the top features in their names
	if test_ids == None:
		return get_poi_top_k_features(ids, conn, top_k_features, args, k_feat, "tokens_index")
	else:
		return get_poi_top_k_features(test_ids, conn, top_k_features, args, k_feat, "tokens_index")
	
def get_features_top_k_ngrams(ids, conn, args, k, test_ids = None):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	""" This function extracts frequent n-grams (n is specified) from the whole 
	corpus of POI names. It selects the top-K most frequent n-gram tokens and creates
	feature positions for each of these terms."""
	
	# get corpus
	corpus = get_corpus(ids, conn, args, n_grams = True)
	
	# find top k features
	top_k_features, k_feat = get_top_k_features(corpus, args, k)
				
	# get boolean values dictating whether pois have or haven't any of the top features in their names
	if test_ids == None:
		return get_poi_top_k_features(ids, conn, top_k_features, args, k_feat, "char_ngrams_index")
	else:
		return get_poi_top_k_features(test_ids, conn, top_k_features, args, k_feat, "char_ngrams_index")
	
	
def get_features_top_k_ngrams_tokens(ids, conn, args, k, test_ids = None):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	""" This function extracts frequent n-grams (n is specified) from the whole 
	corpus of POI names. It selects the top-K most frequent n-gram tokens and creates
	feature positions for each of these terms."""
	
	# get corpus
	corpus = get_corpus(ids, conn, args, n_grams_tokens = True)
	
	# find top k features
	top_k_features, k_feat = get_top_k_features(corpus, args, k)
	
	# get boolean values dictating whether pois have or haven't any of the top features in their names
	if test_ids == None:
		return get_poi_top_k_features(ids, conn, top_k_features, args, k_feat, "token_ngrams_index")
	else:
		return get_poi_top_k_features(test_ids, conn, top_k_features, args, k_feat, "token_ngrams_index")
	
def get_poi_id_to_class_centroid_similarities(ids, poi_id_to_encoded_labels_dict, encoded_labels_set, conn, args, encoded_labels_corpus_dict, test = False):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	if args['pois_tbl_name'] is not None:
		sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
		df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	else:
		df = pd.read_csv(args['pois_csv_name'])
		
	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	all_ids = list(all_ids)
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
	
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			if args['step'] == 1 or args['step'] == 2:
				index_folderpath = latest_experiment_folder + '/' + 'similarity_index_' + str(args['fold_number'])
			else:
				index_folderpath = latest_experiment_folder + '/' + 'similarity_index_train'
			exists = os.path.exists(index_folderpath)
			if not exists:
				os.mkdir(index_folderpath)
				schema = Schema(path=ID(stored=True), class_id=STORED, content=TEXT)
				ix = create_in(index_folderpath, schema)
				writer = ix.writer()
				
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
						
						writer.add_document(path=str(poi_id), class_id=str(poi_id_to_encoded_labels_dict[poi_id][0][0]), content=not_stopwords)
					writer.commit()
			else:
				ix = windex.open_dir(index_folderpath)
	else:
		if args['step'] == 1 or args['step'] == 2:
			index_folderpath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + 'similarity_index_' + str(args['fold_number'])
		else:
			index_folderpath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + 'similarity_index_train'
		ix = windex.open_dir(index_folderpath)
	
	poi_id_to_similarity_per_label = dict.fromkeys(ids)
	
	if args['step'] == 3:
		descriptor_filepath = latest_experiment_folder + '/label_count.txt'
		with open(descriptor_filepath, 'w') as f:
			to_write = len(encoded_labels_set)
			#print(to_write)
			f.write('%d' % to_write)
	if args['step'] == 4:
		descriptor_filepath = latest_experiment_folder + '/label_count.txt'
		with open(descriptor_filepath, 'r') as f:
			encoded_labels_count = f.read()
		encoded_labels_list = [i for i in range(0, int(encoded_labels_count))]
		encoded_labels_set = set(encoded_labels_list)
	
	for poi_id in poi_id_to_similarity_per_label:
		poi_id_to_similarity_per_label[poi_id] = [0 for _ in range(len(encoded_labels_set))]
	
	count = 0
	
	for poi_id, name in zip(df[config.initialConfig.poi_id], df[config.initialConfig.name]):
		# perform stemming based on the language it's written in
		stemmed_word = perform_stemming(name, lang_detect=True)
		# break it in tokens
		not_stopwords, stopwords = normalize_str(name)
		not_stopwords = list(not_stopwords)
		
		with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
			query = qparser.QueryParser("content", ix.schema, group = qparser.OrGroup).parse(name)
			results = searcher.search(query)
			for r in results:
				poi_id_to_similarity_per_label[poi_id][int(r['class_id'])] = r.score
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
	
if __name__ == "__main__":
   main()
