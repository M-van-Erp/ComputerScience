# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 17:45:13 2021

@author: MartijnvanErp
"""

import json
import pandas as pd
import numpy as np
from time import perf_counter
import random
from random import shuffle
from math import comb
import re
import itertools
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

random.seed(251121)
np.random.seed(16111967)


#open on laptop
# with open('C:/Users/marti/Google Drive/MSc Business Analytics & Quantitative Marketing/2 - Computer Science for Business Analytics/TVs-all-merged.json') as f:
#     data = json.load(f)
    
#open on PC    
with open('C:/Users/martijn/Google Drive/MSc Business Analytics & Quantitative Marketing/2 - Computer Science for Business Analytics/TVs-all-merged.json') as f:
    data = json.load(f)
        
    
def create_dataframe(data):
      minData = []
      for k,v in data.items(): #gaat over dictionary (productID)
              for element in v: #gaat over list (items)
                  title = element['title']
                  shop = element['shop']
                  
                  if('Brand' in element['featuresMap'].keys()):
                      brand = element['featuresMap']['Brand']
                  else: brand = None
                  
                  features_str = ""
                  for item in element['featuresMap'].values():
                      features_str += item #perhaps remove spaces from here??
                      features_str += " "
                  
                  features_dict = element['featuresMap']
                                        
                  minData.append((k, title, shop, brand, features_str, features_dict)) #extract title from item
      
      tvDF = pd.DataFrame(minData, columns = ['ID', 'Title', 'Shop', 'Brand', 'Features', 'FeaturesDict'])
      tvDF = cleandata(tvDF)  
      return tvDF

#main function that takes all user specifiable variables as input and predicts duplicates and returns performance metrics
def predict_duplicates(tvDF, band_nr, band_width, dist_threshold): 
    transition_threshold = (1/band_nr)**(1/band_width) #t
    print(f'Transition threshold = {round(transition_threshold, 2)} where #bands = {band_nr} and band_width = {band_width}')

    tvDF, all_mw = onehot_modelwords(tvDF)

    tvDF = add_signatures(tvDF, band_nr, band_width, all_mw)

    global buckets
    global counter
    counter = 0 
    buckets = []
    
    for b in range(band_nr):
        buckets.append({})
       
    for s in tvDF['Signature']:
        hash_sig(s, band_nr, band_width)
    
    candidates = get_candidates(buckets)
    # print(f'Nr of candidate pairs: {len(candidates)}')
    
    total_comparisons = comb(len(tvDF),2)
    frac_comparisons = round(len(candidates)/total_comparisons,6)
    # print(f'total possible comparisons: {total_comparisons}')
    # print(f'Size of current dataframe: {len(tvDF)}')
    print(f'Fraction of comparisons: {frac_comparisons}')
    
    dissim_matrix = create_distancematrix(tvDF, candidates)

    clustering = create_clusters(dissim_matrix, dist_threshold)

    duplicates = get_predicted_duplicates(clustering)
       
    evaluations = list(evaluate(candidates, duplicates, tvDF))
    evaluations.append(frac_comparisons)

    return evaluations


#clean data
def cleandata(tvDF):
    
    for columns in ['Title']:    
        tvDF[columns] = tvDF[columns].map(lambda title: title.lower())
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('-', ''))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace(',', ''))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('.', ''))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace(':', ''))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace(';', ''))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('/', ''))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('newegg.com', '')) #we don't want similarity between shops (as these are NOT duplicates)
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('best buy', ''))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('thenerds.net', ''))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('hertz', 'hz'))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('-hz', 'hz'))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace(' hz', 'hz'))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('inches', 'inch'))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('"', 'inch'))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('-inch', 'inch'))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace(' inch', 'inch'))
        tvDF[columns] = tvDF[columns].map(lambda title: re.sub('\W+', ' ', title))
        tvDF[columns] = tvDF[columns].map(lambda title: title.replace('  ', ' '))
    
    # Leftover of trying to implement something based on key-value pairs as well...
    # for columns in ['Features']:    
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.lower())
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('-', '.'))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace(',', ''))
    #     # tvDF[columns] = tvDF[columns].map(lambda title: title.replace('.', ''))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace(':', ''))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace(';', ''))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace(' x ', ''))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('/', ''))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('newegg.com', '')) #we don't want similarity between shops (as these are NOT duplicates)
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('best buy', ''))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('thenerds.net', ''))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('hertz', 'hz'))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('-hz', 'hz'))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace(' hz', 'hz'))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('inches', 'inch'))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('"', 'inch'))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('-inch', 'inch'))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace(' inch', 'inch'))
    #     tvDF[columns] = tvDF[columns].map(lambda title: title.replace('  ', ' '))
    #     # tvDF[columns] = tvDF[columns].map(lambda title: title.replace(' ', ' '))    
        
    tvDF['Brand'] = tvDF['Brand'].map(lambda brand: brand.lower() if (brand != None) else None)
    return tvDF

def onehot_modelwords(tvDF):
    
    title_regex = "([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)"
    
    MW_arr = [set() for _ in range(len(tvDF))]
    all_mw = set()
    
    for i, row in tvDF.iterrows():
                    
        mw_title = set(itertools.chain.from_iterable(re.findall(title_regex, row['Title'])))        
        
        modelwords = mw_title
        all_mw = all_mw.union(modelwords)
        
        MW_arr[i] = modelwords

    all_mw = list(all_mw)    
    tvDF['MW'] = MW_arr
    
    tvDF.loc[:, 'OneHot_MW'] = tvDF['MW'].map(lambda mw: [1 if a in mw else 0 for a in all_mw])
            
    return tvDF, all_mw        

#MinHashing
#Create hashes (orders in which the vectors are shuffled)
def minhash_func(vocab_size, n):
    hashes = []
    for _ in range(n):
        hash_vec = list(range(1, vocab_size + 1))
        shuffle(hash_vec)
        hashes.append(hash_vec)
    return hashes

#Compute signatures based on OneHot vectors optimized
def signature(minhash_functions, vector):
    idx = np.nonzero(vector)[0].tolist()
    row_nrs = minhash_functions[:, idx]
    # print('arrived signature function')
    signature = np.min(row_nrs, axis=1)
    return signature

def add_signatures(tvDF, band_nr, band_width, all_mw):
    sig_length = band_nr * band_width # n = b*r
    vocab_size = len(all_mw)
    minhash_functions = np.array(minhash_func(vocab_size, sig_length))
    tvDF.loc[:,'Signature'] = tvDF['OneHot_MW'].map(lambda onehot: signature(minhash_functions, onehot))
    return tvDF

#Jaccard similarity
def jaccard(x, y):
    x = set(x)
    y = set(y)
    return float(len(x.intersection(y)) / len(x.union(y)))

#LSH
def split_signature(signature, band_nr, band_width):
    subsigs = []
    for i in range(0, len(signature), band_width):
        subsigs.append(signature[i : i+band_width])
    return subsigs

def hash_sig(signature, band_nr, band_width):
    global counter
    global buckets
    
    subsigs = np.array(split_signature(signature, band_nr, band_width)).astype(str)
    for i, subsig in enumerate(subsigs):
        subsig = ', '.join(subsig)
        if subsig not in buckets[i].keys(): #pairs are (only) potential candidates when their band_width signatures match exactly, too strict? 
            buckets[i][subsig] = []
        buckets[i][subsig].append(counter)
    counter = counter + 1
    
def get_candidates(buckets):
    candidates = []
    for band in buckets:
        keys = band.keys()
        for bucket in keys:
            potential_duplicate_list = band[bucket] #returns the array matching to the key 'band' 
            if len(potential_duplicate_list) > 1: #if len(key_array) > 1 that means 2 signatures are in the same bucket -> potential candidate
                candidates.extend(combinations(potential_duplicate_list, 2))
    return list(set(candidates))

#Performance
def evaluate(candidates, duplicates, tvDF):
    # print(f'Amount of predicted duplicate pairs = {len(duplicates)}')
    total_nr_duplicates = 0
    
    #find nr of true duplicates in the data
    counted = set()
    for modelid in tvDF['ID']:
        if modelid not in counted: #prevents duplicates
            nr_pairs_for_id = len(list(combinations(np.where(tvDF['ID'] == modelid)[0], 2)))    
            total_nr_duplicates += nr_pairs_for_id
            counted.add(modelid)
    # print(f'total #duplicates: {total_nr_duplicates}')            
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    cand_true_pos = 0
    cand_false_pos = 0
    
    nr_comparisons = len(candidates)
    pred_dup = set(duplicates)
    
    for i in range(len(candidates)):
        if (tvDF['ID'][candidates[i][0]] == tvDF['ID'][candidates[i][1]]):
            cand_true_pos += 1
            # list_trueduplicates.append(tvDF['ID'][duplicates[i][0]])
        else:
            cand_false_pos += 1
    
    
    for i in range(len(pred_dup)):
        if (tvDF['ID'][duplicates[i][0]] == tvDF['ID'][duplicates[i][1]]):
            true_pos += 1
            # list_trueduplicates.append(tvDF['ID'][duplicates[i][0]])
        else:
            false_pos += 1
            
    # print(f'True pos: {true_pos}')  
    # print(f'False pos: {false_pos}')
    pair_quality = cand_true_pos/nr_comparisons #precision
    pair_completeness = cand_true_pos/total_nr_duplicates #recall
    
    false_neg = total_nr_duplicates - true_pos
    F1_star = 2*((pair_quality*pair_completeness)/(pair_quality+pair_completeness))
    F1 = true_pos/(true_pos + (false_pos+false_neg)/2)

    # print(f'TP = {true_pos}')
    # print(f'Total #duplicates = {total_nr_duplicates}')
    # print(f'Pair Quality: {round(pair_quality, decimals)}')
    # print(f'Pair Completeness: {round(pair_completeness, decimals)}')
    # print(f'F1*: {round(F1_star, decimals)}')
    # print(f'F1: {round(F1, decimals)}')
    # print('')

    return pair_quality, pair_completeness, F1, F1_star
    
#Creating distance matrix where non-candidate pairs have distance 1000 and same shops/diff brands as well
def create_distancematrix(tvDF, candidates):
        
    dissim_matrix = np.full((len(tvDF), len(tvDF)), 1000.00) #sets distance to 1000.00 for all pairs (then we fill in actual distance for candidate pairs)  

    for pair in candidates:
        
        if (tvDF.loc[pair[0], 'Shop'] == tvDF.loc[pair[1], 'Shop']):
            dissim_matrix[pair[0]][pair[1]] = 1000.00
            dissim_matrix[pair[1]][pair[0]] = 1000.00
        elif (tvDF.loc[pair[0],'Brand'] != None and tvDF.loc[pair[1],'Brand'] != None and tvDF.loc[pair[0],'Brand'] != tvDF.loc[pair[1],'Brand']):
            dissim_matrix[pair[0]][pair[1]] = 1000.00
            dissim_matrix[pair[1]][pair[0]] = 1000.00
        else: 
            dissim_matrix[pair[0]][pair[1]] = 1-jaccard(tvDF.loc[pair[0], 'Signature'], tvDF.loc[pair[1], 'Signature'])
            dissim_matrix[pair[1]][pair[0]] = 1-jaccard(tvDF.loc[pair[0], 'Signature'], tvDF.loc[pair[1], 'Signature'])

    return dissim_matrix

#Clustering
def create_clusters(dissim_matrix, dist_threshold):
    clustering = AgglomerativeClustering(affinity='precomputed', linkage='average', distance_threshold=dist_threshold, n_clusters=None)
    clustering.fit(dissim_matrix)
    # print(f'Number of clusters found: {clustering.n_clusters_}') 
    return clustering

def get_predicted_duplicates(clustering):
    duplicates =[]
    for cluster in range(clustering.n_clusters_): #for all clusters check products in cluster
        products_in_cluster = np.where(clustering.labels_ == cluster)[0]
        if (len(products_in_cluster) > 1):
            duplicates.extend(list(combinations(products_in_cluster, 2)))
    return duplicates


br_pairs = [
            [2, 720],
            [40, 36],
            [45, 32],
            [48, 30], 
            [60, 24], 
            [90, 16], 
            [96, 15],
            [120, 12], 
            [144, 10], 
            [160, 9],
            # [180, 8], similar threshold t as surrounding pairs
            [240, 6],
            # [288, 5], similar threshold t as surrounding pairs
            [360, 4],
            [480, 3],
            [720, 2]
            ]

dist_threshold_list =  [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] #np.arange(0.4, 0.8, 0.05) 

nr_bootstraps = 5
bootstrap_perc = 0.63

def bootstrap(tvDF, nr_bootstraps, bootstrap_perc, dist_threshold_list, 
              br_pairs):
    bootstrap_obs = round(len(tvDF)*bootstrap_perc)

    results_insample = np.empty((1,5))
    results_oos = np.empty((1,5))
    best_thresholds = np.zeros((len(br_pairs),nr_bootstraps))
    for i in range(nr_bootstraps):
        print(f'Bootstrap {i+1}')

        train_indices = np.random.choice(len(tvDF), bootstrap_obs, replace = False)
        
        tvDF_train = tvDF.iloc[train_indices]
        tvDF_test = tvDF.drop(train_indices, axis = 0)
        
        tvDF_train.reset_index(inplace = True)
        tvDF_test.reset_index(inplace = True)
        counter = 0
        for pair in br_pairs:
            F1_max = -1
            band_nr = pair[0]
            band_width = pair[1]  
            PQ_sum = 0
            PC_sum = 0
            F1_star_sum = 0
            for dist_threshold in dist_threshold_list:
                evaluations = predict_duplicates(tvDF_train, band_nr, band_width, dist_threshold)
                PQ = evaluations[0]
                PC = evaluations[1]
                F1 = evaluations[2]
                F1_star = evaluations[3]
                frac_comparisons = evaluations[4]
                print(f'Variables: {[band_nr, band_width, dist_threshold]}')
                # print(f'in-sample F1 score: {round(F1,3)} for variables {[band_nr, band_width, dist_threshold]}')
                # print(f'in-sample F1* score: {round(F1_star,3)} for variables {[band_nr, band_width, dist_threshold]}')
                # print(f'in-sample PQ: {round(PQ,4)}, PC: {round(PC,4)}')
                # print('')
                
                if(F1 > F1_max):
                    F1_max = F1
                    best_thresholds[counter][i] = dist_threshold #counter denotes pair number
                
                PQ_sum += PQ
                PC_sum += PC
                F1_star_sum += F1_star
                
            PQ = PQ_sum/len(dist_threshold_list)
            PC = PC_sum/len(dist_threshold_list)
            F1_star = F1_star_sum/len(dist_threshold_list)
            
            results_insample = np.append(results_insample, [[PQ, PC, F1_max, F1_star, frac_comparisons]], 0)
            results_oos = np.append(results_oos, [predict_duplicates(tvDF_test, band_nr, band_width, best_thresholds[counter][i])], 0)
            counter += 1
            
    return results_oos[1:], results_insample[1:], best_thresholds

tvDF = create_dataframe(data)

start_bootstrap = perf_counter()        
results_oos, results_insample, best_thresholds = bootstrap(tvDF, nr_bootstraps, bootstrap_perc, dist_threshold_list, 
              br_pairs)        
end_bootstrap = perf_counter()
time_bootstrap = end_bootstrap - start_bootstrap
print(f'Nr of bootstraps: {nr_bootstraps}')
print(f'b,r pairs: {br_pairs}')
print(f'Dist Thresholds: {dist_threshold_list}')
print(f'Time: {time_bootstrap}')
    
avg_insample_bootstrap_results = np.empty((len(br_pairs),results_insample.shape[1]))
for i in range(len(br_pairs)):
    avg_insample_bootstrap_results[i] = np.mean(results_insample[i::len(br_pairs)],axis=0)
    
avg_oos_bootstrap_results = np.empty((len(br_pairs),results_oos.shape[1]))
for i in range(len(br_pairs)):
    avg_oos_bootstrap_results[i] = np.mean(results_oos[i::len(br_pairs)],axis=0)

print(f'Max In Sample F1: {np.max(avg_insample_bootstrap_results[:,2])} (avg over {nr_bootstraps} bootstraps)')
print(f'Max In Sample F1* {np.max(avg_insample_bootstrap_results[:,3])} (avg over {nr_bootstraps} bootstraps)')

print(f'Max out of Sample F1: {np.max(avg_oos_bootstrap_results[:,2])} (avg over {nr_bootstraps} bootstraps)')
print(f'Max out of Sample F1*: {np.max(avg_oos_bootstrap_results[:,3])} (avg over {nr_bootstraps} bootstraps)')

#IN SAMPLE PLOTS
#PQ plot
plt.plot(avg_insample_bootstrap_results[:,4], avg_insample_bootstrap_results[:,0], '-o')        
plt.axis([0, 0.2, 0, 0.2])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("PQ")
plt.title("(IS) PQ vs Frac_comp")
plt.show()

#PC plot
plt.plot(avg_insample_bootstrap_results[:,4], avg_insample_bootstrap_results[:,1], '-o')        
plt.axis([0, 1, 0, 1])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("PC")
plt.title("(IS) PC vs Frac_comp")
plt.show()
      
#F1* plot
plt.plot(avg_insample_bootstrap_results[:,4], avg_insample_bootstrap_results[:,3], '-o')        
plt.axis([0, 0.2, 0, 0.2])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("F1*")
plt.title("(IS) F1* vs Frac_comp")
plt.show()

#F1 plot
plt.plot(avg_insample_bootstrap_results[:,4], avg_insample_bootstrap_results[:,2], '-o')        
plt.axis([0, 1, 0, 1])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("F1")
plt.title("(IS) F1 vs Frac_comp (including LSH)")
plt.show()

#OUT OF SAMPLE PLOTS
#PQ plot
plt.plot(avg_oos_bootstrap_results[:,4], avg_oos_bootstrap_results[:,0], '-o')        
plt.axis([0, 0.2, 0, 0.2])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("PQ")
plt.title("(OOS) PQ vs Frac_comp")
plt.show()

#PC plot
plt.plot(avg_oos_bootstrap_results[:,4], avg_oos_bootstrap_results[:,1], '-o')        
plt.axis([0, 1, 0, 1])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("PC")
plt.title("(OOS) PC vs Frac_comp")
plt.show()

#F1* plot
plt.plot(avg_oos_bootstrap_results[:,4], avg_oos_bootstrap_results[:,3], '-o')        
plt.axis([0, 0.2, 0, 0.2])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("F1*")
plt.title("(OOS) F1* vs Frac_comp")
plt.show()

#F1 plot
plt.plot(avg_oos_bootstrap_results[:,4], avg_oos_bootstrap_results[:,2], '-o')        
plt.axis([0, 1, 0, 1])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("F1")
plt.title("(OOS) F1 vs Frac_comp (including LSH)")
plt.show()
