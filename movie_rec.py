#imports
import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from scipy import stats

from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import NMF
from surprise import CoClustering
from surprise import Dataset
from surprise import Reader
from utils import *
import sys


sys.stdout = open("useful_output.log", mode='w', encoding='utf-8')

# constants and initialization
folds = 5
my_seed = 0
rd.seed(my_seed)
np.random.seed(my_seed)
top_fraction = 0.2
user_events_file = 'movieratings.csv'
low_user_file = '10_niche_focused_users.txt'
medium_user_file = '10_diversed_focused_users.txt'
high_user_file = '10_blockbuster_focused_users.txt'

# read user events and users
df_events = pd.read_csv(user_events_file, sep=',', header=0)
df_uid_group = df_events.groupby('USER_MD5').count() >= 10
uid = df_uid_group[df_uid_group.MOVIE_ID].index
df_events = df_events[df_events.USER_MD5.isin(uid)]

print('No. of user ratings: ' + str(len(df_events)))

# read users
low_users = pd.read_csv(low_user_file, sep=',', header=None,  index_col=0)
medium_users = pd.read_csv(medium_user_file, sep=',', header=None, index_col=0)
high_users = pd.read_csv(high_user_file, sep=',', header=None, index_col=0)

no_users = len(low_users) + len(medium_users) + len(high_users)
print('No. of users: ' + str(no_users))
print('No. of ratings per user: ' + str(len(df_events) / no_users))

# get movie distribution
item_dist = df_events['MOVIE_ID'].value_counts()
num_items = len(item_dist)
print('No. movies: ' + str(num_items))


user_dist = df_events['USER_MD5'].value_counts()
df_user_dist = pd.DataFrame(user_dist)
df_user_dist.columns = ['Number_of_Movies']


# create item dataframe with normalized item counts
df_item_dist = pd.DataFrame(item_dist)
df_item_dist.columns = ['Number_of_RATING']
df_item_dist['Number_of_RATING'] /= no_users
print('No. of ratings per movie: ' + str(len(df_events) / num_items))

# sparsity
1 - len(df_events) / (no_users * num_items)

# rating range
print('Min rating: ' + str(df_events['RATING'].min()))
print('Max rating: ' + str(df_events['RATING'].max()))


# get top items (20% as the top threshold)
num_top = int(top_fraction * num_items)
top_item_dist = item_dist[:num_top]
min_top_num = min(top_item_dist)
print('Min ratings of top movie:', min_top_num)
top_item_dist = item_dist[item_dist >= min_top_num]
len_top_num = len(top_item_dist)
print('No. top items: ' + str(len_top_num))
ttt = top_item_dist.index


low_gap = 0
medium_gap = 0
high_gap = 0

# get fractions
user_hist = []  # user history sizes
pop_item_fraq = []  # average popularity of items in user profiles
for u, df in df_events.groupby('USER_MD5'):
    no_user_items = len(df['MOVIE_ID'])  # profile size
    user_hist.append(no_user_items)
    # get popularity (= fraction of users interacted with item) of user items and calculate average of it
    user_pop_item_fraq = sum(item_dist[df['MOVIE_ID']]) / no_users / no_user_items
    pop_item_fraq.append(user_pop_item_fraq)

    if u in low_users.index:
        low_gap += user_pop_item_fraq
    elif u in medium_users.index:
        medium_gap += user_pop_item_fraq
    else:
        high_gap += user_pop_item_fraq


low_gap /= len(low_users)
medium_gap /= len(medium_users)
high_gap /= len(high_users)

print(low_gap, medium_gap, high_gap)


plt.figure()
slope, intercept, r_value, p_value, std_err = stats.linregress(user_hist, pop_item_fraq)
print('R-value: ' + str(r_value))
print('R2-value: ' + str(r_value ** 2))
print('P-value: ' + str(p_value))
print('Slope: ' + str(slope))
print('Intercept: ' + str(intercept))
print(stats.spearmanr(user_hist, pop_item_fraq))

line = slope * np.array(user_hist) + intercept

reader = Reader(rating_scale=(df_events['RATING'].min(), df_events['RATING'].max()))

# The columns must correspond to user id, item id and ratings (in that order).
df_events.MOVIE_ID = df_events.MOVIE_ID.astype(np.int32)
df_events.RATING = df_events.RATING.astype(np.int32)
data = Dataset.load_from_df(df_events, reader)


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


def get_top_n_random(testset, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r in testset:
        if len(top_n[uid]) == 0:
            for i in range(0, 10):
                id = rd.choice(item_dist.index)
                top_n[uid].append((id, i))

    return top_n


def get_top_n_mp(testset, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r in testset:
        if len(top_n[uid]) == 0:
            for iid, count in item_dist[:n].items():
                top_n[uid].append((iid, count))
    return top_n


def get_mae_of_groups(predictions):
    result_df = pd.DataFrame([(uid, true_r, est) for uid, _, true_r, est, _ in predictions],
                             columns=['uid', 'true_r', 'est'])
    low_predictions = (result_df[result_df.uid.isin(low_users_id)].true_r - result_df[result_df.uid.isin(low_users_id)].est).abs()
    med_predictions = (result_df[result_df.uid.isin(medium_users_id)].true_r - result_df[result_df.uid.isin(medium_users_id)].est).abs()
    high_predictions = (result_df[result_df.uid.isin(high_users_id)].true_r - result_df[result_df.uid.isin(high_users_id)].est).abs()
    low_mae = low_predictions.mean()
    med_mae = med_predictions.mean()
    high_mae = high_predictions.mean()
    all_mae = np.mean([low_mae, med_mae, high_mae])
    print('niche_focused: ' + str(low_mae))
    print('diversed_focused: ' + str(med_mae))
    print('blockbuster_focused: ' + str(high_mae))
    print("all_mae:" + str(all_mae))

    print('Low vs. med: ' + str(stats.ttest_ind(low_predictions, med_predictions)))
    print('Low vs. high: ' + str(stats.ttest_ind(low_predictions, high_predictions)))

    return low_mae, med_mae, high_mae, all_mae


def get_users_id():
    low_users = pd.read_csv(low_user_file, sep=',', header=None, names=['ID', 'VALUE'])
    medium_users = pd.read_csv(medium_user_file, sep=',', header=None, names=['ID', 'VALUE'])
    high_users = pd.read_csv(high_user_file, sep=',', header=None, names=['ID', 'VALUE'])

    low_users_id = low_users[~low_users.ID.isin(df_events)].ID.values
    medium_users_id = medium_users[~medium_users.ID.isin(df_events)].ID.values
    high_users_id = high_users[~high_users.ID.isin(df_events)].ID.values

    return low_users_id, medium_users_id, high_users_id


low_users_id, medium_users_id, high_users_id = get_users_id()


sim_users = {'name': 'cosine', 'user_based': True}  # compute cosine similarities between users
algos = []
algos.append(None)  # Random())
algos.append(None)  # MostPopular())
algos.append(BaselineOnly())
algos.append(KNNBasic(sim_options=sim_users, k=40))
algos.append(KNNWithMeans(sim_options=sim_users, k=40))
algos.append(NMF(n_factors=15, random_state=my_seed))
algos.append(CoClustering(n_cltr_u=3, n_cltr_i=3, random_state=my_seed))
algo_names = ['Random',
              'MostPopular',
              "UserItemAvg",
              'KNNBasic',
              'KNNWithMeans',
              'NMF',
              'CoClustering']


low_rec_gap_list = []
medium_rec_gap_list = []
high_rec_gap_list = []


res_dict = dict()
kf = KFold(n_splits=folds, random_state=my_seed)

#for i in range(0, len(algo_names)):
for i in [0, 1, 2, 4, 6]:
# for i in range(2):

    res_dict[algo_names[i]] = dict()

    df_item_dist[algo_names[i]] = 0
    df_user_dist[algo_names[i]] = 0
    low_maes = []
    med_maes = []
    high_maes = []
    all_maes = []
    fold_count = 0
    print(algo_names[i])

    low_rec_gap = 0
    medium_rec_gap = 0
    high_rec_gap = 0

    low_count = 0
    med_count = 0
    high_count = 0

    for trainset, testset in kf.split(data):

        # calculate and evaluate recommendations
        # get top-n items and calculate gaps for all algorithms
        if algo_names[i] == 'Random':
            top_n = get_top_n_random(testset, n=10)
        elif algo_names[i] == 'MostPopular':
            top_n = get_top_n_mp(testset, n=10)
        else:
            algos[i].fit(trainset)
            predictions = algos[i].test(testset)
            top_n = get_top_n(predictions, n=10)

            low_mae, med_mae, high_mae, all_mae = get_mae_of_groups(predictions)
            low_maes.append(low_mae)
            med_maes.append(med_mae)
            high_maes.append(high_mae)
            all_maes.append(all_mae)

        res_dict[algo_names[i]].update(top_n)

        for uid, user_ratings in top_n.items():

            pop_count = 0
            iid_list = []
            for (iid, _) in user_ratings:
                
                df_item_dist.loc[iid, algo_names[i]] += 1
                iid_list.append(iid)
                if iid in top_item_dist.index:
                    
                    pop_count += 1
            pop_ratio = pop_count / len(iid_list)

            gap = sum(item_dist[iid_list]) / no_users / len(iid_list)
            df_user_dist.loc[uid, algo_names[i]] = pop_ratio

            if uid in low_users.index:
                low_rec_gap += gap
                low_count += 1
            elif uid in medium_users.index:
                medium_rec_gap += gap
                med_count += 1
            elif uid in high_users.index:
                high_rec_gap += gap
                high_count += 1

    low_rec_gap_list.append(low_rec_gap / low_count)
    medium_rec_gap_list.append(medium_rec_gap / med_count)
    high_rec_gap_list.append(high_rec_gap / high_count)

    if i not in [0, 1]:
        print('LowMS: ' + str(np.mean(low_maes)))
        print('MedMS: ' + str(np.mean(med_maes)))
        print('HighMS: ' + str(np.mean(high_maes)))
        print('All: ' + str(np.mean(all_maes)))

    print('\n')


for name, d in res_dict.items():
    for uid, t in d.items():
        d[uid] = [x[0] for x in t]
res_df = pd.DataFrame.from_dict(res_dict)
res_df.to_csv("res_df.csv", index=True, index_label="uid")

df_item_dist.to_csv('item_dist.csv')
df_user_dist.to_csv("user_dist.csv")


for i in range(0, len(algo_names)):
    plt.figure(figsize=[10, 10])
    plt.tick_params(labelsize=20)
    x = df_item_dist['Number_of_RATING']
    y = df_item_dist[algo_names[i]]
    # plt.gca().set_ylim(0, 300)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    print(algo_names[i])
    print(stats.spearmanr(x, y))
    plt.plot(x, y, 'o', x, line)
    plt.xlabel('Item popularity', fontsize='15')
    plt.ylabel('Recommendation frequency', fontsize='15')
    plt.xticks(fontsize='13')
    plt.yticks(fontsize='13')
    plt.title(algo_names[i])
    plt.savefig('rec_' + algo_names[i] + '.png', dpi=300, bbox_inches='tight')

