# prepare data for analyzing
import pandas as pd
res_df = pd.read_csv('res_df.csv')
print(len(res_df))

group_labeled_data = pd.read_csv('group_labeled_data.csv')
group_labeled_data_new = group_labeled_data.drop_duplicates(subset=['USER_MD5'])
group_labeled_data_new = group_labeled_data_new.drop(columns=['MOVIE_ID', 'RATING'])
group_labeled_data_new = group_labeled_data_new.rename(columns={'USER_MD5':'uid', 'USER_CATEGORY':'label'})

res_df_new = res_df.merge(group_labeled_data_new, how='left', on='uid')
res_df_new = res_df_new.reset_index(drop = True)
print(res_df_new.head())

df_events_new = pd.read_csv('10_movieratings_with_flag.csv', sep=',', header=0)
df_events_new = df_events_new.rename(columns={'MOVIE_ID':'movie_id'})
df_events_new = df_events_new.drop(columns=['USER_MD5'])
df_events_new = df_events_new.drop_duplicates(subset=['movie_id'])
print(df_events_new.head())



import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc

k = 10 # number of topK elements returned (value should be between 10 and 400)
p = 0.25 # proportion of protected candidates in the topK elements (value should be between 0.02 and 0.98) 
alpha = 0.1 # significance level (value should be between 0.01 and 0.15)

fair = fsc.Fair(k, p, alpha)



# check fairness of different algorithms & Rerank
res = pd.DataFrame(res_df_new[['uid', 'label']])

algo_names = ['Random',
              'MostPopular',
              "UserItemAvg",
              'KNNWithMeans',
              'CoClustering']
			  
for index in range(5):
    fair_res = []
    rerank_fair_res = []

    for i in range(len(res)):
    #for i in range(1):

        id = []
        rating = []
        flag = []
        fair_data = []

        ids = res_df_new[algo_names[index]][i][1:-1].split(', ')

        for j in range(10):
            id.append(int(ids[j]))
            rating.append(df_events_new['RATING'][df_events_new[df_events_new['movie_id']==int(ids[j])].index.tolist()[0]])
            flag.append(df_events_new['nich_protected_flag'][df_events_new[df_events_new['movie_id']==int(ids[j])].index.tolist()[0]])
            fair_data.append(FairScoreDoc(id[j], rating[j], flag[j]))
    
        fair_res.append(fair.is_fair(fair_data))
        rerank_fair_res.append(fair.is_fair(fair.re_rank(fair_data)))

    res[str(algo_names[index]) + '_fair'] = fair_res
    res[str(algo_names[index]) + '_Rerank_fair'] = rerank_fair_res



# analyze fairness rate of three groups
low_fair = res[res['label'] == 'niche-focused']
medium_fair = res[res['label'] == 'diversed-focused']
high_fair = res[res['label'] == 'blockbuster-focused']



LOW_RES = []
MEDIUM_RES = []
HIGH_RES = []

new_LOW_RES = []
new_MEDIUM_RES = []
new_HIGH_RES = []

for index in range(5):
    LOW_RES.append(len(low_fair[low_fair[str(algo_names[index]) + '_fair'] == False]) / len(low_fair))
    MEDIUM_RES.append(len(medium_fair[medium_fair[str(algo_names[index]) + '_fair'] == False]) / len(medium_fair))
    HIGH_RES.append(len(high_fair[high_fair[str(algo_names[index]) + '_fair'] == False]) / len(high_fair))

    new_LOW_RES.append(len(low_fair[low_fair[str(algo_names[index]) + '_Rerank_fair'] == False]) / len(low_fair))
    new_MEDIUM_RES.append(len(medium_fair[medium_fair[str(algo_names[index]) + '_Rerank_fair'] == False]) / len(medium_fair))
    new_HIGH_RES.append(len(high_fair[high_fair[str(algo_names[index]) + '_Rerank_fair'] == False]) / len(high_fair))



# show res table
res_table = pd.DataFrame()
res_table['algo_names'] = algo_names
res_table['Low_RS'] = LOW_RES
res_table['Low_reranked'] = new_LOW_RES
res_table['MEDIUM_RS'] = MEDIUM_RES
res_table['MEDIUM_reranked'] = new_MEDIUM_RES
res_table['HIGH_RS'] = HIGH_RES
res_table['HIGH_reranked'] = new_HIGH_RES
res_table



# plot unfairness ratio (RS)
import matplotlib.pyplot as plt
import numpy as np

barWidth = 0.1

plt.figure()

bars1 = [LOW_RES[0], MEDIUM_RES[0], HIGH_RES[0]]
bars2 = [LOW_RES[1], MEDIUM_RES[1], HIGH_RES[1]]
bars3 = [LOW_RES[2], MEDIUM_RES[2], HIGH_RES[2]]
bars4 = [LOW_RES[3], MEDIUM_RES[3], HIGH_RES[3]]
bars5 = [LOW_RES[4], MEDIUM_RES[4], HIGH_RES[4]]

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

plt.bar(r1, bars1, width=barWidth, label='Random')
plt.bar(r2, bars2, width=barWidth, label='MostPopular')
plt.bar(r3, bars3, width=barWidth, label='UserItemAvg')
plt.bar(r4, bars4, width=barWidth, label='KNNWithMeans')
plt.bar(r5, bars5, width=barWidth, label='CoClustering')

plt.xlabel('User group', fontsize='14')
plt.ylabel('% Unfair Recommendation Ratio', fontsize='14')
plt.xticks([r + barWidth + 0.15 for r in range(len(bars1))], ['Low', 'Medium', 'High'], fontsize='13')
plt.yticks(fontsize='13')
plt.title("The Unfair Ratio (RS)", fontsize='15')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., framealpha=1, fontsize='13')
plt.savefig('unfair_ratio_1_RS.png', dpi=300, bbox_inches='tight')



# plot unfairness ratio (reranked)
barWidth = 0.1
plt.figure()

bars1 = [new_LOW_RES[0], new_MEDIUM_RES[0], new_HIGH_RES[0]]
bars2 = [new_LOW_RES[1], new_MEDIUM_RES[1], new_HIGH_RES[1]]
bars3 = [new_LOW_RES[2], new_MEDIUM_RES[2], new_HIGH_RES[2]]
bars4 = [new_LOW_RES[3], new_MEDIUM_RES[3], new_HIGH_RES[3]]
bars5 = [new_LOW_RES[4], new_MEDIUM_RES[4], new_HIGH_RES[4]]

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

plt.bar(r1, bars1, width=barWidth, label='Random')
plt.bar(r2, bars2, width=barWidth, label='MostPopular')
plt.bar(r3, bars3, width=barWidth, label='UserItemAvg')
plt.bar(r4, bars4, width=barWidth, label='KNNWithMeans')
plt.bar(r5, bars5, width=barWidth, label='CoClustering')

plt.xlabel('User group', fontsize='14')
plt.ylabel('% Unfair Recommendation Ratio', fontsize='14')
plt.xticks([r + barWidth + 0.15 for r in range(len(bars1))], ['Low', 'Medium', 'High'], fontsize='13')
plt.yticks(fontsize='13')
plt.title("The Unfair Ratio (reranked)", fontsize='15')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., framealpha=1, fontsize='13')
plt.savefig('unfair_rate_2_reranked', dpi=300, bbox_inches='tight')