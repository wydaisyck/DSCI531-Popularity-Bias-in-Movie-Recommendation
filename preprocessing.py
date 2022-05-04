import pandas as pd
movie = pd.read_csv('movies.csv', sep=',', encoding='utf-8')
print(movie.columns.values)

movies = movie[['MOVIE_ID','NAME','GENRES','REGIONS','STORYLINE', 'DOUBAN_SCORE','DOUBAN_VOTES']]

users = pd.read_csv('users.csv', sep=',', encoding='utf-8')
print(users.columns.values)

rating = pd.read_csv('ratings.csv', sep=',', encoding='utf-8')
print(rating.columns.values)

ratings = pd.merge(movies,rating,on='MOVIE_ID')
ratings = pd.merge(ratings,users,on='USER_MD5')



data = ratings.drop(ratings[(ratings['DOUBAN_SCORE'] == 0) | (ratings['DOUBAN_VOTES'] == 0)].index)
user_filter = data.groupby('USER_MD5').count() >= 10
uid = user_filter[user_filter.MOVIE_ID].index
data = data[data.USER_MD5.isin(uid)]
print(len(data))



extract = data.groupby('USER_MD5').agg({'DOUBAN_SCORE':'mean','DOUBAN_VOTES':'mean','RATING':'mean'})

# normalize "DOUBAN_VOTES"
extract['DOUBAN_VOTES'] = (extract['DOUBAN_VOTES'] - extract['DOUBAN_VOTES'].min()) / (extract['DOUBAN_VOTES'].max() - extract['DOUBAN_VOTES'].min())
print(extract['DOUBAN_VOTES'].describe())



# divide users to three groups
listBins = [0, 0.0037, 0.026, 1]
listLabels = ['niche-focused','diversed-focused','blockbuster-focused']
extract['USER_CATEGORY'] = pd.cut(extract['DOUBAN_VOTES'], bins=listBins, labels=listLabels)
extract['USER_MD5'] = extract.index
extract = extract.reset_index(drop=True)



new_data = pd.merge(data, extract[['USER_CATEGORY','USER_MD5']], on='USER_MD5')
use_data = new_data[['USER_MD5', 'MOVIE_ID', 'RATING']]
use_data.to_csv('10_movieratings.csv', index=False)

group_labeled_data = new_data[['USER_MD5', 'USER_CATEGORY', 'MOVIE_ID', 'RATING']]
group_labeled_data.to_csv('group_labeled_data.csv', index=False)



niche_focused_users = {}
blockbuster_focused_users = {}
diversed_focused_users = {}
for i in range(len(extract['USER_MD5'])):
    if extract['USER_CATEGORY'][i] == 'niche-focused':
        niche_focused_users[extract['USER_MD5'][i]] = extract['DOUBAN_VOTES'][i]
    if extract['USER_CATEGORY'][i] == 'blockbuster-focused':
        blockbuster_focused_users[extract['USER_MD5'][i]] = extract['DOUBAN_VOTES'][i]
    if extract['USER_CATEGORY'][i] == 'diversed-focused':
        diversed_focused_users[extract['USER_MD5'][i]] = extract['DOUBAN_VOTES'][i]

print(len(niche_focused_users))
print(len(blockbuster_focused_users))
print(len(diversed_focused_users))



# save data of three groups of users
file1 = open('10_niche_focused_users.txt', 'w') 
for k,v in niche_focused_users.items():
    file1.write(str(k) + ',' + str(v)+ '\n')
file1.close()

file2 = open('10_blockbuster_focused_users.txt', 'w') 
for k,v in blockbuster_focused_users.items():
    file2.write(str(k) + ',' + str(v)+ '\n')
file2.close()

file3 = open('10_diversed_focused_users.txt', 'w') 
for k,v in diversed_focused_users.items():
    file3.write(str(k) + ',' + str(v)+ '\n')
file3.close()



# prepare data for check fairness
nich_protected_flag = []

for i in range(len(new_data)):
    if new_data['USER_CATEGORY'][i] == 'niche-focused':
        nich_protected_flag.append(True)
    else:
        nich_protected_flag.append(False)

new_data['nich_protected_flag'] = nich_protected_flag
fairness_use_data = new_data[['USER_MD5', 'MOVIE_ID', 'RATING', 'nich_protected_flag']]
fairness_use_data.to_csv('10_movieratings_with_flag.csv',index=False)