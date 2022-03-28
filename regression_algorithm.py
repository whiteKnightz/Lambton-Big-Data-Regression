import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('.\data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('.\data\games_of_all_time.csv')

print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.info())
df.fillna('NULL', inplace = True)
print(df.isnull().sum())
df.drop(['description','url','type'],axis = 1, inplace = True)
print(df.head())
# df[['genre','platform']]
print(df[['genre','platform']])


def find_minmax(x):
    min_index = df[x].idxmin()
    high_index = df[x].idxmax()
    high = pd.DataFrame(df.loc[high_index, :])
    low = pd.DataFrame(df.loc[min_index, :])

    print('Game with the Highest ' + x + ':', df['game_name'][high_index])
    print('Game with the Lowest ' + x + ':', df['game_name'][min_index])
    return pd.concat([high, low], axis=1)


find_minmax('meta_score')
find_minmax('user_score')

print(df['developer'].value_counts())
print(df['platform'].value_counts())




info = pd.DataFrame(df['meta_score'].sort_values(ascending=False))
info['game_name'] = df['game_name']
data = list(map(str,(info['game_name'])))
x = list(data[:10])
y = list(info['meta_score'][:10])

ax = sns.pointplot(x=y,y=x)

sns.set(rc={'figure.figsize':(20,10)})
ax.set_title('Top 10 Games per MetaScore', fontsize = 30)
ax.set_xlabel('Meta Score',fontsize = 20)
sns.set_style('darkgrid')





info1 = pd.DataFrame(df['user_score'].sort_values(ascending=False))
info1['game_name'] = df['game_name']
data1 = list(map(str,(info['game_name'])))
x = list(data1[:10])
y = list(info1['user_score'][:10])

ax = sns.pointplot(x=y,y=x)

sns.set(rc={'figure.figsize':(20,10)})
ax.set_title('Top 10 Games per UserScore', fontsize = 30)
ax.set_xlabel('Meta Score',fontsize = 20)
sns.set_style('darkgrid')





info = pd.DataFrame(df['meta_score'].sort_values(ascending=False))
info['developer'] = df['developer']
data = list(map(str,(info['developer'])))
x = list(data[:10])
y = list(info['meta_score'][:10])

ax = sns.pointplot(x=y,y=x)

sns.set(rc={'figure.figsize':(20,10)})
ax.set_title('Top 10 Developers per MetaScore', fontsize = 30)
ax.set_xlabel('Meta Score',fontsize = 20)
sns.set_style('darkgrid')





info = pd.DataFrame(df['meta_score'].sort_values(ascending=False))
info['platform'] = df['platform']
data = list(map(str,(info['platform'])))
x = list(data[:10])
y = list(info['meta_score'][:10])

ax = sns.lineplot(x=y,y=x)

sns.set(rc={'figure.figsize':(20,10)})
ax.set_title('Top 10 platforms per MetaScore', fontsize = 30)
ax.set_xlabel('Meta Score',fontsize = 20)
sns.set_style('darkgrid')






# Make a function that will split the string
# and return a count of each genre
def count_genre(x):
    # Concatenate all the rows of the genre
    data_plot = df[x].str.cat(sep='|')
    data = pd.Series(data_plot.split('|'))
    info = data.value_counts(ascending = False)
    return info

total_genre = count_genre('genre')[:10]
total_genre.plot(kind='barh',figsize=(20,10),
                fontsize=20, colormap = 'tab20c')
plt.title('Genre with Highest amount', fontsize = 35)
plt.xlabel('Number of Games', fontsize = 20)
plt.ylabel('Genres', fontsize = 20)
sns.set_style('whitegrid')

# will make a pie chart for genre's
i = 0
genre_count = []
for genre in total_genre.index:
    genre_count.append([genre, total_genre[i]])
    i = i + 1

plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(20, 20))
genre_count.sort(key=lambda x: x[1], reverse=True)
labels, sizes = zip(*genre_count)
labels_selected = [n if v > sum(sizes) * 0.01 else '' for n, v in genre_count]
ax.pie(sizes, labels=labels_selected, autopct=lambda x: '{:2.0f}%'.format(x) if x > 1 else '', shadow=False,
       startangle=0)
ax.axis('equal')
plt.tight_layout()
