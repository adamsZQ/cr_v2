import json
import os
import sqlite3

from sklearn.model_selection import train_test_split

db_file = os.path.expanduser('~/cr_repo/movie_sql')


def insert(id, critic_rating, audience_rating, director, country, genres):
    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        c.execute("INSERT INTO movies (id, critic_rating, audience_rating, director, country, genres) VALUES (?,?,?,?,?,?)", (id, critic_rating, audience_rating, director, country, genres,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()


def insert_genre(movie,genre):
    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        c.execute("INSERT INTO movie_genres (movie,genre) VALUES (?,?)", (movie,genre,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()


def select_genres(movie):
    sql = 'SELECT genre FROM movie_genres WHERE movie = ?'
    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        data = c.execute(sql, (movie,))
        data = [x[0] for x in data]
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()


def select_all_movie_genres():
    sql = 'SELECT * FROM movie_genres'
    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        data = c.execute(sql)
        data = [x for x in data]
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()


def select_by_attributes(attributes):
    sql = 'SELECT id FROM movies WHERE '
    for key in list(attributes.keys()):
        sql = sql + key + ' = ? AND '
    sql = sql[:-4]

    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        data = c.execute(sql, list(attributes.values()))
        data = [x[0] for x in data]
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()


def select_all():
    sql = 'SELECT * FROM movies'
    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        data = c.execute(sql)
        data = [x for x in data]
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()

if __name__ == '__main__':
    genres = [3,5]
    attributes = {'director':1227}
    a = select_by_attributes(attributes)
    print(a)

    for id in a:
        b = select_genres(id)
        print(id,':',b)
# if __name__ == '__main__':
#     data_list = []
#     with open('/path/mv/movie_rating', 'r') as f:
#         for line in f:
#             line = json.loads(line)
#             data_list.append(line)
#     trainset, testset, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.2, random_state=1)
#     data = testset
#
#     for d in data:
#         print(d)
#         data = select_by_attributes(d)
#     # print(data)
