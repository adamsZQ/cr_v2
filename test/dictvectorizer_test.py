from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    v = CountVectorizer()
    a = []
    for i in range(100):
        a.append('dt_'+str(i))

    X = v.fit_transform(a)
    b = X.toarray()
    print(b)
    t = v.transform(['dt_2'])
    # y = v.transform({'fff':'2'})
    # z = v.transform({'fff':None})
    # aaa = t + y
    print(t.toarray())
    # print(y.toarray())
    # print(z.toarray())
