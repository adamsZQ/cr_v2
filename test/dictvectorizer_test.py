from sklearn.feature_extraction import DictVectorizer

if __name__ == '__main__':
    v = DictVectorizer()
    a = []
    for i in range(100):
        a.append({'fff': str(i)})

    X = v.fit_transform(a)
    print(X.toarray())