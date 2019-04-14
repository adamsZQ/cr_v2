
'''
    array transfer to chunks
    import:array, number of elements
    output:chunks
'''
def chunks(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]


# zip (five sentences, user, movie)
def zip_data(sentences_prepared, user_list, movie_list):
    i = 1
    five_sentences = []
    data_list = []
    for sentence, user, movie in zip(sentences_prepared, user_list, movie_list):
        five_sentences.append(sentence)
        if i % 5 == 0:
            data = {'five_sentences': five_sentences, 'user': user, 'movie': movie}
            data_list.append(data)
            five_sentences = []
        i = i + 1

    return data_list