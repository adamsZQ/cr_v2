
'''
    array transfer to chunks
    import:array, number of elements
    output:chunks
'''
def chunks(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]
