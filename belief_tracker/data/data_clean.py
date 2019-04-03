

lines = []
with open('/path/bt'+ '/data_labeled/data_filled') as d:
    for line in d:
        # ','->' '
        line = line.replace(',', ' ')
        line = line.replace('.', '')
        line = line.replace('。', '')
        line = line.replace('?', '')
        try:
            index_in = line.index('in')
            if line[index_in + 2] != ' ' and line[index_in - 1] == ' ':
                line = line[:index_in] + 'in ' + line[index_in + 2:]
                print(line)
        except Exception as e:
            pass

        line = ' '.join(line.split())
        lines.append(line)

with open('/path/bt'+'/data_labeled/data_cleaned', 'w') as f:
    for a in lines:
        print(a)
        f.write(a)
        f.write('\n')
