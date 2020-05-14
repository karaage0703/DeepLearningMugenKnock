_chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっー１２３４５６７８９０！？、。"
chars = [c for c in _chars]

def data_load():
    fname = 'sandwitchman.txt'

    xs = []
    
    with open(fname, 'r') as f:
        for l in f.readlines():
            l = l.strip()
            for c in l:
                x = [0 for _ in range(len(chars))]
                x[chars.index(c)] = 1
                xs.append(x)

    return xs

print(data_load())
