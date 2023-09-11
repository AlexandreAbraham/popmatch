def cumulate(l):
    r = []
    for i in l:
        r.append(i)
        yield r
