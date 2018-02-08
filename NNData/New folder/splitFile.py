from itertools import zip_longest

def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

n = 80

with open('11-12-1p.txt') as f:
    for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
        with open('11-12-1pos{0}'.format(i * n) + '.txt', 'w') as fout:
            fout.writelines(g)