from itertools import zip_longest


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

batchsize=32
for batch in grouper(list(range(0,40)), batchsize):
    batch = [x for x in batch if x is not None]
    print(len(batch), batch)