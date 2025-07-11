import itertools as iter

def keep_repeats_and_group(xs: list) -> list:
    # remove non-repeats
    repeats = []
    prevPrev = None
    prev = None
    for x in xs:
        # only added to list if repeated twice
        # cause one repeat might happen by sheer coincidence
        if prev == x and prevPrev == x:
            repeats.append(x)
        prevPrev = prev
        prev = x
 
    # map each group to the value of the group
    return list(map(lambda x: x[0], iter.groupby(repeats)))

def count_increase(xs) -> int:
    count = 0
    for i in range(len(xs)):
        # stop when end or elements don't increase anymore
        if i + 1 >= len(xs) or not xs[i] < xs[i + 1]:
            break
        count += 1
    return count
