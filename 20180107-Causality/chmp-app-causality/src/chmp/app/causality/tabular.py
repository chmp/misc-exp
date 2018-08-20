def prob(counts, variables=None, condition=(), transform=False):
    """Calculate (conditional) probabilites from counts.

    The levels of the result series, will allways be sorted to ensure proper behavior
    under multiplication and addition.

    :param counts:
        a series of counts with the variables as index. Typically,
        it is the result of an operation similar to
        ``df.groupby(variables).size()``.
    :param variables:
        the variables for which to calculate the probability. If
        not given all variables are used.
    :param conditon:
        the variables to condition on.
    :param transform:
        if ``True`` return the result with the original index, similar
        to how ``.groupby().transform()`` operates.
    """
    condition = list(condition)

    if variables is None:
        variables = [*counts.index.names]

    else:
        variables = list(variables)

    if transform:
        complement = [name for name in counts.index.names if name not in {*variables}]

    else:
        complement = condition

    grouped = counts.groupby(level=condition + variables)

    counts = grouped.agg("sum") if not transform else grouped.transform("sum")
    if not complement:
        result = counts / counts.sum()

    else:
        result = counts / counts.groupby(level=complement).transform("sum")

    if len(result.index.names) > 1:
        return result.reorder_levels(sorted(result.index.names))

    else:
        return result
