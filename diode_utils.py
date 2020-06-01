""" General helper functions to handle measurement data
"""


def crop_data_range_to_x(xdata, ydata, lower, upper):
    """Crop two data data vectors so that the second corresponds to the first

    Args:
        xdata (1D array/list):
        ydata (1D array/list):
        lower (float): desired lower bound of xdata
        upper (float): desired upper bound of xdata

    Raises:
        ValueError: Length of xdata and ydata must be equal
        ValueError: Lower bound needs to be >= xdata[0]
        ValueError: Upper bound needs to <= xdata[-1]
        ValueError: xdata needs to be a monotonously rising sequence

    Returns:
        tuple of 1D arrays: cropped xdata and ydata
    """
    if (len(xdata) != len(ydata)):
        raise ValueError('Length of xdata and ydata must be equal!')

    if (lower < xdata[0]):
        raise ValueError('Lower bound needs to be equal or larger than first value of xdata !')

    if (upper > xdata[-1]):
        raise ValueError('Upper bound needs to equal or smaller than last value of xdata!')

    x_previous = xdata[0]
    for i in range(1, len(xdata)):
        if (xdata[i] <= x_previous):
            raise ValueError('xdata', xdata, 'needs to be a monotonously rising sequence!')
        else:
            x_previous = xdata[i]

    for i in range(len(xdata)):
        if (xdata[i] >= lower):
            index_lower = i
            break

    for i in (range(len(xdata) -1, -1, -1)):
        if (xdata[i] <= upper):
            index_upper = i
            break

    xdata_cropped = xdata[index_lower:index_upper]
    ydata_cropped = ydata[index_lower:index_upper]

    return (xdata_cropped, ydata_cropped)

