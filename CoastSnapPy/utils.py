def divide_chunks(l, n):
    """
    Splits a list into chunks of length n. Used to process the images in chunks.
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]
