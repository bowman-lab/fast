class DataInvalid(Exception):
    """The data provided is structurally invalid (i.e. mismatched array
    lengths, numbers that are out of range, or unexpected data type)."""
    pass

class MissingData(Exception):
    """Data is not present or is in the wrong location. """
    pass

