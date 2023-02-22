"""Custom exceptions.
"""

class DataInvalid(Exception):
    """The data provided is structurally invalid (i.e. mismatched array
    lengths, numbers that are out of range, or unexpected data type)."""
    pass


class ImproperlyConfigured(Exception):
    """The given configuration is incomplete or not appropriate."""


class MissingData(Exception):
    """Data is not present or is in the wrong location."""
    pass


class UnexpectedResult(Exception):
    """The process has given an unexpected result that was not
    accounted for."""
    pass


class ConvergenceWarning(Exception):
    """The iterative proceedure failed to converge"""
    pass
