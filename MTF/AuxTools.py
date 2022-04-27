# coding=utf-8
"""
This module provides access to variables and functions meant to be used
across different modules and projects.
"""

# Import built-in modules:
import inspect
import math
from typing import Callable, List, Optional, Tuple, TypeVar


IntListOpt = TypeVar("IntListOpt", Optional[int], Optional[List[int]])
FunList = TypeVar("FunList", Callable, List[Callable])
ObjList = TypeVar("ObjList", Optional[object], Optional[List[object]])
ObjListOpt = TypeVar("ObjListOpt", Optional[object], Optional[List[object]])
StrList = TypeVar("StrList", str, List[str])
StrListOpt = TypeVar("StrListOpt", Optional[str], Optional[List[str]])


def errbadfields(args, fields):
    """
    Raise a ValueError due to invalid field names provided as arguments.

    :param StrList args:    Names of the troublesome arguments.
    :param StrList fields:  Names of the invalid fields.
    :rtype:                 None
    """
    func = inspect.stack(0)[1].function
    if len(glist(args)) > 1:
        args = ", ".join([f"'{a}'" for a in args])
        fields = ", ".join([f"'{f}'" for f in fields])
        message = f"The chosen fields {fields} are not valid options for the" \
                  f" {args} arguments of the '{func}' function. "
    else:
        message = f"The chosen field '{fields}' is not a valid option for " \
                  f"the '{args}' argument of the '{func}' function."
    raise ValueError(message)


def errmissargs(*args):
    """
    Raise a ValueError due to non-provided arguments.

    :param StrList args:    Names of the non-provided arguments.
    :rtype:                 None
    """
    func = inspect.stack(0)[1].function
    if len(glist(args)) > 1:
        args = ", ".join([f"'{a}'" for a in args])
        message = f"The required arguments {args} of the '{func}' function " \
                  f"were not provided."
    else:
        message = f"The required argument '{args}' of the '{func}' function " \
                  f"was not provided."
    raise ValueError(message)


def errnofields(fields):
    """
    Raise a ValueError due to nonexistent field names.

    :param StrList fields:  Names of the nonexistent fields.
    :rtype:                 None
    """
    func = inspect.stack(0)[1].function
    if len(glist(fields)) > 1:
        fields = ", ".join([f"'{a}'" for a in fields])
        message = f"The field names {fields} provided to the '{func}' " \
                  f"function do not exist."
    else:
        message = f"The field name '{fields}' provided to the '{func}' " \
                  f"function does not exist."
    raise ValueError(message)


def errnofieldtypes(fieldtypes):
    """
    Raise a ValueError due to nondefined field types.

    :param StrList fieldtypes:  Names of the nondefined field types.
    :rtype:                     None
    """
    func = inspect.stack(0)[1].function
    if len(glist(fieldtypes)) > 1:
        fieldtypes = ", ".join([f"'{a}'" for a in fieldtypes])
        message = f"The field types {fieldtypes} requested by the '{func}' " \
                  f"function are not defined."
    else:
        message = f"The field type {fieldtypes} requested by the '{func}' " \
                  f"function is not defined."
    raise ValueError(message)


def errvalues(args, values):
    """
    Raise a ValueError due to invalid values given to function arguments.

    :param StrList args:    List of names of the troublesome arguments.
    :param StrList values:  List of faulty values passed.
    :rtype:                 None
    """
    func = inspect.stack(0)[1].function
    if len(glist(args)) > 1:
        args = ", ".join([f"'{a}'" for a in args])
        values = ", ".join([f"'{f}'" for f in values])
        message = f"The provided values {values} are not valid options for " \
                  f"the {args} arguments of the '{func}' function. "
    else:
        message = f"The provided values '{values}' are not valid options for " \
                  f"the '{args}' arguments of the '{func}' function. "
    raise ValueError(message)


def erryesfields(fields):
    """
    Raise a ValueError due to already existent field names.

    :param StrList fields:  Names of the already existent fields.
    :rtype:                 None
    """
    func = inspect.stack(0)[1].function
    if len(glist(fields)) > 1:
        fields = ", ".join([f"'{a}'" for a in fields])
        message = f"The field names {fields} provided to the '{func}' " \
                  f"function already exist."
    else:
        message = f"The field name '{fields}' provided to the '{func}' " \
                  f"function already exists."
    raise ValueError(message)


def flist(ruggedlist):
    """
    Extract elements from sublists and append them to the main list.

    :param list ruggedlist: List possibly containing list elements.
    :return:                Flat list with sublists elements collapsed into
                            the main list.
    :rtype:                 list
    """
    smoothedlist = [glist(elem) for elem in ruggedlist]
    flattenedlist = [elem for sublist in smoothedlist for elem in sublist]
    return flattenedlist


def glist(suspectedlist):
    """
    Ensure that the provided object is really a list of objects (even with
    only one element), and not an individual non-list object. Tuples are
    converted to lists, then checked.

    :param object suspectedlist:    Input object which may be a list of
                                    objects or a non-list object.
    :return:                        Guaranteed list of objects.
    :rtype:                         list
    """
    if isinstance(suspectedlist, tuple):
        suspectedlist = list(suspectedlist)
    guaranteedlist = suspectedlist if isinstance(suspectedlist, list) \
        else [suspectedlist, ]
    return guaranteedlist


def icoords(i, ncols, nrows, byrows=True, i1base=False, rc1base=False):
    """
    Compute the row and column coordinates of the i-th element of a array-like
    structure.

    :param int i:           Array i-th element whose coordinates are to be
                            found.
    :param int ncols:       Total number of columns.
    :param int nrows:       Total number of rows.
    :param bool byrows:     If True, array is filled by rows.
    :param bool i1base:     If True, the array elements numeration is expected
                            to be 1-index based, instead of 0-index based.
    :param bool rc1base:    If True, the row and column elements numerations
                            are expected to be 1-index based, instead of
                            0-index based.
    :return:                Row and column number of the passed item.
    :rtype:                 Tuple[int, int]
    """
    # Index based coefficients:
    a = 1 if i1base else 0
    b = 1 if rc1base else 0
    c = (1 + (ncols if byrows else nrows)) if rc1base else 0
    if byrows:
        row = math.floor((i - a) / ncols) + b
        col = math.ceil(((i - a) / ncols - row) * ncols) + c
    else:
        col = math.floor((i - a) / nrows) + b
        row = math.ceil(((i - a) / nrows - col) * nrows) + c
    return row, col


def multisearchin(values, checklists):
    """
    Check which passed values are present inside every sublist elements of a
    given list of lists, and return them.

    :param ObjList values:              Values to be sought.
    :param List[ObjList] checklists:    List of lists where values are to be
                                        sought.
    :return:                            Values present inside every sublist.
    :rtype:                             ObjList
    """
    presentlist = list()
    for value in glist(values):
        addvalue = True
        for check in checklists:
            if value not in check:
                addvalue = False
                break
        if addvalue:
            presentlist.append(value)
    return presentlist


def sflist(disasterlist):
    """
    Remove None elements and extract elements from sublists.

    :param list disasterlist:   List possibly containing list and/or None
                                elements.
    :return:                    List with sublists elements collapsed into
                                the main list and without None elements.
    :rtype:                     list
    """
    fixedlist = slist(flist(disasterlist))
    return fixedlist


def slist(hollowlist):
    """
    Remove None elements from list.

    :param list hollowlist: List possibly containing None elements.
    :return:                List without None elements.
    :rtype:                 list
    """
    solidlist = [element for element in hollowlist if element is not None]
    return solidlist


def ulist(duplicatedlist):
    """
    Remove duplicated elements from a list, keeping original order even if its
    elements are not hashable.

    :param list duplicatedlist: List possibly containing duplicated elements.
    :return:                    List without duplicated elements.
    :rtype:                     list
    """
    try:
        uniquelist = list(dict.fromkeys(duplicatedlist))
    except TypeError:
        uniquelist = list()
        for new in duplicatedlist:
            addnew = True
            for known in uniquelist:
                if known == new:
                    addnew = False
                    break
            if addnew:
                uniquelist.append(new)
    return uniquelist
