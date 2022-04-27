# coding=utf-8
"""
This module provides access to the MeteoFields class and instances. It is
meant to be encapsulate the storage and retrieval operations of fields names
and types used in the MeteoFrame class.
"""

# TODO: exhaustively check returning types.

# Import built-in modules:
from typing import List
# Import project scripts:
from AuxTools import *


class MeteoFields:
    """
    Class to store and retieve fields of MeteoFrame objects.
    """
    # TODO: expand MeteoFields class documentation.

    #########
    # Slots #
    #########
    __slots__ = ["__dictionary"]

    ####################
    # Class attributes #
    ####################
    type_codefield = "codefield"
    type_timefield = "timefield"
    type_lonfield = "lonfield"
    type_latfield = "latfield"
    type_meteofield = "meteofields"

    ########################
    # Instance constructor #
    ########################
    def __init__(self, fields, codefield=None, timefield=None, lonfield=None,
                 latfield=None):
        # Create fields dictionary with index fields:
        self.__dictionary = {self.type_codefield: codefield,
                             self.type_timefield: timefield,
                             self.type_lonfield: lonfield,
                             self.type_latfield: latfield}
        # Add meteorological fields to dictionary:
        meteofields = [field for field in fields
                       if field not in self.indexing]
        self.__dictionary.update({self.type_meteofield: meteofields})

    ##############
    # Properties #
    ##############
    @property
    def codefield(self):
        """
        Get the stored code index field name.

        :return:    Code field name.
        :rtype:     str or None
        """
        return self.__dictionary[self.type_codefield]

    @property
    def comparing(self):
        """
        Get the stored index fields names, sorted by comparing priority order,
        removing possible None elements from non defined fields.

        :return:    Comparing index fields names.
        :rtype:     List[str] or None
        """
        return slist([self.timefield, self.codefield, self.lonfield,
                      self.latfield])

    @property
    def comparing_exhaustive(self):
        """
        Get the stored index fields names, sorted by comparing priority order,
        without removing possible None elements from non defined fields.

        :return:    Comparing index fields names.
        :rtype:     List[str]
        """
        return [self.timefield, self.codefield, self.lonfield, self.latfield]

    @property
    def coords(self):
        """
        Get the stored coordinates index fields names (aka latfield and
        lonfield types), removing possible None elements from non defined
        fields.

        :return:    Coordinates fields names.
        :rtype:     List[str]
        """
        return slist([self.lonfield, self.latfield])

    @property
    def indexing(self):
        """
        Get the stored index fields names, removing possible None elements from
        non defined fields.

        :return:    Index fields names.
        :rtype:     List[str]
        """
        return slist([self.codefield, self.timefield, self.lonfield,
                      self.latfield])

    @property
    def latfield(self):
        """
        Get the stored latitude index field name.

        :return:    Latitude field name.
        :rtype:     str or None
        """
        return self.__dictionary[self.type_latfield]

    @property
    def lonfield(self):
        """
        Get the stored longitude index field name.

        :return:    Longitude field name.
        :rtype:     str or None
        """
        return self.__dictionary[self.type_lonfield]

    @property
    def meteofields(self):
        """
        Get the stored meteorological values field names.

        :return:    Meteorological values field names.
        :rtype:     StrListOpt
        """
        return self.__dictionary[self.type_meteofield]

    @property
    def sorting(self):
        """
        Get the stored index fields names, sorted by field sorting order.

        :return:    Sorting index fields names.
        :rtype:     List[str] or None
        """
        return slist([self.timefield, self.codefield, self.lonfield,
                      self.latfield])

    @property
    def storing(self):
        """
        Get all the stored fields names, sorted by storage order.

        :return:    All fields names.
        :rtype:     List[str] or None
        """
        return sflist([self.codefield, self.timefield, self.lonfield,
                       self.latfield, self.meteofields])

    @property
    def timefield(self):
        """
        Get the stored time index field name.

        :return:    Time field name.
        :rtype:     str or None
        """
        return self.__dictionary[self.type_timefield]

    ##################
    # Public methods #
    ##################
    def add(self, fields, fieldtypes):
        """
        Add new fields to the field dictionary.

        :param StrList fields:      New field names.
        :param StrList fieldtypes:  New field types.
        """
        fields = glist(fields)
        fieldtypes = [fieldtype.lower() for fieldtype in glist(fieldtypes)]
        for field, fieldtype in zip(fields, fieldtypes):
            self.__checkvalidtype(fieldtype)
            if fieldtype == self.type_meteofield:
                self.__dictionary[fieldtype].append(field)
            else:
                self.__dictionary[fieldtype] = field

    def delete(self, fields):
        """
        Delete old fields from the field dictionary.

        :param StrList fields:      Old field names.
        """
        fields = glist(fields)
        for field in fields:
            fieldtype = self.name2type(field)
            if fieldtype == self.type_meteofield:
                meteofields = [f for f in self.meteofields if f != field]
                self.__dictionary[fieldtype] = meteofields
            else:
                self.__dictionary[fieldtype] = None

    def name2type(self, field):
        """
        Get the field type associated to a field name.

        :param str field:   Field name.
        :return:            Field type.
        :rtype:             str
        """
        try:
            return [key for key, values in list(self.__dictionary.items())
                    if values is not None and field in glist(values)][0]
        except IndexError:
            self.__errunknownname(field)

    def rename(self, namesmap):
        """
        Rename fields on the field dictionary.

        :param Dict[str, str] namesmap: Dictionary with key-value pairs of
                                        old-new names for renamed fields.
        """
        for key, values in namesmap.items():
            if self.codefield == key:
                self.__dictionary[self.type_codefield] = values
            elif self.timefield == key:
                self.__dictionary[self.type_timefield] = values
            elif self.lonfield == key:
                self.__dictionary[self.type_lonfield] = values
            elif self.latfield == key:
                self.__dictionary[self.type_latfield] = values
            else:
                meteofields = [values if field == key else field
                               for field in self.meteofields]
                self.__dictionary[self.type_meteofield] = meteofields
        # TODO: ensure only individual str objects are inserted into index
        #  fields.

    def type2name(self, fieldtype):
        """
        Get the field name associated to a field type.

        :param str fieldtype:   Field type.
        :return:                Field name.
        :rtype:                 StrList
        """
        fieldtype = fieldtype.lower()
        self.__checkvalidtype(fieldtype)
        return self.__dictionary[fieldtype]

    def update(self, fields):
        """
        Update fields dictionary based on a new list of available fields,
        removing missing index fields and assigning the meteorological type to
        unknown fields.

        :param StrList fields:      Available field names.
        """
        fields = glist(fields)
        # Delete missing index fields:
        if self.codefield is not None and self.codefield not in fields:
            self.delete(self.codefield)
        if self.timefield is not None and self.timefield not in fields:
            self.delete(self.timefield)
        if self.lonfield is not None and self.lonfield not in fields:
            self.delete(self.lonfield)
        if self.latfield is not None and self.latfield not in fields:
            self.delete(self.latfield)
        # Assign meteorological type to provided unknown fields:
        meteofields = [field for field in fields if field not in self.indexing]
        self.__dictionary[self.type_meteofield] = meteofields

    ###################
    # Private methods #
    ###################
    def __checkvalidtype(self, fieldtype):
        """
        Check if the provided field type is valid, and raise an error if not.

        :param str fieldtype:   Checked field type.
        """
        fieldtype = fieldtype.lower()
        validtypes = [self.type_codefield, self.type_timefield,
                      self.type_lonfield, self.type_latfield,
                      self.type_meteofield]
        if fieldtype not in validtypes:
            self.__errunknowntype(fieldtype, inspect.stack(0)[1].function)

    @staticmethod
    def __errunknownname(fieldname, func=None):
        """
        Raise a ValueError due to a unknown field name.

        :param str fieldname:   Name of the unknown field type.
        :param str func:        Original calling function.
        :rtype:                 None
        """
        func = inspect.stack(0)[1].function if func is not None else func
        message = f"The field name {fieldname} used in the '{func}' " \
                  f"function is not defined."
        raise ValueError(message)
        # TODO: remove "func" argument and generate a list of the last n
        #  (3-5) callers.

    @staticmethod
    def __errunknowntype(fieldtype, func=None):
        """
        Raise a ValueError due to a unknown field type.

        :param str fieldtype:   Name of the unknown field type.
        :param str func:        Original calling function.
        :rtype:                 None
        """
        func = inspect.stack(0)[1].function if func is not None else func
        message = f"The field type {fieldtype} used in the '{func}' " \
                  f"function is not defined."
        raise ValueError(message)
        # TODO: remove "func" argument and generate a list of the last n
        #  (3-5) callers.
