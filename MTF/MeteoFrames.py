# coding=utf-8
"""
This module provides access to the MeteoFrame class and instances. It is
meant to be an specialisation of pandas.DataFrame class, designed to safely
operate with well-identified meteorological tabular data. Well-identified
meteorological data requires each value having assigned an unique index
linked to a specific geographic location (aka a code), a datetime value,
and geographic coordinates (longitude and latitude) if possible.

The main feature of a MeteoFrame object is its internal dataframe,
where the meteorological data is stored as a regular pandas.DataFrame object.
pandas.DataFrame labeled axes, namely index/rows and columns, are referred
within this module as rows and fields. Within a MeteoFrame object, different
types of fields can be found, grouped into two main categories: index fields,
which act as identifiers to allow proper storage and access to meteorological
information, and value fields, which contain the actual meteorological data.
There are different subcategories, which are:

- Code: Index field storing an unique identifier for a given
  longitude-latitude pair. They are usually referred to as "codefield".
- Time: Index field storing a valid numpy.datetime64 datetime. It is usually
  referred to as "timefield".
- Longitude: Index field storing a longitude coordinate. It is usually
  referred to as "lonfield".
- Latitude: Index field storing a latitude coordiate. It is usually referred
  to as "latfield".
- Meteorological: Value field storing a meteorological field value. It is
  usually referred to as "meteofields".

Although is actually possible to directly change the index and meteorological
fields names stored inside the corresponding attribute, this is strongly not
recommended as the corresponding dataframe columns labels are nor updated
this way, and following operations where those fields are involved can raise
unexpected errors. Instead, the provided renaming method should be always used.

Standalone functions, class methods and auxiliary classes are provided to
generate MeteoFrame objects from standard pandas.DataFrames or from data
files, and also to safely manipulate them. Safely manipulate means a
particular meteorological measurement or result at a given location and
at a given datetime should never be mingled with meteorological data from a
different location and datetime. The internal dataframe object should not be
manipulated directly, as it would potentially result in a violation of its
structural rules, which are:

- Missing values are always indicated by the numpy.nan sentinel value. This
  value is stored inside the class attribute "navalue".
- Fields are always sorted following the order established by the fields sort
  function. General rule is first index fields (in a predetermined order),
  next meteorological fields (in the same order they were added to the
  meteoframe).
- Rows are always sorted following the order established by the fieldssort
  function. General rule is rows are sorted by all the available index fields
  (with a predetermined order).
- Rows index are always a filled and sorted interval, from 0 to the internal
  dataframe's number of rows minus one.

This rules are to be enforced when a new meteoframe is created, and after
each operation where any row or field is gained, any row is lost, or any
field is renamed.
"""

# Import built-in modules:
from copy import deepcopy
import os
import types
from typing import Dict, List, Tuple
# Import external packages:
import numpy
import pandas
import tabulate
# Import project scripts:
from AuxTools import *
from MeteoFields import MeteoFields

# TODO: insert hyperlinks in module and objects documentation.


class MeteoFrame:
    """
    Class to generate and manipulate meteorological tabular data in a safely
    way.

    Almost all public available instance methods (all that do not have a
    declared return value and do not export or print data) are designed to
    operate changes over the calling meteoframe instance, and thus are meant
    to be used without assignating a new name to the object. As a rule of
    thumb, you should not do an assignment of a new name to an already created
    meteoframe instance, unless you are actually looking for a copy or a
    derived, independent object (like a aggregated meteoframe). The
    expressions "on self" and "to self" found on the following methods
    documentation refer to changes done over the calling object.

    Attributes
        - __fields: MeteoFields object with fields names.
        - __dataframe: Pandas.DataFrame object with meteorological data.
        - __shortname: Short name for referring to a meteoframe.
        - __navalue: Sentinel value for missing data.

    Public methods
        - __init__: Meteoframe instances constructor.
        - __add__: Implement arithmetic addition operation between meteoframes.
        - __getitem__: Select and return subset data from self's dataframe.
        - __repr__: Generate a string with key information about self.
        - __sub__: Implement arithmetic subtraction operation between
          meteoframes.
        - __str__: Generate a string with table-like visual representation of
          self's dataframe.
        - add_field: Add a new field to self.
        - add_rows: Append new rows to self.
        - aggregatemeteo: Aggregate self's meteorological fields by day on a
          new meteoframe.
        - apply_scalar: Apply a scalar-returning function on self's fields.
        - apply_series: Apply a series-returning function on self's fields.
        - autocodefield: Automatically generate a code field on self using
          unique pairs of coordinates fields.
        - coordsmapping: Map values to a new or existing field on self by means
          of closest distance between coordinates.
        - copy: Create a new instance with a deep copy of self's attributes.
        - csv: Export self's dataframe as a comma-separated values (csv) file.
        - drop_duplicaterows: Remove duplicated rows on self.
        - drop_fields: Drop fields selected by field name on self.
        - drop_rows: Drop rows selected by row number or by filtering
          conditions on self.
        - enforceframe: Modify self's dataframe field order, row order and
          values to enforce basic structural rules on self.
        - griupbyindex: Group self's dataframe using an available index field.
        - info: Print the returning string from the __repr__ method.
        - merge: Merge self with another meteoframe and replace self.
        - print: Print the returning string from the __str__ method.
        - rename_fields: Change selected fields names on self.
        - replace_self: Fully replace self with a new meteoframe.
        - unique_rows: Select and return unique rows from self's dataframe.
    """
    # TODO: define and use longname attribute as default value when defining
    #  plot titles.

    #########
    # Slots #
    #########
    __slots__ = ["__dataframe", "__fields", "__shortname", "__navalue"]

    ########################
    # Instance constructor #
    ########################
    def __init__(self, sources, sourcemode, codefield=None, autocode=False,
                 timefield=None, lonfield=None, latfield=None,
                 navalue=None, datetimeformat="%Y-%m-%d  %H:%M:%S",
                 units_temp="degC", units_humi="%", units_pres="Pa",
                 units_radi="W/m2", units_wisp="m/s", units_widi="deg",
                 units_prec="mm", units_heat="kWh", shortname="default"):
        """
        Meteoframe instances constructor.

        :param StrList sources:             Source objects to be used for
                                            building the meteoframe.
        :param str sourcemode:              Source objects processing mode.
                                            Current accepted modes are "csv" for
                                            reading .csv files using "sources"
                                            as fullpaths, and "dataframe", for
                                            directly appending "sources"
                                            objects as the internal dataframe.
        :param Optional[str] codefield:     Name of the code index field.
        :param bool autocode:               Attempt to automatically generate a
                                            code field using unique pairs of
                                            coordinates fields (aka lonfield
                                            and latfield) as references. Ignored
                                            if "codefield" argument is provided.
                                            If no coordinates fields are
                                            provided, an error is raised.
        :param Optional[str] timefield:     Name of the time index field.
        :param Optional[str] lonfield:      Name of the longitude index field.
        :param Optional[str] latfield:      Name of the latitude index field.
        :param object navalue:              Custom missing data sentinel value
                                            used on the input data.
        :param datetimeformat:              Strftime used to parse the time
                                            index field. Passed to
                                            pandas.to_datetime.
        :type datetimeformat:               Optional[str]
        :param Optional[str] units_temp:    Units of the provided temperature
                                            fields values.
        :param Optional[str] units_humi:    Units of the provided humidity
                                            fields values.
        :param Optional[str] units_pres:    Units of the provided pressure
                                            fields values.
        :param Optional[str] units_radi:    Units of the provided radiation
                                            fields values.
        :param Optional[str] units_wisp:    Units of the provided wind speed
                                            fields values.
        :param Optional[str] units_widi:    Units of the provided wind direction
                                            fields values.
        :param Optional[str] units_prec:    Units of the provided precipitation
                                            fields values.
        :param Optional[str] units_heat:    Units of the provided heating
                                            demands fields values.
        :param str shortname:               Short name for referring to the
                                            meteoframe.
        """
        # Read original files/objects and store dataframe:
        sources = glist(sources)
        self.__dataframe = pandas.DataFrame()
        for s in sources:
            if sourcemode.lower().replace(".", "") == "csv":
                self.__dataframe = self.dataframe.append(pandas.read_csv(s))
            elif sourcemode.lower() == "dataframe":
                self.__dataframe = self.dataframe.append(s)
            else:
                errvalues("sourcemode", sourcemode)
        # Store other attributes:
        self.__fields = MeteoFields(fields=list(self.dataframe),
                                    codefield=codefield, timefield=timefield,
                                    lonfield=lonfield, latfield=latfield)
        self.__shortname = shortname
        self.__navalue = numpy.nan
        # Autogenerate code field if not provided:
        if codefield is None and autocode:
            if lonfield is None or latfield is None:
                errmissargs("lonfield", "latfield")
            else:
                self.autocodefield(lonfield, latfield)
        # Enforce dataframe rules:
        self.enforceframe(navalue=navalue)
        # Convert time field to datetime format:
        if timefield is not None:
            if self.dataframe[timefield].dtype is not numpy.datetime64:
                self.dataframe[timefield] = \
                    pandas.to_datetime(self.dataframe[timefield],
                                       format=datetimeformat)
        # Convert meteofields units:
        inputunits = {
            "temperature": units_temp,
            "humidity": units_humi,
            "pressure": units_pres,
            "radiation": units_radi,
            "wind speed": units_wisp,
            "wind direction": units_widi,
            "precipitation": units_prec,
            "heating": units_heat}
        for i, field in enumerate(self.fields.meteofields):
            variable = MeteoVariables.alias2var(field)
            if variable is not None:
                inpunits = inputunits.get(variable)
                outunits = MeteoVariables.defunits(variable)
                self.dataframe[field] = MeteoVariables.convert(
                    self.dataframe[field], variable, inpunits, outunits)

    ##############
    # Properties #
    ##############
    @property
    def dataframe(self):
        """
        Get the dataframe object attribute.

        :return:    Internal dataframe.
        :rtype:     pandas.DataFrame
        """
        return self.__dataframe

    @property
    def fields(self):
        """
        Get the meteofields object attribute.

        :return:    Internal meteofields.
        :rtype:     MeteoFields
        """
        return self.__fields

    @property
    def shortname(self):
        """
        Get the short name string attribute.

        :return:    Short name.
        :rtype:     str
        """
        return self.__shortname

    @shortname.setter
    def shortname(self, name):
        """
        Set the short name string attribute.

        :param str name:    Short name.
        """
        self.__shortname = name

    @property
    def navalue(self):
        """
        Get the missing data sentinel value attribute.

        :return:    Missing data sentinel value.
        :rtype:     object
        """
        return self.__navalue

    ##################
    # Public methods #
    ##################
    def __add__(self, other):
        """
        Implement arithmetic addition operation between meteoframes.

        Operation is done element-wise for all common meteorological fields,
        using index fields to ensure compatibility.

        :param MeteoFrame other:    Meteoframe to be added with self.
        :return:                    New meteoframe whose meteorological
                                    fields are the addition of the input
                                    meteoframes common meteorological fields.
        :rtype:                     MeteoFrame
        """
        # Merge meteoframes and drop non-common meteorological fields:
        suffixes = ("_x", "_y")
        mf = merge(self, other, how="outer", sort=False,
                   returntype="meteoframe", keepindexfields="left",
                   suffixes=suffixes)
        mf.drop_fields(commonmeteofields(self, other, noncommon=True))
        # Operate series of common meteorological fields:
        combfields = [field for field in self.fields.meteofields if field in
                      other.fields.meteofields]
        for field in combfields:
            combval = mf[f"{field}_x"] + mf[f"{field}_y"]
            mf.add_field(field=field, value=combval)
        # Drop temporal suffixed fields:
        oldfields = [field for field in mf.fields.meteofields if field[-2:]
                     in suffixes]
        mf.drop_fields(oldfields)
        return mf

    def __getitem__(self, key):
        """
        Select and return subset data from self's dataframe.

        :param object key:  Key object for indexing. Passed to
                            pandas.DataFrame.__getitem__.
        :return:            Subset data.
        :rtype:             pandas.DataFrame or pandas.Series
        """
        return self.dataframe[key]

    def __repr__(self):
        """
        Generate a string with key information about self.

        :return:    String with key information.
        :rtype:     str
        """
        module = os.path.basename(os.path.splitext(__file__)[0])
        message = str(f""
                      f" .. Basic Info\n"
                      f" .... Short name: {self.shortname}\n"
                      f" .... Class:      {self.__class__.__name__}\n"
                      f" .... Module:     {module}\n"
                      f" .. Fields\n"
                      f" .... Code:        {self.fields.codefield}\n"
                      f" .... Time:        {self.fields.timefield}\n"
                      f" .... Longitude:   {self.fields.lonfield}\n"
                      f" .... Latitude:    {self.fields.latfield}\n"
                      f" .... Meteorology: {self.fields.meteofields}\n"
                      f" .. Dataframe\n"
                      f" .... Number of rows, columns:    "
                      f" {self.dataframe.shape[0]}, {self.dataframe.shape[1]}\n"
                      f" .... Missing data sentinel value: {self.navalue}")\
            .replace("None", "-")
        return message

    def __sub__(self, other):
        """
        Implement arithmetic subtraction operation between meteoframes.

        Operation is done element-wise for all common meteorological fields,
        using index fields to ensure compatibility.

        :param MeteoFrame other:    Meteoframe to be substracted from self.
        :return:                    New meteoframe whose meteorological
                                    fields are the subtraction of the input
                                    meteoframes common meteorological fields.
        :rtype:                     MeteoFrame
        """
        # Merge MeteoFrames and drop non-common meteorological fields:
        suffixes = ("_x", "_y")
        mf = merge(self, other, how="outer", sort=False,
                   returntype="meteoframe", keepindexfields="left",
                   suffixes=suffixes)
        mf.drop_fields(commonmeteofields(self, other, noncommon=True))
        # Operate series of common meteorological fields:
        combfields = [field for field in self.fields.meteofields if field in
                      other.fields.meteofields]
        for field in combfields:
            combval = mf[f"{field}_x"] - mf[f"{field}_y"]
            mf.add_field(field=field, value=combval)
        # Drop temporal suffixed fields:
        oldfields = [field for field in mf.fields.meteofields if field[-2:]
                     in suffixes]
        mf.drop_fields(oldfields)
        return mf

    def __str__(self):
        """
        Generate a string with table-like visual representation of self's
        dataframe.

        :return:    Message with table-like visual representation.
        :rtype:     str
        """
        return tabulate.tabulate(self.dataframe, headers="keys",
                                 showindex=False, tablefmt="rst",
                                 stralign="center", floatfmt=f".3f")

    def add_field(self, field, value, loc=None, fieldtype="meteofields",
                  keepoldspecial=False, navalue=None):
        """
        Add a new field to self.

        :param str field:           Name of the newly inserted field.
        :param pandas.Series value: Value of the newly inserted field.
        :param int loc:             Location where to insert new field. If
                                    None, field is appended after the last
                                    existent field. If inserting an index
                                    field, the inserting location will most
                                    certainly end being ignored. Passed to
                                    pandas.DataFrame.insert.
        :param str fieldtype:       Field type.
        :param bool keepoldspecial: If True, keep old index fieldtype field
                                    as a meteorological type. If False, drop it.
        :param object navalue:      Sentiel value indicating missing values on
                                    newly inserted data.
        """
        # Insert new field into dataframe:
        loc = len(self.dataframe.columns) if loc is None else loc
        self.dataframe.insert(loc=loc, column=field, value=value,
                              allow_duplicates=False)
        # Update fields:
        old = self.fields.type2name(fieldtype=fieldtype)
        self.fields.add(field, fieldtype)
        if old is not None and fieldtype.lower() != "meteofields":
            if keepoldspecial:
                self.fields.add(old, "meteofields")
            else:
                self.drop_fields(fields=old)
        # Enforce dataframe rules:
        self.enforceframe(navalue=navalue)

    def add_rows(self, newelements, navalue=None):
        """
        Append new rows to self.

        :param newelements:     Object whose rows are to be appended.
        :type newelements:      MeteoFrame or pandas.DataFrame
        :param object navalue:  Sentiel value indicating missing values on
                                newly inserted data.
        """
        if isinstance(newelements, MeteoFrame):
            newelements = newelements.dataframe
        self.dataframe.append(newelements, ignore_index=True, sort=False)
        self.fields.update(fields=list(self.dataframe))
        self.enforceframe(navalue=navalue)

    def aggregatemeteo(self, meteofields=None):
        """
        Aggregate self's meteorological fields by day on a new meteoframe.

        :param StrListOpt meteofields:  Meteorological fields to be aggregated.
                                        If None, all the current MeteoFrame's
                                        meteorological fields are used.
        :return:                        New meteoframe with inherited index
                                        fields and aggregated meteorological
                                        fields.
        :rtype:                         MeteoFrame
        """
        df = self.dataframe.copy()
        timeseries = df[self.fields.timefield].dt.floor("d")
        compfields = [f for f in self.fields.comparing
                      if not f == self.fields.timefield]
        meteofields = glist(self.fields.meteofields) if meteofields is None \
            else glist(meteofields)
        aggregated = None
        for i, field in enumerate(meteofields):
            var = MeteoVariables.alias2var(field)
            names = MeteoVariables.aggregnames(var)
            funcs = MeteoVariables.aggregfuns(var)
            natreshs = MeteoVariables.aggregnastreshold(var)
            for j, name in enumerate(names):
                name = f"{field}.{name}"
                func = funcs[j]
                natresh = natreshs[j]
                navalue = self.navalue
                def __superfunc(x): return navalue if \
                    (x.isnull().sum() * 100 / len(x)) >= natresh else func(x)
                dser = pandas.pivot_table(df, index=sflist([timeseries,
                                                            compfields]),
                                          values=field, dropna=True,
                                          aggfunc=__superfunc).reset_index()
                dser.rename({field: name}, axis="columns", inplace=True)
                aggregated = dser if i == 0 and j == 0 else \
                    pandas.merge(aggregated, dser, on=self.fields.comparing,
                                 how="left")
        aggregated = MeteoFrame(aggregated, sourcemode="dataframe",
                                codefield=self.fields.codefield,
                                timefield=self.fields.timefield,
                                lonfield=self.fields.lonfield,
                                latfield=self.fields.latfield,
                                shortname=self.shortname)
        return aggregated

    def apply_scalar(self, func, fields=None, name=None, fields2col=False,
                     *args, **kwargs):
        """
        Apply a scalar-returning function on self's fields.

        A scalar, pandas.Series or pandas.DataFrame is returned with an
        aggregation value for each operated field.

        For series-returning functions, use apply_series.

        :param callable func:   Function to which self's fields will be passed
                                as pandas.Series. Passed to
                                pandas.DataFrame.apply.
        :param str fields:      Names of the fields to be operated. If
                                None, all fields are operated.
        :param str name:        Optional name for the resulting pandas.Series.
                                Ignored if result is a scalar.
        :param bool fields2col: Add a new column with the name of the operated
                                fields, efectively returning a pandas.DataFrame.
        :param args:            Positional arguments to pass to func.
        :param kwargs:          Keyword arguments to pass to func.
        :return:                Aggregated value or series of aggregated values.
        :rtype:                 scalar or pandas.Series or pandas.DataFrame
        """
        fields = self.fields.storing if fields is None else fields
        scalars = self.dataframe[fields].aggregate(func=func, axis="index",
                                                   *args, **kwargs)
        if name is not None and isinstance(scalars, pandas.Series):
            scalars.rename(name, inplace=True)
        if fields2col:
            scalars = scalars.reset_index()
            scalars.rename(columns={"index": "Field"}, inplace=True)
        return scalars

    def apply_series(self, func, fields=None, *args, **kwargs):
        """
        Apply a series-returning function on self's fields.

        A new dataframe attribute is stored, with transformed values and the
        same axis length than the original.

        For scalar-returning functions, use apply_scalar.

        :param callable func:   Function to which self's fields will be passed
                                as pandas.Series. Passed to
                                pandas.DataFrame.apply.
        :param str fields:      Names of the fields to be operated. If
                                None, all fields are operated.
        :param args:            Positional arguments to pass to func.
        :param kwargs:          Keyword arguments to pass to func.
        """
        fields = self.fields.storing if fields is None else fields
        self.dataframe[fields].transform(func=func, axis="index",
                                         *args, **kwargs)
        self.fields.update(fields=list(self.dataframe))
        self.enforceframe()

    def autocodefield(self, lonfield, latfield):
        """
        Automatically generate a code field on self using unique pairs of
        coordinates fields.

        :param str lonfield:    Name of the longitude index field.
        :param str latfield:    Name of the latitude index field.
        """
        codefield = "Code"
        codeframe = self.dataframe.groupby([lonfield, latfield]).size()\
            .reset_index().drop(columns=0)
        numdigits = len(str(max(codeframe.index.values)))
        codes = [f"P{str(ind).zfill(numdigits)}"
                 for ind in codeframe.index.values]
        codeframe.insert(loc=0, column=codefield, value=codes)
        # Assign codes to dataframe attribute:
        self.__dataframe = self.dataframe.merge(codeframe, how="left",
                                                on=[lonfield, latfield])
        self.fields.rename({self.fields.type_codefield: codefield})

    def coordsmapping(self, coordsmap, newfield, newfieldtype="meteofields",
                      maplonfield="Longitude", maplatfield="Latitude",
                      mapvaluesfield="Values"):
        """
        Map values to a new or existing field on self by means of closest
        distance between coordinates.

        Coordinate index fields must be all defined. If not, an errnofieldtypes
        error is raised.

        :param pandas.DataFrame coordsmap:  Dataframe containing columns of
                                            mapping coordinates and values.
        :param str newfield:                Name of the new mapped field. If
                                            mapping would result in a duplicated
                                            meteorological field name, an
                                            errorvalue error is raised.
        :param str newfieldtype:            Type of the new field. If an
                                            already existing index field type is
                                            provided, the old index field is
                                            removed before mapping. Coordinate
                                            fields (aka lonfield or latfield
                                            types) are not allowed.
        :param str maplonfield:
        :param str maplatfield:
        :param str mapvaluesfield:
        """
        # Check coordinates fields:
        if self.fields.lonfield is None or self.fields.latfield is None:
            errnofieldtypes(["lonfield", "latfield"])
        # Define duplicated index field to be removed:
        if newfieldtype.lower() == "codefield" \
                and self.fields.codefield is not None:
            self.drop_fields(self.fields.codefield)
        elif newfieldtype.lower() == "timefield" \
                and self.fields.timefield is not None:
            self.drop_fields(self.fields.timefield)
        elif newfieldtype.lower() == "lonfield" or newfieldtype.lower() == \
                "latfield":
            errbadfields("newfieldtype", newfieldtype)
        else:
            if newfield in self.fields.meteofields:
                erryesfields(newfield)
        # Map values to closest MeteoFrame unique coordinate pairs:
        coordsframe = self.unique_rows(fields=self.fields.coords)
        assignframe = list()
        for row in coordsframe.itertuples(index=False):
            dx = getattr(row, self.fields.lonfield) - coordsmap[maplonfield]
            dy = getattr(row, self.fields.latfield) - coordsmap[maplatfield]
            mindistrow = numpy.sqrt(dx ** 2 + dy ** 2).idxmin()
            assignframe.append((getattr(row, self.fields.lonfield),
                                getattr(row, self.fields.latfield),
                                coordsmap[mapvaluesfield].iloc[mindistrow]))
        assignframe = pandas.DataFrame(assignframe, columns=[maplonfield,
                                                             maplatfield,
                                                             newfield])
        # Merge mapped values back to MeteoFrame structure:
        self.__dataframe = pandas.merge(self.dataframe, assignframe, how="left",
                                        sort=False,
                                        left_on=[self.fields.lonfield,
                                                 self.fields.latfield],
                                        right_on=[maplonfield, maplatfield])
        self.fields.add(newfield, newfieldtype)
        # Enforce dataframe rules:
        self.enforceframe()

    def copy(self):
        """
        Create a new instance with a deep copy of self's attributes.

        Modifications to any attribute of the copy will not be reflected in
        the original field. A shallow copy option is not provided, as
        meteoframe instances critically rely on their mutable attributes,
        which would still be references to their original versions when using
        shallow copies. With this in mind (and bearing the large difference
        in copying times), a scenario where a shallow copy would be more
        appropiate than a deep copy was not found.

        :return:    New meteoframe instance identical to self.
        :rtype:     MeteoFrame
        """

        return deepcopy(self)

    def csv(self, fullpath, mode="w+"):
        """
        Export self's dataframe as a comma-separated values (csv) file.

        :param str fullpath:    Full path for the generated .csv file. If a
                                filename and/or file extension are not found
                                inside, self's short name attribute and/or
                                ".csv" values are used, respectively.
                                Passed to pandas.DataFrame.to_csv.
        :param str mode:        Write mode passed to pandas.DataFrame.to_csv.
        """
        path, filename = os.path.split(fullpath)
        filename, extension = os.path.splitext(filename)
        filename = self.shortname if filename == "" else filename
        extension = ".csv" if extension == "" else extension
        fullpath = os.path.join(path, filename + extension)
        self.dataframe.to_csv(fullpath, index=False, mode=mode)

    def drop_duplicaterows(self, subset=None, keep="first"):
        """
        Remove duplicated rows on self.

        :param StrListOpt subset:   Columns to be removed. Passed to
                                    pandas.DataFrame.drop_duplicates.
        :param str keep:            Keep "first", "last" or "none" duplicated
                                    ocurrences. Passed to
                                    pandas.DataFrame.drop_duplicates.
        """
        self.dataframe.drop_duplicates(subset=subset, keep=keep, inplace=True)
        self.enforceframe()

    def drop_fields(self, fields):
        """
        Drop fields selected by field name on self.

        :param StrList fields:  Field or list of fields to be dropped. Passed
                                to pandas.DataFrame.drop.
        """
        self.dataframe.drop(fields, axis="columns", inplace=True)
        self.fields.delete(fields=fields)
        self.enforceframe()

    def drop_rows(self, rownums=None, filterdict=None, dropinlist=True):
        """
        Drop rows selected by row number or by filtering conditions on self.

        :param IntListOpt rownums:  Index or list of indexes to be dropped.
                                    Passed to pandas.DataFrame.drop.
        :param filterdict:          Filter to select columns to be dropped.
        :type filterdict:           Optional[Dict[str, Union[str,
                                    List[object]]]]
        :param bool dropinlist:     Drop or keep selected rows.
        """
        # Check switched rownums and filterdict arguments:
        if isinstance(rownums, dict):
            if isinstance(filterdict, list):
                rownums, filterdict = filterdict, rownums
            else:
                filterdict = rownums
                rownums = None
        # Select rows that meet conditions:
        if rownums is not None:
            rownums = glist(rownums)
            rownums = [n for n in self.dataframe.index.values.tolist()
                       if n in rownums]
        elif filterdict is not None:
            rownums = list()
            for key in filterdict:
                if key in self.fields.storing:
                    vals = filterdict[key]
                    vals = glist(vals)
                    rownums.append(self.dataframe[self.dataframe[key].isin(
                        vals)].index.values.tolist())
                else:
                    errbadfields("filterdict", key)
            rownums = flist(rownums)
        # Invert row selection if conditions are for keeping and not dropping:
        if not dropinlist:
            rownums = [n for n in self.dataframe.index.values.tolist()
                       if n not in rownums]
        # Drop dataframe rows:
        self.dataframe.drop(index=rownums, inplace=True)
        # Enforce dataframe rules:
        self.enforceframe()

    def enforceframe(self, navalue=None):
        """
        Modify self's dataframe field order, row order and values to enforce
        basic structural rules on self.

        Rules enforcement is done always in the same order as they have being
        defined. Rules compliance is checked when possible before enforced to
        avoid unrequired operations.

        :param object navalue:  Custom sentinel value indicating a missing
                                entry on not processed data.
        """
        # Missing values rule:
        if self.dataframe.isin([numpy.inf, -numpy.inf, navalue]).any().any():
            self.dataframe.replace(sflist([numpy.inf, -numpy.inf, navalue]),
                                   self.navalue, inplace=True)
        # Fields sorting rule:
        if sflist(list(self.dataframe)) != self.fields.storing:
            self.__dataframe = self.dataframe[self.fields.storing]
        # Rows sorting rule:
        self.dataframe.sort_values(self.fields.sorting, inplace=True)
        # Row index rule:
        if self.dataframe.index.values.tolist() != \
                list(range(len(self.dataframe.index))):
            self.dataframe.reset_index(drop=True, inplace=True)

    def groupbyindex(self, indextype="timefield", timestr=None, stat=None,
                     func=None):
        """
        Group self's dataframe using an available index field.

        :param str indextype:       Type of the index field used for grouping.
        :param str timestr:         Optional datetime format string used when
                                    grouping by a time index field.
        :param str stat:            Apply a known pandas.DataFrame.groupby
                                    stat computation method instead of the
                                    provided function. These methods are usually
                                    significantly faster.
        :param callable func:       Function to be applied over the generated
                                    groups. Not used if a stat is also provided.
        :return:                    Dataframe containing grouped data
        :rtype:                     pandas.DataFrame
        """
        # Group fields by key:
        indexfield = self.fields.type2name(fieldtype=indextype)
        groupkey = self[indexfield].dt.strftime(timestr) if indextype.lower(
        ) == "timefield" and timestr is not None else indexfield
        dfgroupby = self.dataframe[self.fields.meteofields]\
            .groupby(groupkey, as_index=True)
        # Apply function by key:
        df = None
        if stat is not None:
            if stat.lower() == "sum":
                df = dfgroupby.sum()
            elif stat.lower() == "mean":
                df = dfgroupby.mean()
            elif stat.lower() == "median":
                df = dfgroupby.median()
            elif stat.lower() == "count":
                df = dfgroupby.count()
            elif stat.lower() == "size":
                df = dfgroupby.size()
            elif stat.lower() == "max":
                df = dfgroupby.max()
            elif stat.lower() == "min":
                df = dfgroupby.min()
            else:
                errvalues("stat", stat)
        elif func is not None:
            df = dfgroupby.apply(func)
        else:
            errmissargs("pandasmethod", "func")
        df.reset_index(inplace=True)
        return df

    def info(self):
        """
        Print the returning string from the __repr__ method.
        """
        print(repr(self))

    def merge(self, right, how="inner", suffixes=("_x", "_y"), shortname=None):
        """
        Merge self with another meteoframe and replace self.

        :param MeteoFrame right:            Right side meteoframe.
        :param str how:                     Type of merge to be performed.
                                            Passed to pandas.merge().
        :param Tuple[str, str] suffixes:    Suffixes to be applied to
                                            overlapping field names. Passed to
                                            pandas.merge().
        :param Optional[str] shortname:     Short name for referring to the
                                            new merged MeteoFrame. If None, the
                                            left MeteoFrame shortname will be
                                            used unless 'how'=='right'.
        """
        df = merge(self, right, how=how, sort=False, returntype="meteoframe",
                   keepindexfields="left", suffixes=suffixes,
                   shortname=shortname)
        self.replace_self(df)

    def print(self):
        """
        Print the returning string from the __str__ method.
        """
        print(self)

    def rename_fields(self, dictmap):
        """
        Change selected fields names on self.

        :param Dict[str, str] dictmap:  Dictionary containing key-value pairs of
                                        old-new names for renamed fields.
                                        Passed to pandas.DataFrame.rename.
        """
        self.dataframe.rename(dictmap, axis="columns", inplace=True)
        self.fields.rename(namesmap=dictmap)

    def replace_self(self, new):
        """
        Fully replace self with a new meteoframe.

        :param MeteoFrame new:  Meteoframe to use as replacement.
        """
        self.__dict__.update(new.__dict__)

    def unique_rows(self, fields=None, subset=None):
        """
        Select and return unique rows from self's dataframe.

        :param StrListOpt fields:   Fields to be returned.
        :param StrListOpt subset:   Names of fields considered for identifying
                                    unique rows. Passed to
                                    pandas.DataFrame.drop_duplicates.
        :return:                    New dataframe or series containing unique
                                    rows.
        :rtype:                     pandas.DataFrame or pandas.Series
        """
        fields = self.fields.meteofields if fields is None else fields
        df = self.dataframe[fields]
        df = df.drop_duplicates(subset=subset, keep="first", inplace=False)
        return df


class MeteoVariables:
    """
    Class to handle meteorological fields units conversions.

    Class attributes:
        - meteodict: Nested dictioraries storing fields aliases, units
          conversion functions and default units for different meteorological
          fields.

    Class methods:
        - aggregfuns: Search the meteorological dictionary for the given
          variable's aggregators functions.
        - aggregnames: Search the meteorological dictionary for the given
          variable's aggregators names.
        - aggregnastreshold: Search the meteorological dictionary for the given
          variable's aggregators treshold percentage of na values.
        - alias2var: Search for the meteorological variable to which a field is
          associated.
        - convert: Convert series data between two compatible sets of
          units.
        - defunits: Search the meteorological dictionary for the given
          variable's default units.
        - unitsfun: Search the meteorological dictionary for one of the given
          variable's units conversion function.
    """
    meteodict = {
        "temperature": {
            "alias": ["TA", ],
            "units": {
                "degC": [lambda x: x, lambda x: x],
                "K": [lambda x: x - 273.15, lambda x: x + 273.15],
                "degF": [lambda x: (x - 32) / 1.8, lambda x: 1.8 * x + 32]},
            "defunits": "degC",
            "aggregators": {
                "names": ["max", "mean", "min"],
                "functions": [numpy.max, numpy.mean, numpy.min],
                "naspercentagetreshold": [100, 100, 100]}},
        "humidity": {
            "alias": ["HR", ],
            "units": {
                "%": [lambda x: x, lambda x: x]},
            "defunits": "%",
            "aggregators": {
                "names": ["max", "mean", "min"],
                "functions": [numpy.max, numpy.mean, numpy.min],
                "naspercentagetreshold": [100, 100, 100]}},
        "pressure": {
            "alias": ["PR", "SLP"],
            "units": {
                "Pa": [lambda x: x, lambda x: x],
                "hPa": [lambda x: x * 10 ** 2, lambda x: x / 10 ** 2],
                "bar": [lambda x: x * 10 ** 5, lambda x: x / 10 ** 5],
                "atm": [lambda x: x * 101325, lambda x: x / 101325]},
            "defunits": "Pa",
            "aggregators": {
                "names": ["max", "mean", "min"],
                "functions": [numpy.max, numpy.mean, numpy.min],
                "naspercentagetreshold": [100, 100, 100]}},
        "radiation": {
            "alias": ["RS", ],
            "units": {
                "W/m2": [lambda x: x, lambda x: x],
                "kW/m2": [lambda x: x * 10 ** 3, lambda x: x / 10 ** 3],
                "J/m2": [lambda x: x / 3600, lambda x: x * 3600],
                "kJ/m2": [lambda x: x / 3.6, lambda x: x * 3.6]},
            "defunits": "W/m2",
            "aggregators": {
                "names": ["max", "total"],
                "functions": [numpy.max, numpy.sum],
                "naspercentagetreshold": [100, 100]}},
        "wind speed": {
            "alias": ["VV", "VR", "VN", "VE"],
            "units": {
                "m/s": [lambda x: x, lambda x: x],
                "km/h": [lambda x: x / 3.6, lambda x: x * 3.6],
                "kn": [lambda x: x * 1.852 / 3.6, lambda x: x / 1.852 * 3.6]},
            "defunits": "m/s",
            "aggregators": {
                "names": ["max", "mean", "min"],
                "functions": [numpy.max, numpy.mean, numpy.min],
                "naspercentagetreshold": [100, 100, 100]}},
        "wind direction": {
            "alias": ["DV", "DR"],
            "units": {
                "deg": [lambda x: x, lambda x: x],
                "rad": [lambda x: x * 180 / numpy.pi,
                        lambda x: x / 180 * numpy.pi],
                "g": [lambda x: x * 0.9, lambda x: x / 0.9]},
            "defunits": "deg",
            "aggregators": {
                "names": ["mean", ],
                "functions": [numpy.mean, ],
                "naspercentagetreshold": [100, ]}},
        "precipitation": {
            "alias": ["PP", ],
            "units": {
                "mm": [lambda x: x, lambda x: x]},
            "defunits": "mm",
            "aggregators": {
                "names": ["max", "total"],
                "functions": [numpy.max, numpy.sum],
                "naspercentagetreshold": [100, 100]}},
        "heating": {
            "alias": ["Cal", ],
            "units": {
                "kWh": [lambda x: x, lambda x: x],
                "Wh": [lambda x: x / 10 ** 3, lambda x: x * 10 ** 3]},
            "defunits": "kWh",
            "aggregators": {
                "names": ["max", "total"],
                "functions": [numpy.max, numpy.sum],
                "naspercentagetreshold": [100, 100]}}}

    @classmethod
    def aggregfuns(cls, variable):
        """
        Search the meteorological dictionary for the given variable's
        aggregators functions.

        :param str variable:    Meteorological variable to be searched.
        :return:                Aggregators functions for the given variable.
        :rtype:                 List[callable]
        """
        names = cls.meteodict[variable]["aggregators"]["functions"]
        return names

    @classmethod
    def aggregnames(cls, variable):
        """
        Search the meteorological dictionary for the given variable's
        aggregators names.

        :param str variable:    Meteorological variable to be searched.
        :return:                Aggregators names for the given variable.
        :rtype:                 List[str]
        """
        names = cls.meteodict[variable]["aggregators"]["names"]
        return names

    @classmethod
    def aggregnastreshold(cls, variable):
        """
        Search the meteorological dictionary for the given variable's
        aggregators treshold percentage of na values.

        :param str variable:    Meteorological variable to be searched.
        :return:                Aggregators functions treshold for the given
                                variable.
        :rtype:                 List[bool]
        """
        names = cls.meteodict[variable]["aggregators"]["naspercentagetreshold"]
        return names

    @classmethod
    def alias2var(cls, alias):
        """
        Search for the meteorological variable to which a field is associated.

        :param str alias:   Field name associated to the objective variable.
        :return:            Meteorological variable name.
        :rtype:             str
        """
        namelist = [key for key in cls.meteodict.keys() if alias.lower() in
                    [stored.lower() for stored in
                     cls.meteodict.get(key).get("alias")]]
        name = namelist[0] if len(namelist) > 0 else None
        return name

    @classmethod
    def convert(cls, series, variable, fromunits, tounits):
        """
        Convert series data between two compatible sets of units.

        :param pandas.Series series:    Series of data to be converted.
        :param str variable:            Meteorological variable of the given
                                        series.
        :param str fromunits:           Current units of the given series.
        :param str tounits:             Objective units for the given series.
        :return:                        Series converted to the objective units.
        :rtype:                         pandas.Series
        """
        if fromunits != tounits:
            tobase_func, frombase_func = None, None
            try:
                tobase_func = cls.unitsfun(variable, fromunits)
            except KeyError:
                errvalues(["variable", "fromunits"], [variable, fromunits])
            try:
                frombase_func = cls.unitsfun(variable, tounits, False)
            except KeyError:
                errvalues(["variable", "fromunits"], [variable, fromunits])
            series = series.apply(tobase_func).apply(frombase_func)
        return series

    @classmethod
    def defunits(cls, variable):
        """
        Search the meteorological dictionary for the given variable's default
        units.

        :param str variable:    Meteorological variable to be searched.
        :return:                Default units for the given variable.
        :rtype:                 str
        """
        fun = cls.meteodict[variable]["defunits"]
        return fun

    @classmethod
    def unitsfun(cls, variable, units, converttobase=True):
        """
        Search the meteorological dictionary for one of the given variable's
        units conversion function.

        :param str variable:        Meteorological variable to be searched for.
        :param str units:           Units to be searched for.
        :param bool converttobase:  If True, convert from given units to
                                    default units. If False, convert from
                                    default units to given units.
        :return:                    Units conversion function.
        :rtype:                     types.FunctionType
        """
        fun = cls.meteodict[variable]["units"][units][not converttobase]
        return fun


def commonindexfields(left, right):
    """
    Return names of equivalent index fields defined for two meteoframes.

    :param MeteoFrame left:     Left side meteoframe.
    :param MeteoFrame right:    Right side meteoframe.
    :return:                    Pair of lists with index fields.
    :rtype:                     Tuple[StrList, List[str]]
    """
    # Select relevant fields:
    leftlist = left.fields.comparing_exhaustive
    rightlist = right.fields.comparing_exhaustive
    leftfields = [lfield for lfield, rfield in zip(leftlist, rightlist)
                  if lfield is not None and rfield is not None]
    rightfields = [rfield for lfield, rfield in zip(leftlist, rightlist)
                   if lfield is not None and rfield is not None]
    return leftfields, rightfields


def commonmeteofields(left, right, noncommon=False):
    """
    Return names of meteorological fields available for two meteoframes.

    :param MeteoFrame left:     Left side meteoframe.
    :param MeteoFrame right:    Right side meteoframe.
    :param bool noncommon:      If True, return non-common meteorological
                                fields instead.
    :return:                    List of meteorological fields.
    :rtype:                     StrList
    """
    if noncommon:
        fields = [field for field in
                  left.fields.meteofields + right.fields.meteofields
                  if field not in left.fields.meteofields
                  or field not in right.fields.meteofields]
    else:
        fields = [field for field in left.fields.meteofields
                  if field in right.fields.meteofields]
    return fields


def comparableseries(left, right, field):
    """
    Merge two meteoframes and return a pair of directly comparable series for a
    given field.

    :param MeteoFrame left:     Left side meteoframe.
    :param MeteoFrame right:    Right side meteoframe.
    :param str field:           Field to be returned.
    :return:                    Tuple of directly comparable series.
    :rtype:                     Tuple[pandas.Series, pandas.Series]
    """
    df = merge(left, right, how="outer", sort=False, returntype="dataframe",
               keepindexfields="left", suffixes=("_x", "_y"))
    series1, series2 = df[f"{field}_x"], df[f"{field}_y"]
    return series1, series2


def merge(left, right, how="inner", sort=False, returntype="meteoframe",
          keepindexfields="left", suffixes=("_x", "_y"), shortname=None):
    """
    Merge dataframes elements from two meteoframes by their index fields.
    Row coherence between merged columns is guaranteed by means of commonly
    available index fields.

    :param MeteoFrame left:             Left side meteoframe.
    :param MeteoFrame right:            Right side meteoframe.
    :param str how:                     Type of merge to be performed. Passed to
                                        pandas.merge().
    :param bool sort:                   Option to sort resulting keys.
                                        MeteoFrame results will always be
                                        sorted during their creation.
                                        Passed to pandas.merge().
    :param str returntype:              Type of the result structure.
    :param str keepindexfields:         MeteoFrame from which index fields will
                                        be kept.
    :param Tuple[str, str] suffixes:    Suffixes to be applied to
                                        overlapping field names. Passed to
                                        pandas.merge().
    :param str shortname:               Short name for referring to the new
                                        merged MeteoFrame. If None, the left
                                        MeteoFrame shortname will be used unless
                                        'how'=='right'.
    :return:                            Merged meteoframe or dataframe.
    :rtype:                             pandas.DataFrame or MeteoFrame
    """
    # Define duplicated index fields to be removed:
    if keepindexfields.lower() == "right":
        oldindex = [ind for ind in left.fields.indexing
                    if ind not in right.fields.indexing]
        refmf = right
    elif keepindexfields.lower() == "both":
        oldindex = None
        refmf = left
    elif keepindexfields.lower() == "left":
        oldindex = [ind for ind in right.fields.indexing
                    if ind not in left.fields.indexing]
        refmf = left
    else:
        errvalues("keepindexfields", keepindexfields)
        oldindex, refmf = None, None
    # Merge dataframes:
    leftindices, rightindices = commonindexfields(left, right)
    frame = pandas.merge(left.dataframe, right.dataframe, how=how, sort=sort,
                         left_on=leftindices, right_on=rightindices,
                         suffixes=suffixes)
    # Drop duplicated index fields:
    frame.drop(oldindex, axis="columns", inplace=True)
    # Return mode check:
    if returntype.lower() == "meteoframe":
        shortname = refmf.shortname if shortname is None else shortname
        frame = MeteoFrame(frame, sourcemode="dataframe",
                           codefield=refmf.fields.codefield,
                           timefield=refmf.fields.timefield,
                           lonfield=refmf.fields.lonfield,
                           latfield=refmf.fields.latfield, shortname=shortname)
    return frame
