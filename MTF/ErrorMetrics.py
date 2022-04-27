# coding=utf-8
"""
This module provides access to the ErrorMetric class. It is meant to be used
for comparing MeteoFrame objects storing meteorological measurements and/or
estimations data.
"""

# Import built-in modules:
from functools import reduce
# Import external packages:
import numpy
import pandas
import tabulate
# Import project scripts:
from AuxTools import *
from MeteoFrames import MeteoFrame, MeteoVariables


class ErrorMetrics:
    """
    Class to compute and compare meteorological data stored inside MeteoFrame
    objects.

    Class methods:
        - __applicator: apply an error metric function over a difference
          values MeteoFrame.
        - __equalizetime: drop rows with from a reference data MeteoFrame and
          multiple estimations data MeteoFrames whose timefield value is outside
          of a common overlapping temporal interval.
        - compare: Compute and compare error metrics between a reference data
          MeteoFrame and multiple estimations data MeteoFrames.
        - count: Count the number of valid non-missing values on a data Series.
        - cvrmse: Compute the Coefficient of Variation of the Root Mean Square
          Error metric on a data Series containing differences between
          estimation results and observation, reference data.
        - mae: Compute the Mean Absolute Error metric on a data Series
          containing differences between estimation results and observation,
          reference data.
        - mbe: Compute the Mean Bias Error metric on a data Series containing
          differences between estimation results and observation, reference
          data.
        - rmse: Compute the Root Mean Square Error metric on a data Series
          containing differences between estimation results and observation,
          reference data.
    """

    #########################
    # Private class methods #
    #########################
    @classmethod
    def __applicator(cls, meteoframe, func, name, grouped=False,
                     indextype="timefield", timestr=None):
        """
        Apply an error metric function over a difference values MeteoFrame.

        :param MeteoFrame meteoframe:   MeteoFrame providing the difference
                                        values to be converted into an error
                                        metric.
        :param callable func:           Error metric function to be applied.
        :param str name:                Name of the column storing func
                                        result values.
        :param bool grouped:            Group differences by a given index type
                                        when computing error metrics.
        :param str indextype:           Type of the index field used for
                                        grouping. Passed to
                                        MeteoFrame.groupbyindex.
        :param str timestr:             Optional datetime format string used
                                        when grouping by a time index field.
                                        Passed to MeteoFrame.groupbyindex.
        :return:                        DataFrame with error metric values.
        :rtype:                         pandas.DataFrame
        """
        if grouped:
            result = pandas.melt(meteoframe.groupbyindex(
                indextype=indextype, timestr=timestr, func=func)
                .rename(columns={meteoframe.fields.timefield: "Group"}),
                id_vars="Group", var_name="Field", value_name=name)
        else:
            result = meteoframe.apply_scalar(
                func=func, name=name, fields2col=True,
                fields=meteoframe.fields.meteofields)
        return result

    @classmethod
    def __equalizetime(cls, data, results):
        """
        Drop rows with from a reference data MeteoFrame and multiple
        estimations data MeteoFrames whose timefield value is outside of a
        common overlapping temporal interval.

        :param MeteoFrame data:         MeteoFrame providing the measurements
                                        used as reference data.
        :param results:                 MeteoFrames providing the results used
                                        as estimations data.
        :type results:                  List[MeteoFrame]
        """
        mfs = flist([data, ] + results)
        start = [mf[mf.fields.timefield].min() for mf in mfs]
        start = pandas.to_datetime(max(start), format="%Y-%m-%d %H:%M:%S")
        end = [mf[mf.fields.timefield].max() for mf in mfs]
        end = pandas.to_datetime(min(end), format="%Y-%m-%d %H:%M:%S")
        instants = [str(i) for i in pandas.date_range(start, end, freq="H")]
        [mf.drop_rows(filterdict={mf.fields.timefield: instants},
                      dropinlist=False) for mf in mfs]

    ########################
    # Public class methods #
    ########################
    @ classmethod
    def compare(cls, data, results, filepath=None, aggregatedfields=False,
                groupbyindex=False, indextype="timefield", timestr=None):
        """
        Compute and compare error metrics between a reference data MeteoFrame
        and multiple estimations data MeteoFrames.

        :param MeteoFrame data:         MeteoFrame providing the measurements
                                        used as reference data.
        :param results:                 MeteoFrames providing the results used
                                        as estimations data.
        :type results:                  List[MeteoFrame]
        :param str filepath:            Full path for the file containing
                                        metrics comparison results. If None,
                                        results are prety-printed on console.
        :param bool aggregatedfields:   Include aggregated meteorological fields
                                        in the comparison.
        :param bool groupbyindex:       Group differences by a given index type
                                        when computing error metrics.
        :param str indextype:           Type of the index field used for
                                        grouping. Passed to
                                        MeteoFrame.groupbyindex.
        :param str timestr:             Optional datetime format string used
                                        when grouping by a time index field.
                                        Passed to MeteoFrame.groupbyindex.
        """
        # Compute differences meteoframes:
        cls.__equalizetime(data, results)
        difs = [res - data for res in results]
        aggs = [res.aggregatemeteo() - data.aggregatemeteo()
                for res in results] if aggregatedfields else None
        mfs = flist([difs, aggs]) if aggregatedfields else difs
        # Generate indices dataframe:
        fields = list()
        for field in difs[0].fields.meteofields:
            variable = MeteoVariables.alias2var(field)
            if aggregatedfields:
                fields.append([field, [f"{field}.{agg}" for agg in
                                       MeteoVariables.aggregnames(variable)]])
            else:
                fields.append(field)
        fields = flist(flist(fields))
        sources = [dif.shortname for dif in difs]
        groups = data.groupbyindex(indextype=indextype, timestr=timestr,
                                   func=cls.count)[
            data.fields.type2name(indextype)] if groupbyindex else ["-", ]
        metrics = pandas.DataFrame([(field, source, group) for field in fields
                                    for source in sources for group in groups])\
            .rename(columns={0: "Field", 1: "Source", 2: "Group"})
        # Compute metrics:
        functions = [cls.count, cls.mbe, cls.mae, cls.rmse]
        names = ["N", "MBE", "MAE", "RMSE"]
        mergeindex = ["Field", "Source", "Group"] if groupbyindex \
            else ["Field", "Source"]
        partials = [pandas.concat([cls.__applicator(
            mf, func=func, name=name, grouped=groupbyindex, indextype=indextype,
            timestr=timestr).assign(Source=mf.shortname) for mf in mfs])
                    for func, name in zip(functions, names)]
        metrics = reduce(lambda left, right:
                         pandas.merge(left, right, on=mergeindex, how="left"),
                         flist([metrics, partials]))
        # Drop rows with all empty metrics:
        metrics.dropna(axis="index", how="all", subset=names, inplace=True)
        metrics.reset_index(drop=True, inplace=True)
        # Save or print results:
        if filepath is not None:
            metrics.to_csv(filepath, header=True, index=False, mode="w+")
        else:
            print(tabulate.tabulate(metrics, headers="keys", tablefmt="rst",
                                    showindex=False))

    @classmethod
    def count(cls, series, sentinel=None):
        """
        Count the number of valid non-missing values on a data Series.

        :param pandas.Series series:    Input data series.
        :param object sentinel:         Use a custom missing data sentinel
                                        value. If None, default pandas sentinel
                                        value is used.
        :return:                        Number of valid values.
        :rtype:                         int32
        """
        n = series.count() if sentinel is None \
            else len(series) - series.value_counts(dropna=False)[sentinel]
        return n

    @classmethod
    def cvrmse(cls, series, meanobservation):
        """
        Compute the Coefficient of Variation of the Root Mean Square Error
        metric on a data Series containing differences between estimation
        results and observation, reference data.

        :param pandas.Series series:    Input differences Series.
        :param float64 meanobservation: Mean of the observation data values.
        :return:                        CV(RMSE) metric error value.
        :rtype:                         float64
        """
        rmse = cls.rmse(series)
        error = rmse / meanobservation
        return error

    @classmethod
    def mae(cls, series):
        """
        Compute the Mean Absolute Error metric on a data Series containing
        differences between estimation results and observation, reference data.

        :param pandas.Series series:    Input differences Series.
        :return:                        MAE metric error value.
        :rtype:                         float64
        """
        error = series.abs().mean(skipna=True)
        return error

    @classmethod
    def mbe(cls, series):
        """
        Compute the Mean Bias Error metric on a data Series containing
        differences between estimation results and observation, reference data.

        :param pandas.Series series:    Input differences Series.
        :return:                        MBE metric error value.
        :rtype:                         float64
        """
        error = series.mean(skipna=True)
        return error

    @classmethod
    def rmse(cls, series):
        """
        Compute the Root Mean Square Error metric on a data Series containing
        differences between estimation results and observation, reference data.

        :param pandas.Series series:    Input differences Series.
        :return:                        RMSE metric error value.
        :rtype:                         float64
        """
        error = numpy.sqrt(numpy.square(series).mean(skipna=True))
        return error
