"""Collection of functions to operate on data from University of Oregon Solar
Radiation Monitoring Laboratory (SRML) data.
"""
import numpy as np
import pandas as pd
import urllib
import warnings

# VARIABLE_MAP is a dictionary mapping SRML data element numbers to their
# pvlib names. For most variables, only the first three digits are used,
# the fourth indicating the instrument. Spectral data (7xxx) uses all
# four digits to indicate the variable. See a full list of data element
# numbers `here. <http://solardata.uoregon.edu/DataElementNumbers.html>`_

VARIABLE_MAP = {
'016': 'voltage_15_180',
'024': 'voltage_25_270',
'026': 'voltage_25_180',
'028': 'voltage_25_90',
'036': 'voltage_30_180',
'056': 'voltage_45_180',
'066': 'voltage_60_180',
'100': 'ghi',
'109': 'gri',
'116': 'poa_global_15_180',
'124': 'poa_global_25_270',
'125': 'poa_global_25_225',
'126': 'poa_global_25_180',
'128': 'poa_global_25_90',
'136': 'poa_global_30_180',
'146': 'poa_global_35_180',
'156': 'poa_global_45_180',
'166': 'poa_global_60_180',
'192': 'poa_global_90_0',
'196': 'poa_global_90_180',
'201': 'dni',
'300': 'dhi',
'416': 'current_15_180',
'424': 'current_25_270',
'426': 'current_25_180',
'428': 'current_25_90',
'436': 'current_30_180',
'456': 'current_45_180',
'466': 'current_60_180',
'516': 'power_15_180',
'524': 'power_25_270',
'526': 'power_25_180',
'528': 'power_25_90',
'536': 'power_30_180',
'556': 'power_45_180',
'566': 'power_60_180',
'616': 'voltage_15_180',
'624': 'voltage_25_270',
'626': 'voltage_25_180',
'628': 'voltage_25_90',
'636': 'voltage_30_180',
'656': 'voltage_45_180',
'666': 'voltage_60_180',
'7000': 'horizontal_og_570',
'7001': 'horizontal_rg_630',
'7002': 'horizontal_rg_695',
'7003': 'horizontal_uva',
'7004': 'horizontal_uvb',
'7005': 'horizontal_illum_zenith',
'7006': 'horizontal_illum_diffuse',
'7007': 'horizontal_illum_global',
'7008': 'horizontal_illum_max',
'7009': 'horizontal_illum_min',
'7010': 'beam_og_570',
'7011': 'beam_rg_630',
'7012': 'beam_rg_695',
'7013': 'beam_uva',
'7014': 'beam_uvb',
'7017': 'beam_illum_norm',
'7018': 'beam_illum_max',
'7019': 'beam_illum_min',
'910': 'sky_condition',
'911': 'ceiling_height',
'912': 'visibility',
'913': 'weather',
'915': 'rainfall',
'917': 'bp',
'920': 'wind_direction',
'921': 'wind_speed',
'922': 'wind_direction_standard_deviation',
'930': 'temp_air',
'931': 'temp_dew',
'933': 'relative_humidity',
'937': 'temp_cell',
'938': 'temp_air_min',
'939': 'temp_air_max',
'940': 'bp_average',
'951': 'total_sky_cover',
'952': 'opaque_sky_cover',
'953': 'precipitable_water',
'954': 'aerosol_optical_depth',
'955': 'snow_depth',
'956': 'albedo',
'965': 'days_since_last_snowfall',
}


def read_srml(filename, map_variables=True):
    """
    Read University of Oregon SRML 1min .tsv file into pandas dataframe.

    The SRML is described in [1]_.

    Parameters
    ----------
    filename: str
        filepath or url to read for the tsv file.
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.

    Returns
    -------
    data: Dataframe
        A dataframe with datetime index

    Notes
    -----
    The time index is shifted back by one interval to account for the
    daily endtime of 2400, and to avoid time parsing errors on leap
    years. The returned data values are labeled by the left endpoint of
    interval, and should be understood to occur during the interval from
    the time of the row until the time of the next row. This is consistent
    with pandas' default labeling behavior.

    See [2]_ for more information concerning the file format.

    References
    ----------
    .. [1] University of Oregon Solar Radiation Monitoring Laboratory
       http://solardata.uoregon.edu/
    .. [2] `Archival (short interval) data files
       <http://solardata.uoregon.edu/ArchivalFiles.html>`_
    """
    tsv_data = pd.read_csv(filename, delimiter='\t')
    data = _format_index(tsv_data)
    # Drop day of year and time columns
    data = data[data.columns[2:]]

    if map_variables:
        data = data.rename(columns=_map_columns)

    # Quality flag columns are all labeled 0 in the original data. They
    # appear immediately after their associated variable and are suffixed
    # with an integer value when read from the file. So we map flags to
    # the preceding variable with a '_flag' suffix.
    #
    # Example:
    #   Columns ['ghi_0', '0.1', 'temp_air_2', '0.2']
    #
    #   Yields a flag_label_map of:
    #       { '0.1': 'ghi_0_flag',
    #         '0.2': 'temp_air_2'}
    #
    columns = data.columns
    flag_label_map = {flag: columns[columns.get_loc(flag) - 1] + '_flag'
                      for flag in columns[1::2]}
    data = data.rename(columns=flag_label_map)

    # Mask data marked with quality flag 99 (bad or missing data)
    for col in columns[::2]:
        missing = data[col + '_flag'] == 99
        data[col] = data[col].where(~(missing), np.nan)
    return data


def _map_columns(col):
    """Map data element numbers to pvlib names.

    Parameters
    ----------
    col: str
        Column label to be mapped.

    Returns
    -------
    str
        The pvlib label if it was found in the mapping,
        else the original label.
    """
    if col.startswith('7'):
        # spectral data
        try:
            return VARIABLE_MAP[col]
        except KeyError:
            return col
    try:
        variable_name = VARIABLE_MAP[col[:3]]
        variable_number = col[3:]
        return variable_name + '_' + variable_number
    except KeyError:
        return col


def _format_index(df):
    """Create a datetime index from day of year, and time columns.

    Parameters
    ----------
    df: pd.Dataframe
        The srml data to reindex.

    Returns
    -------
    df: pd.Dataframe
        The Dataframe with a DatetimeIndex localized to 'Etc/GMT+8'.
    """
    # Name of the second column indicates the year of the file, but
    # the column contains times.
    year = int(df.columns[1])
    df_doy = df[df.columns[0]]
    # Times are expressed as integers from 1-2400, we convert to 0-2359 by
    # subracting the length of one interval and then correcting the times
    # at each former hour. interval_length is determined by taking the
    # difference of the first two rows of the time column.
    # e.g. The first two rows of hourly data are 100 and 200
    #      so interval_length is 100.
    interval_length = df[df.columns[1]][1] - df[df.columns[1]][0]
    df_time = df[df.columns[1]] - interval_length
    if interval_length == 100:
        # Hourly files do not require fixing the former hour timestamps.
        times = df_time
    else:
        # Because hours are represented by some multiple of 100, shifting
        # results in invalid values.
        #
        # e.g. 200 (for 02:00) shifted by 15 minutes becomes 185, the
        #      desired result is 145 (for 01:45)
        #
        # So we find all times with minutes greater than 60 and remove 40
        # to correct to valid times.
        old_hours = df_time % 100 > 60
        times = df_time.where(~old_hours, df_time - 40)
    times = times.apply(lambda x: '{:04.0f}'.format(x))
    doy = df_doy.apply(lambda x: '{:03.0f}'.format(x))
    dts = pd.to_datetime(str(year) + '-' + doy + '-' + times,
                         format='%Y-%j-%H%M')
    df.index = dts
    df = df.tz_localize('Etc/GMT+8')
    return df


def get_srml(station, start, end, filetype='PO', map_variables=True,
             url="http://solardata.uoregon.edu/download/Archive/"):
    """Request data from UoO SRML and read it into a Dataframe.

    The University of Oregon Solar Radiation Monitoring Laboratory (SRML) is
    described in [1]_. A list of stations can be found in [2]_.

    Data is returned for the entire months between and including start and end.

    Parameters
    ----------
    station : str
        Two letter station abbreviation.
    start : datetime-like
        First day of the requested period
    end : datetime-like
        Last day of the requested period
    filetype : string, default: 'PO'
        SRML file type to gather. See notes for explanation.
    map_variables : bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, default: 'http://solardata.uoregon.edu/download/Archive/'
        API endpoint URL

    Returns
    -------
    data : pd.DataFrame
        Dataframe with data from SRML.
    meta : dict
        Metadata.

    Notes
    -----
    File types designate the time interval of a file and if it contains
    raw or processed data. For instance, `RO` designates raw, one minute
    data and `PO` designates processed one minute data. The availability
    of file types varies between sites. Below is a table of file types
    and their time intervals. See [1] for site information.

    ============= ============ ==================
    time interval raw filetype processed filetype
    ============= ============ ==================
    1 minute      RO           PO
    5 minute      RF           PF
    15 minute     RQ           PQ
    hourly        RH           PH
    ============= ============ ==================

    Warning
    -------
    SRML data has nighttime data prefilled with 0s through the end of the
    current month (i.e., values are provided for data in the future).

    References
    ----------
    .. [1] University of Oregon Solar Radiation Measurement Laboratory
       http://solardata.uoregon.edu/
    .. [2] Station ID codes - Solar Radiation Measurement Laboratory
       http://solardata.uoregon.edu/StationIDCodes.html
    """
    # Use pd.to_datetime so that strings (e.g. '2021-01-01') are accepted
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # Generate list of months
    months = pd.date_range(
        start, end.replace(day=1) + pd.DateOffset(months=1), freq='1M')
    months_str = months.strftime('%y%m')

    # Generate list of filenames
    filenames = [f"{station}{filetype}{m}.txt" for m in months_str]

    dfs = []  # Initialize list of monthly dataframes
    for f in filenames:
        try:
            dfi = read_srml(url + f, map_variables=map_variables)
            dfs.append(dfi)
        except urllib.error.HTTPError:
            warnings.warn(f"The following file was not found: {f}")

    data = pd.concat(dfs, axis='rows')

    meta = {'filetype': filetype,
            'station': station,
            'filenames': filenames}

    return data, meta
