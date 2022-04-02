import numpy as np
import pandas as pd
from scipy import optimize

def split_word_to_list(input_string):
    return [str_char for str_char in input_string]


def intersect_of_2_lists(list1, list2):
    return list(set(list1).intersection(list2))


def dataframe_filter_only_above_nas(df, header_column):
    '''
    Filter to store only relevant list values, i.e. only part of dataframe above first nan entry

    :param pd.DataFrame df: Dataframe to be filtered
    :param str header_column: Column in dataframe to search for NaN values

    :return: Filtered dataframe
    :rtype: pd.DataFrame
    '''
    na_indexes = list(np.flatnonzero(df.loc[:, header_column].isna()))  # Find NaN cells to define where lists end

    return df.iloc[0:na_indexes[0], :] if len(na_indexes) > 0 else df


def friction_torsion_resistance_swivel_t1(friction, force, r_out, r_in):
    '''
    Function returning friction resistance moment due to reaction force at swivel / cleat interface.

    :param float friction: Friction coefficient
    :param float force: Transversal force component at insulator attachment point
    :param float r_out: Outer radius of swivel area in contact with cleat
    :param float r_in: Inner radius of swivel area in contact with cleat

    :return: Swivel force reaction moment due to reaction force at swivel / cleat interface
    :rtype: float
    '''

    return 2. * friction * force * (r_out ** 3 - r_in ** 3) / (r_out ** 2 - r_in ** 2) / 3.


def friction_torsion_resistance_swivel_t2(friction, f_y, f_z, b_swivel, h_swivel, r_pin):
    '''
    Function returning friction resistance moment due to force couple in set up in swivel in case of transversal force
    components. Returns value non-zero value only if reaction force couple is of opposing sign.

    :param float friction: Friction coefficient
    :param float f_y: Transversal force component at insulator attachment point
    :param float f_z: Vertical force component at insulator attachment point
    :param float b_swivel: Width of swivel
    :param float h_swivel: Height of swivel, from pin bolt centre line to cold end attachment centre point
    :param float r_pin: Swivel pin bolt radius

    :return: Swivel force reaction moment due to reaction force couple
    :rtype: float
    '''

    r_y = (f_z / 2. - f_y * h_swivel / b_swivel)

    return 2. * friction * abs(r_y) * r_pin if r_y < 0. else 0.


def friction_moment_at_critical_section_swivel(f_x, force_arm, fraction_to_critical_section, friction_moment):
    '''
    Function finding the moment at the selected insulator cold end section to be evaluated for bending stresses

    :param float f_x: Longitudinal swivel attachment force
    :param float force_arm: Distance from swivel pin bolt centre line to effective force arm. Can be further out than
    clevis / ball contact, depending on friction in the insulator ball links
    :param float fraction_to_critical_section: Distance (pin -> critical section) / (effective force arm)
    :param float friction_moment: Friction resistance moment (T1 + T2)

    :return: Moment at critical section
    :rtype: float
    '''

    m_applied = abs(f_x) * force_arm * fraction_to_critical_section
    m_friction = friction_moment * fraction_to_critical_section

    return m_applied if m_applied < m_friction else m_friction


def stress_stem_roark_17(f_axial, m_bending, d_outer, d_inner, r_notch):
    '''
    Function to convert force and bending actions to stress at a stem diameter transition

    :param float f_axial: Axial force
    :param float m_bending: Bending moment
    :param float d_outer: Stem maximum diameter
    :param float d_inner: Stem minimum diameter
    :param float r_notch: Notch radius at diameter transition

    :return: Stress including SCF's from axial and bending actions
    :rtype: float
    '''
    stress_axial, stress_bending = 0., 0.
    if abs(f_axial) > 0.:
        stress_axial = scf_roark_17a(d_outer, d_inner, r_notch) * abs(f_axial)
    if abs(m_bending) > 0.:
        stress_bending = scf_roark_17b(d_outer, d_inner, r_notch) * abs(m_bending)

    return stress_axial + stress_bending


def scf_roark_17a(d_outer, d_inner, r_notch):
    '''
    Roark's Formulas for stress and strain, table 17.1, formula 17a for axial stress

    :param float d_outer: Stem maximum diameter
    :param float d_inner: Stem minimum diameter
    :param float r_notch: Notch radius at diameter transition

    :return: Axial stress SCF
    :rtype: float
    '''
    h = (d_outer - d_inner) / 2.
    ratio = h / r_notch
    h_d = 2. * h / d_outer
    root_ratio = np.sqrt(ratio)

    C1 = [0.927 + 1.149 * root_ratio - 0.086 * ratio, 1.125 + 0.831 * root_ratio - 0.01 * ratio]
    C2 = [0.011 - 3.029 * root_ratio + 0.948 * ratio, -1.831 - 0.318 * root_ratio - 0.049 * ratio]
    C3 = [-0.304 + 3.979 * root_ratio - 1.737 * ratio, 2.236 - 0.522 * root_ratio + 0.176 * ratio]
    C4 = [0.366 - 2.098 * root_ratio + 0.875 * ratio, -0.63 + 0.009 * root_ratio - 0.117 * ratio]

    col = 0
    if 0.25 <= ratio <= 2.:
        col = 0
    elif 2. < ratio <= 20.:
        col = 1
    else:
        raise "h / r outside bounds. SCF not valid"

    scf = C1[col] + C2[col] * h_d + C3[col] * h_d ** 2 + C4[col] * h_d ** 3

    return 4. * scf / (np.pi * (d_outer - 2. * h) ** 2)


def scf_roark_17b(d_outer, d_inner, r_notch):
    '''
    Roark's Formulas for stress and strain, table 17.1, formula 17b for bending stress

    :param float d_outer: Stem maximum diameter
    :param float d_inner: Stem minimum diameter
    :param float r_notch: Notch radius at diameter transition

    :return: Axial stress SCF
    :rtype: float
    '''
    h = (d_outer - d_inner) / 2.
    ratio = h / r_notch
    h_d = 2. * h / d_outer
    root_ratio = np.sqrt(ratio)

    C1 = [0.927 + 1.149 * root_ratio - 0.086 * ratio, 1.125 + 0.831 * root_ratio - 0.01 * ratio]
    C2 = [0.015 - 3.281 * root_ratio + 0.837 * ratio, -3.79 + 0.958 * root_ratio - 0.257 * ratio]
    C3 = [0.847 + 1.716 * root_ratio - 0.506 * ratio, 7.374 - 4.834 * root_ratio + 0.862 * ratio]
    C4 = [-0.79 + 0.417 * root_ratio - 0.246 * ratio, -3.809 + 3.046 * root_ratio - 0.595 * ratio]

    col = 0
    if 0.25 <= ratio <= 2.:
        col = 0
    elif 2. < ratio <= 20.:
        col = 1
    else:
        raise "h / r outside bounds. SCF not valid"

    scf = C1[col] + C2[col] * h_d + C3[col] * h_d ** 2 + C4[col] * h_d ** 3

    return 32. * scf / (np.pi * (d_outer - 2. * h) ** 3)


def en_1991_1_4_b_9(x):
    return 0.7 * np.log10(x) ** 2 - 17.4 * np.log10(x) + 100


def en_1991_1_4_b_9_solve(x, y):
    return en_1991_1_4_b_9(x) - y


def stress_histogram_en_1991_1_4(n_histograms, n_delta, coefficient=1):
    '''

    :param int n_histograms:
    :param int n_delta:
    :param int coefficient:
    :return:
    '''
    'Define boundaries for stress histogram'
    n_max = optimize.root_scalar(
                en_1991_1_4_b_9_solve,
                x0=1e6,
                bracket=[1., 1e10],
                args=(0.)
            ).root
    n_start = optimize.root_scalar(
        en_1991_1_4_b_9_solve,
        x0=1e6,
        bracket=[1., 1e10],
        args=(100.)
    ).root

    'Find factor by which to increase each n_delta such that final value in n_list equals n_max'
    # alpha = (n_max / n_delta - n_histograms) / (n_histograms * (n_histograms + 1.)) * 2.
    alpha = (n_max / n_delta - n_histograms) / sum([i ** coefficient for i in range(n_histograms + 1)])
    'Find upper and lower bound for each histogram column'
    n_list = [1.]
    for i in range(0, n_histograms):
        n_list.append(n_list[i] + n_delta * (1 + i ** coefficient * alpha))
    'Find histograms'
    n_list = np.array(n_list)
    delta_list = n_list[1:] - n_list[:-1]

    'Find stress values at middle of histograms'
    n_middle = [(x + y) / 2 for (x, y) in zip(n_list[:-1], n_list[1:])]
    stresses = [en_1991_1_4_b_9(x) for x in n_middle]

    return delta_list, stresses


def dataframe_select_suspension_insulator_sets(df, tow_string, set_string, set_factor, column_texts):
    '''
    Function to search dataframe only for unique tower entries in column "tow_string" where all columns containing
    text in "column_texts" contain non-zero values.

    :param pd.DataFrame df: Dataframe to be sorted.
    :param str tow_string: Column to find unique values for
    :param str set_string: Column to find unique values for
    :param float set_factor: Factor that average loading of Phase wires / EW is assumed
    :param list column_texts: Text to search for in matching columns

    :return: Dataframe storing only unique row entries in "column_string" where all columns with entries from
    "column_texts" are non-zero
    :rtype: pd.DataFrame
    '''
    'Store original column ids'
    columns_ids = df.columns
    columns = [[x for x in list(df.columns) if y in x] for y in column_texts]
    col_names = []

    'Add resultant to ahead / back loading'
    for i, column_matches in enumerate(columns):
        col_names.append(f"susp{i + 1}")
        df[col_names[i]] = df.apply(lambda x: np.sqrt(sum([x[col] ** 2 for col in column_matches])), axis=1)

    'Check to store only row entries that are loaded both from ahead and back spans'
    df["match"] = df.apply(lambda x: 1 if all([1 if x[y] != 0 else 0 for y in col_names]) else 0, axis=1)

    'Perform second check, to identify attachment points with substantially lower loads, which are assumed as EW'
    tow_sets = {}
    for tow_id, item in df.groupby([tow_string]):
        set_sums = {}
        'Find average resultant load per attachment point'
        for set_id, item_set in item.groupby([set_string]):
            set_sums[set_id] = item_set.apply(lambda x: sum([x[y] for y in col_names]), axis=1).mean()
        'Find sets to remove, if their average is lower by more than "set_factor"'
        remove_sets = list(
            {
                x: y for x, y in set_sums.items() if (np.average(list(set_sums.values())) / y) > set_factor
            }.keys()
        )
        keep_sets = [x for x in item.loc[:, set_string].unique() if x not in remove_sets]
        tow_sets[tow_id] = keep_sets

    df["keep"] = df.apply(
        lambda x: 1 if (x[tow_string] in tow_sets.keys()) and (x[set_string] in tow_sets[x[tow_string]]) else 0,
        axis=1
    )
    'Keep only towers and sets that match both checks, i.e. if both ahead / back loading and loads larger'
    'than a fraction of the average attachment point loading'
    df = df.loc[(df.loc[:, "match"] == 1) & (df.loc[:, "keep"] == 1), :]
    df = df.loc[:, columns_ids]

    return df


def dataframe_add_swing_angle(df):
    df["swing_angle"] = df.apply(lambda x: np.degrees(np.arctan2(x["transversal"], x["vertical"])), axis=1)
    return df
