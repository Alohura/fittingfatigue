import numpy as np
import pandas as pd
from scipy import optimize
from collections import OrderedDict
from copy import copy

# Ryan's taking over

def split_word_to_list(input_string):
    return [str_char for str_char in input_string]


def intersect_of_2_lists(list1, list2):
    return list(set(list1).intersection(list2))


def dataframe_filter_only_above_nas(df, header_column):
    '''
    Filter to store only relevant list values, i.e. only part of dataframe above first nan entry

    :param pd.DataFrame df: Dataframe to be filtered
    :param int header_column: Column in dataframe to search for NaN values

    :return: Filtered dataframe
    :rtype: pd.DataFrame
    '''
    na_indexes = list(np.flatnonzero(df.iloc[:, header_column].isna()))  # Find NaN cells to define where lists end

    return df.iloc[0:na_indexes[0], :] if len(na_indexes) > 0 else df


def dataframe_filter_only_columns_without_nas(df):
    '''
    Filter to store only relevant list values, i.e. only part of dataframe above first nan entry

    :param pd.DataFrame df: Dataframe to be filtered

    :return: Filtered dataframe
    :rtype: pd.DataFrame
    '''

    return df.loc[:, ~df.isna().all()]


def friction_torsion_resistance_swivel_rolling(angle, angle_max, force, r_pin):
    '''
    Function returning friction resistance moment due to rolling resistance at swivel / pin interface.

    :param float angle: Angle distance to swivel / pin contact point
    :param float force: Resultant force at insulator attachment point in the longitudinal - vertical plane
    :param float angle_max: Max angle distance to swivel / pin contact point, i.e. angle above which sliding occurs
    :param float r_pin: Radius of swivel pin

    :return: Swivel reaction moment due to reaction pin / swivel rotation resistance
    :rtype: float
    '''

    angles = [abs(np.radians(angle)), abs(np.radians(angle_max))]

    return abs(force) * min(angles) * r_pin


def friction_torsion_resistance_swivel_t1(friction, force, r_out, r_in, lift_off=False):
    '''
    Function returning friction resistance moment due to reaction force at swivel / cleat interface.

    :param float friction: Friction coefficient
    :param float force: Transversal force component at insulator attachment point
    :param float r_out: Outer radius of swivel area in contact with cleat
    :param float r_in: Inner radius of swivel area in contact with cleat
    :param bool lift_off: If transverse force and swivel configuration leads to lift-off on one cleat side

    :return: Swivel force reaction moment due to reaction force at swivel / cleat interface
    :rtype: float
    '''

    t_friction = 2. * friction * abs(force) * (r_out ** 3 - r_in ** 3) / (r_out ** 2 - r_in ** 2) / 3.
    if lift_off:
        t_friction = friction * abs(force) * r_out - t_friction

    return t_friction


def friction_torsion_resistance_swivel_t2(friction, f_y, f_z, b_swivel, h_swivel, r_out, r_in):
    '''
    Function returning friction resistance moment due to force couple in set up in swivel in case of transversal force
    components. The force couple is assumed to act such that the effective T1 moment arm is "r_out".
    Returns value non-zero value only if reaction force couple is of opposing sign.

    :param float friction: Friction coefficient
    :param float f_y: Transversal force component at insulator attachment point
    :param float f_z: Vertical force component at insulator attachment point
    :param float b_swivel: Width of swivel
    :param float h_swivel: Height of swivel, from pin bolt centre line to cold end attachment centre point
    :param float r_in: Inner radius of swivel area in contact with cleat
    :param float r_out: Outer radius of swivel area in contact with cleat

    :return: Swivel force reaction moment due to reaction force couple
    :rtype: float
    '''

    r_y = (f_z / 2. - abs(f_y) * h_swivel / b_swivel)

    return friction_torsion_resistance_swivel_t1(friction, f_y, r_out, r_in, True) if r_y < 0. else 0.


def friction_torsion_resistance_swivel_t2_old(friction, f_y, f_z, b_swivel, h_swivel, r_pin):
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

    r_y = (f_z / 2. - abs(f_y) * h_swivel / b_swivel)

    return 2. * friction * abs(r_y) * r_pin if r_y < 0. else 0.


def friction_moment_at_critical_section_swivel(f_x, force_arm, fraction_to_critical_section, friction_moment):
    '''
    Function finding the moment at the selected insulator cold end section to be evaluated for bending stresses

    :param float f_x: Longitudinal swivel attachment force
    :param float force_arm: Distance from swivel pin bolt centre line to effective force arm. Can be further out than
    clevis / ball contact, depending on friction in the insulator ball links
    :param float fraction_to_critical_section: Distance (pin -> critical section) / (effective force arm)
    :param float friction_moment: Friction resistance moment (T1 + T2 + T3)

    :return: Moment at critical section
    :rtype: float
    '''

    m_applied = abs(f_x) * force_arm * fraction_to_critical_section
    m_friction = friction_moment * fraction_to_critical_section

    return m_applied if m_applied < m_friction else m_friction


def stress_stem_roark_17_old(f_axial, m_bending, d_outer, d_inner, r_notch):
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
        stress_axial = stress_roark_17a(d_outer, d_inner, r_notch) * abs(f_axial)
    if abs(m_bending) > 0.:
        stress_bending = stress_roark_17b(d_outer, d_inner, r_notch) * abs(m_bending)

    return stress_axial + stress_bending


def stress_stem_roark_17(f_axial, m_bending, scf_a, scf_b, d_outer, d_inner):
    '''
    Function to convert force and bending actions to stress at a stem diameter transition

    :param float f_axial: Axial force
    :param float m_bending: Bending moment
    :param float scf_a: Axial stress concentration factor
    :param float scf_b: Bending stress concentration factor
    :param float d_outer: Stem maximum diameter
    :param float d_inner: Stem minimum diameter
    :param float r_notch: Notch radius at diameter transition

    :return: Stress including SCF's from axial and bending actions
    :rtype: float
    '''
    stress_axial, stress_bending = 0., 0.
    h = (d_outer - d_inner) / 2.
    if abs(f_axial) > 0.:
        stress_axial = scf_a * abs(f_axial) * 4. / (np.pi * (d_outer - 2. * h) ** 2)
    if abs(m_bending) > 0.:
        stress_bending = scf_b * abs(m_bending) * 32. / (np.pi * (d_outer - 2. * h) ** 3)

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

    C1 = [0.927 + 1.149 * root_ratio - 0.086 * ratio, 1.225 + 0.831 * root_ratio - 0.01 * ratio]
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

    return C1[col] + C2[col] * h_d + C3[col] * h_d ** 2 + C4[col] * h_d ** 3


def stress_roark_17a(d_outer, d_inner, r_notch):
    '''
    Roark's Formulas for stress and strain, table 17.1, formula 17a for axial stress.

    :param float d_outer: Stem maximum diameter
    :param float d_inner: Stem minimum diameter
    :param float r_notch: Notch radius at diameter transition

    :return: Axial stress SCF / Area
    :rtype: float
    '''

    h = (d_outer - d_inner) / 2.

    return 4. * scf_roark_17a(d_outer, d_inner, r_notch) / (np.pi * (d_outer - 2. * h) ** 2)


def scf_roark_17b(d_outer, d_inner, r_notch):
    '''
    Roark's Formulas for stress and strain, table 17.1, formula 17b for bending stress

    :param float d_outer: Stem maximum diameter
    :param float d_inner: Stem minimum diameter
    :param float r_notch: Notch radius at diameter transition

    :return: Bending stress SCF
    :rtype: float
    '''
    h = (d_outer - d_inner) / 2.
    ratio = h / r_notch
    h_d = 2. * h / d_outer
    root_ratio = np.sqrt(ratio)

    C1 = [0.927 + 1.149 * root_ratio - 0.086 * ratio, 1.225 + 0.831 * root_ratio - 0.01 * ratio]
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

    return C1[col] + C2[col] * h_d + C3[col] * h_d ** 2 + C4[col] * h_d ** 3


def stress_roark_17b(d_outer, d_inner, r_notch):
    '''
    Roark's Formulas for stress and strain, table 17.1, formula 17b for bending stress

    :param float d_outer: Stem maximum diameter
    :param float d_inner: Stem minimum diameter
    :param float r_notch: Notch radius at diameter transition

    :return: Bending stress SCF / section modulus
    :rtype: float
    '''

    h = (d_outer - d_inner) / 2.

    return 32. * scf_roark_17b(d_outer, d_inner, r_notch) / (np.pi * (d_outer - 2. * h) ** 3)


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


def dataframe_remove_columns(df, column_texts):
    '''
    Function to remove columns containing certain texts

    :param pd.DataFrame df: Dataframe to be sorted.
    :param list column_texts: Text to search for in matching columns

    :return: Dataframe without columns containing text from "column_texts"
    :rtype: pd.DataFrame
    '''
    columns_ids = df.columns
    columns = [x for x in list(df.columns) for y in column_texts if y in x]
    columns_ids = [x for x in columns_ids if x not in columns]
    df = df.loc[:, columns_ids]

    return df


def dataframe_select_suspension_insulator_sets(df, tow_string, set_string, set_factor, column_texts, ew_sets):
    '''
    Function to search dataframe only for unique tower entries in column "tow_string" where all columns containing
    text in "column_texts" contain non-zero values.

    :param pd.DataFrame df: Dataframe to be sorted.
    :param str tow_string: Column to find unique values for
    :param str set_string: Column to find unique values for
    :param float set_factor: Factor that average loading of Phase wires / EW is assumed
    :param list column_texts: Text to search for in matching columns
    :param list ew_sets: EW set IDs according to Transpower, to be removed from dataframe.

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
    df.loc[:, "match"] = df.apply(lambda x: 0 if x["set_no"] in ew_sets else x["match"], axis=1)

    'Perform second check, to identify attachment points with substantially lower loads, which are assumed as EW'
    df["keep"] = 1
    if set_factor > 1.:
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


def dataframe_add_swing_angle(df, col_label, components=["vertical", "transversal"]):
    df[f"swing_angle_{col_label}"] = df.apply(
        lambda x: np.degrees(np.arctan2(x[components[1]], x[components[0]])), axis=1
    )
    return df


def excel_sheet_find_input_rows_by_string(excel_object, sheet, column, offset=0, start_string="#"):
    '''
    Read PLS_Input sheet to find row indexes which starts with "start_string" in the relevant tables in sheet

    :param pd.ExcelFile excel_object: Input ExcelFile object
    :param str sheet: Name of sheet to read
    :param str column: Column to search for NaN's in
    :param offset: Offset relative to the location of "start_string". - upwards, + downwards
    :param str start_string: String by which headers all start
    
    :return: Lists storing indexes and headers of input 
    :rtype: (list, list)
    '''
    inp_df = excel_object.parse(
        skiprows=0,
        sheet_name=sheet,
    )

    header_rows = inp_df.iloc[:, column].fillna("").str.startswith(start_string).dropna()
    header_row_indexes = list(np.flatnonzero(header_rows) + offset)
    header_row_labels = list(inp_df.iloc[:, column][header_row_indexes])

    return header_row_indexes, header_row_labels


def dataframe_from_excel_object(excel_object, sheet, row, column=0):
    '''


    :param pd.ExcelFile excel_object: Input ExcelFile object
    :param str sheet: Name of sheet to read
    :param int row: First row in Excel sheet to extract data
    :param int column: First column in Excel sheet to extract data

    :return: Dataframe
    :rtype: pd.DataFrame
    '''
    df = excel_object.parse(
        skiprows=row,
        sheet_name=sheet,
    )
    df = dataframe_filter_only_above_nas(df, column)
    df = dataframe_filter_only_columns_without_nas(df)

    return df


def fatigue_damage_from_histogram(stress_max, histogram, sn_curve):
    stresses = [stress_max * x / 100. for x in histogram[1]]
    return sum([n / fatigue_cycle_constant_stress_range_nzs3404(stress, sn_curve) for n, stress in zip(histogram[0], stresses)])


def fatigue_cycle_constant_stress_range_nzs3404(stress, sn_curve):
    '''
    Function to find allowable number of cycles at one stress state, in accordance with NZS 3404, 10.6.

    :param float stress: Stress at one stress state
    :param dict sn_curve: Containing all SN parameters

    :return: Allowable cycles at stress level
    :rtype: float
    '''
    if stress > sn_curve["s2"]:
        return sn_curve["s1"] ** sn_curve["m1"] * sn_curve["n_s1"] / stress ** sn_curve["m1"]
    elif stress > sn_curve["s3"]:
        return sn_curve["s3"] ** sn_curve["m2"] * sn_curve["n_cut_off"] / stress ** sn_curve["m2"]
    else:
        return 1.e99


def dataframe_subtract_one_load_case(df, nominal, parameter, tower_column):
    '''

    :param pd.DataFrame df: Dataframe to be edited
    :param pd.DataFrame nominal: Dataframe containing nominal load case
    :param str parameter:
    :param str tower_column:

    :return:
    :rtype: pd.DataFrame
    '''
    df[f"{parameter}_range"] = df.apply(
        lambda x: abs(x[parameter] - nominal.loc[x[tower_column], parameter]),
        axis=1
    )
    return df


def dataframe_aggregate_by_specific_column(df, group_column):
    '''
    Function aggregate dataframe values with same index into one line, where values in group_column are aggregated in
    list (of length 1 if only 1 entry)

    :param pd.DataFrame df: Dataframe to aggregate
    :param str group_column: Column to group by

    :return: Aggregated dataframe
    :rtype: pd.DataFrame
    '''

    'Find indexes of rows with duplicated entries'
    rows = np.array(df.index)
    index_lists = [list((rows == x).nonzero()[0]) for x in np.unique(rows)]
    'Find indexes of first entry of each unique row entry and store in dataframe'
    rows = list(rows)
    indexes = [rows.index(x) for x in np.unique(rows)]
    df_new = df.iloc[indexes, :]
    'Reset indexes (to integers starting from 0) such that indexes can be used directly '
    'to look up from dataframe (index row)'
    df = df.reset_index()
    df_new = df_new.reset_index()
    columns = list(df.columns)
    'Add duplicated information to "group_column" for each unique row entry'
    df_new.loc[:, group_column] = df_new.apply(
        lambda x: list(df.iloc[index_lists[x.name], columns.index(group_column)]),
        axis=1
    )
    df_new = df_new.set_index(columns[0])

    return df_new


def dataframe_add_nominal_values(df, lc_nom, col_search="lc_description", set_column="line_tow_set"):
    '''
    Function to add loads for nominal load case for each entry.

    :param pd.DataFrame df: Dataframe to be processed
    :param pd.DataFrame lc_nom: Column in which to search for nominal load case
    :param str col_search:
    :param str set_column: Identifier for column containing tower set numbers / IDs for a line

    :return: Dataframe with all information necessary for processing fatigue damage, and nominal dataframe
    :rtype: tuple
    '''

    df_nominal = df.loc[df.loc[:, col_search] == lc_nom, :]
    df_nom_dict = df_nominal.set_index(set_column).transpose().to_dict()

    df["f_long_nom"] = df.loc[:, set_column].map(lambda x: df_nom_dict[x]["longitudinal"])
    df["f_trans_nom"] = df.loc[:, set_column].map(lambda x: df_nom_dict[x]["transversal"])
    df["f_vert_nom"] = df.loc[:, set_column].map(lambda x: df_nom_dict[x]["vertical"])

    return df, df.loc[df.loc[:, col_search] == lc_nom, :]


def dataframe_add_nominal_values_old(df, df_nom, set_column="set_no", tow_column="structure_number"):
    '''
    Function to add loads for nominal load case for each entry.

    :param pd.DataFrame df: Dataframe to be processed
    :param pd.DataFrame df_nom: Dataframe containing values for nominal load case
    :param str set_column: Identifier for column containing set numbers / IDs
    :param str tow_column: Identifier for column containing tower numbers / IDs

    :return: Dataframe with all information necessary for processing fatigue damage
    :rtype: pd.DataFrame
    '''
    for set_no, item in df.groupby([set_column]):
        'Find dataframe entries for nominal case'
        item_nominal = df_nom.loc[
                       df_nom.loc[:, set_column] == set_no, :
                       ].set_index(tow_column)

        'Add nominal values'
        item["f_long_nom"] = item.apply(lambda x: item_nominal.loc[x[tow_column], "longitudinal"], axis=1)
        item["f_trans_nom"] = item.apply(lambda x: item_nominal.loc[x[tow_column], "transversal"], axis=1)
        item["f_vert_nom"] = item.apply(lambda x: item_nominal.loc[x[tow_column], "vertical"], axis=1)

        if "df_return" not in locals():
            df_return = item
        else:
            df_return = pd.concat([df_return, item])

    return df_return


def lists_compare_contents(list1, list2):
    '''
    Function to compare lists and return list of strings describing each entry where the list differ

    :param list list1:
    :param list list2:

    :return: List of strings
    :rtype: (list, list)
    '''

    item_list = [x for x in list1 if x not in list2]
    string_list = [f"File: '{x}' not in input list, please update.\n" for x in item_list]

    return item_list, string_list


def file_objects_defined_in_input_file(file_objects, input_list, exit_code=True):
    '''
    Function to check if list of file objects to be processed are specified in input list. Exit if not, with
    printout of missing objects in input list.

    :param list file_objects: List of file objects to be processed
    :param list input_list: Input list connecting file objects with IDs specified in input sheet
    :param bool exit_code: Specify if code to exit or continue running

    '''

    file_list, string_list = lists_compare_contents(file_objects, input_list)
    if len(file_list) > 0:
        if exit_code:
            print(exit_code)
            print("".join(string_list))
            exit()


def list_items_move(input_list, sorting_items):
    '''
    Function to reorganize input list based on sorting items in separate list
    [item_to_move (str), item_to_replace (str), remove (bool)].

    :param list input_list: List to be reorganized
    :param list sorting_items: List with entries to search for and move, with 3rd item deciding if replaced value shall
    be stored or not

    :return: Reorganized list
    :rtype: list
    '''
    list_edit = input_list.copy()
    for item_from, item_to, item_replace in sorting_items:
        # index_from, index_to = input_list.index(item_from), input_list.index(item_to)

        item_to_fill, item_from_fill = [], [item_from]
        index_to = list_edit.index(item_to)
        index_from = list_edit.index(item_from)

        if item_replace:
            list_edit[index_to] = item_from
            list_edit[index_from] = item_to
        else:
            list_edit = list_edit[:index_to] + [item_from] + list_edit[index_to:]
            if index_from > index_to:
                list_edit = list_edit[:index_from+1] + list_edit[index_from+2:]
            else:
                list_edit = list_edit[:index_from] + list_edit[index_from + 1:]

        # a=1
        # if item_replace:
        #     list_edit[index_to]
        #     item_to_fill = [item_to]
        #     list_edit.pop(index_to)
        # list_edit.pop(index_from)
        # list_edit = list_edit[:index_to] + item_from_fill + list_edit[index_to:]
        #
        # # index_from = list_edit.index(item_from)
        # a = list_edit.pop(index_from+1)
        # list_edit = list_edit[:index_from] + item_to_fill + list_edit[index_from+1:]

    return list_edit


def convert_names(x, convert):
    return convert[x] if x in convert else x


def sn_from_ca_values(ca, sn_curves, default_curve=-1):
    '''
    Function looking up SN parameters given a condition assessment (CA) value

    :param float ca: Transpower condition assessment value, between 0 - 100
    :param dict sn_curves: Dictionary storing SN parameters for all defined SN curves
    :param int default_curve: 0 is highest SN curve, 1 second highest etc., whereas -1 is last curve

    :return:
    :rtype: dict
    '''
    curve_keep = list(sn_curves.keys())[default_curve]
    for curve, sn in sn_curves.items():
        if sn["ca_list"][0] < float(ca) <= sn["ca_list"][1]:
            curve_keep = curve

    return curve_keep


def sn_curve_add_ca_list(sn_curves):
    '''
    Function to split CA value string into two-value list

    :param dict sn_curves:

    :return: SN curve with [to, from] CA value list
    :rtype: dict
    '''

    for curve, sn in sn_curves.items():
        ca_vals = [float(x) for x in sn["ca"].replace(" ", "").split("-")]
        if sum(ca_vals) == 0:
            ca_vals = [-2., -1.]
        sn_curves[curve].update({"ca_list": ca_vals})

    return sn_curves


def ca_and_set_value_from_dict(key, inp_dict, return_column, default_value):
    '''
    Function to look up set and condition assessment (CA) information

    :param str key: Key in dictionary, for instance "line_structure"
    :param dict inp_dict: Dictionary containing necessary CA or set info
    :param str return_column: What column to look up values from
    :param float default_value: Default value (CA, set) in case key is not in inp_dict

    :return: Condition assessment number
    :rtype: float
    '''

    return inp_dict[key][return_column] if key in inp_dict else default_value


def lists_add_and_flatten(lists_input):
    '''

    :param list lists_input: List of lists to be added and flattened

    :return: Sum of input lists, flattened in case of 2D lists
    :rtype: list
    '''
    'Initialize columns to be read and stored'
    columns_start = ["line_id", "structure_number"]
    columns_end = ["circuit1", "circuit2"]
    list_return = []
    for lst in lists_input:
        list_return += lst

    return remove_duplicates_from_list(
        [x for x in list_return if type(x) is not list] + [x for x in list_return if type(x) is list][0]
    )


def remove_duplicates_from_list(input_list):
    '''
    Function to remove duplicates in list

    :param list input_list:
    :return: Reorganized list
    :rtype: list
    '''

    return list(OrderedDict.fromkeys(input_list))


def check_overlap_between_two_lists(lst1, lst2):  # , line_id):
    '''

    :param list lst1:
    :param list lst2:

    :return: 0 if no overlap between two lists, 1 if overlap
    :rtype: int
    '''

    set1, set2 = set(lst1), set(lst2)
    diff = set1 - set2

    return 0 if len(diff) == len(set1) else 1


def find_max_value_all_types(val1, val2):
    '''
    Function to find maximum value, and if val1 and val2 of different types, converted to strings.

    :param val1:
    :param val2:

    :return: Maximum value
    '''
    ret_list = [val1, val2]
    if type(val1) == type(val2):
        return max(val1, val2)
    else:
        return max([str(x) for x in ret_list])


def dict_of_dicts_max_item(dct, item_key):
    '''
    Function to return dictionary item with largest value "item_key"

    :param dict dct: Dictionary to be evaluate
    :param str item_key: Item key to compare

    :return: Item of dictionary of dictionaries
    :rtype: dict
    '''
    max_val = -1.e99
    store_item = {}
    for key, item in dct.items():
        if item[item_key] > max_val:
            store_item = item
            max_val = item[item_key]

    return {"D.F.": store_item}


def swivel_rotation_angle(angle, d_pin, d_hole, sliding_factor):
    '''
    Function to use pin and hole geometry together with applied load angle to calculate how the point of contact
    changes between the two cylinders. The change of contact is represented by an angle, measured about the centre
    of rotation of the outer cylinder.

    :param float angle: Load angle
    :param float d_pin: Pin diameter, i.e. inner cylinder
    :param float d_hole: Hole diameter, i.e. outer cylinder
    :param float sliding_factor: Factor to account for inner cylinder not rolling perfectly inside outer cylinder, i.e.
    it accounts for some sliding below the friction limit.

    :return: Angle to new contact point between the two cylinders
    :rtype: float
    '''
    'Factor to account for how contact point moves as a function of inner cylinder rotation'
    rolling_multiplier = d_pin / (2. * (d_hole - d_pin))

    return rolling_multiplier * np.abs(angle) / sliding_factor


def swivel_rotation_angle_max(friction, d_pin, d_hole, sliding_factor):
    '''
    Function to calculate maximum rotation of one cylinder inside another before the inner cylinder starts to slide.

    :param float friction: Friction factor
    :param float d_pin: Pin diameter, i.e. inner cylinder
    :param float d_hole: Hole diameter, i.e. outer cylinder
    :param float sliding_factor: Factor to account for inner cylinder not rolling perfectly inside outer cylinder, i.e.
    it accounts for some sliding below the friction limit.

    :return: Angle at which inner cylinder overcomes friction and starts to slide inside the other cylinder.
    :rtype: float
    '''
    'Factor to account for how contact point moves as a function of inner cylinder rotation'
    rolling_multiplier = d_pin / (2. * (d_hole - d_pin))

    return np.degrees(np.arctan(friction) / (1. - 1. / (rolling_multiplier * sliding_factor)))


def insulator_to_cantilever_beam_length(t_friction, angle, h, h_min, EI, force, power_factor):
    '''
    Function to convert an insulator (soft cantilever beam) loaded in both transversal (x) and longitudinal (y)
    directions to an equivalent beam only loaded in the x direction.

    :param float t_friction: Torsion resistance moment
    :param float angle: Friction factor
    :param float h: Length of insulator assembly
    :param float h_min: Minimum length of insulator assembly outside svivel + clevis length, usually roughly 100 mm.
    :param float EI: Bending stiffness, typically taken based on insulator stem diameter
    :param float force: Applied tensile load in insulator (resultant).
    :param float power_factor: Factor describing moment evolution along effective cantilever beam
    (length swivel -> effective rotation centre). 1 = linear, 2=square root function, 1.5=good fit with test results

    :return: Length of equivalent cantilever beam.
    :rtype: float
    '''

    a_rad = np.abs(np.radians(angle))

    angle_max = optimize.root_scalar(
        force_arm_minimize_function,
        x0=0.,
        x1=np.radians(5.),
        args=(
            t_friction, force, EI, h, power_factor
        ),
        maxiter=20,
        method='secant'  # method='brentq'
    )

    if not angle_max.converged:
        h_eff = h_min
    else:
        if angle_max.root < a_rad:
            a_rad = angle_max.root
        h_eff = h / np.cos(a_rad) * (1. - 1. / (3. * (EI / (force * h ** 2) + np.cos(a_rad) /
                                                      (2. + 1. / power_factor))))

    return max(h_eff, h_min)  # h_eff if h_eff > h_min else h_min


def force_arm_minimize_function(angle, torsion, force, EI, h, power_factor):
    '''
    Function used to find the force angle which results in torsion moment "torsion".
    The variable to miminize the expression is "angle". Once a value of zero is returned, force * angle = torsion

    :param float torsion: Torsion moment to which force times displacement shall equal
    :param float force: Applied resultant force
    :param float angle: Angle of applied force [radians]
    :param float EI: Bending stiffness of beam
    :param float h: Height of beam
    :param float power_factor: Power factor to define moment distribution from the longitudinal force component

    :return: Function value that shall be minimized
    :rtype: float
    '''

    return torsion - cantilever_beam_deformation(force, angle, EI, h, power_factor) * force


def cantilever_beam_deformation(force, angle, EI, h, power_factor):
    '''
    Function to calculate deformation for cantilever beam with force applied at an angle, i.e. with two force
    components.

    :param float force: Applied resultant force
    :param float angle: Angle of applied force [radians]
    :param float EI: Bending stiffness of beam
    :param float h: Height of beam
    :param float power_factor: Power factor to define moment distribution from the longitudinal force component

    :return: Beam deformation
    :rtype: float
    '''

    return h * np.tan(angle) * (1. - 1. / (3. * (EI / (force * h ** 2) + abs(np.cos(angle)) / (2. + 1. / power_factor))))


def bending_stiffness_cylinder(e_modulus, d):
    '''
    
    
    :param float e_modulus: Elastic modulus 
    :param float d: Outer diameter of cylinder
    
    :return: Bending stiffness, EI
    :rtype: float
    '''

    return e_modulus * np.pi / 64. * d ** 4


def notch_factor_neuber_beta(ys):
    """
    Beta factor for use in Neuber notch factor for steels, taken from N. E. Dowling 'Mechanical Behaviour of
    Materials', Third Edition, formula 10.11 in section 10.3

    :param float ys: Yield strength, in MPa
    :return: Beta factor, to be used with notch radius in calculating notch sensitivity factor, in mm.
    """
    return 10. ** (-1.079 * ys ** 3 * 1.e-9 + 2.74 * ys ** 2 * 1.e-6 - 3.74 * ys * 1.e-3 + 0.6404)


def notch_factor_neuber_q(r, beta):
    """
    Notch sensitivity factor, q, for use in Neuber notch factor for steels, taken from N. E. Dowling
    'Mechanical Behaviour of Materials', Third Edition, formula 10.10 in section 10.3

    :param float r: Notch radius, in mm
    :param float beta: Material sensitive factor
    :return: Notch sensitivity factor, q
    """

    return 1. / (1. + np.sqrt(beta / r))
