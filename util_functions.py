import numpy as np
import pandas as pd
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
