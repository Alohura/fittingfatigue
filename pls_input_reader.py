import pandas as pd
import numpy as np
import os
from pathlib import Path
from util_functions import *  # split_word_to_list, intersect_of_2_lists, dataframe_filter_only_above_nas


def main():
    path = r"C:\Users\AndreasLem\OneDrive - Groundline\Projects\NZ-6500 - Clevis fatigue load evaluations"
    input_file = "ClevisFatigue_Input.xlsx"
    results_file = "raft_moments_test.xlsx"
    TowerColdEndFittingFatigue.fatigue_damage(path, input_file, results_file)


class TowerColdEndFittingFatigue:
    def __init__(self, path, file_name, results_file):
        self.path_input = path
        self.file_name = file_name
        self.results_file_name = results_file
        'Dictionary to convert from naming in Excel sheet to format consistent with python code'
        self.convert_names_pls = {
            'Row #': 'row',
            'Str. No.': 'structure_number',
            'Str. Name': 'structure_name',
            'LC #': 'lc',
            'WC #': 'wc',
            'Load Case Description': 'lc_description',
            'Set No.': 'set_no',
            'Phase No.': 'phase_no',
            'Attach. Joint Labels': 'joint',
            'Structure Loads Vert. (N)': 'vertical',
            'Structure Loads Trans. (N)': 'transversal',
            'Structure Loads Long. (N)': 'longitudinal',
            'Loads from back span Vert. (N)': 'vertical_back',
            'Loads from back span Trans. (N)': 'transversal_back',
            'Loads from back span Long. (N)': 'longitudinal_back',
            'Loads from ahead span Vert. (N)': 'vertical_ahead',
            'Loads from ahead span Trans. (N)': 'transversal_ahead',
            'Loads from ahead span Long. (N)': 'longitudinal_ahead',
            'Warnings': 'warnings'
        }
        self.convert_names_general = {
            'Height [mm]:': 'height',
            'Width [mm]:': 'width',
            'Outer diameter [mm]:': 'd_outer',
            'Inner diameter [mm]:': 'd_inner',
            'Pin diameter [mm]:': 'd_pin',
            'Stem diameter [mm]:': 'd_stem',
            'Transition radius [mm]:': 'r_notch',
            'Ball effective diameter [mm]:': 'd_ball',
            'Friction coefficient [-]:': 'friction',
            'Force arm distance [mm]:': 'force_arm',
            'Unit:': 'unit',
            'Load case at rest:': 'lc_rest',
            'Line name:': 'line_id',
            'File name:': 'file_name'
        }
        self.unit_conversion = {
            'mm': 0.001,
            'cm': 0.01,
            'm': 1.,
            'mm2': 1.e-6,
            'cm2': 1.e-4,
            'm2': 1.
        }
        # columns_keep = list(self.convert_names_pls.values())[:-7]
        columns_keep = list(self.convert_names_pls.values())[:-1]
        self.convert_names_pls = {x: y for x, y in self.convert_names_pls.items() if y in columns_keep}
        self.convert_names_pls_back = {y: x for x, y in self.convert_names_pls.items()}
        'Setup for Excel and csv input'
        self.excel_object = self._init_excel_object()
        self.csv_objects = self._init_csv_objects()
        'Input variables'
        self.swivel_info = {}
        self.clevis_info = {}
        self.general_info = {}
        self.line_file_name_info = {}
        self.sn_info = {}
        self.sn_input_columns = ['Curve', 'm1', 'm2', 's1', 's2', 's3', 'n_s1', 'n_cut_off', 'CA']

    @classmethod
    def fatigue_damage(cls, path, input_file, results_file):
        cls_obj = cls(path, input_file, results_file)
        df = cls_obj._read_attachment_loads()
        df = cls_obj._ca_value_map_to_sn_detail(df)
        df = cls_obj._calculate_fatigue_damage(df)
        a=1

    def _calculate_fatigue_damage(self, df):
        '''

        :param pd.DataFrame df:

        :return:
        :rtype: pd.DataFrame
        '''
        histogram = stress_histogram_en_1991_1_4(300, 2, 3)

        df["damage"] = df.apply(
            lambda x: fatigue_damage_from_histogram(x["stress_range"], histogram, self.sn_info[x["sn_curve"]]),
            axis=1
        )

        return df.sort_values(by="damage", ascending=False)

    def _ca_value_map_to_sn_detail(self, df):
        '''

        :param df:

        :return:
        :rtype: pd.DataFrame
        '''
        df["sn_curve"] = "160"
        return df

    def _read_attachment_loads(self):
        '''

        :return:
        :rtype: pd.DataFrame
        '''
        # cls_obj = cls(path, input_file, results_file)
        'Find position of data entries'
        header_indexes, header_labels = excel_sheet_find_input_rows(self.excel_object, "GeneralInput", 0)
        'Get input for necessary for friction calculations'
        self.swivel_info = self._get_input_info("GeneralInput", header_indexes[0] + 2, ["Parameter", "Value"])
        self.clevis_info = self._get_input_info("GeneralInput", header_indexes[1] + 2, ["Parameter", "Value"])
        self.general_info = self._get_input_info("GeneralInput", header_indexes[2] + 2, ["Parameter", "Value"])
        self.sn_info = self._get_input_info(
            "GeneralInput",
            header_indexes[3] + 2,
            self.sn_input_columns,
            True
        )
        self.sn_info = {f"{x}": y for x, y in self.sn_info.items()}
        self.line_file_name_info = self._get_input_info("FileInput", 0, ["Line name:", "File name:"])
        self.line_file_name_info = {y: x for x, y in self.line_file_name_info.items()}

        'Read loads from csv files. calculate stem stresses and store maximum stress ranges'
        df = self._dataframe_setup()

        'Add swing angles'
        df = self._add_swing_angles(df)

        return df

    def _add_swing_angles(self, df):
        '''

        :param pd.DataFrame df:

        :return:
        :rtype: pd.DataFrame
        '''
        df = dataframe_add_swing_angle(df, "trans", ["vertical", "transversal"])
        df = dataframe_add_swing_angle(df, "trans_orig", ["f_vert_nom", "f_trans_nom"])
        df["swing_angle_trans_range"] = df.loc[:, "swing_angle_trans"] - df.loc[:, "swing_angle_trans_orig"]
        df = dataframe_add_swing_angle(df, "long", ["vertical", "longitudinal"])
        df = dataframe_add_swing_angle(df, "long_orig", ["f_vert_nom", "f_long_nom"])
        df["swing_angle_long_range"] = df.loc[:, "swing_angle_long"] - df.loc[:, "swing_angle_long_orig"]

        return df

    def _get_input_info(self, sheet, header_row, header_columns, transpose=False):
        '''
        Function to read swivel, clevis or friction info, as specified in Excel sheet, and store in self

        :param str sheet: Sheet name which data shall be extracted from
        :param int header_row: Index to specify header row of Dataframe
        :param list header_columns: Columns specifying header column of Dataframe
        :param bool transpose: Transpose dictionary in case of S-N data

        :return: Information for friction calculations
        :rtype: dict
        '''
        df = self.excel_object.parse(
            index_col=None,
            skiprows=header_row,
            sheet_name=sheet,
            usecols=header_columns
        )
        df = dataframe_filter_only_above_nas(df, header_columns[1])
        df.loc[:, header_columns[0]] = df.loc[:, header_columns[0]].map(
            lambda x: self.convert_names_general[x] if x in self.convert_names_general else x
        )
        df = df.set_index([header_columns[0]])
        df.columns = df.columns.map(lambda x: x.lower())
        return df.transpose().to_dict() if transpose else df.to_dict()[header_columns[1].lower()]

    def _dataframe_setup(self):
        '''

        :return:
        :rtype: pd.DataFrame
        '''
        for file_name in self.csv_objects:
            df = pd.read_csv(os.path.join(self.path_input, file_name))
            'Convert names to code names, ref. dictionaries for converting names in __init__'
            df.columns = df.columns.map(lambda x: self.convert_names_pls[x] if x in self.convert_names_pls else x)
            df = df.loc[:, list(self.convert_names_pls.values())]
            'Sort only suspension towers'
            df = dataframe_select_suspension_insulator_sets(
                df, "structure_number", "set_no", 3., ["ahead", "back"]
            )

            df["resultant"] = df.apply(
                lambda x: np.sqrt(x["longitudinal"] ** 2 + x["transversal"] ** 2 + x["vertical"] ** 2),
                axis=1
            )
            df["line_id"] = self.line_file_name_info[file_name]
            df = self._add_swivel_torsion_moments(df)
            df = self._add_stem_stresses(df)
            df = self._maximum_force_and_stress_range(df)

        return df

    def _maximum_force_and_stress_range(self, df, tow_column="structure_number"):
        '''
        Function to find clevis stem stresses based on force, moment and SCF

        :param pd.DataFrame df: Dataframe containing all force information
        :param str tow_column: Identifier for column containing tower numbers

        :return: Dataframe with stem stresses
        :rtype: pd.DataFrame
        '''
        lc_rest = self.general_info["lc_rest"]
        for set_no, item in df.groupby(["set_no"]):
            'Store dataframe entries for nominal case'
            df_nominal = item.loc[item.loc[:, "lc_description"] == lc_rest, :].set_index(tow_column)

            'Convert to stress ranges by subtracting the EDS (no wind) weather case'
            item = dataframe_subtract_one_loadcase(item, df_nominal, "stress_axial", tow_column)
            item = dataframe_subtract_one_loadcase(item, df_nominal, "stress_bending", tow_column)

            'Assume worst case where bending completely reverses, i.e. multiply MW - EDS by 2'
            item["stress_range"] = item.loc[:, "stress_axial_range"] + 2. * item.loc[:, "stress_bending_range"]
            'Find maximum stress ranges per tower'
            tow_list = list(item.loc[:, tow_column].unique())
            indx_list = []
            for tow in tow_list:
                indx_list.append(item.loc[item.loc[:, tow_column] == tow, "stress_range"].idxmax())

            item_max = item.loc[indx_list, :]

            'Add nominal values'
            item_max["f_long_nom"] = item_max.apply(lambda x: df_nominal.loc[x[tow_column], "longitudinal"], axis=1)
            item_max["f_trans_nom"] = item_max.apply(lambda x: df_nominal.loc[x[tow_column], "transversal"], axis=1)
            item_max["f_vert_nom"] = item_max.apply(lambda x: df_nominal.loc[x[tow_column], "vertical"], axis=1)

            if "df_return" not in locals():
                df_return = item_max
            else:
                df_return = pd.concat([df_return, item_max])

        return df_return

    def _add_stem_stresses(self, df):
        '''
        Function to find clevis stem stresses based on force, moment and SCF

        :param pd.DataFrame df: Dataframe containing all force information

        :return: Dataframe with stem stresses
        :rtype: pd.DataFrame
        '''
        d_out = self.clevis_info["d_ball"] * self.unit_conversion[self.general_info["unit"]]
        d_in = self.clevis_info["d_stem"] * self.unit_conversion[self.general_info["unit"]]
        r_notch = self.clevis_info["r_notch"] * self.unit_conversion[self.general_info["unit"]]
        df["stress_axial"] = df.apply(
            lambda x: stress_stem_roark_17(x["resultant"], 0., d_out, d_in, r_notch) / 1.e6,
            axis=1
        )
        df["stress_bending"] = df.apply(
            lambda x: stress_stem_roark_17(0., x["m_section"], d_out, d_in, r_notch) / 1.e6,
            axis=1
        )
        df["stress"] = df.loc[:, "stress_axial"] + df.loc[:, "stress_bending"]

        return df

    def _add_swivel_torsion_moments(self, df):
        '''
        Function to find swivel torsion resistance moments based on applied force angle

        :param pd.DataFrame df: Dataframe containing all force information

        :return: Dataframe with torsion resistance moments
        :rtype: pd.DataFrame
        '''
        'Input parameters to friction calculations'
        my = self.general_info["friction"]
        r_out = self.swivel_info["d_outer"] * self.unit_conversion[self.general_info["unit"]] / 2.
        r_in = self.swivel_info["d_inner"] * self.unit_conversion[self.general_info["unit"]] / 2.
        width = self.swivel_info["width"] * self.unit_conversion[self.general_info["unit"]]
        height_swivel = self.swivel_info["height"] * self.unit_conversion[self.general_info["unit"]]
        height_clevis = self.clevis_info["height"] * self.unit_conversion[self.general_info["unit"]]
        r_pin = self.swivel_info["d_pin"] * self.unit_conversion[self.general_info["unit"]] / 2.
        force_arm = self.general_info["force_arm"] * self.unit_conversion[self.general_info["unit"]]
        fraction_to_section = (height_clevis + height_swivel) / force_arm
        'Calculate friction moments from swivel / cleat interface (T1) and from swivel / pin reaction force couple (T2)'
        df["t1"] = df.apply(
            lambda x: friction_torsion_resistance_swivel_t1(my, x["transversal"], r_out, r_in),
            axis=1
        )
        df["t2"] = df.apply(
            lambda x: friction_torsion_resistance_swivel_t2(
                my, x["transversal"], x["vertical"], width, height_swivel, r_pin
            ),
            axis=1
        )
        df["t_friction"] = df.loc[:, "t1"] + df.loc[:, "t2"]
        df["m_section"] = df.apply(lambda x: friction_moment_at_critical_section_swivel(
            x["longitudinal"],
            force_arm,
            fraction_to_section,
            x["t_friction"]
        ),
                                   axis=1
                                   )

        return df

    def _init_excel_object(self):
        '''
        Open Excel object for read

        :return All Excel sheets stored in object for data extraction
        :rtype pandas.ExcelFile ExcelObj
        '''
        excel_obj = pd.ExcelFile(os.path.join(self.path_input, self.file_name))
        return excel_obj

    def _init_csv_objects(self):
        """
        Function to return all 'csv' files in folder

        :return: 'csv' files to be evaluated
        :rtype: list
        """
        file_names = os.listdir(self.path_input)
        return_names = []
        for file_name in file_names:
            if Path(file_name).suffix.strip(".") == "csv":
                return_names.append(file_name)
        return return_names

    def _write_foundation_forces_to_excel(self, df):
        '''Write to file for all towers'''
        file_name = os.path.join(self.path_input, self.results_file_name)
        counter, counter2 = 0, 0
        with pd.ExcelWriter(file_name) as writer:
            counter =+1
            for tow, res in df.groupby("structure_number"):
                counter2 += 1
                res = res.sort_values(by=["m_tot_res", "foundation"], ascending=False)
                res.columns = res.columns.map(lambda x: self.convert_names_pls_back[x] if x in self.convert_names_pls_back else x)
                res.to_excel(
                    writer,
                    sheet_name=str(tow)
                )


if __name__ == '__main__':
    main()