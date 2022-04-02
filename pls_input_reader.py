import pandas as pd
import numpy as np
import os
from pathlib import Path
from util_functions import *  # split_word_to_list, intersect_of_2_lists, dataframe_filter_only_above_nas


def main():
    path = r"C:\Users\AndreasLem\OneDrive - Groundline\Projects\NZ-6500 - Clevis fatigue load evaluations"
    input_file = "ClevisFatigue_Input.xlsx"
    results_file = "raft_moments_test.xlsx"
    df_loads = TowerLoads.read_attachment_loads(path, input_file, results_file)


class TowerLoads:
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

    @classmethod
    def read_attachment_loads(cls, path, input_file, results_file):
        cls_obj = cls(path, input_file, results_file)
        'Get input for necessary for friction calculations'
        cls_obj.swivel_info = cls_obj._get_friction_info("GeneralInput", 1, ["Parameter", "Value"])
        cls_obj.clevis_info = cls_obj._get_friction_info("GeneralInput", 9, ["Parameter", "Value"])
        cls_obj.general_info = cls_obj._get_friction_info("GeneralInput", 16, ["Parameter", "Value"])
        cls_obj.line_file_name_info = cls_obj._get_friction_info("FileInput", 0, ["Line name:", "File name:"])
        cls_obj.line_file_name_info = {y: x for x, y in cls_obj.line_file_name_info.items()}
        'Read loads from csv files and calculate stem stresses'
        df = cls_obj._dataframe_setup()

        return df

    def _get_friction_info(self, sheet, header_row, header_columns):
        '''
        Function to read swivel, clevis or friction info, as specified in Excel sheet, and store in self

        :param str sheet: Sheet name which data shall be extracted from
        :param int header_row: Index to specify header row of Dataframe
        :param list header_columns: Columns specifying header column of Dataframe

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
        return df.to_dict()[header_columns[1]]

    @classmethod
    def get_maximum_lc_moments(cls, path, file_name, results_file):
        cls_obj = cls(path, file_name, results_file)
        # cls_obj._get_foundation_distance("JointSupportReactions", 1, 5)
        cls_obj._get_foundation_distance("VerificationLoads", 1, 5)
        cls_obj._get_foundation_moment_arms()
        # df = cls_obj._find_most_loaded_lcs_for_raft_foundations("JointSupportReactions", 7)
        df = cls_obj._find_most_loaded_lcs_for_raft_foundations("VerificationLoads", 7)
        cls_obj._write_foundation_forces_to_excel(df)

    def _dataframe_setup(self):
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
            df = dataframe_add_swing_angle(df)
            df = self._add_stem_stresses(df)
            df = self._maximum_stress_range(df)

        return df

    def _maximum_stress_range(self, df):
        '''
        Function to find clevis stem stresses based on force, moment and SCF

        :param pd.DataFrame df: Dataframe containing all force information

        :return: Dataframe with stem stresses
        :rtype: pd.DataFrame
        '''
        lc_rest = self.general_info["lc_rest"]
        for set_no, item in df.groupby(["set_no"]):
            'Convert to stress ranges by subtracting the EDS (no wind) weather case'
            stress_nominal = item.loc[item.loc[:, "lc_description"] == lc_rest, :].set_index("structure_number")
            item["stress_range_axial"] = item.apply(
                lambda x: abs(x["stress_axial"] - stress_nominal.loc[x["structure_number"], "stress_axial"]),
                axis=1
            )
            'Assume worst case where bending completely reverses, i.e. multiply MW - EDS by 2'
            item["stress_range_bending"] = item.apply(
                lambda x: 2. * abs(x["stress_bending"] - stress_nominal.loc[x["structure_number"], "stress_bending"]),
                axis=1
            )
            item["stress_range"] = item.loc[:, "stress_range_axial"] + 2. * item.loc[:, "stress_range_bending"]
            'Find maximum stress ranges per tower'
            tow_list = list(item.loc[:, "structure_number"].unique())
            indx_list = []
            for tow in tow_list:
                indx_list.append(item.loc[item.loc[:, "structure_number"] == tow, "stress_range"].idxmax())

            item_max = item.loc[indx_list, :]
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