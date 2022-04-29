import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from time import time
from util_functions import *  # split_word_to_list, intersect_of_2_lists, dataframe_filter_only_above_nas

time0 = time()
time1 = time()


def main():
    # path_pls_input = r"C:\temp\clevis"
    # path_ca_input = r"C:\temp\clevis\input"
    path_pls_input = r"C:\Users\AndreasLem\Groundline\NZ-6500 Insulator Cold End Failure Investigation - Documents\03 Operations\04_Analyses\Load input"
    path_ca_input = r"C:\Users\AndreasLem\Groundline\NZ-6500 Insulator Cold End Failure Investigation - Documents\03 Operations\01_Inputs\01_Line Asset"
    input_file = "ClevisFatigue_Input.xlsx"
    results_file = "line_summary.xlsx"
    time0 = time()
    TowerColdEndFittingFatigue.fatigue_damage(path_pls_input, path_ca_input, input_file, results_file, True, False)


class TowerColdEndFittingFatigue:
    def __init__(self, path, path_ca, file_name, results_file):
        self.path_input = path
        self.path_ca = path_ca
        self.file_name = file_name
        self.results_file_name = results_file
        'Dictionaries to convert from naming in Excel sheet to format consistent with python code'
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
            'Parameter': 'parameter',
            'Value': 'value',
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
            'File name:': 'file_name',
            'Insulator sets:': 'insulator'
        }
        self.convert_names_code = {
            "Fatigue damage": "damage",
            "SN Curve (NZS 3404)": "sn_curve",
            "Transversal swing angle [deg]": "swing_angle_trans",
            "Transversal swing angle - at rest [deg]": "swing_angle_trans_orig",
            " Change in transversal swing angle [deg]": "swing_angle_trans_range",
            "Longitudinal swing angle [deg]": "swing_angle_long",
            "Longitudinal swing angle - at rest [deg]": "swing_angle_long_orig",
            " Change in longitudinal swing angle [deg]": "swing_angle_long_range",
            "Resultant force [N]": "resultant",
            "Nominal EDS Longitudinal load [N]": "f_long_nom",
            "Nominal EDS Transversal load [N]": "f_trans_nom",
            "Nominal EDS Vertical load [N]": "f_vert_nom",
            "Stress [MPa]": "stress",
            "Stress - axial [MPa]": "stress_axial",
            "Stress - bending [MPa]": "stress_bending",
            "Stress range [MPa] (Axial + 2 x Bending stress ranges)": "stress_range",
            "Stress range - axial [MPa]": "stress_axial_range",
            "Stress range - bending [MPa]": "stress_bending_range",
            "Friction resistance moment - T1 [Nm]": "t1",
            "Friction resistance moment - T2 [Nm]": "t2",
            "Total friction resistance moment [Nm]": "t_friction",
            "Moment at critical stem section [Nm]": "m_section",
            "SCF axial": "SCF_axial",
            "SCF bending": "SCF_bending"
        }
        self.convert_names_ca_set = {
            'Asset': 'asset',
            'Asset Description': 'asset_description',
            'Device Position': 'device_position',
            'Asset Location': 'asset_location',
            'Year Of Manufacture': 'year',
            'Meter Name': 'position_id',
            'Meter Description': 'position_name',
            'Measurement': 'measurement',
            'Totals': 'total',
            'Measurement Date': 'date',
            'Measurement Comment': 'comment',
            'Circuit': 'circuit',
            'Strain Fwd Std Assy': 'strain_forward',
            'Strain Back Std Assy': 'strain_back',
            'Susp Std Assy': 'suspension',
            'Susp Ins Type Desc': 'suspension_description',
            'Condition Assessment': 'ca',
            'Set Identifyer': 'set_name',
            'Flag if set identified as critical (1) or not (0)': 'critical_set'
        }
        self.convert_names_all = {
            **self.convert_names_pls,
            **self.convert_names_general,
            **self.convert_names_code,
            **self.convert_names_ca_set
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
        self.convert_names_all_back = {y: x for x, y in self.convert_names_all.items()}
        'Input variables'
        self.swivel_info = {}
        self.clevis_info = {}
        self.general_info = {}
        self.line_file_name_info = {}
        self.sn_info = {}
        self.sn_input_columns = ['Curve', 'm1', 'm2', 's1', 's2', 's3', 'n_s1', 'n_cut_off', 'CA']
        'Setup for Excel and csv input'
        self.excel_object = self._init_excel_object()
        self.csv_objects = self._init_csv_files()

    @classmethod
    def fatigue_damage(cls, path, path_ca, input_file, results_file, read_pickle_ca=False, read_pickle=False):
        '''
        Calculate fatigue damage of insulators from PLS-CADD structure attachment loading, input values from
        "input_file", condition assessment (CA) values from files in folder "path_ca" and store to "results_file"

        :param str path: Path with PLS-CADD input and structure loading ".csv" files
        :param str path_ca: Path to folder with OBIEE reports containing CA values and insulator set information
        :param str input_file: Excel input file manually filled out prior to analysis
        :param str results_file: Name of Excel file results are written to
        :param bool read_pickle_ca: Allows for reading of pickled Excel OBIIE input, as reading is quite time consuming
        :param bool read_pickle: Allows for reading of pickled results, as reading is quite time consuming

        '''
        cls_obj = cls(path, path_ca, input_file, results_file)
        cls_obj._excel_input_read()

        'Read pickle or process OBIEE CA and set information dataframe'
        dataframe_process = False
        if read_pickle_ca:
            if os.path.exists(os.path.join(cls_obj.path_ca, "pickle_ca.p")):
                with open(os.path.join(cls_obj.path_ca, "pickle_ca.p"), "rb") as f:
                    [df_ca, df_set] = pickle.load(f)
            else:
                dataframe_process = True
        else:
            dataframe_process = True

        if dataframe_process:
            df_ca, df_set = ReadCAAndSetInformation.setup(cls_obj)
            with open(os.path.join(cls_obj.path_ca, "pickle_ca.p"), "wb") as f:
                pickle.dump([df_ca, df_set], f)

        'Read pickle or process main dataframe'
        dataframe_process = False
        if read_pickle:
            if os.path.exists(os.path.join(cls_obj.path_input, "pickle_df.p")):
                with open(os.path.join(cls_obj.path_input, "pickle_df.p"), "rb") as f:
                    df = pickle.load(f)
            else:
                dataframe_process = True
        else:
            dataframe_process = True

        if dataframe_process:
            df = cls_obj._read_attachment_loads()
            with open(os.path.join(cls_obj.path_input, "pickle_df.p"), "wb") as f:
                pickle.dump(df, f)

        df = cls_obj._line_and_tower_to_set_name_map(df, df_set)
        df = cls_obj._ca_value_map_to_sn_detail(df, df_ca, ca_default=100.)
        df = cls_obj._calculate_fatigue_damage(df)
        cls_obj._write_to_excel(df, "line_id", ["damage"])
        print(f"----------Time to finish of all calculations: {time() - time0}----------")

    def _calculate_fatigue_damage(self, df):
        '''

        :param pd.DataFrame df:

        :return:
        :rtype: pd.DataFrame
        '''

        print(f"----------Time to start of fatigue damage calculation: {time() - time0}----------")
        histogram = stress_histogram_en_1991_1_4(300, 2, 3)

        df["damage"] = df.apply(
            lambda x: fatigue_damage_from_histogram(x["stress_range"], histogram, self.sn_info[x["sn_curve"]]),
            axis=1
        )

        return df.sort_values(by="damage", ascending=False)

    def _line_and_tower_to_set_name_map(self, df, df_set):
        '''
        Function to map line and tower id's to to set names

        :param pd.DataFrame df: Dataframe to be processed
        :param pd.DataFrame df_set: Dataframe containing set names for a range of lines

        :return: Dataframe updated with set names
        :rtype: pd.DataFrame
        '''

        print(f"----------Time to start of mapping of set names: {time() - time0}----------")

        'Set up dictionary to speed up lookup operations'
        df_set = dataframe_aggregate_by_specific_column(df_set, "set_name")

        set_dict = df_set.transpose().to_dict()

        'Map set information from set dataframe'
        df["set_name"] = df.apply(
            lambda x: ca_and_set_value_from_dict(x["line_id"].split("_")[0] + "_" + x["structure_number"], set_dict, "set_name", []),
            axis=1
        )

        'Search for set for set names in critical set list'
        df["critical_set"] = df.apply(
            lambda x: check_overlap_between_two_lists(
                x["set_name"],
                self.line_file_name_info["insulator"][x["line_id"]],
                x["line_id"]
            ),
            axis=1
        )

        return df

    def _ca_value_map_to_sn_detail(self, df, df_ca, ca_default=100., sn_default=1):
        '''
        Function to map CA values to SN details

        :param pd.DataFrame df: Dataframe to be processed
        :param pd.DataFrame df_ca: Dataframe containing CA values for a range of lines
        :param float ca_default: Default CA value, in case it is not given
        :param int sn_default: 0 is highest SN curve, 1 second highest etc., whereas -1 is last curve

        :return: Dataframe updated with SN details
        :rtype: pd.DataFrame
        '''

        print(f"----------Time to start of mapping of CA values: {time() - time0}----------")

        'Set up dictionary to speed up lookup operations'
        df_ca["line_structure"] = df_ca.loc[:, "line_id"] + "_" + df_ca.loc[:, "structure_number"]
        df_ca = df_ca.set_index("line_structure")

        ca_dict = df_ca.transpose().to_dict()

        'Map CA values'
        df["ca"] = df.apply(
            lambda x: ca_and_set_value_from_dict(x["line_id"] + "_" + x["structure_number"], ca_dict, "ca_max", ca_default),
            axis=1
        )
        df.loc[:, "ca"] = df.loc[:, "ca"].astype(float)

        df["sn_curve"] = df.loc[:, "ca"].map(lambda x: sn_from_ca_values(x, self.sn_info, sn_default))

        return df

    def _excel_input_read(self):
        '''
        Read excel input sheet and check verify that all .csv files to be processed are listen in input

        '''
        'Find position of data entries'
        header_indexes, header_labels = excel_sheet_find_input_rows_by_string(self.excel_object, "GeneralInput", 0)

        'Get input for necessary for friction calculations'
        self.swivel_info = self._get_input_info("GeneralInput", header_indexes[0] + 2)["value"]
        self.clevis_info = self._get_input_info("GeneralInput", header_indexes[1] + 2)["value"]
        self.general_info = self._get_input_info("GeneralInput", header_indexes[2] + 2)["value"]
        self.sn_info = self._get_input_info(
            "GeneralInput",
            header_indexes[3] + 2,
            True
        )
        self.sn_info = sn_curve_add_ca_list(self.sn_info)
        self.sn_info = {f"{x}": y for x, y in self.sn_info.items()}

        'Get input on file names and check if files are present'
        self.line_file_name_info = self._get_input_info("FileInput", 0)
        self.line_file_name_info["insulator"] = {
            x: y.replace(" ", "").split(",") for x, y in self.line_file_name_info["insulator"].items()
        }

        'Exit code if input list is not complete'
        file_objects_defined_in_input_file(
            self.csv_objects,
            np.array([[x, y] for x, y in self.line_file_name_info["file_name"].items()])[:, 1],
            False
        )
        self.excel_object.close()

    def _read_attachment_loads(self):
        '''
        Read and store all data relevant for fatigue calculations of insulator cold end clevis

        :return: Dataframe containing all necessary information for fatigue calculations
        :rtype: pd.DataFrame
        '''
        'Read loads from csv files. calculate stem stresses and store maximum stress ranges'
        df = self._dataframe_setup()

        'Add swing angles'
        df = self._add_swing_angles(df)

        return df

    def _add_swing_angles(self, df):
        '''
        Function to calculate swing angles (longitudinal, transversal) based on attachment point loads

        :param pd.DataFrame df: Input dataframe

        :return: Dataframe with swing angle information added
        :rtype: pd.DataFrame
        '''
        df = dataframe_add_swing_angle(df, "trans", ["vertical", "transversal"])
        df = dataframe_add_swing_angle(df, "trans_orig", ["f_vert_nom", "f_trans_nom"])
        df["swing_angle_trans_range"] = df.loc[:, "swing_angle_trans"] - df.loc[:, "swing_angle_trans_orig"]
        df = dataframe_add_swing_angle(df, "long", ["vertical", "longitudinal"])
        df = dataframe_add_swing_angle(df, "long_orig", ["f_vert_nom", "f_long_nom"])
        df["swing_angle_long_range"] = df.loc[:, "swing_angle_long"] - df.loc[:, "swing_angle_long_orig"]

        return df

    def _get_input_info(self, sheet, header_row, transpose=False):
        '''
        Function to read swivel, clevis or friction info, as specified in Excel sheet, and store in self

        :param str sheet: Sheet name which data shall be extracted from
        :param int header_row: Index to specify header row of Dataframe
        :param bool transpose: Transpose dictionary in case of S-N data

        :return: Information for friction calculations
        :rtype: dict
        '''
        df = self.excel_object.parse(
            index_col=None,
            skiprows=header_row,
            sheet_name=sheet
        )
        df = dataframe_filter_only_above_nas(df, 0)
        df = dataframe_filter_only_columns_without_nas(df)

        'Convert to coding names, i.e. without spaces and special signs. See __init__ for conversion dictionary'
        df.iloc[:, 0] = df.iloc[:, 0].map(
            lambda x: self.convert_names_general[x] if x in self.convert_names_general else x
        )

        df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: x.lower() if type(x) is str else x)
        df = df.set_index(df.columns[0])
        df.columns = df.columns.map(
            lambda x: self.convert_names_general[x] if x in self.convert_names_general else x
        ).str.lower()

        return df.transpose().to_dict() if transpose else df.to_dict()

    def _dataframe_setup(self):
        '''
        Read PLS-CADD structure attachment loading reports and process loads. The function stores swivel torsion
        moments, clevis stem stresses and swing angles.

        :return: Dataframe containing all necessary input for fatigue evaluations and control checks
        :rtype: pd.DataFrame
        '''
        print(f"----------Time to start of csv file read: {time() - time0}----------")
        df_list = [
            0 for file_name in self.csv_objects if file_name in self.line_file_name_info["file_name"].values()
        ]
        i = 0
        for file_name in self.csv_objects:
            'Check if CSV file in input list'
            if file_name not in self.line_file_name_info["file_name"].values():
                print(f"File type '{file_name}' not in excel input sheet. Skipped.")
                continue

            'Read CSV file'
            df = pd.read_csv(os.path.join(
                self.path_input, file_name),
                low_memory=False,
                na_values=["nan", np.nan, "NaN", ""]
            )

            'Convert names to code names, ref. dictionaries for converting names in __init__ module'
            df.columns = df.columns.map(lambda x: self.convert_names_pls[x] if x in self.convert_names_pls else x)
            df = df.loc[:, list(self.convert_names_pls.values())]

            'Remove rows with missing structure information'
            df = df.loc[df.loc[df.loc[:, "structure_number"].notna(), :].index]

            'Store only suspension towers and remove columns no longer needed'
            df = dataframe_select_suspension_insulator_sets(
                df, "structure_number", "set_no", 1., ["ahead", "back"], [7, 8]  # Set factor = 1 -> no ew filtering
            )
            df = dataframe_remove_columns(df, ["ahead", "back"])

            'Convert structure_number column, to keep consistent across lines, i.e. in case of "a", "b" extensions etc.'
            df.loc[:, "structure_number"] = df.loc[:, "structure_number"].map(str)
            df.loc[:, "structure_number"] = df.loc[:, "structure_number"].str.lower()

            'Add line ID to uniquely identify each line'
            file_to_line_id = {y: x for x, y in self.line_file_name_info["file_name"].items()}
            df["line_id"] = file_to_line_id[file_name]

            'Process data and calculate moments'
            # df["resultant"] = df.apply(
            #     lambda x: np.sqrt(x["longitudinal"] ** 2 + x["transversal"] ** 2 + x["vertical"] ** 2),
            #     axis=1
            # )
            df["resultant"] = np.sqrt(
                df.loc[:, "longitudinal"] ** 2 + df.loc[:, "transversal"] ** 2 + df.loc[:, "vertical"] ** 2
            )
            df = self._add_swivel_torsion_moments(df)

            'Calculate stem stresses'
            df = self._add_stem_stresses(df)

            'Store nominal values'
            df, df_nom_dict = dataframe_add_nominal_values(df, self.general_info["lc_rest"], "lc_description")

            'Convert to stress ranges'
            df = self._maximum_force_and_stress_range(df, df_nom_dict)

            # 'Store nominal values'
            # df = dataframe_add_nominal_values(df, df_nom)

            df_list[i] = df
            i += 1

            print(f"Time after csv read of {file_name}: {time() - time0}")

        return pd.concat(df_list)

    def _maximum_force_and_stress_range(self, df, df_nom_dict, tow_set_column="tow_set"):
        '''
        Function to find clevis stem stresses based on force, moment and SCF.

        :param pd.DataFrame df: Dataframe containing all force information
        :param dict df_nom_dict: Dictionary of entire dataframe, used to look up nominal values
        :param str tow_set_column: Identifier for column containing tower and set numbers / IDs

        :return: Dataframe with stem stresses
        :rtype: pd.DataFrame
        '''
        'Calculate stress ranges'
        df["stress_axial_range"] = abs(df.loc[:, "stress_axial"] - df.loc[:, tow_set_column].map(
            lambda x: df_nom_dict[x]["stress_axial"]))
        df["stress_bending_range"] = abs(df.loc[:, "stress_bending"] - df.loc[:, tow_set_column].map(
            lambda x: df_nom_dict[x]["stress_bending"]))
        df["stress_range"] = df.loc[:, "stress_axial_range"] + 2. * df.loc[:, "stress_bending_range"]

        'Find maximum LC per "tow_set"'
        df = df.sort_values("stress_range", ascending=False).drop_duplicates([tow_set_column])

        return df

    def _maximum_force_and_stress_range_old(self, df, set_column="set_no", tow_column="structure_number"):
        '''
        Function to find clevis stem stresses based on force, moment and SCF.

        :param pd.DataFrame df: Dataframe containing all force information
        :param str set_column: Identifier for column containing set numbers / IDs
        :param str tow_column: Identifier for column containing tower numbers / IDs

        :return: Dataframe with stem stresses
        :rtype: (pd.DataFrame, pd.DataFrame)
        '''
        'Store dataframe entries for nominal case'
        df_nominal = df.loc[df.loc[:, "lc_description"] == self.general_info["lc_rest"], :]
        df_list = [0 for x in df.loc[:, set_column].unique()]
        for i, (set_no, item) in enumerate(df.groupby([set_column])):
            'Find dataframe entries for nominal case'
            item_nominal = item.loc[
                           item.loc[:, "lc_description"] == self.general_info["lc_rest"], :
                           ].set_index(tow_column)

            'Convert to stress ranges by subtracting the EDS (no wind) weather case'
            item = dataframe_subtract_one_load_case(item, item_nominal, "stress_axial", tow_column)
            item = dataframe_subtract_one_load_case(item, item_nominal, "stress_bending", tow_column)

            'Assume worst case where bending completely reverses, i.e. multiply MW - EDS by 2'
            item["stress_range"] = item.loc[:, "stress_axial_range"] + 2. * item.loc[:, "stress_bending_range"]

            'Find maximum stress ranges per tower'
            tow_list = list(item.loc[:, tow_column].unique())
            indx_list = []
            for tow in tow_list:
                indx_list.append(item.loc[item.loc[:, tow_column] == tow, "stress_range"].idxmax())

            item_max = item.loc[indx_list, :]

            df_list[i] = item_max

        return pd.concat(df_list), df_nominal

    def _add_stem_stresses(self, df):
        '''
        Function to find clevis stem stresses based on force, moment and SCF

        :param pd.DataFrame df: Dataframe containing all force information

        :return: Dataframe with stem stresses in MPa
        :rtype: pd.DataFrame
        '''
        d_out = self.clevis_info["d_ball"] * self.unit_conversion[self.general_info["unit"]]
        d_in = self.clevis_info["d_stem"] * self.unit_conversion[self.general_info["unit"]]
        r_notch = self.clevis_info["r_notch"] * self.unit_conversion[self.general_info["unit"]]
        df["SCF_axial"] = scf_roark_17a(d_out, d_in, r_notch)
        df["SCF_bending"] = scf_roark_17b(d_out, d_in, r_notch)
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
        fraction_to_section = 1. - (height_clevis + height_swivel) / force_arm
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
        df["m_section"] = df.apply(
            lambda x: friction_moment_at_critical_section_swivel(
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
        try:
            excel_obj = pd.ExcelFile(os.path.join(self.path_input, self.file_name))
        except PermissionError:
            print(
                f"File '{os.path.join(self.path_input, self.file_name)}' could not be opened. "
                f"See if it is already open in another process."
            )
            exit()

        return excel_obj

    def _init_csv_files(self):
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

    def _write_to_excel(self, df, group_col, sort_cols=[]):
        '''
        Write results to Excel.

        :param pd.DataFrame df: Dataframe containing all information to be written to file
        :param str group_col: Dataframe columns to group into separate sheets
        :param list sort_cols: Columns to sort on, in prioritized order.

        '''

        print(f"----------Time to start of Excel results write: {time() - time0}----------")
        file_name = os.path.join(self.path_input, self.results_file_name)
        'Reorganize order of columns'
        columns = list_items_move(
            list(df.columns),
            [
                ["line_id", "structure_number", False],
                ["set_no", "lc", False],
                ["resultant", "phase_no", True],
                ["joint", "resultant", False],
                ["line_id", "row", True],
                ["f_long_nom", "tow_set", True],
                ["f_vert_nom", "t1", False],
                ["f_trans_nom", "t1", False],
                ["f_long_nom", "t1", False],
            ]
        )
        'Store reorganized columns'
        df = df.loc[:, columns]

        with pd.ExcelWriter(file_name) as writer:
            'Store summary sheet'
            df_tmp = df.copy()
            df_tmp = df_tmp.sort_values(
                by=["critical_set", "damage", "line_id", "structure_number"],
                ascending=[False, False, True, True]
            )
            df_tmp.columns = df_tmp.columns.map(
                lambda x: self.convert_names_all_back[x] if x in self.convert_names_all_back else x
            )
            df_tmp.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )
            for group, res in df.groupby(group_col):
                if len(sort_cols) > 0:
                    res = res.sort_values(by=sort_cols, ascending=False)
                res = res.sort_values(
                    by=["critical_set", "damage", "line_id", "structure_number"],
                    ascending=[False, False, True, True]
                )
                res.columns = res.columns.map(
                    lambda x: self.convert_names_all_back[x] if x in self.convert_names_all_back else x
                )
                res.to_excel(
                    writer,
                    sheet_name=str(group),
                    index=False
                )


class ReadCAAndSetInformation:
    def __init__(self, input_object):
        '''

        :param TowerColdEndFittingFatigue input_object:
        '''
        self.input = input_object
        # self.line_file_name_info = input_object.line_file_name_info
        self.file_types = {
            "Lines CA - With Detailed Data_Detailed Lines CA Data": "ca_info",
            "Lines Overview Dashboard_Lines - Asset Attributes": "set_info"
        }
        self.file_name_info = {}

    @classmethod
    def setup(cls, input_object):
        '''

        :param TowerColdEndFittingFatigue input_object: Class object to inherit input data from

        :return:
        :rtype: (pd.DataFrame, pd.DataFrame)
        '''
        cls_obj = cls(input_object)
        cls_obj._init_excel_file_list()
        'Check if lines listed in input is present in CA input folder'
        file_objects_defined_in_input_file(
            np.array([[x, y] for x, y in cls_obj.input.line_file_name_info["file_name"].items()])[:, 0],
            list(cls_obj.file_name_info.keys()),
            False,
        )

        'Read CA values'
        columns_lookup = {
            "structure_number": "device_position",
            "circuit": "asset_location",
            "position_column": "position_name",
            "lookup": "measurement",  # Defining ca or set columns
            "val_max": "ca_max",
            "val_1": "ca_1",
            "val_2": "ca_2",
            "additional": ["year", "date", "comment"]
        }
        df_ca = cls_obj._excel_read_ca_set_file_all(
            "Detailed Lines CA Data",
            "Asset",
            -1,
            "Insulators & Hardware",
            "Susp Cold End Condition",
            columns_lookup,
            "ca"
        )

        'Read set information'
        columns_lookup = {
            "structure_number": "device_position",  # 'Device Position'
            "circuit": "circuit",  # 'Circuit'
            "position_column": "suspension",  # Defining ca or set columns  # 'Susp Std Assy'
            "lookup": "suspension",  # Defining ca or set  # 'Susp Std Assy'
            "val_max": "set_name",
            "val_1": "set_1",
            "val_2": "set_2",
            "additional": ["suspension_description"]  # 'Susp Ins Type Desc'
        }
        df_set = cls_obj._excel_read_ca_set_file_all(
            "Lines - Asset Attributes",
            "Asset",
            -2,
            "Insulators and Accessories",
            "",
            columns_lookup,
            "set"
        )

        return df_ca, df_set

    def _excel_read_ca_set_file_all(self, sheet, header_search, offset, table_string, position, columns_lookup, ca_set):
        '''
        Function to search for files defined in "__init__" as CA or Set info files, and read information into dataframe.

        :param str sheet: Name of Excel sheet where data is read
        :param str header_search: Unique string value that defines the start of each table in OBIEE sheet
        :param int offset: Offset from row with "header_search", - upwards, + downwards
        :param table_string: Table in sheet to get data from
        :param str position: Measurement position to be chosen, i.e. specific text in a column.
        If blank, only non-empty values
        :param dict columns_lookup: Dictionary specifying the columns to be read
        :param str ca_set: "ca" or "set"

        :return: Dataframe containing information for all lines in input folder
        :rtype: pd.DataFrame
        '''

        print(f"----------Time to start of Excel {ca_set} read: {time() - time0}---------")
        'Define ca or set file to be read'
        excel_lookup = "ca_file" if ca_set == "ca" else "set_file"

        'Setup for total dataframe concatenation'
        df_list = [0 for x, item in self.file_name_info.items() if excel_lookup in item.keys()]

        'Loop through all files'
        i = 0
        for line_id, item in self.file_name_info.items():
            'Set up excel object'
            if excel_lookup not in item.keys():
                print(f"File type '{excel_lookup}' does not exist for line {line_id}. Skipped.")
                continue
            excel_object = self._init_excel_object(self.file_name_info[line_id][excel_lookup])
            df_list[i] = self.dataframe_get_ca_and_set_data_from_excel(
                excel_object,
                sheet,
                header_search,
                offset,
                table_string,
                line_id,
                position,
                columns_lookup,
                self.input.convert_names_ca_set
            )
            excel_object.close()
            i += 1
            print(f"Time after Excel {ca_set} read of {line_id}: {time() - time0}")

        return pd.concat(df_list)

    @staticmethod
    def dataframe_get_ca_and_set_data_from_excel(
            excel_object, sheet, header_search, offset, table_string, line_name, position, lookup_info, convert_names
    ):
        '''
        File to loop through Transpower OBIEE reports, to return CA and set data if dataframe format

        :param pandas.ExcelFile ExcelObj excel_object: All Excel sheets stored in object for data extraction
        :param str sheet: Name of Excel sheet where data is read
        :param str header_search: Unique string value that defines the start of each table in OBIEE sheet
        :param int offset: Offset from row with "header_search", - upwards, + downwards
        :param table_string: Table in sheet to get data from
        :param str line_name: Line name ID
        :param str position: Measurement position to be chosen, i.e. specific text in a column.
        If blank, only non-empty values
        :param dict lookup_info: Dictionary specifying the columns to be read
        :param dict convert_names: Dictionary converting from Excel to code name format

        :return: Dataframe containing CA or set data
        :rtype: pd.DataFrame
        '''
        'Initialize columns to be read and stored'
        columns_return = lists_add_and_flatten(
            [
                ["line_id", "structure_number"],
                list(lookup_info.values()),
                ["circuit1", "circuit2"]
            ]
        )

        'Find position of data entries'
        header_indexes, header_labels = excel_sheet_find_input_rows_by_string(
            excel_object, sheet, 0, offset, header_search
        )
        row = header_indexes[header_labels.index(table_string)]

        'Create dataframe from Excel object and convert to code name format'
        df = dataframe_from_excel_object(excel_object, sheet, row + 1 - offset, 1)
        df.columns = df.columns.map(
            lambda x: convert_names[x] if x in convert_names else x
        )
        df.fillna(0)

        'Add columns to keep relevant information'
        for col in columns_return:
            if col not in df.columns:
                df[col] = ""

        'Store only relevant columns and rows'
        df = df.loc[:, columns_return]

        if len(position) > 0:
            df = df.loc[df.loc[:, lookup_info["position_column"]] == position, :]
        else:
            df = df.loc[df.loc[:, lookup_info["position_column"]].notna(), :]

        'Check if dataframe is empty'
        if df.shape[0] == 0:
            return pd.DataFrame(columns=columns_return)

        'Sort on line, structure and circuit information'
        df.loc[:, "line_id"] = line_name
        df.loc[:, "structure_number"] = df.apply(
            lambda x: x[lookup_info["structure_number"]].lower().replace(x["line_id"], ""),
            axis=1
        )

        'Strip leading zeros from structure_number column'
        df.loc[:, "structure_number"] = df.loc[:, "structure_number"].str.lstrip("0")

        df["circuit1"] = df.loc[:, lookup_info["circuit"]].map(lambda x: x.split("-")[-1].replace("0", ""))
        df = df.loc[:, columns_return]
        df["line_structure"] = df.loc[:, "line_id"] + "_" + df.loc[:, "structure_number"]
        df = df.sort_values(by=["structure_number", "circuit1"])

        'Lookup values from "lookup_info" and find maximum value'
        'If more than one circuit, store maximum ca_value for both circuits'
        circuits = df.loc[:, "circuit1"].unique()

        if len(circuits) > 1:
            df1 = df.groupby("circuit1").get_group(circuits[0]).set_index("line_structure")
            df2 = df.groupby("circuit1").get_group(circuits[1]).set_index("line_structure")
            shapes = [df1.shape[0], df2.shape[0]]
            dfs = [df1, df2]
            'Sort to use dataframe with most entries as basis'
            df1, df2 = dfs[shapes.index(max(shapes))], dfs[shapes.index(min(shapes))]
            missing_list, _ = lists_compare_contents(list(df2.index.unique()), list(df1.index.unique()))
            if len(missing_list) > 0:
                df1 = pd.concat([df1, df2.loc[missing_list, :]])
                print(f"Different towers in the circuits for line {line_name}:", missing_list)
            df1[lookup_info["val_1"]] = df1.loc[:, lookup_info["lookup"]]
            df1[lookup_info["val_2"]] = df2.loc[:, lookup_info["lookup"]]
            df1.loc[:, lookup_info["val_2"]].fillna(0, inplace=True)
            df1[lookup_info["val_max"]] = df1.apply(
                lambda x: find_max_value_all_types(x[lookup_info["val_1"]], x[lookup_info["val_2"]]),
                axis=1
            )
            df1["circuit2"] = df2.loc[:, "circuit1"]
            df1.loc[:, "circuit2"].fillna(0, inplace=True)
            df = df1
        else:
            df[lookup_info["val_1"]] = df.loc[:, lookup_info["lookup"]]
            df[lookup_info["val_2"]] = df.loc[:, lookup_info["lookup"]]
            df[lookup_info["val_max"]] = df.loc[:, lookup_info["lookup"]]
            df["circuit2"] = df.loc[:, "circuit1"]
            df = df.set_index("line_structure")

        'Reorder columns'
        df = df.loc[:, columns_return]

        return df

    def _init_excel_object(self, file_name):
        '''
        Open Excel object for read

        :param str file_name: File to be processed

        :return All Excel sheets stored in object for data extraction
        :rtype pandas.ExcelFile ExcelObj
        '''
        try:
            excel_obj = pd.ExcelFile(os.path.join(self.input.path_ca, file_name))
        except PermissionError:
            print(
                f"File '{os.path.join(self.input.path_ca, file_name)}' could not be opened. "
                f"See if it is already open in another process."
            )
            exit()

        return excel_obj

    def _init_excel_file_list(self):
        """
        Function to return all Excel files in folder and store in self.file_name_info

        """
        file_names = os.listdir(self.input.path_ca)
        file_types = list(self.file_types.keys())
        line_info = {}

        'Store only files containing CA and insulator information'
        for file_name in file_names:
            if os.path.isdir(os.path.join(self.input.path_ca, file_name)):
                continue
            if Path(file_name).suffix.strip(".") != "xlsx":
                continue
            f = Path(file_name).stem
            file_type = f[f.index(" ")+1:]
            line_name = f[:f.index(" ")].lower()
            if file_type in file_types:
                if line_name not in line_info.keys():
                    line_info[line_name] = {}
                if "ca" in file_type.lower():
                    line_info[line_name].update({"ca_file": file_name.lower()})
                else:
                    line_info[line_name].update({"set_file": file_name.lower()})

        self.file_name_info = line_info


if __name__ == '__main__':
    main()
