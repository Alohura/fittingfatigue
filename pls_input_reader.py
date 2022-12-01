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
    path_input = r"C:\Users\AndreasLem\Groundline\NZ-6529 - Ball clevis test and analysis support - Documents\03 Operations\Analysis"
    path_inputs = {
        "ca": "Asset input - 1.12.22",
        "loads": "Load input"
    }
    input_file = "ClevisFatigue_Input.xlsx"
    results_file = "line_summary.xlsx"
    TowerColdEndFittingFatigue.fatigue_damage(path_input, path_inputs, input_file, results_file, True, False)


class TowerColdEndFittingFatigue:
    def __init__(self, path_setup, path_inputs, file_name, results_file):
        self.path_setup = path_setup
        self.path_input = os.path.join(path_setup, path_inputs["loads"])
        self.path_ca = os.path.join(path_setup, path_inputs["ca"])
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
            'Swivel ID:': 'swivel_id',
            'Swivel type:': 'swivel_type',
            'Height [mm]:': 'height',
            'Width [mm]:': 'width',
            'Bracket height [mm]:': 'height_bracket',
            'Outer diameter [mm]:': 'd_outer',
            'Inner diameter [mm]:': 'd_inner',
            'Pin diameter [mm]:': 'd_pin',
            'Stem diameter [mm]:': 'd_stem',
            'Transition radius [mm]:': 'r_notch',
            'Ball effective diameter [mm]:': 'd_ball',
            'Ball clevis length [mm]:': 'l_clevis',
            'Rolling angle factor [-]:': 'rolling_factor',
            'Power factor [-]:': 'power_factor',
            'Friction coefficient [-]:': 'friction',
            'Force arm minimum distance [mm]:': 'force_arm_min',
            'Force arm distance [mm]:': 'force_arm',
            'Insulator length [mm]:': 'length_insulator',
            'Elastic modulus - steel [MPa]:': 'e_modulus',
            'Unit:': 'unit',
            'Load case at rest:': 'lc_rest',
            'Line name:': 'line_id',
            'Unique identifier [line_tow_set]:': 'line_tow_set',
            'File name:': 'file_name',
            'Insulator sets:': 'insulator'
        }
        self.convert_names_code = {
            "Fatigue damage": "damage",
            "SN Curve (NZS 3404)": "sn_curve",
            "Transversal swing angle [deg]": "swing_angle_trans",
            "Transversal swing angle - at rest [deg]": "swing_angle_trans_orig",
            "Change in transversal swing angle [deg]": "swing_angle_trans_range",  #
            "Longitudinal swing angle [deg]": "swing_angle_long",
            "Longitudinal swing angle - at rest [deg]": "swing_angle_long_orig",
            "Change in longitudinal swing angle [deg]": "swing_angle_long_range",  #
            "Resultant force [N]": "resultant",
            "Nominal EDS Longitudinal load [N]": "f_long_nom",
            "Nominal EDS Transversal load [N]": "f_trans_nom",
            "Nominal EDS Vertical load [N]": "f_vert_nom",
            "Stress [MPa]": "stress",
            "Stress - axial [MPa]": "stress_axial",
            "Stress - bending [MPa]": "stress_bending",
            "Stress range [MPa] (Axial + Bending stress ranges)": "stress_range",
            "Stress range [MPa] (Axial + 2 x Bending stress ranges)": "stress_range",
            "Stress range - axial [MPa]": "stress_axial_range",
            "Stress range - bending [MPa]": "stress_bending_range",
            "Friction rolling resistance moment - T1 [Nm]": "t1",
            "Friction resistance moment - T2 [Nm]": "t2",
            "Friction resistance moment - T3 [Nm]": "t3",
            "Total friction resistance moment [Nm]": "t_friction",
            "Moment at critical stem section [Nm]": "m_section",
            "Moment fraction at stem section [-]:": "m_fraction",
            "Hanger bracket check - 1 if bracket, 0 if not": "hbk",
            "SCF axial": "SCF_axial",
            "SCF bending": "SCF_bending",
            "Angle of swivel / pin contact point [deg]:": "rolling_angle_swivel",
            "Maximum angle of swivel / pin contact point [deg]:": "rolling_angle_swivel_max",
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
            'Insulator AttachType': 'swivel_hbk',
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
        self.hbk_info = {}
        self.insulator_set_sw = {}
        self.insulator_set_hbk = {}
        self.clevis_info = {}
        self.general_info = {}
        self.line_file_name_info = {}
        self.sn_info = {}
        self.sn_input_columns = ['Curve', 'm1', 'm2', 's1', 's2', 's3', 'n_s1', 'n_cut_off', 'CA']
        'Setup for Excel and csv input'
        self.excel_object = self._init_excel_object()
        self.csv_objects = self._init_csv_files()

    @classmethod
    def fatigue_damage(cls, path_setup, path_inputs, input_file, results_file, read_pickle_ca=False, read_pickle=False):
        '''
        Calculate fatigue damage of insulators from PLS-CADD structure attachment loading, input values from
        "input_file", condition assessment (CA) values from files in folder "path_ca" and store to "results_file"

        :param str path_setup: Path to root of project, to input Excel sheet
        :param dict path_inputs: Paths to folders;
        {"ca": OBIEE reports containing CA values and insulator set information,
        "loads": PLS-CADD input and structure loading ".csv" files}
        :param str input_file: Excel input file manually filled out prior to analysis
        :param str results_file: Name of Excel file results are written to
        :param bool read_pickle_ca: Allows for reading of pickled Excel OBIIE input, as reading is quite time consuming
        :param bool read_pickle: Allows for reading of pickled results, as reading is quite time consuming

        '''
        cls_obj = cls(path_setup, path_inputs, input_file, results_file)
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
        print(f"Time to finish OBIEE read: {time() - time0}----------")
        # exit()
        'Read pickle or process main dataframe'
        dataframe_process = False
        if read_pickle:
            if os.path.exists(os.path.join(cls_obj.path_input, "pickle_df.p")):
                with open(os.path.join(cls_obj.path_input, "pickle_df.p"), "rb") as f:
                    [df, df_nominal] = pickle.load(f)
            else:
                dataframe_process = True
        else:
            dataframe_process = True

        if dataframe_process:
            df, df_nominal = cls_obj._read_attachment_loads()
            with open(os.path.join(cls_obj.path_input, "pickle_df.p"), "wb") as f:
                pickle.dump([df, df_nominal], f)

        print(f"Time to finish csv read: {time() - time0}----------")
        df = cls_obj._line_and_tower_to_set_name_map(df, df_set)
        df_nominal = cls_obj._line_and_tower_to_set_name_map(df_nominal, df_set)
        print(f"Time to finish set mapping: {time() - time0}----------")
        df = cls_obj._ca_value_map_to_sn_detail(df, df_ca, ca_default=100.)
        print(f"Time to finish S-N mapping: {time() - time0}----------")
        df = cls_obj._calculate_moment_and_stress(df, df_nominal)
        print(f"Time to finish moment and stress calculations: {time() - time0}----------")
        df = cls_obj._calculate_fatigue_damage(df)
        print(f"Time to finish fatigue calculation: {time() - time0}----------")
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
        Function to map line and tower id's to set names

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
            lambda x: ca_and_set_value_from_dict(
                x["line_id"].split("_")[0] + "_" + x["structure_number"], set_dict, "set_name", []
            ),
            axis=1
        )

        'Map hanger bracket information from set dataframe'
        df["hbk"] = df.apply(
            lambda x: ca_and_set_value_from_dict(
                x["line_id"].split("_")[0] + "_" + x["structure_number"], set_dict, "hbk", 0
            ),
            axis=1
        )
        # a = df.loc[df.loc[:,"hbk"] == 1, "hbk"]

        'Search for set for set names in critical set list'
        df["critical_set"] = df.apply(
            lambda x: check_overlap_between_two_lists(
                x["set_name"],
                self.line_file_name_info["insulator"][x["line_id"]]
            ),
            axis=1
        )

        return df.reset_index(drop=True)

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
        self.swivel_info = self._get_input_info("GeneralInput", header_indexes[0] + 2, True)
        self.hbk_info = self._get_input_info("GeneralInput", header_indexes[1] + 2, True)
        self.clevis_info = self._get_input_info("GeneralInput", header_indexes[2] + 2)["value"]
        self.general_info = self._get_input_info("GeneralInput", header_indexes[3] + 2)["value"]
        self.sn_info = self._get_input_info(
            "GeneralInput",
            header_indexes[4] + 2,
            True
        )
        self.sn_info = sn_curve_add_ca_list(self.sn_info)
        self.sn_info = {f"{x}": y for x, y in self.sn_info.items()}

        'Convert swivel and hanger bracket (hbk) info to insulator set dictionaries, to enable fast lookup'
        self._insulator_set_input_setup()

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

    def _insulator_set_input_setup(self):
        '''
        Function to enable quick lookup of properties for each insulator set number when calculating bending stresses
        '''
        dict_key = "insulator"
        'Swivel info'
        for key, item in self.swivel_info.items():
            for ins in item[dict_key].split(","):
                if ins not in self.insulator_set_sw:
                    self.insulator_set_sw[ins] = {x: y for x, y in item.items() if x != dict_key}
        'Hanger bracket information'
        for key, item in self.hbk_info.items():
            for ins in item[dict_key].split(","):
                if ins not in self.insulator_set_hbk:
                    self.insulator_set_hbk[ins] = {x: y for x, y in item.items() if x != dict_key}

    def _read_attachment_loads(self):
        '''
        Read and store all data relevant for fatigue calculations of insulator cold end clevis

        :return: Dataframe containing all necessary information for fatigue calculations
        :rtype: tuple
        '''
        'Read loads from csv files. calculate stem stresses and store maximum stress ranges'
        df, df_nominal = self._dataframe_setup()

        'Add swing angles'
        df = self._add_swing_angles(df)
        df_nominal = self._add_swing_angles(df_nominal)

        return df, df_nominal

    @staticmethod
    def _add_swing_angles(df):
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
        'For rows'
        df.iloc[:, 0] = df.iloc[:, 0].map(
            lambda x: self.convert_names_general[x] if x in self.convert_names_general else x
        )
        'For columns'
        df.columns = df.columns.map(
            lambda x: self.convert_names_general[x] if x in self.convert_names_general else x
        ).str.lower()

        'Remove Notes column'
        df = df.loc[:, [x for x in df.columns if x != "notes"]]

        df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: x.lower() if type(x) is str else x)
        df = df.set_index(df.columns[0])

        return df.transpose().to_dict() if transpose else df.to_dict()

    def _dataframe_from_csv(self, file_name):
        '''
        Function to read csv file and return as dataframe if file is present in read folder

        :param str file_name: File to be processed

        :return: Dataframe to be processed with attachment loads
        :rtype: pd.DataFrame
        '''
        file_check = True
        'Check if CSV file in input list'
        if file_name not in self.line_file_name_info["file_name"].values():
            print(f"File type '{file_name}' not in excel input sheet. Skipped.")
            file_check = False

        'Read CSV file'
        if file_check:
            df = pd.read_csv(os.path.join(
                self.path_input, file_name),
                low_memory=False,
                na_values=["nan", np.nan, "NaN", ""]
            )
        else:
            df = pd.DataFrame()

        return file_check, df

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
        df_nom_list = df_list.copy()
        i = 0
        tower_count = 0
        for file_name in self.csv_objects:
            'Read csv file, process only if file is present'
            file_check, df = self._dataframe_from_csv(file_name)
            if not file_check:
                continue

            'Convert names to code names, ref. dictionaries for converting names in __init__ module'
            df.columns = df.columns.map(lambda x: self.convert_names_pls[x] if x in self.convert_names_pls else x)
            df = df.loc[:, list(self.convert_names_pls.values())]

            'Remove rows with missing structure information'
            df = df.loc[df.loc[df.loc[:, "structure_number"].notna(), :].index]
            df.loc[:, "structure_number"] = df.loc[:, "structure_number"].map(
                lambda x: str(int(x)) if type(x) is float else str(x)
            )

            'Count towers processed'
            tower_count += len(df.loc[:, "structure_number"].unique())

            'Store only suspension towers and remove columns no longer needed'
            df = dataframe_select_suspension_insulator_sets(
                df, "structure_number", "set_no", 1., ["ahead", "back"], [7, 8]  # Set factor = 1 -> no ew filtering
            )

            'Preprocess dataframe for moment and stress calculations'
            df, df_nominal = self._dataframe_preprocess(file_name, df)

            df_list[i] = df
            df_nom_list[i] = df_nominal
            i += 1

            print(f"Time after csv read of {file_name}: {time() - time0}")

        print(f"-----{tower_count} towers processed-----")

        return pd.concat(df_list), pd.concat(df_nom_list)

    def _dataframe_preprocess(self, file_name, df):
        '''
        Function to set up dataframe with columns needed for swivel and moment stress calculations

        :param str file_name:
        :param pd.DataFrame df: Dataframe with attachment points

        :return: Dataframe with all results and with only nominal values
        :rtype: tuple
        '''
        'Remove columns not needed'
        df = dataframe_remove_columns(df, ["ahead", "back"])

        'Convert structure_number column, to keep consistent across lines, i.e. in case of "a", "b" extensions etc.'
        df.loc[:, "structure_number"] = df.loc[:, "structure_number"].map(str)
        df.loc[:, "structure_number"] = df.loc[:, "structure_number"].str.lower()

        'Add line ID to uniquely identify each line'
        file_to_line_id = {y: x for x, y in self.line_file_name_info["file_name"].items()}
        line_id = file_to_line_id[file_name]
        df["line_id"] = line_id

        'Add set information, unique to tower and line'
        df["line_tow_set"] = df.loc[:, "line_id"] + "_" + df.loc[:, "structure_number"] + "_" \
                             + df.loc[:, "set_no"].fillna(0).map(lambda x: str(int(x)))

        'Add load resultant'
        df["resultant"] = np.sqrt(
            df.loc[:, "longitudinal"] ** 2 + df.loc[:, "transversal"] ** 2 + df.loc[:, "vertical"] ** 2
        )

        'Store nominal values'
        df, df_nominal = dataframe_add_nominal_values(df, self.general_info["lc_rest"], "lc_description")

        return df, df_nominal

    def _calculate_moment_and_stress(self, df, df_nominal, set_column="line_tow_set"):
        '''
        Function to evaluate torsion moments and swivel stresses, taking into account different swivel geometries

        :param pd.DataFrame df:
        :param pd.DataFrame df_nominal:
        :param str set_column: Identifier for column containing tower set numbers / IDs for a line

        :return: Dataframe with maximum stress ranges for each tower set in the lines evaluated
        :rtype: pd.DataFrame
        '''

        'Calculate moments'
        df = self._add_swivel_torsion_moments(df)
        df_nominal = self._add_swivel_torsion_moments(df_nominal)

        'Calculate stem stresses'
        df = self._add_stem_stresses(df)
        df_nominal = self._add_stem_stresses(df_nominal)

        'Convert to stress ranges'
        df_nom_dict = df_nominal.set_index(set_column).transpose().to_dict()
        df = self._maximum_force_and_stress_range(df, df_nom_dict)

        return df

    @staticmethod
    def _maximum_force_and_stress_range(df, df_nom_dict, line_tow_set_column="line_tow_set"):
        '''
        Function to find clevis stem stresses based on force, moment and SCF. Stores line_tower_set entry with
        maximum stress range, i.e. worst load case.

        :param pd.DataFrame df: Dataframe containing all force information
        :param dict df_nom_dict: Dictionary of entire dataframe, used to look up nominal values
        :param str line_tow_set_column: Identifier for column containing tower and set numbers / IDs for a line

        :return: Dataframe with stem stresses
        :rtype: pd.DataFrame
        '''

        'Calculate stress ranges'
        df["stress_axial_range"] = abs(df.loc[:, "stress_axial"] - df.loc[:, line_tow_set_column].map(
            lambda x: df_nom_dict[x]["stress_axial"]))
        df["stress_bending_range"] = abs(df.loc[:, "stress_bending"] - df.loc[:, line_tow_set_column].map(
            lambda x: df_nom_dict[x]["stress_bending"]))
        df["stress_range"] = df.loc[:, "stress_axial_range"] + 2. * df.loc[:, "stress_bending_range"]

        'Find maximum LC per "line_tow_set"'
        df = df.sort_values("stress_range", ascending=False).drop_duplicates([line_tow_set_column])

        return df

    def _add_stem_stresses(self, df):
        '''
        Function to find clevis stem stresses based on force, moment and SCF

        :param pd.DataFrame df: Dataframe containing all force information

        :return: Dataframe with stem stresses in MPa
        :rtype: pd.DataFrame
        '''
        r_notch = self.clevis_info["r_notch"] * self.unit_conversion[self.general_info["unit"]]
        d_stem = self.clevis_info["d_stem"] * self.unit_conversion[self.general_info["unit"]]
        d_ball = self.clevis_info["d_ball"] * self.unit_conversion[self.general_info["unit"]]
        # df["SCF_axial"] = df.apply(lambda x: scf_roark_17a(x["r_out"] * 2., x["r_in"] * 2., r_notch), axis=1)
        # df["SCF_bending"] = df.apply(lambda x: scf_roark_17b(x["r_out"] * 2., x["r_in"] * 2., r_notch), axis=1)
        df["SCF_axial"] = scf_roark_17a(d_ball, d_stem, r_notch)
        df["SCF_bending"] = scf_roark_17b(d_ball, d_stem, r_notch)
        df["stress_axial"] = df.apply(
            lambda x: stress_stem_roark_17(
                x["resultant"], 0., x["SCF_axial"], 0., d_ball, d_stem
            ) / 1.e6,
            axis=1
        )
        df["stress_bending"] = df.apply(
            lambda x: stress_stem_roark_17(
                0., x["m_section"], 0., x["SCF_bending"], d_ball, d_stem
            ) / 1.e6,
            axis=1
        )
        df["stress"] = df.loc[:, "stress_axial"] + df.loc[:, "stress_bending"]

        return df

    def _swivel_hbk_set_lookup(self, set_name, hbk, dcts_default):
        '''
        Function used to map swivel data onto results dataframe for each "set_no"

        :param int hbk: 1 if hanger bracket, 0 if not
        :param list set_name: Set ID to find dictionary item. May contain more set names in list.
        :param list dcts_default: List of default dictionaries [hbk, sw], used in case set "set_no" is not in dicts

        :return: Dictionary item
        :rtype: dict
        '''
        if hbk == 1:
            dct_look_up = self.insulator_set_hbk
            dct_default = dcts_default[0]
        else:
            dct_look_up = self.insulator_set_sw
            dct_default = dcts_default[1]

        'Store set if in critical set list, otherwise store first entry in list'
        if len(set_name) == 0:
            set_name = list(dct_default.keys())[0]
        else:
            if check_overlap_between_two_lists(set_name, list(dct_look_up.keys())):
                set_name = [x for x in set_name if x in dct_look_up][0]
            else:
                set_name = "D.F."

        dct_look_up.update(dct_default)
        # print(f"-----------{set_name}-----------")

        return dct_look_up[set_name]

    def _add_swivel_torsion_moments_old(self, df):
        '''
        Function to find swivel torsion resistance moments based on applied force angle

        :param pd.DataFrame df: Dataframe containing all force information

        :return: Dataframe with torsion resistance moments
        :rtype: pd.DataFrame
        '''
        'General input parameters to friction calculations'
        unit_conversion = self.unit_conversion[self.general_info["unit"]]
        my = self.general_info["friction"]
        force_arm = self.general_info["force_arm_min"] * unit_conversion
        swivel_default = dict_of_dicts_max_item(self.swivel_info, "d_pin")
        hbk_default = dict_of_dicts_max_item(self.hbk_info, "d_pin")
        dcts_default = [hbk_default, swivel_default]

        'Add columns for moment calculations'
        df = self._dataframe_add_cols_for_moment_calcs(df, dcts_default, unit_conversion, force_arm)

        'Calculate friction moments from swivel / cleat interface (T1) and from swivel / pin reaction force couple (T2)'
        df["t1"] = df.apply(
            lambda x: friction_torsion_resistance_swivel_t1(my, x["transversal"], x["r_out"], x["r_in"]),
            axis=1
        )
        df["t2"] = df.apply(
            lambda x: friction_torsion_resistance_swivel_t2(
                my, x["transversal"], x["vertical"], x["width"], x["height_swivel"], x["r_pin"]
            ),
            axis=1
        )

        df["t_friction"] = df.loc[:, "t1"] + df.loc[:, "t2"]
        df["m_section"] = df.apply(
            lambda x: friction_moment_at_critical_section_swivel(
                x["longitudinal"],
                x["force_arm"],
                x["m_fraction"],
                x["t_friction"]
            ),
            axis=1
        )

        return df

    def _add_swivel_torsion_moments(self, df):
        '''
        Function to find swivel torsion resistance moments based on applied force angle

        :param pd.DataFrame df: Dataframe containing all force information

        :return: Dataframe with torsion resistance moments
        :rtype: pd.DataFrame
        '''
        'General input parameters to friction calculations'
        unit_conversion = self.unit_conversion[self.general_info["unit"]]
        my = self.general_info["friction"]
        force_arm_min_dist = self.general_info["force_arm_min"] * unit_conversion
        swivel_default = dict_of_dicts_max_item(self.swivel_info, "d_pin")
        hbk_default = dict_of_dicts_max_item(self.hbk_info, "d_pin")
        dcts_default = [hbk_default, swivel_default]

        'Add columns for moment calculations'
        df = self._dataframe_add_cols_for_moment_calcs(df, dcts_default, unit_conversion, force_arm_min_dist)

        'Calculate friction moments from swivel / cleat interface (T1) and from swivel / pin reaction force couple (T2)'
        df["t1"] = df.apply(
            lambda x: friction_torsion_resistance_swivel_rolling(
                x["rolling_angle_swivel"],
                x["rolling_angle_swivel_max"],
                x["transversal"],
                x["r_pin"]
            ),
            axis=1
        )
        df["t2"] = df.apply(
            lambda x: friction_torsion_resistance_swivel_t1(my, x["transversal"], x["r_out"], x["r_in"]),
            axis=1
        )
        df["t3"] = df.apply(
            lambda x: friction_torsion_resistance_swivel_t2(
                my, x["transversal"], x["vertical"], x["width"], x["height_swivel"], x["r_out"], x["r_in"]
            ),
            axis=1
        )
        # df["t2"] = df.apply(
        #     lambda x: friction_torsion_resistance_swivel_t2_old(
        #         my, x["transversal"], x["vertical"], x["width"], x["height_swivel"], x["r_pin"]
        #     ),
        #     axis=1
        # )

        df["t_friction"] = df.loc[:, "t1"] + df.loc[:, "t2"] + df.loc[:, "t3"]
        df["m_section"] = df.apply(
            lambda x: friction_moment_at_critical_section_swivel(
                x["longitudinal"],
                x["force_arm"],
                x["m_fraction"],
                x["t_friction"]
            ),
            axis=1
        )

        return df

    def _dataframe_add_cols_for_moment_calcs(
            self, df, dcts_default, unit_conversion, default_dist=(50. / 1000.)
    ):
        '''
        Function to add columns to dataframe to enable torsion moment calculations

        :param pd.DataFrame df: Input dataframe
        :param list dcts_default: Default values to use in case set numbers are not defined in input Excel sheet
        :param float unit_conversion: Factor to convert such that units are consistent
        :param float default_dist: Minimum distance from critical ball clevis section to effective rotation centre

        :return: Processed dataframe
        :rtype: pd.DataFrame
        '''
        '---Swivel dimensions---'
        'Outer radius'
        df["r_out"] = df.apply(
            lambda x: self._swivel_hbk_set_lookup(x["set_name"], x["hbk"], dcts_default)["d_outer"] / 2.,
            axis=1
        ) * unit_conversion
        'Inner radius'
        df["r_in"] = df.apply(
            lambda x: self._swivel_hbk_set_lookup(x["set_name"], x["hbk"], dcts_default)["d_inner"] / 2.,
            axis=1
        ) * unit_conversion
        'Swivel width'
        df["width"] = df.apply(
            lambda x: self._swivel_hbk_set_lookup(x["set_name"], x["hbk"], dcts_default)["width"],
            axis=1
        ) * unit_conversion
        'Swivel height'
        df["height_swivel"] = df.apply(
            lambda x: self._swivel_hbk_set_lookup(x["set_name"], x["hbk"], dcts_default)["height"],
            axis=1
        ) * unit_conversion
        'Swivel pin diameter '
        df["r_pin"] = df.apply(
            lambda x: self._swivel_hbk_set_lookup(x["set_name"], x["hbk"], dcts_default)["d_pin"] / 2.,
            axis=1
        ) * unit_conversion

        '---Distance to ball socket---'
        'Clevis height'
        df["height_clevis"] = df.apply(
            lambda x: self._swivel_hbk_set_lookup(x["set_name"], x["hbk"], dcts_default)["l_clevis"],
            axis=1
        ) * unit_conversion
        'Hanger bracket height'
        df["height_hbk"] = df.apply(
            lambda x: self._swivel_hbk_set_lookup(x["set_name"], x["hbk"], dcts_default)["height_bracket"],
            axis=1
        ) * unit_conversion
        'Total height to section'
        df["height_total"] = df.loc[:, "height_swivel"] + df.loc[:, "height_hbk"] + df.loc[:, "height_clevis"]

        '---Rotation angles---'
        'Force arm'
        df["rolling_angle_swivel"] = df.apply(
            lambda x: swivel_rotation_angle(
                x["swing_angle_long_range"], x["r_pin"] * 2., x["r_in"] * 2., self.general_info["rolling_factor"]),
            axis=1
        )
        df["rolling_angle_swivel_max"] = df.apply(
            lambda x: swivel_rotation_angle_max(
                self.general_info["friction"], x["r_pin"] * 2., x["r_in"] * 2., self.general_info["rolling_factor"]),
            axis=1
        )

        'Force arm'
        'Bending stiffness of insulator, assume ball clevis stem is representative for insulator'
        EI = bending_stiffness_cylinder(
            self.general_info["e_modulus"], self.clevis_info["d_stem"]
        ) * unit_conversion ** 2
        df["force_arm"] = df.apply(
            lambda x: insulator_to_cantilever_beam_length(
                x["swing_angle_long_range"],
                self.general_info["length_insulator"] * unit_conversion * np.cos(np.radians(x["swing_angle_trans"])),  # Vertical length of insulator
                x["height_total"] + default_dist,
                EI,
                np.sqrt(x["vertical"] ** 2 + x["longitudinal"] ** 2),  # Resultant in longitudinal / vertical plane
                self.general_info["power_factor"]
            ),
            axis=1
        )

        'Calculate moment fraction at critical clevis intersection'
        df["m_fraction"] = df.apply(
            lambda x: 1. - x["height_total"] / x["force_arm"],
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
            excel_obj = pd.ExcelFile(os.path.join(self.path_setup, self.file_name))
        except PermissionError:
            print(
                f"File '{os.path.join(self.path_setup, self.file_name)}' could not be opened. "
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

    @staticmethod
    def _write_to_excel_df_edit(df):
        '''

        :param df:
        :return:
        '''
        'Reorganize order of columns'
        columns = list_items_move(
            list(df.columns),
            [
                ["line_id", "structure_number", False],
                ["set_no", "lc", False],
                ["resultant", "phase_no", True],
                ["joint", "resultant", False],
                ["line_id", "row", True],
                ["f_long_nom", "line_tow_set", True],
                ["f_vert_nom", "t1", False],
                ["f_trans_nom", "t1", False],
                ["f_long_nom", "t1", False],
                ["critical_set", "damage", False],
                ["line_tow_set", "line_id", False]
            ]
        )

        'Store reorganized columns'
        df = df.loc[:, columns]

        'Delete some columns'
        df = dataframe_remove_columns(
            df,
            [
                "r_out",
                "r_in",
                "width",
                "r_pin",
                "height_swivel",
                "height_clevis",
                "height_hbk",
                "height_total",
                "row"
            ]
        )
        'PLS-CADD sometimes requires all towers to be stored in different models. These are typically named ".._1", '
        '".._2" etc. Below such models are merged to one'
        df.loc[:, "line_id"] = df.loc[:, "line_id"].map(
            lambda x: x[:-2] if (x[-1].isnumeric() and x[-2] == "_") else x
        )

        'Change force arm distance to mm'
        df.loc[:, "force_arm"] = df.loc[:, "force_arm"] * 1000.

        return df

    def _write_to_excel(self, df, group_col, sort_cols=[]):
        '''
        Write results to Excel.

        :param pd.DataFrame df: Dataframe containing all information to be written to file
        :param str group_col: Dataframe columns to group into separate sheets
        :param list sort_cols: Columns to sort on, in prioritized order.

        '''

        print(f"----------Time to start of Excel results write: {time() - time0}----------")
        file_name = os.path.join(self.path_input, self.results_file_name)
        df = self._write_to_excel_df_edit(df)

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

        'Read swivel CA information'
        columns_lookup = {
            "structure_number": "device_position",  # Column identifying line and tower id
            "circuit": "asset_description",  # Column from which circuit numbers can be extracted
            "position_column": "position_name",  # Column used to specify rows to keep (for CA values, only keep
            #                                       'Suspension Cold End' entries, as only these are ov interest)
            "lookup": "measurement",  # Defining column to extract values
            "val_max": "ca_max",  # Specify title of value column where lookup values are placed
            "val_1": "ca_1",  # Allow for columns to keep values for both circuits if two are present
            "val_2": "ca_2",
            "additional": []  # ''
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
            "structure_number": "device_position",  # See description for "ca_lookup"
            "circuit": "circuit",  # See description for "ca_lookup"
            "position_column": "",  # See description for "ca_lookup"
            "lookup": "suspension",  # See description for "ca_lookup"
            "val_max": "set_name",  # See description for "ca_lookup"
            "val_1": "set_1",  # See description for "ca_lookup"
            "val_2": "set_2",
            "additional": ["suspension_description"]
        }
        df_set = cls_obj._excel_read_ca_set_file_all(
            "Lines - Asset Attributes",
            "Asset",
            -2,
            "Insulators and Accessories",
            "",
            columns_lookup,
            "set"
        ).sort_index()

        'Read tower information to find hanger bracket locations'
        columns_lookup = {
            "structure_number": "device_position",  # See description for "ca_lookup"
            "circuit": "circuit",  # See description for "ca_lookup"
            "position_column": "swivel_hbk",  # See description for "ca_lookup"
            "lookup": "swivel_hbk",  # See description for "ca_lookup"
            "val_max": "swivel_name",  # See description for "ca_lookup"
            "val_1": "swivel_1",  # See description for "ca_lookup"
            "val_2": "swivel_2",
            "additional": ["suspension_description"]
        }
        df_hbk = cls_obj._excel_read_ca_set_file_all(
            "Lines - Asset Attributes",
            "Asset",
            -2,
            "Towers",
            "",
            columns_lookup,
            "set"
        ).sort_index()

        'Add hanger bracket information'
        df_set["hbk"] = df_hbk.loc[:, "swivel_hbk"].map(lambda x: 1 if str(x).lower() == "hbk" else 0)

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
        if table_string in header_labels:
            row = header_indexes[header_labels.index(table_string)]
        else:
            row = -1

        'Create dataframe from Excel object and convert to code name format'
        if row == -1:
            df = pd.DataFrame(columns=columns_return)
        else:
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
        df = df.loc[df.loc[:, lookup_info["lookup"]].notna(), :]
        if len(position) > 0:
            df = df.loc[df.loc[:, lookup_info["position_column"]] == position, :]

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

        'Remove duplicates'
        df["line_tow_circ"] = df.loc[:, "line_id"] + "_" + df.loc[:, "structure_number"] + "_" + df.loc[:, "circuit1"]
        df = df.loc[~df.loc[:, "line_tow_circ"].duplicated(keep="last"), :]
        df = df.drop(columns=["line_tow_circ"])

        if len(circuits) > 1:
            df1 = df.groupby("circuit1").get_group(circuits[0]).set_index("line_structure")
            df2 = df.groupby("circuit1").get_group(circuits[1]).set_index("line_structure")
            shapes = [df1.shape[0], df2.shape[0]]
            dfs = [df1, df2]
            'Sort to use dataframe with most entries as basis'
            indexes = [0, 1] if shapes[0] == shapes[1] else [shapes.index(max(shapes)), shapes.index(min(shapes))]
            df1, df2 = dfs[indexes[0]], dfs[indexes[1]]
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
