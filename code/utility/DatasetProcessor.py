# Imports
import ast
from datetime import datetime
from pydoc import locate

import dateutil.utils
import pandas as pd
import sklearn.preprocessing as sk_prep
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter
from sklearn.model_selection import KFold

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from typing import List
from torch.nn.utils.rnn import pad_sequence
from ast import literal_eval

from torch.utils.data import TensorDataset, DataLoader

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

END_ABORTED_MSG = ") - Aborted."
STATE_FEAS_MSG = "_state_feasible() - The process "
DATE_FORMAT = "%d-%m-%Y"

class DatasetProcessor:
    # preceding_constraints = { method_name : { 'state' : minimum state, 'precedings' : method and there argument to execute before }} #
    preceding_constraints = {

        # Preprocessing
        'import_dataset': {'state': 0, 'modif' : True},
        'import_dataset_from_object': {'state': 0, 'modif' : True},
        'vec_to_record': {'state': 1, 'modif' : True },
        'rename_attributes': {'state': 2, 'modif' : True},
        'group_attributes': {'state': 2, 'modif': True},
        'clean_na': {'state':3, 'modif' : True},
        'clean_value': {'state': 3, 'modif': True},
        'replace': {'state': 4, 'modif' : True},
        'remove_duplicates': {'state': 6, 'modif' : True},
        'sample': {'state': 7, 'modif' : True},
        'clean_sparse_data': {'state': 7, 'modif' : True},
        'enc_categories_feat': {'state': 8, 'modif' : True},
        'create_ir2index': {'state': 9, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif': True},

        # Processing
        'shuffle': {'state': 10, 'modif' : True},
        'records_to_vec': {'state': 10, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif': True},
        'create_know_vec': {'state': 10, 'precedings': ('enc_categories_feat', ['skill_id']), 'modif': True},

        'distribute_items': {'state': 11, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif' : True},
        'train_test_split': {'state': 11, 'precedings': ('enc_categories_feat', ['user_id']), 'modif' : True},

        'compute_mean_skills': {'state': 12, 'precedings': ('enc_categories_feat', ['skill_id', 'user_id']),
                                'modif': True},


        'ir2index_array': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif' : False},
        'ir2index' : {'state': 12, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif' : False},
        'index2ir': {'state': 12, 'precedings': ('create_ir2index', None), 'modif' : False},
        'index2ir_array': {'state': 12, 'precedings': ('create_ir2index', None), 'modif' : False},
        'index_complement': {'state': 12, 'precedings': ('create_ir2index', None), 'modif' : False},


        # todo : remove or modify it attention : pas au point! 'replace_vec_values': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif': True},
        'generate_train_map': {'state': 12, 'precedings': [('train_test_split', None), ('records_to_vec', None)], 'modif' : True},
        'generate_train_valid_map': {'state': 12, 'precedings': [('train_test_split', None), ('records_to_vec', None)], 'modif' : True},
        'generate_test_map': {'state': 12, 'precedings': [('train_test_split', None), ('records_to_vec', None)], 'modif' : True},
        'generate_test_list': {'state': 12, 'precedings': [('train_test_split', None)], 'modif' : True},
        'generate_test_matrix': {'state': 12, 'precedings': [('train_test_split', None)], 'modif': True},
        'generate_valid_matrix': {'state': 12, 'precedings': [('train_test_split', None)], 'modif': True},
        'generate_valid_list': {'state': 12, 'precedings': [('train_test_split', None)], 'modif' : True},

        'transform': {'state': 12, 'precedings': ('train_test_split', None), 'modif': True},
        'get_triplet': {'state': 12, 'precedings': ('train_test_split', None), 'modif': True},

        'DINA_transform': {'state': 13, 'precedings': [('train_test_split', None),('create_know_vec',None)], 'modif': True},
        'NCDM_transform': {'state': 13, 'precedings': [('train_test_split', None), ('create_know_vec', None)],
                           'modif': True},
        'IRR_transform': {'state': 13, 'precedings': [('train_test_split', None), ('create_know_vec', None)],
                           'modif': True},

        # Post-processing
        'result_landmarks': {'state':12,'precedings': ('train_test_split', None), 'modif' : False},

        # Plotting
        'display_responses_per_user': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif' : False},
        'display_responses_per_item': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif' : False},
        'display_response_map': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif' : False},
        'display_skill_map': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif' : False},
        'plot_distribution': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id', 'user_id']), 'modif' : False},
        'plot_response_prop_per_item': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id']), 'modif' : False},
        'plot_response_prop_given_item_count': {'state': 12, 'precedings': ('enc_categories_feat', ['item_id']), 'modif' : False},
    }

    def __init__(self, experiment_nb, data_path="./Data", experiment_path="./Experiments",
                 logging_level=logging.INFO):

        # ---- Attributes declaration
        ## -- Datasets attributes
        self.dataset_state = {}  # State of preprocessing of each dataset. Used to implement preceding constraints in
        # the preprocessing of each dataset. Format : {key= (str) dataset_name, value= (int) state}
        self.metadata = {}
        self.datasets = {}  # row datasets, format : pd.Dataframe
        self.maps = {}  # matrices of responses for each dataset, size = (user_id,item_id), format : np.ndarray
        self.train_maps = {}  # matrices of train and valid responses for each dataset, size = (user_id,item_id), format : np.ndarray
        self.train_valid_maps = {} # matrices of train and valid responses for each dataset, size = (user_id,item_id), format : np.ndarray
        self.train_valid_size_per_user = {}  # arrays of the number of train logs for each user
        self.valid_steps = {} # Vectors of the number of responses of each valid response's user in the train datasets
        self.valid_logs = {} # lists of the logs in the testing set
        self.valid_mats = {}  # matrices of valid responses for each dataset, size = (test_log_nb, 3), format : np.ndarray, column : user_id,item_id,response
        self.test_maps = {} # matrices of test responses for each dataset, size = (user_id,item_id), format : np.ndarray
        self.test_mats = {} # matrices of test responses for each dataset, size = (test_log_nb, 3), format : np.ndarray, column : user_id,item_id,response
        self.test_logs = {}  # lists of the logs in the testing set
        self.test_steps = {} # Vectors of the number of responses of each test response's user in the train and valid datasets
        self.skills_maps = {}  # matrices of the skills per user, size = (user_id,skill_id), format : np.ndarray
        self.users = {}  # data information of each dataset's user, format : pd.Dataframe
        self.skills_nb = {} # dictionaries for each dataset of the nb of different values for skill_id. Necessary because this attribute is aggregated into list


        ## -- Dataprocessing attributes
        self.methods = dir(self)
        self.data_path = data_path
        self.xp_name = experiment_nb
        self.exp_group_path = experiment_path + "/" + self.xp_name
        self.xp_path = self.exp_group_path + "/" + str(datetime.now().strftime("%d-%m-%Y-%Hh%M"))

        self.logger = logging.getLogger('experimentLogger')
        self.logger.handlers.clear()

        # ---- Initialisation routine
        self._create_directory(self.exp_group_path)
        self._create_directory(self.xp_path)

        log_name = '/running.log'
        try:
            f = open(self.xp_path + log_name, 'w')
            f.close()
        except OSError as error:
            self.logger.warning(
                "create logs - handled error when creating a file at " + self.xp_path + log_name + ": " + str(
                    error))

        fh = logging.FileHandler(self.xp_path + log_name)
        fh.setLevel(logging_level)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        self.logger.setLevel(logging_level)
        self.logger.addHandler(fh)

        self.logger.info("__init__()")
        self._metadata_update(added_value={'exp_nb':self.xp_name})

    # PRIVATE METHODS -----------------------------------------------

    def _create_directory(self, path: str):
        """Create a directory to store all the data and metadata from the experiment to ensure reproducibility.

        Returns:
            void
        """
        self.logger.debug("_create_directory(" + path + ")")
        try:
            os.mkdir(path)
        except OSError as error:
            self.logger.warning(
                "_create_directory() - handled error when creating a directory at " + path + ": " + str(error))

    def _write_metadata(self, dataset_name=None):
        if dataset_name is not None :
            metadata = self.metadata[dataset_name]
        else :
            metadata = self.metadata
        json_object = json.dumps(metadata)
        with open(self.xp_path + "/metadata_"+str(dataset_name) + "_" + str(datetime.now().strftime(DATE_FORMAT))+".json", 'w') as outfile:
            outfile.write(json_object)

    def _metadata_update(self, dataset_name: str=None, added_value=None, saving=False):
        """Update the *operation_name* entry of the *metadata* dictionary by adding the *added_value* or replacing everything by *cleaning_value*

        Args:
            dataset_name (str): Name of the *dataset_name* entry
            added_value : Value to add to the *operation_name* entry

        Returns:
            void
        """
        self.logger.debug("_metadata_update(dataset_name=" + str(dataset_name) + ",added_value=" + str(
            added_value) )
        if dataset_name is not None :
            if dataset_name in self.metadata.keys():
                self.metadata[dataset_name].update(added_value)
            else:
                self.metadata[dataset_name] = added_value
        else :
            self.metadata.update(added_value)
        if saving:
            self._write_metadata()

    def _matching_requirements(self, process_name: str, dataset_name: str,
                               parameters: List[str] = None) -> bool:
        """Returns whether the *process_name* preprocessing has been applied on the *dataset_name* dataset  with the given parameters (True) or not (False).

        Args:
            process_name (str): Name of the process to check
            dataset_name (str): Name of the dataset to work on
            parameters (List[str]): List of the attributes name used in the process

        Returns:
            void
        """

        self.logger.debug(
            "_matching_requirements(process_name=" + process_name + ",dataset_name=" + dataset_name + ",parameters=" + str(
                parameters))

        if dataset_name in self.metadata.keys():
            if process_name == 'enc_categories_feat' or process_name == 'enc_categories_feat_agg':
                if process_name in self.metadata[dataset_name].keys():
                    if parameters is None:

                        return True
                    else:
                        for feat in parameters:
                            if feat not in self.metadata[dataset_name][process_name].keys():
                                self.logger.error(
                                    "Unsatisfied requirements : the dataset's " + feat + " attribute(s) need to be re-encoded.")
                                return False

                    return True
                else:
                    self.logger.error(
                        "Unsatisfied requirements : the " + str(dataset_name) + " dataset attribute(s) " + str(
                            parameters) + " need to be preprocessed with " + str(process_name) + ".")
                    return False

            elif process_name in ['distribute_items','train_test_split','records_to_vec','create_ir2index','create_know_vec','get_triplet'] :
                    if process_name in self.metadata[dataset_name].keys():
                        return True
                    else:
                        self.logger.error("Unsatisfied requirements : the process " + str(process_name) + " has not been used for dataset "+str(dataset_name))
                        return False
            else:
                self.logger.error('Unhandeled process checking')

                return False
        else :
            self.logger.error("Unsatisfied requirements : the " + str(
                dataset_name) + " dataset need to be preprocessed with " + str(process_name) + ". No process were yet done.")
            return False

    def _state_feasible(self, process_name: str, dataset_name: str) -> bool:

        if dataset_name not in self.datasets.keys() and dataset_name not in self.maps.keys() :
            if self.preceding_constraints[process_name]['state'] != 0:  # If the process is not import_dataset
                return False
            else:
                return True
        elif self.preceding_constraints[process_name]['state'] < self.dataset_state[dataset_name]:
            self.logger.error(STATE_FEAS_MSG + str(process_name) + " is not feasible while " + str(
                dataset_name) + " is in state " + str(self.dataset_state[dataset_name]))
            return False
        elif 'precedings' in self.preceding_constraints[process_name].keys():
            if isinstance(self.preceding_constraints[process_name]['precedings'],list) :
                for pc in self.preceding_constraints[process_name]['precedings'] :
                    if not self._matching_requirements(pc[0],dataset_name,pc[1]):
                        self.logger.error(
                            STATE_FEAS_MSG + str(
                                process_name) + " is not feasible with unmatched requirements method " + str(pc[0])
                            + "() with parameters " + str(pc[1]))
                        return False
            else :
                if not self._matching_requirements(self.preceding_constraints[process_name]['precedings'][0], dataset_name,
                                                   self.preceding_constraints[process_name]['precedings'][1]):
                    self.logger.error(
                        STATE_FEAS_MSG + str(
                            process_name) + " is not feasible with unmatched requirements method " + str(
                            self.preceding_constraints[process_name]['precedings'][0]) + "() with parameters " + str(
                            self.preceding_constraints[process_name]['precedings'][1]))
                    return False

        if self.preceding_constraints[process_name]['modif'] : # State is updated only if the process modified a dataset attribute
            self.dataset_state[dataset_name] = self.preceding_constraints[process_name]['state']
        return True

    # PUBLIC METHODS -----------------------------------------------

    ## Imports -------------

    def import_dataset_from_object(self,dataset,dataset_name: str = None, data_format: str = "array"):
        self.logger.info("import_dataset_from_object( dataset_name = " + dataset_name + ")")
        if self._state_feasible('import_dataset_from_object', dataset_name):
            if data_format == "array" :
                self.maps[dataset_name] = dataset
                self.dataset_state[dataset_name] = 0
        else :
            self.logger.error("import_dataset_from_object( dataset_name= " + dataset_name + END_ABORTED_MSG)

    def import_dataset(self, dataset_file_name: str = None, metadata_file_name:str =None, dataset_name: str = None,
                       data_format: str = "records", item_id_names: List[str] = None, user_id_column: str = None,encoding="ISO-8859-15",low_memory=True,header='infer'):
        """Import the dataset from the *dataset_file_name* at *self.data_path*

        Args:
            dataset_file_name (str): Name of the file containing the dataset to import

        Returns:
            void
        """

        self.logger.info("import_dataset(" + str(dataset_file_name) + ", metadata_file_name=" + str(metadata_file_name)+", dataset_name="+ str(dataset_name) +
                       ", data_format=" +str(data_format) + ", item_id_names=" + str(item_id_names) + "; user_id_column : " +str(user_id_column)+ ")")

        if self._state_feasible('import_dataset', dataset_name):

            if item_id_names is None:
                item_id_names = []
            if user_id_column is None:
                user_id_column = []

            [name, file_format] = dataset_file_name.split(".")

            if dataset_name is not None:
                name = dataset_name
            if data_format == 'records':
                if file_format == 'csv':
                    self.datasets[name] = pd.read_csv(self.data_path + "/" + dataset_file_name, encoding="ISO-8859-15",
                                                      low_memory=False)
                elif file_format == 'xls' or file_format == 'xlsx':
                    self.datasets[name] = pd.read_excel(self.data_path + "/" + dataset_file_name)
                else:
                    self.logger.error(
                        "import_dataset() - unhandled format " + file_format + " of file " + dataset_file_name)

            elif data_format == 'map':
                if file_format == 'csv':
                    self.maps[name] = pd.read_csv(self.data_path + "/" + dataset_file_name, encoding=None,low_memory=low_memory,header=header).to_numpy()
                elif file_format == 'xls' or file_format == 'xlsx':
                    self.maps[name] = pd.read_excel(self.data_path + "/" + dataset_file_name).to_numpy()
                elif file_format == 'tab':
                    self.maps[name] = pd.read_table(self.data_path + "/" + dataset_file_name).to_numpy()
                else:
                    self.logger.error(
                        "import_dataset() - unhandled format " + (file_format) + " of file " + dataset_file_name)

            else:
                self.logger.error(
                    "import_dataset() - unhandled data format " + str(data_format))

            self.dataset_state[name] = 0
            self._metadata_update(dataset_name=dataset_name, added_value={"import_dataset":name})

            if metadata_file_name is not None :
                with open(self.data_path +"/"+metadata_file_name, 'r') as outfile:
                    json_object = json.load(outfile)
                self._metadata_update(dataset_name=dataset_name,added_value=json_object)

        else:
            self.logger.error("import_dataset(" + dataset_file_name + END_ABORTED_MSG)

    def import_datasets(self, datasets_file_name: List[str]):
        """Import the datasets from the *datasets_file_name* list at *self.data_path*

        Args:
            datasets_file_name (str): Name of the file containing the dataset to import

        Returns:
            void

        """
        self.logger.debug("import_dataset(" + str(datasets_file_name) + ")")

        for file in datasets_file_name:
            self.import_dataset(file)

    ## Data preprocessing -------------

    def process_from_file(self, dataset_name: str, process_description_filepath: str):
        """Apply a processing pipeline, defined in the *process_description_file*, on the *dataset_name*

        Args:
            process_description_filepath (str): Path of the file describing the canonical processing pipeline to apply to the dataset
            dataset_name (str): Name of the dataset to preprocess

        Returns:
            void

        """
        self.logger.info(
            "process_from_file(" + dataset_name + ", process_description_filepath=" + process_description_filepath + ")")

        with open(process_description_filepath, 'r') as input_file:
            processes = json.loads(input_file.read())

        for process in processes:
            process_name = list(process.keys())[0]
            attribute_dict = list(process.values())[0]

            method = getattr(self, self.methods[self.methods.index(process_name)])

            #TODO : remove this
            #attribute_dict.update({'dataset_name': dataset_name})
            method(**attribute_dict)

    def rename_attributes(self, dataset_name: str, modifs):

        self.logger.info("rename_attributes(" + dataset_name + ", attributes=" + str(modifs.keys()) + ")")
        if self._state_feasible('rename_attributes', dataset_name):

            modifs_dict = dict(modifs)
            self.datasets[dataset_name].rename(columns=modifs_dict, inplace=True)
            self._metadata_update(dataset_name=dataset_name,added_value={'rename_attributes': {'modifs': modifs}})
        else:
            self.logger.error(
                "rename_attributes(" + dataset_name + ", attributes=" + str(modifs.keys()) + END_ABORTED_MSG)

    def group_attributes(self, dataset_name: str, attributes:List[str],new_name:str):
        """
        Concatenate all the the attributes value into the single attribute named new_name
        """

        self.logger.info("group_attributes(" + dataset_name + ", attributes=" + str(attributes)+ ", new_name=" + str(new_name) + ")")
        if self._state_feasible('group_attributes', dataset_name):

            # Concatenate selected columns into a new column
            self.datasets[dataset_name][new_name] = self.datasets[dataset_name][attributes].astype(str).agg(
                ''.join, axis=1)

            self._metadata_update(dataset_name=dataset_name,added_value={'group_attributes': {'attributes': attributes,'new_name':new_name}})
        else:
            self.logger.error(
                "group_attributes(" + dataset_name + ", attributes=" + str(attributes)+ ", new_name=" + str(new_name) + END_ABORTED_MSG)

    def remove_duplicates(self, dataset_name: str, attributes: List[str],agg_attributes: List[str]=list()):

        self.logger.info("remove_duplicates(" + dataset_name + ", attributes=" + str(attributes)+ ", agg_attributes=" + str(agg_attributes) + ")")

        if self._state_feasible('remove_duplicates', dataset_name):

            special_attributes = attributes.copy()
            special_attributes.extend(agg_attributes)
            d = {}
            for agg_attr in agg_attributes :
                 d.update({agg_attr: set})
            d.update({col: 'first' for col in self.datasets[dataset_name].columns if col not in special_attributes})
            self.datasets[dataset_name] = self.datasets[dataset_name].groupby(attributes).agg(d).reset_index()

            self._metadata_update(dataset_name, {'remove_duplicates': {'attributes': attributes,"agg_attributes" :agg_attributes}})
        else:
            self.logger.error("remove_duplicates(" + dataset_name + ", attributes=" + str(attributes) + ", agg_attributes=" + str(agg_attributes)+ END_ABORTED_MSG)

    def replace(self, dataset_name: str, attributes: List[str], values: List[tuple] or List[str], types: List[str]):

        self.logger.info("replace(" + dataset_name + ", attributes=" + str(attributes) + ", values=" + str(
            values) + ", types=" + str(types) + ")")

        if self._state_feasible('replace', dataset_name):

            if type(attributes) == List[str]:
                attr2 = []
                for attr in attributes:
                    attr2.append(tuple(attr))

            for i_attr, attr in enumerate(attributes):
                self.datasets[dataset_name][attr].replace(to_replace=values[i_attr]['old'],
                                                          value=locate(types[i_attr])(values[i_attr]['new']),
                                                          inplace=True)
            for i_attr, attr in enumerate(attributes):
                self.datasets[dataset_name] = self.datasets[dataset_name].astype({attr: locate(types[i_attr])})

            self._metadata_update(dataset_name, {'replace': {'attributes': attributes, 'values': values}})
        else:
            self.logger.error("replace(" + dataset_name + ", attributes=" + str(attributes) + ", values=" + str(
                values) + END_ABORTED_MSG)

    def replace_vec_values(self, dataset_name: str, values: List[tuple] or List[str], types: List[str]):

        self.logger.info("replace_vec_values(" + dataset_name + ", values=" + str(
            values) + ", types=" + str(types) + ")")

        if self._state_feasible('replace_vec_values', dataset_name):

            new_array = np.copy(self.maps[dataset_name])
            for value in enumerate(values):
                old = value['old']
                new = value['new']
                mask = np.isin(self.maps[dataset_name], old)
                new_array[mask] = new

            self.maps[dataset_name] = new_array

            self._metadata_update(dataset_name, {'replace': {'values': values}})
        else:
            self.logger.error("replace_vec_values(" + dataset_name + ", values=" + str(
                values) + END_ABORTED_MSG)

    def clean_na(self, dataset_name: str, attributes: List[str]):
        """Remove rows from the *dataset_name* dataset in which the *attributes* are unknown (NaN value)

        Args:
            dataset_name (str): Name of the dataset to work on
            attributes (list): The attributes to check


        Returns:
            void
        """

        self.logger.info("clean_na(" + dataset_name + ", attributes=" + str(attributes) + ")")

        if self._state_feasible('clean_na', dataset_name):
            self.datasets[dataset_name] = self.datasets[dataset_name].dropna(subset=attributes, axis='index')

            self._metadata_update(dataset_name, {'clean_na': {'attributes': attributes}})
        else:
            self.logger.error("clean_na(" + dataset_name + ", attributes=" + str(attributes) + END_ABORTED_MSG)

    def clean_value(self, dataset_name: str, attributes: List[str], values: List[str]):
        """Remove rows from the *dataset_name* dataset in which the *attributes* are unknown (NaN value)

        Args:
            dataset_name (str): Name of the dataset to work on
            attributes (list): The attributes to check


        Returns:
            void
        """

        self.logger.info("clean_value(" + dataset_name + ", attributes=" + str(attributes)+ ", values=" + str(values) + ")")

        if self._state_feasible('clean_value', dataset_name):
            for i,attr in enumerate(attributes) :
                self.datasets[dataset_name] = self.datasets[dataset_name][self.datasets[dataset_name][attr]!=values[i]]

            self._metadata_update(dataset_name, {'clean_value': {'attributes': attributes,'values':values}})
        else:
            self.logger.error("clean_value(" + dataset_name + ", attributes=" + str(attributes)+ ", values=" + str(attributes) + END_ABORTED_MSG)

    def clean_sparse_data(self, dataset_name: str, attributes: List[str], t: List[int], iterate=False):
        """Remove row data when each seperated attribute's value has less than T records.

        Requirements :
            - Not being used after "enc_categories_feat" because it can break the regularity of the encoding
            - Not being used after "distribute_items" because it unrandomizes the support and query sets sampling

        Args:
            dataset_name (str): Name of the dataset to work on
            attributes (str): Name of the attribute to have in sufficient number
            t (int): The minimum number of records required per attribute
            iterate (bool): Whether to iterate on the conditions until all of them are simultaneously fulfilled

        Returns:
            void
        """

        self.logger.info(
            "clean_sparse_data(" + dataset_name + ", attributes=" + str(attributes) + ", t=" + str(
                t) + ", iterate=" + str(iterate) + ")")

        if self._state_feasible('clean_sparse_data', dataset_name):

            for i_attr, attr in enumerate(attributes):
                s = self.datasets[dataset_name][attr].value_counts()
                s2 = s[s >= t[i_attr]]
                self.datasets[dataset_name] = self.datasets[dataset_name][
                    self.datasets[dataset_name][attr].isin(s2.index)]

            if len(attributes) > 1 and iterate:  # if several attributes , there is a risk that performing one cleaning affect the other one. We therefor iterate the cleaning task until all conditions are matched
                matched_conditions = False
                while not matched_conditions:

                    for i_attr, attr in enumerate(attributes):
                        s = self.datasets[dataset_name][attr].value_counts()
                        s2 = s[s >= t[i_attr]]
                        self.datasets[dataset_name] = self.datasets[dataset_name][
                            self.datasets[dataset_name][attr].isin(s2.index)]

                    matched_conditions = True
                    for i_attr, attr in enumerate(attributes):
                        s = self.datasets[dataset_name][attr].value_counts()
                        s2 = s[s < t[i_attr]]

                        if s2.size > 0:
                            matched_conditions = False
                            break

            self._metadata_update(dataset_name, {'clean_sparse_data': {'attributes': attributes, 'T': t}})
        else:
            self.logger.error(
                "clean_sparse_data(" + dataset_name + ", attributes=" + str(attributes) + ", iterate=" + str(
                    iterate) + ", t=" + str(
                    t) + END_ABORTED_MSG)

    def sample(self, dataset_name: str, nb_logs,rgn_seed=123):
        """Randomly select nb_logs among all the logs.

        Requirements :
            - Not being used after "enc_categories_feat" because it can break the regularity of the encoding
            - Not being used after "distribute_items" because it unrandomizes the support and query sets sampling

        Args:
            dataset_name (str): Name of the dataset to work on
            nb_logs (int): The nb of logs to keep

        Returns:
            void
        """

        self.logger.info("sample(" + dataset_name + ",nb_logs=" + str(nb_logs)+")")

        if self._state_feasible('sample', dataset_name):
            if self.datasets[dataset_name].shape[0] < nb_logs :
                self.logger.warning("sample() - The original number of logs ("+str(self.datasets[dataset_name].shape[0])+" logs) is inferior to the nb of expected samples ("+str(nb_logs)+" logs) therefore we sample with the possibility of choosing several times a same row.")
                #self.datasets[dataset_name] = self.datasets[dataset_name].sample(n=nb_logs, random_state=rgn_seed, replace=False)
            else :
                self.datasets[dataset_name] = self.datasets[dataset_name].sample(n=nb_logs,random_state=rgn_seed)

            self._metadata_update(dataset_name, {'sample': {'nb_logs': nb_logs}})
        else:
            self.logger.error("sample(" + dataset_name + ", nb_logs=" + str(nb_logs) + END_ABORTED_MSG)

    def shuffle(self, dataset_name: str, attributes: List[str], group_attributes: List[str],rgn_seed=123):
        """
        """

        self.logger.info("shuffle(" + str(dataset_name) + ",attributes=" + str(attributes) + ")")

        if self._state_feasible('shuffle', dataset_name):

            np.random.seed(rgn_seed)

            for i_attr, attr in enumerate(attributes):
                for name, group in self.datasets[dataset_name].groupby(group_attributes[i_attr]):
                    values = group[attr].to_numpy()
                    np.random.shuffle(values)
                    #group.loc[:, attr] = values
                    self.datasets[dataset_name].loc[group.index,attr]=values
                self.datasets[dataset_name] = self.datasets[dataset_name].sort_values(by=attr)

            self._metadata_update(dataset_name, {'shuffle': {'attributes': attributes}})
        else:
            self.logger.error("shuffle(" + dataset_name + ", attributes=" + str(attributes) + END_ABORTED_MSG)

    def enc_categories_feat(self, dataset_name: str, attributes: list, encoding_missing_value: float = None):
        """Encode the categorical features of the attributes of the dataset named dataset_name. Series attributes required.

        Args:
            dataset_name (str): Name of the dataset to work on
            attributes (List[str]): List of the attributes name to encode
            encoding_missing_value (float) : value to replace the dataset empty data of the *attributes* with. If None, the entire row of the missing data is deleted

        Returns:
            void
        """
        self.logger.info("enc_categories_feat(" + dataset_name + "," + str(attributes) + ")")
        if self._state_feasible('enc_categories_feat', dataset_name):


            if encoding_missing_value is None:
                self.logger.warning(
                    "If the method clean_na has not been used, there may still be nan values for attributes " + str(
                        attributes))
                enc = sk_prep.OrdinalEncoder()
            else:
                enc = sk_prep.OrdinalEncoder(encoded_missing_value=encoding_missing_value)

            for i, a in enumerate(attributes):
                self.datasets[dataset_name] = self.datasets[dataset_name].explode(a).reset_index(drop=True)

            categorized_data = enc.fit_transform(self.datasets[dataset_name][attributes])
            categorized_data = categorized_data.astype(np.longlong)

            caract = {}
            for i, a in enumerate(attributes):
                #print(a)
                #print(self.datasets[dataset_name][a].unique().shape)
                #print(self.datasets[dataset_name].loc[0,"skill_id"])

                #print(self.datasets[dataset_name].loc[0, "skill_id"])
                self.datasets[dataset_name][a] = categorized_data[:, i]
                caract[str(a)] = {"categories_nb": self.datasets[dataset_name][a].unique().shape[0]}

            d = {"skill_id": set}
            d.update({col: 'first' for col in self.datasets[dataset_name].columns if col not in ['skill_id']})
            self.datasets[dataset_name] = self.datasets[dataset_name].groupby(["user_id", "item_id"]).agg(d).reset_index(
                drop=True)
            #self.datasets[dataset_name] = self.datasets[dataset_name].groupby(["user_id", "item_id"]).agg({"skill_id": set}).reset_index(drop=True)
            self._metadata_update(dataset_name=dataset_name, added_value={'enc_categories_feat': caract})
        else:
            self.logger.error("enc_categories_feat(" + dataset_name + "," + str(attributes) + END_ABORTED_MSG)

    ## Learning preparation -------------

    def distribute_items(self, dataset_name: str, nb_support_items=5, nb_test_items=0, rgn_seed=1234):
        """Add an attribute to the dataset *dataset_name* which randomly assign each record to the support_set (0), the query_set (1) or the test_set (2) with a specific number of items per user in each set

        Requirements : item_id and user_id features have been re-encoded.

        Args:
            dataset_name (str): Name of the dataset to work on
            nb_support_items (int): Number of items per user in the support set
            nb_test_items (int): Number of items per user in the test set

        Returns:
            void
        """

        self.logger.info("distribute_items(" + dataset_name + ", nb_support_items=" + str(
            nb_support_items) + ", nb_test_items=" + str(nb_test_items) + ")")

        if self._state_feasible('distribute_items', dataset_name):

            for user in self.datasets[dataset_name]['user_id'].unique():
                items = np.array(self.datasets[dataset_name][self.datasets[dataset_name]['user_id'] == user].index)
                rng = np.random.default_rng(rgn_seed)
                rng.shuffle(items)
                self.datasets[dataset_name].loc[items[0:nb_support_items], 'item_distribution'] = 0  # support
                self.datasets[dataset_name].loc[
                    items[nb_support_items:len(items) - nb_test_items], 'item_distribution'] = 1  # query
                self.datasets[dataset_name].loc[
                    items[len(items) - nb_test_items:len(items)], 'item_distribution'] = 2  # test

            support_set = self.datasets[dataset_name][self.datasets[dataset_name]['item_distribution'] == 0]
            query_set = self.datasets[dataset_name][self.datasets[dataset_name]['item_distribution'] == 1]
            test_set = self.datasets[dataset_name][self.datasets[dataset_name]['item_distribution'] == 2]

            self._metadata_update(dataset_name, {
                'distribute_items': {'nb_users_QS': len(query_set['user_id'].unique()),
                               'nb_users_SS': len(support_set['user_id'].unique()),
                               'nb_users_TS': len(test_set['user_id'].unique()),
                               'nb_items_QS': len(query_set['item_id']),
                               'nb_items_SS': len(support_set['item_id']),
                               'nb_items_TS': len(test_set['item_id']),
                               'nb_unique_items_QS': len(query_set['item_id'].unique()),
                               'nb_unique_items_SS': len(support_set['item_id'].unique()),
                               'nb_unique_items_TS': len(test_set['item_id'].unique()),
                               'nb_support_items': nb_support_items,
                               'random_generator_seed': rgn_seed
                               }})
        else:
            self.logger.error("distribute_items(" + dataset_name + ", nb_support_items=" + str(
                nb_support_items) + ", nb_test_items=" + str(nb_test_items) + END_ABORTED_MSG)

    def train_test_split(self, dataset_name: str, test_proportion: float = 0.2, valid_proportion: float = 0.2, profile_len = 10,
                         n_folds: int = 5, i_fold: int=None, rgn_seed: int = 1):
        """

        Requirements :

        Args:
            dataset_name (str): Name of the dataset t)o work on

        Returns:
            void
        """

        self.logger.info("train_test_split(" + dataset_name + ", test_proportion=" + str(
            test_proportion) + ", valid_proportion=" + str(valid_proportion) + ")")

        self.dataset_state[dataset_name] = self.preceding_constraints['train_test_split']['state']
        if self._state_feasible('train_test_split', dataset_name):

            self.datasets[dataset_name]['split'] = pd.Series(dtype=str)

            user_list = []
            train_valid_log_nb_list = []
            train_log_nb_list = []
            kf = KFold(n_splits=n_folds,shuffle=False)

            self.datasets[dataset_name] = self.datasets[dataset_name].sort_values(by='start_time')
            for i_group, group in self.datasets[dataset_name].groupby('user_id') :
                group_idxs = np.array(group.index)
                folds = list(kf.split(group_idxs))

                train_valid_fold_idx, test_fold_idx = folds[i_fold]
                train_valid_item_idx = group_idxs[train_valid_fold_idx]
                test_item_idx = group_idxs[test_fold_idx]

                train_item_idx, valid_item_idx = train_test_split(train_valid_item_idx, test_size=float(valid_proportion)/(1.0-float(test_proportion)),
                                                                 random_state=int(rgn_seed), shuffle=False)

                #todo : big train test split modif to test
                # self.datasets[dataset_name].loc[group_idxs[-1], 'split'] = 'test'
                # self.datasets[dataset_name].loc[group_idxs[-2], 'split'] = 'valid'
                # self.datasets[dataset_name].loc[group_idxs[:-2], 'split'] = 'train'

                self.datasets[dataset_name].loc[test_item_idx, 'split'] = 'test'
                self.datasets[dataset_name].loc[valid_item_idx, 'split'] = 'valid'
                self.datasets[dataset_name].loc[train_item_idx, 'split'] = 'train'

                self.datasets[dataset_name].loc[test_item_idx, 'train_valid_log_nb'] = train_valid_item_idx.shape[0]
                self.datasets[dataset_name].loc[valid_item_idx, 'train_log_nb'] = train_item_idx.shape[0]

                self.valid_steps[dataset_name] = self.get_valid_dataset(dataset_name=dataset_name)['train_log_nb'].to_numpy().astype(int)
                self.test_steps[dataset_name] = self.get_test_dataset(dataset_name=dataset_name)['train_valid_log_nb'].to_numpy().astype(int)

                user_list.append(i_group)
                train_valid_log_nb_list.append(train_valid_item_idx.shape[0])
                train_log_nb_list.append(train_item_idx.shape[0])

            if 'skill_id' in self.datasets[dataset_name].keys() and not isinstance(self.datasets[dataset_name]['skill_id'].iloc[0],int) :
                if isinstance(self.datasets[dataset_name]['skill_id'].iloc[0],str) :
                    self.datasets[dataset_name]['skill_id'] = self.datasets[dataset_name]['skill_id'].apply(
                        lambda x: literal_eval(x))
                self.datasets[dataset_name] = self.datasets[dataset_name].explode('skill_id').reset_index(drop=True)

            self.users[dataset_name] = pd.DataFrame({'user_id':user_list,'train_valid_log_nb':train_valid_log_nb_list,'train_log_nb':train_log_nb_list })

            # # User list creation
            # self.users[dataset_name] = pd.DataFrame(self.datasets[dataset_name]['user_id'].unique())
            # self.users[dataset_name].rename(columns={0: 'user_id'}, inplace=True)
            # user_id_index = np.array(self.users[dataset_name].index)
            #
            # np.random.seed(rgn_seed)
            # np.random.shuffle(user_id_index)
            #
            # # Horizontal split (Users)
            # ## train, valid <-> test
            # train_valid_user_id_idx, test_user_id_idx = train_test_split(user_id_index,
            #                                                        test_size=float(test_proportion),
            #                                                        random_state=int(rgn_seed))
            #
            # self.users[dataset_name].loc[test_user_id_idx, 'subset'] = 'test'
            #
            # ## train <-> valid
            # train_user_idx, valid_user_idx = train_test_split(train_valid_user_id_idx,
            #                                                   test_size=float(valid_proportion),
            #                                                   random_state=int(rgn_seed))
            # self.users[dataset_name].loc[train_user_idx, 'subset'] = "train"
            # self.users[dataset_name].loc[valid_user_idx, 'subset'] = "valid"
            #
            # ## users prof_len=1 <-> ... <->users prof_len=n
            # user_id_idx_list_of_prof_len = np.split(test_user_id_idx,profile_len)
            #
            # for profile_len,user_id_idx in enumerate(user_id_idx_list_of_prof_len) :
            #     self.users[dataset_name].loc[user_id_idx, 'profile_len'] = profile_len+1
            #
            # # Vertical split (Logs)
            # for user in self.get_train_user_id(dataset_name) :
            #     print("hello")





            # # Folding
            # if n_folds is not None:
            #     np.random.shuffle(train_valid_user_id_idx)
            #
            #     for fold in range(n_folds):
            #         fold_users_idx = train_valid_user_id_idx[int((fold) * len(train_valid_user_id_idx) / n_folds):int(
            #             (fold + 1) * len(train_valid_user_id_idx) / n_folds)]
            #
            #         self.users[dataset_name].loc[fold_users_idx, 'subset'] = "cross_valid"
            #         self.users[dataset_name].loc[fold_users_idx, 'fold'] = fold
            # else:
            #     train_user_idx, valid_user_idx = train_test_split(train_valid_user_id_idx,
            #                                                       test_size=float(valid_proportion),
            #                                                       random_state=int(rgn_seed))
            #     self.users[dataset_name].loc[train_user_idx, 'subset'] = "train"
            #     self.users[dataset_name].loc[valid_user_idx, 'subset'] = "valid"
            #     self.users[dataset_name].loc[train_valid_user_id_idx, 'fold'] = 0
            #
            # # Report folds from users to log
            # for user in self.datasets[dataset_name]['user_id'].unique():
            #     self.datasets[dataset_name].loc[self.datasets[dataset_name]['user_id'] == user, 'fold'] = \
            #         self.users[dataset_name][self.users[dataset_name]['user_id'] == user]['fold'].to_numpy()[0]

            self._metadata_update(dataset_name, {
                'train_test_split': {'test_proportion': test_proportion, 'valid_proportion': valid_proportion,
                               'n_folds': n_folds, 'rgn_seed': rgn_seed}})
        else:
            self.logger.error("train_test_split(" + dataset_name + ", test_proportion=" + str(
                test_proportion) + ", valid_proportion=" + str(valid_proportion) + END_ABORTED_MSG)


    def train_test_split(self, dataset_name: str, test_proportion: float = 0.2, valid_proportion: float = 0.2, profile_len = 10,
                         n_folds: int = 5, i_fold: int=None, rgn_seed: int = 1):
        """

        Requirements :

        Args:
            dataset_name (str): Name of the dataset t)o work on

        Returns:
            void
        """

        self.logger.info("train_test_split(" + dataset_name + ", test_proportion=" + str(
            test_proportion) + ", valid_proportion=" + str(valid_proportion) + ")")

        self.dataset_state[dataset_name] = self.preceding_constraints['train_test_split']['state']
        if self._state_feasible('train_test_split', dataset_name):

            self.datasets[dataset_name]['split'] = pd.Series(dtype=str)

            user_list = []
            train_valid_log_nb_list = []
            train_log_nb_list = []
            kf = KFold(n_splits=n_folds,shuffle=False)

            self.datasets[dataset_name] = self.datasets[dataset_name].sort_values(by='start_time')
            for i_group, group in self.datasets[dataset_name].groupby('user_id') :
                group_idxs = np.array(group.index)
                folds = list(kf.split(group_idxs))

                train_valid_fold_idx, test_fold_idx = folds[i_fold]
                train_valid_item_idx = group_idxs[train_valid_fold_idx]
                test_item_idx = group_idxs[test_fold_idx]

                train_item_idx, valid_item_idx = train_test_split(train_valid_item_idx, test_size=float(valid_proportion)/(1.0-float(test_proportion)),
                                                                 random_state=int(rgn_seed), shuffle=False)

                #todo : big train test split modif to test
                # self.datasets[dataset_name].loc[group_idxs[-1], 'split'] = 'test'
                # self.datasets[dataset_name].loc[group_idxs[-2], 'split'] = 'valid'
                # self.datasets[dataset_name].loc[group_idxs[:-2], 'split'] = 'train'

                self.datasets[dataset_name].loc[test_item_idx, 'split'] = 'test'
                self.datasets[dataset_name].loc[valid_item_idx, 'split'] = 'valid'
                self.datasets[dataset_name].loc[train_item_idx, 'split'] = 'train'

                self.datasets[dataset_name].loc[test_item_idx, 'train_valid_log_nb'] = train_valid_item_idx.shape[0]
                self.datasets[dataset_name].loc[valid_item_idx, 'train_log_nb'] = train_item_idx.shape[0]

                self.valid_steps[dataset_name] = self.get_valid_dataset(dataset_name=dataset_name)['train_log_nb'].to_numpy().astype(int)
                self.test_steps[dataset_name] = self.get_test_dataset(dataset_name=dataset_name)['train_valid_log_nb'].to_numpy().astype(int)

                user_list.append(i_group)
                train_valid_log_nb_list.append(train_valid_item_idx.shape[0])
                train_log_nb_list.append(train_item_idx.shape[0])

            if 'skill_id' in self.datasets[dataset_name].keys() and not isinstance(self.datasets[dataset_name]['skill_id'].iloc[0],int) :
                if isinstance(self.datasets[dataset_name]['skill_id'].iloc[0],str) :
                    self.datasets[dataset_name]['skill_id'] = self.datasets[dataset_name]['skill_id'].apply(
                        lambda x: literal_eval(x))
                self.datasets[dataset_name] = self.datasets[dataset_name].explode('skill_id').reset_index(drop=True)

            self.users[dataset_name] = pd.DataFrame({'user_id':user_list,'train_valid_log_nb':train_valid_log_nb_list,'train_log_nb':train_log_nb_list })

            # # User list creation
            # self.users[dataset_name] = pd.DataFrame(self.datasets[dataset_name]['user_id'].unique())
            # self.users[dataset_name].rename(columns={0: 'user_id'}, inplace=True)
            # user_id_index = np.array(self.users[dataset_name].index)
            #
            # np.random.seed(rgn_seed)
            # np.random.shuffle(user_id_index)
            #
            # # Horizontal split (Users)
            # ## train, valid <-> test
            # train_valid_user_id_idx, test_user_id_idx = train_test_split(user_id_index,
            #                                                        test_size=float(test_proportion),
            #                                                        random_state=int(rgn_seed))
            #
            # self.users[dataset_name].loc[test_user_id_idx, 'subset'] = 'test'
            #
            # ## train <-> valid
            # train_user_idx, valid_user_idx = train_test_split(train_valid_user_id_idx,
            #                                                   test_size=float(valid_proportion),
            #                                                   random_state=int(rgn_seed))
            # self.users[dataset_name].loc[train_user_idx, 'subset'] = "train"
            # self.users[dataset_name].loc[valid_user_idx, 'subset'] = "valid"
            #
            # ## users prof_len=1 <-> ... <->users prof_len=n
            # user_id_idx_list_of_prof_len = np.split(test_user_id_idx,profile_len)
            #
            # for profile_len,user_id_idx in enumerate(user_id_idx_list_of_prof_len) :
            #     self.users[dataset_name].loc[user_id_idx, 'profile_len'] = profile_len+1
            #
            # # Vertical split (Logs)
            # for user in self.get_train_user_id(dataset_name) :
            #     print("hello")





            # # Folding
            # if n_folds is not None:
            #     np.random.shuffle(train_valid_user_id_idx)
            #
            #     for fold in range(n_folds):
            #         fold_users_idx = train_valid_user_id_idx[int((fold) * len(train_valid_user_id_idx) / n_folds):int(
            #             (fold + 1) * len(train_valid_user_id_idx) / n_folds)]
            #
            #         self.users[dataset_name].loc[fold_users_idx, 'subset'] = "cross_valid"
            #         self.users[dataset_name].loc[fold_users_idx, 'fold'] = fold
            # else:
            #     train_user_idx, valid_user_idx = train_test_split(train_valid_user_id_idx,
            #                                                       test_size=float(valid_proportion),
            #                                                       random_state=int(rgn_seed))
            #     self.users[dataset_name].loc[train_user_idx, 'subset'] = "train"
            #     self.users[dataset_name].loc[valid_user_idx, 'subset'] = "valid"
            #     self.users[dataset_name].loc[train_valid_user_id_idx, 'fold'] = 0
            #
            # # Report folds from users to log
            # for user in self.datasets[dataset_name]['user_id'].unique():
            #     self.datasets[dataset_name].loc[self.datasets[dataset_name]['user_id'] == user, 'fold'] = \
            #         self.users[dataset_name][self.users[dataset_name]['user_id'] == user]['fold'].to_numpy()[0]

            self._metadata_update(dataset_name, {
                'train_test_split': {'test_proportion': test_proportion, 'valid_proportion': valid_proportion,
                               'n_folds': n_folds, 'rgn_seed': rgn_seed}})
        else:
            self.logger.error("train_test_split(" + dataset_name + ", test_proportion=" + str(
                test_proportion) + ", valid_proportion=" + str(valid_proportion) + END_ABORTED_MSG)


    def records_to_vec(self, dataset_name: str, coordinate_name: str = 'correct',
                       init_value: int = -1, dtype=None) -> None:
        """Transform a dataset of records into a matrix of responses of size user_id*item_id , with *init_value* for unknown values. Equivalent to creating horizontal vectors of responses for each user.

        Requirements : item_id and user_id features have been re-encoded.

        Args:
            dataset_name (str): Name of the dataset to work with
            coordinate_name : Name of the attribute in the dataset to fill the matrix with
            init_value : value of the unknown responses
            dtype : type of the elements in the matrix

        Returns:
            void
        """
        self.logger.info("records_to_vec(" + dataset_name + ", coordinate_name=" + str(
            coordinate_name) + ",init_value=" + str(init_value) + ",dtype=" + str(dtype) + ")")

        if self._state_feasible('records_to_vec', dataset_name):
            mat = np.ones(shape=(len(self.datasets[dataset_name]['user_id'].unique()),
                                 len(self.datasets[dataset_name]['item_id'].unique()))) * init_value

            for i, r in self.datasets[dataset_name].iterrows():
                mat[int(r['user_id']), int(r['item_id'])] = int(r[coordinate_name])

            if dtype is not None:
                mat = mat.astype(dtype)

            self.maps[dataset_name] = mat

            self._metadata_update(dataset_name, {
                'records_to_vec': {'coordinate_name': coordinate_name,
                               'init_value': init_value,
                               'dtype': dtype,
                               }})
        else:
            self.logger.error("records_to_vec(" + dataset_name + ", coordinate_name=" + str(
                coordinate_name) + ",init_value=" + str(init_value) + ",dtype=" + str(dtype) + END_ABORTED_MSG)

    def vec_to_record(self, dataset_name: str, dtype=None) -> None:
        """Transform a matrix of responses of size user_id*item_id with *init_value* for unknown values into a dataset of record.

        Args:
            dataset_name (str): Name of the dataset to work with
            init_value : value of the unknown responses
            dtype : type of the elements in the matrix

        Returns:
            void
        """
        self.logger.info(
            "vec_to_record(" + dataset_name  + ",dtype=" + str(dtype) + ")")

        if self._state_feasible('vec_to_record', dataset_name):
            self.datasets[dataset_name] = pd.DataFrame(columns=['user_id', 'item_id', 'correct'], dtype=dtype)

            user_ids = []
            item_ids = []
            correcteness=[]
            for i, r in enumerate(self.maps[dataset_name]):
                for j, e in enumerate(r):
                    user_ids.append(i)
                    item_ids.append(j)
                    correcteness.append(e)

            self.datasets[dataset_name]['user_id'] = user_ids
            self.datasets[dataset_name]['item_id'] = item_ids
            self.datasets[dataset_name]['correct'] = correcteness


            self.datasets[dataset_name]['user_id'] = self.datasets[dataset_name]['user_id'].astype(dtype)
            self.datasets[dataset_name]['item_id'] = self.datasets[dataset_name]['item_id'].astype(dtype)


    def transform(self, dataset_name: str, batch_size,**params):
        self.logger.info(
            "transform(" + dataset_name + ",batch_size=" + str(batch_size) + ")")

        if self._state_feasible('transform', dataset_name):

            #self.datasets[dataset_name]['skill_id'] = self.datasets[dataset_name]['skill_id'].apply(lambda x: literal_eval(x))
            #self.datasets[dataset_name] = self.datasets[dataset_name].explode('skill_id')

            train_data = self.get_train_dataset(dataset_name)
            valid_data = self.get_valid_dataset(dataset_name)
            train_valid_data = self.get_train_valid_dataset(dataset_name)
            test_data = self.get_test_dataset(dataset_name)

            dataloaders = []
            for d in [train_data,valid_data,train_valid_data,test_data] :

                dataset = TensorDataset(
                    torch.tensor(d['user_id'].to_numpy(), dtype=torch.int64),
                    torch.tensor(d['item_id'].to_numpy(), dtype=torch.int64),
                    torch.tensor(d['correct'].to_numpy(), dtype=torch.float),
                    torch.tensor(d['knowledge'].to_list(), dtype=torch.float32),
                    torch.tensor(d['skill_id'].to_numpy().astype(int), dtype=torch.int)
                )
                dataloaders.append(DataLoader(dataset, batch_size=batch_size, **params))

            self.dataloaders = dataloaders

            self._metadata_update(dataset_name, {
                'transform': {'batch_size': batch_size}})
        else:
            self.logger.error("transform(" + dataset_name + ",batch_size=" + str(batch_size) + END_ABORTED_MSG)

    def BPR_transform(self, dataset_name: str, batch_size,alpha,**params):
        self.logger.info(
            "BPR_transform(" + dataset_name + ",batch_size=" + str(batch_size)+ ",alpha=" + str(alpha) + ")")

        if self._state_feasible('transform', dataset_name):

            #self.datasets[dataset_name]['skill_id'] = self.datasets[dataset_name]['skill_id'].apply(lambda x: literal_eval(x))
            #self.datasets[dataset_name] = self.datasets[dataset_name].explode('skill_id')
            self.datasets[dataset_name]['neg_user'] = self.datasets[dataset_name]['user_id']

            train_data = self.get_train_dataset(dataset_name).copy()
            valid_data = self.get_valid_dataset(dataset_name).copy()
            train_valid_data = self.get_train_valid_dataset(dataset_name).copy()
            test_data = self.get_test_dataset(dataset_name).copy()

            dataloaders = []
            for d in [train_data,valid_data,train_valid_data,test_data] :
                items_with_0_response = d[d['correct'] == 0]['item_id'].unique()
                positive_rows = d[d['correct'] == 1]
                positive_rows = d[positive_rows['item_id'].isin(items_with_0_response)]
                new_rows = positive_rows.loc[positive_rows.index.repeat(alpha)].copy()
                neg_users = d[d['correct'] == 0].groupby('item_id')['user_id'].unique().apply(lambda x: np.concatenate([x])).reset_index()#d[d['correct'] == 0].groupby('item_id')['user_id'].unique().apply(lambda x: np.concatenate([x])).to_numpy()
                new_rows['neg_user'] = neg_users[new_rows['item_id']]
                new_rows['neg_user'] = new_rows.apply(lambda x: np.random.choice(x['neg_user']), axis=1)
                d = pd.concat([d, new_rows], ignore_index=True)

                dataset = TensorDataset(
                    torch.tensor(d['user_id'].to_numpy(), dtype=torch.int),
                    torch.tensor(d['item_id'].to_numpy(), dtype=torch.int),
                    torch.tensor(d['correct'].to_numpy(), dtype=torch.float),
                    torch.tensor(d['knowledge'].to_list(), dtype=torch.float32),
                    torch.tensor(d['skill_id'].to_numpy().astype(int), dtype=torch.int),
                    torch.tensor(d['neg_user'].to_numpy(), dtype=torch.int)
                )
                dataloaders.append(DataLoader(dataset, batch_size=batch_size, **params))

            self.dataloaders = dataloaders

            self._metadata_update(dataset_name, {
                'ransform': {'batch_size': batch_size}})
        else:
            self.logger.error("BPR_transform(" + dataset_name + ",batch_size=" + str(batch_size) + ",alpha=" + str(alpha)+ END_ABORTED_MSG)

    def create_know_vec(self,dataset_name:str) :
        """
        Creates a column with vectors of size (skill_id_nb) that contains zeros everywhere and ones at the coordinates of the row skills
        """
        self.logger.info(
            "create_know_vec(" + dataset_name+ ")")

        if self._state_feasible('create_know_vec', dataset_name):
            series_from_rows = pd.Series(data=[code2vector(ast.literal_eval(row), self.get_skills_nb(dataset_name)) for row in
                                               self.datasets[dataset_name]['skill_id']], name="knowledge").to_numpy()
            self.datasets[dataset_name]['knowledge'] = series_from_rows

            self._metadata_update(dataset_name, {
                    'create_know_vec': None})
        else :
            self.logger.error("create_know_vec(" + dataset_name + END_ABORTED_MSG)


    def IRR_transform(self,dataset_name:str, batch_size, **params):
        self.logger.info(
            "IRR_transform(" + dataset_name + ",batch_size=" + str(batch_size) + ")")

        if self._state_feasible('IRR_transform', dataset_name):

            train_data = self.get_train_dataset(dataset_name)
            valid_data = self.get_valid_dataset(dataset_name)
            train_valid_data = self.get_train_valid_dataset(dataset_name)
            test_data = self.get_test_dataset(dataset_name)

            dataloaders = []
            for d in [valid_data,test_data] :

                dataset = TensorDataset(
            torch.tensor(d['user_id'].to_numpy(), dtype=torch.int64) ,  # (1, user_n) to (0, user_n-1)
                    torch.tensor(d['item_id'].to_numpy(), dtype=torch.int64) ,  # (1, item_n) to (0, item_n-1)
                    torch.tensor(d['knowledge'].to_list(), dtype=torch.float32),
                    torch.tensor(d['correct'].to_numpy(), dtype=torch.float32)
                )
                dataloaders.append(DataLoader(dataset, batch_size=batch_size, **params))



            self.dataloaders = dataloaders

            self._metadata_update(dataset_name, {
                'IRR_transform': {'batch_size': batch_size}})
        else:
            self.logger.error("IRR_transform(" + dataset_name + ",batch_size=" + str(batch_size) + END_ABORTED_MSG)


    def generate_train_valid_map(self, dataset_name, init_value = -1):
        """Generate a matrix of responses of size nb_users*nb_items with only the training and validation data, and only one sample from per user from the test dataset

                Args:
                    dataset_name (str): Name of the dataset to work with

                Returns:
                    void
                """
        self.logger.info(
            "generate_train_valid_map(" + dataset_name + ",init_value=" + str(init_value)+")")

        if self._state_feasible('generate_train_valid_map', dataset_name):

            self.logger.warning(
                "The initialisation value is critical for test data -> train data leaks. It must be identical to the map init_value. Current init_value=" + str(init_value) + ")")

            self.train_valid_maps[dataset_name] = self.maps[dataset_name].copy()
            for i_row, row in self.get_test_dataset(dataset_name=dataset_name).iterrows():
                self.train_valid_maps[dataset_name][int(row['user_id']), int(row['item_id'])] = init_value
        else:
            self.logger.error("generate_train_valid_map(" + dataset_name  + ",init_value=" + str(init_value) + END_ABORTED_MSG)

    def generate_train_map(self, dataset_name, init_value = -1):
        """Generate a matrix of responses of size nb_users*nb_items with only the training and validation data, and only one sample from per user from the test dataset

                Args:
                    dataset_name (str): Name of the dataset to work with

                Returns:
                    void
                """
        self.logger.info(
            "generate_train_map(" + dataset_name + ",init_value=" + str(init_value)+")")

        if self._state_feasible('generate_train_map', dataset_name):

            self.logger.warning(
                "The initialisation value is critical for test data -> train data leaks. It must be identical to the map init_value. Current init_value=" + str(init_value) + ")")

            self.train_maps[dataset_name] = self.maps[dataset_name].copy()
            for i_row, row in self.get_test_dataset(dataset_name=dataset_name).iterrows():
                self.train_maps[dataset_name][int(row['user_id']), int(row['item_id'])] = init_value
            for i_row, row in self.get_valid_dataset(dataset_name=dataset_name).iterrows():
                self.train_maps[dataset_name][int(row['user_id']), int(row['item_id'])] = init_value
        else:
            self.logger.error("generate_train_map(" + dataset_name  + ",init_value=" + str(init_value) + END_ABORTED_MSG)

    def generate_test_map(self, dataset_name, init_value = -1):
        """Generate a matrix of responses of size nb_users*nb_items with only the testing data, and only one sample from per user from the test dataset

                Args:
                    dataset_name (str): Name of the dataset to work with

                Returns:
                    void
                """
        self.logger.info(
            "generate_test_map(" + dataset_name + ",init_value=" + str(init_value) + ")")

        if self._state_feasible('generate_test_map', dataset_name):

            self.logger.warning(
                "The initialisation value is critical for test data -> train data leaks. It must be identical to the map init_value. Current init_value=" + str(init_value) + ")")

            self.test_maps[dataset_name] = np.ones(shape=self.maps[dataset_name].shape)*init_value
            for i_row, row in self.get_test_dataset(dataset_name=dataset_name).iterrows():
                self.test_maps[dataset_name][int(row['user_id']), int(row['item_id'])] = int(row['correct'])
        else:
            self.logger.error("generate_test_map(" + dataset_name  + ",init_value=" + str(init_value) + END_ABORTED_MSG)

    def generate_test_list(self, dataset_name):
        """Generate a list of the test record dict :  {'user_id': row['user_id'], 'item_id': row['item_id'], 'score': row['correct']}

                Args:
                    dataset_name (str): Name of the dataset to work with

                Returns:
                    void
                """
        self.logger.info(
            "generate_test_list(" + dataset_name + ")")

        if self._state_feasible('generate_test_list', dataset_name):

            self.test_logs[dataset_name] = []
            #test_steps = []

            for i_row, row in self.get_test_dataset(dataset_name=dataset_name).iterrows():
                    self.test_logs[dataset_name].append({'user_id': row['user_id'], 'item_id': row['item_id'], 'score': row['correct']})
                    #test_steps.append(self.users[dataset_name][self.users[dataset_name]['user_id']==row['user_id']]['train_valid_log_nb'])

            #self.test_steps[dataset_name] = np.array(test_steps).reshape((len(test_steps),))
        else:
            self.logger.error("generate_test_list(" + dataset_name + END_ABORTED_MSG)

    def generate_test_matrix(self, dataset_name):
        """Generate a matrix of the test records :  col 0 : user_id; col 1 : item_id; col2 : score; lines : logs

                Args:
                    dataset_name (str): Name of the dataset to work with

                Returns:
                    void
                """
        self.logger.info(
            "generate_test_matrix(" + dataset_name + ")")

        if self._state_feasible('generate_test_matrix', dataset_name):

            self.test_mats[dataset_name] = self.get_test_dataset(dataset_name=dataset_name)[['user_id','item_id','correct']].to_numpy()

        else:
            self.logger.error("generate_test_matrix(" + dataset_name + END_ABORTED_MSG)

    def generate_valid_matrix(self, dataset_name):
        """Generate a matrix of the valid records :  col 0 : user_id; col 1 : item_id; col2 : score; lines : logs

                Args:
                    dataset_name (str): Name of the dataset to work with

                Returns:
                    void
                """
        self.logger.info(
            "generate_valid_matrix(" + dataset_name + ")")

        if self._state_feasible('generate_valid_matrix', dataset_name):

            self.valid_mats[dataset_name] = self.get_valid_dataset(dataset_name=dataset_name)[['user_id','item_id','correct']].to_numpy()

        else:
            self.logger.error("generate_valid_matrix(" + dataset_name + END_ABORTED_MSG)

    def generate_valid_list(self, dataset_name):
        """Generate a list of the valid record dict :  {'user_id': row['user_id'], 'item_id': row['item_id'], 'score': row['correct']}

                Args:
                    dataset_name (str): Name of the dataset to work with

                Returns:
                    void
                """
        self.logger.info(
            "generate_valid_list(" + dataset_name + ")")

        if self._state_feasible('generate_valid_list', dataset_name):

            self.valid_logs[dataset_name] = []
            #valid_steps = []

            for i_row, row in self.get_valid_dataset(dataset_name=dataset_name).iterrows():
                    self.valid_logs[dataset_name].append({'user_id': row['user_id'], 'item_id': row['item_id'], 'score': row['correct']})
                    #valid_steps.append(self.users[dataset_name][self.users[dataset_name]['user_id']==row['user_id']]['train_log_nb'])

            #self.valid_steps[dataset_name] = np.array(valid_steps).reshape((len(valid_steps),))
        else:
            self.logger.error("generate_valid_list(" + dataset_name + END_ABORTED_MSG)

    def save_dataset(self, dataset_name: str, attributes: List[str]=None, metadata=False, rebus_format=False ):

        self.logger.info(
            "save_dataset(" + dataset_name + ",attributes=" + str(attributes) + ",metadata="+str(metadata)+",rebus_format="+str(rebus_format)+ ")")

        if not rebus_format :
            if attributes is None :
                with open(self.xp_path+ "/preprocessed_" + dataset_name + "_" + str(
                        datetime.now().strftime(DATE_FORMAT)) + ".csv", 'w') as outfile:
                    self.datasets[dataset_name].to_csv(path_or_buf=outfile)


            else :
                with open(self.xp_path+ "/preprocessed_" + dataset_name + "_" + str(
                        datetime.now().strftime(DATE_FORMAT)) + ".csv", 'w') as outfile:
                    self.datasets[dataset_name][attributes].to_csv(path_or_buf=outfile)

        else :
            with open(self.xp_path + "/preprocessed_" + dataset_name + "_" + str(
                    datetime.now().strftime(DATE_FORMAT)) + ".txt", 'w') as outfile :
                df_string = self.datasets[dataset_name][["user_id", "qr", "correct", "start_time"]].to_string(header=False, index=False)
                outfile.write(df_string)

        if metadata :
            self._write_metadata(dataset_name=dataset_name)

        self._metadata_update(dataset_name, {'save_dataset': {'dataset_name': dataset_name,
                                                              'attributes': attributes,
                                                              }})
    def get_train_dataset(self, dataset_name: str):

        return self.datasets[dataset_name][self.datasets[dataset_name]['split']=='train']

    def get_valid_dataset(self, dataset_name: str):
        return self.datasets[dataset_name][self.datasets[dataset_name]['split']=='valid']

    def get_test_dataset(self, dataset_name: str):
        return self.datasets[dataset_name][self.datasets[dataset_name]['split']=='test']

    def get_train_valid_dataset(self, dataset_name: str):
        return self.datasets[dataset_name][self.datasets[dataset_name]['split'].isin(['train','valid'])]
    def get_attr_nb(self, dataset_name: str, attributes):
        self.logger.debug(
            "get_resp_nb(" + str(dataset_name) + ")")
        return len(self.datasets[dataset_name][attributes].unique())

    def get_resp_nb(self, dataset_name:str):
        """Gives the number of different responses in the dataset"""

        self.logger.debug(
            "get_resp_nb(" + str(dataset_name) + ")")

        return len(self.datasets[dataset_name]["correct"].unique())

    def get_irs_nb(self,dataset_name:str):
        """Gives the number of different item-response pairs in the dataset"""
        self.logger.debug(
            "get_irs_nb(" + str(dataset_name) + ")")

        try:
            nb_responses = self.get_resp_nb(dataset_name)
            nb_items = len(self.datasets[dataset_name]["item_id"].unique())
            return int(nb_responses*nb_items)

        except:
            self.logger.warning(
                "get_irs_nb() - data should exist in qr form in the dataset. "+END_ABORTED_MSG)

    def get_skills_nb(self,dataset_name:str):
        """Gives the number of different item-response pairs in the dataset"""
        self.logger.debug(
            "get_skills_nb(" + str(dataset_name) + ")")

        try:
            skills = self.metadata[dataset_name]["enc_categories_feat"]["skill_id"]["categories_nb"]
            if isinstance(skills,int) :
                nb_skills = skills
            elif isinstance(skills,list) :
                nb_skills = skills[0]
            else :
                self.logger.error(
                    "get_skills_nb() - unreadable metadata. " + END_ABORTED_MSG)
            return int(nb_skills)
        except :
            self.logger.warning(
                "get_skills_nb() - the dataset needs to have skill_id attributes. "+END_ABORTED_MSG)




    def create_ir2index(self, dataset_name: str):
        """
        Creates qr and qr_compl columns in the dataset
        """
        self.logger.info(
            "create_ir2index(" + str(dataset_name) + ")")

        if self._state_feasible('create_ir2index', dataset_name):

            item_array = self.datasets[dataset_name]["item_id"].to_numpy()
            response_array = self.datasets[dataset_name]["correct"].to_numpy()
            compl_response_array = np.ones(shape=response_array.shape) - response_array

            index_array = self.ir2index_array(dataset_name, items =item_array, responses = response_array)
            compl_index_array = self.ir2index_array(dataset_name, items=item_array, responses=compl_response_array)

            self.datasets[dataset_name]["qr"] = index_array
            self.datasets[dataset_name]["qr_compl"] = compl_index_array

            self._metadata_update(dataset_name, {'create_ir2index': {'dataset_name': dataset_name}})
        else:
            self.logger.error("create_ir2index(" + dataset_name + END_ABORTED_MSG)

    def get_triplet(self, dataset_name: str):
        """
        Returns [[u,i_pos,i_neg]]
        """
        self.logger.info(
            "get_triplet(" + str(dataset_name) + ")")

        if self._state_feasible('get_triplet', dataset_name):

            train_data = self.get_train_dataset(dataset_name)
            valid_data = self.get_valid_dataset(dataset_name)
            train_valid_data = self.get_train_valid_dataset(dataset_name)
            test_data = self.get_test_dataset(dataset_name)

            dataloaders = []
            for d in [train_data, valid_data, train_valid_data, test_data] :
                dataloaders.append(d[["user_id","qr","qr_compl"]].to_numpy())

            self.dataloaders = dataloaders

            self._metadata_update(dataset_name, {'get_triplet': {'dataset_name': dataset_name}})
        else:
            self.logger.error("get_triplet(" + dataset_name + END_ABORTED_MSG)


    def ir2index(self,dataset_name:str, ir: dict):
        """Bijective function : {'item':i,'response':r} -> index"""
        self.logger.debug(
            "ir2index(" + "dataset_name : " + str(dataset_name) + ",ir : " + str(ir) + ")")

        if self._state_feasible('ir2index', dataset_name):
            nb_responses = self.get_resp_nb(dataset_name)
            return int((ir['item'] * nb_responses) + ir['response'])
        else:
            self.logger.error("ir2index(" + "dataset_name : " + str(dataset_name) + ",ir : " + str(ir) + END_ABORTED_MSG)

    def ir2index_array(self,dataset_name:str, items : np.array, responses : np.array):
        """Bijective function : (items,responses) -> indexes"""
        self.logger.debug(
            "ir2index_array("+"dataset_name : "+str(dataset_name)+",items : " + str(items) +", responses"+str(responses)+ ")")

        if self._state_feasible('ir2index_array', dataset_name):
            nb_responses = self.get_resp_nb(dataset_name)
            return np.add((items * nb_responses), responses).astype(int)
        else:
            self.logger.error("ir2index_array(" +"dataset_name : "+str(dataset_name)+",items : " + str(items) +", responses"+str(responses) + END_ABORTED_MSG)


    def index2ir(self,dataset_name:str, index: int) -> dict:
        """Bijective function : index -> {'item':i,'response':r}"""
        self.logger.debug(
            "index2ir(" + "dataset_name : " + str(dataset_name) + ",index : " + str(index) + ")")

        if self._state_feasible('index2ir', dataset_name):
            nb_responses = self.get_resp_nb(dataset_name)
            return {'item': int(index //nb_responses),
                    'response': int(index % nb_responses)}
        else:
            self.logger.error("index2ir(" + "dataset_name : " + str(dataset_name)+ ",index : " + str(index) +END_ABORTED_MSG)



    def index2ir_array(self, dataset_name:str,indexes: np.array) -> dict :
        """Bijective function : indexes -> {'item':i,'response':r}"""
        self.logger.debug(
            "index2ir_array(" + "dataset_name : " + str(dataset_name) + ",indexes shape : " + str(indexes.shape) + ")")

        if self._state_feasible('index2ir_array', dataset_name):
            nb_responses = self.get_resp_nb(dataset_name)
            return {'item': (indexes // nb_responses),
                    'response': indexes % nb_responses}
        else:
            self.logger.error(
                "index2ir_array(" + "dataset_name : " + str(dataset_name) + ",indexes shape: " + str(indexes) + END_ABORTED_MSG)



    def index_complement(self, dataset_name:str, index: int):
        """Input : index related to pair (i,r). Output : the list of indexes related to the pair of item i and all
        the other possible responses : [(i,r'),...]"""
        self.logger.debug(
            "index_complement("+"dataset_name : "+str(dataset_name)+",index : " + str(index) + ")")

        if self._state_feasible('index_complement', dataset_name):
            nb_responses = self.get_resp_nb(dataset_name)
            return [(index // nb_responses) * nb_responses + x for x in
                    range(nb_responses) if x != (index % nb_responses)]
        else:
            self.logger.error("index_complement(" +"dataset_name : "+str(dataset_name)+",index : " + str(index) + END_ABORTED_MSG)


    ## Data analysis -------------
    def compute_mean_skills(self, dataset_name: str):
        """
        Requirements : skill_id have been re-encoded.

        Args:
            dataset_name (str): Name of the dataset to work on

        Returns:
            void
        """

        self.logger.info("compute_mean_skills(" + dataset_name + ")")

        if self._state_feasible('compute_mean_skills', dataset_name):

            y_dataframe = self.datasets[dataset_name][['user_id', 'skill_id', 'correct']].groupby(
                by=['user_id', 'skill_id']).mean()
            self.skills_maps[dataset_name] = np.zeros(shape=(
                len(self.datasets[dataset_name]['user_id'].unique()),
                len(self.datasets[dataset_name]['skill_id'].unique())))

            for i, row in y_dataframe.iterrows():
                self.skills_maps[dataset_name][i] = row

            self._metadata_update(dataset_name, {'compute_mean_skills':'done'})
        else:
            self.logger.error("compute_mean_skills(" + dataset_name + END_ABORTED_MSG)

    def generate_random_dataset(self):

        dataset_name = "acc_control"
        nb_items = 200
        nb_users = 500
        nb_responses = 2
        test_proportion = 0.35
        valid_proportion = 0.2
        accuracy = 0.7

        nb_true = int(accuracy * test_proportion * nb_users)
        nb_wrong = int(test_proportion * nb_users - nb_true)

        dataset = np.ones(shape=(nb_users, nb_items))

        train_pattern = np.random.randint(low=0, high=nb_responses, size=(1, nb_items))

        # Fill the dataset
        ## Training pattern 01 + right answers of the Test set
        dataset[0:int(nb_users * (1 - test_proportion) + nb_true), :] = dataset[0:int(
            nb_users * (1 - test_proportion) + nb_true), :] * train_pattern

        ## Test pattern 10 for the wrong answer
        dataset[int(nb_users * (1 - test_proportion) + nb_true):, :] = np.random.randint(low=0, high=nb_responses,
                                                                                         size=(nb_wrong,
                                                                                               nb_items))  # dataset[int(nb_users*(1-test_proportion)+nb_true):,:]*test_pattern

        # Experiment save #todo : removes this very ugly code that has nothing to do here
        self.import_dataset_from_object(np.array(dataset), dataset_name=dataset_name)
        self.vec_to_record(dataset_name=dataset_name, dtype=np.int64)

        # split_users

        self.train_test_split(dataset_name, test_proportion, valid_proportion)
        self.datasets[dataset_name].loc[:, "start_time"] = np.random.rand(self.datasets[dataset_name].shape[0])

    def generate_group_dataset(self,nb_users=500,nb_items=400):
        # Parameters
        dataset_name = "group_dataset"

        test_proportion = 0.35
        valid_proportion = 0.2

        nb_categories = 3
        cat_users_prop = [0, 0.35, 0.58, 1]
        cat_items_prop = [0, 0.1, 0.64, 1]
        noise = 0.3

        # Computation
        dataset = np.zeros(shape=(nb_users, nb_items))

        cat_users_num = (np.array(cat_users_prop) * nb_users).astype(int)
        cat_items_num = (np.array(cat_items_prop) * nb_items).astype(int)

        users_cat = np.zeros(shape=nb_users, dtype=int)
        items_cat = np.zeros(shape=nb_items, dtype=int)
        gen = np.random.default_rng(seed=0)
        for cat in range(nb_categories):
            dataset[cat_users_num[cat]:cat_users_num[cat + 1],
            cat_items_num[cat]:cat_items_num[cat + 1]] = gen.binomial(1, 1 - noise, size=(
            cat_users_num[cat + 1] - cat_users_num[cat], cat_items_num[cat + 1] - cat_items_num[cat]))
            users_cat[cat_users_num[cat]:cat_users_num[cat + 1]] = np.ones(
                shape=cat_users_num[cat + 1] - cat_users_num[cat]) * cat
            items_cat[cat_items_num[cat]:cat_items_num[cat + 1]] = np.ones(
                shape=cat_items_num[cat + 1] - cat_items_num[cat]) * cat

        # Experiment save
        self.import_dataset_from_object(np.array(dataset), dataset_name=dataset_name)
        self.vec_to_record(dataset_name=dataset_name, dtype=np.int64)

        self.datasets[dataset_name]['items_category'] = self.datasets[dataset_name].apply(
            lambda row: items_cat[int(row['item_id'])], axis=1)
        self.datasets[dataset_name]['users_category'] = self.datasets[dataset_name].apply(
            lambda row: users_cat[int(row['user_id'])], axis=1)

        self.enc_categories_feat(dataset_name, attributes=['item_id', 'user_id'])
        # self.display_response_map(dataset_name)
        # split_users
        self.train_test_split(dataset_name,test_proportion,valid_proportion)
        self.datasets[dataset_name].loc[:, "start_time"] = gen.random(size=self.datasets[dataset_name].shape[0])



    def generate_rand_noisy_groups_dataset(self, dataset_name="rngd",nb_users=3000, nb_items=100, nb_logs=60000, nb_items_groups=10,
                                           nb_users_groups=3, min_user_group_proportion=0.2,min_item_group_proportion=0.2, max_noise=0.3,alpha_max=9):
        """ Generate a dataset with user groups responding correctly to certain items groups and incorrectly to the others, with a certain probability regulated by the noise

        Requirements :
            - item_id and user_id features have been re-encoded.
            - The support and query sets have been created

        Args:
            nb_users (int):
            nb_items (int):
            nb_logs (int)
            nb_items_groups (int)
            nb_users_groups (int)
            max_noise (float)
            alpha_max (int) : Minimum number of responses per users

        Returns:
            list
        """
        self.logger.info("generate_rand_noisy_groups_dataset(" + "dataset_name"+str(dataset_name)+"nb_users="+str(nb_users)+",nb_items="+str(nb_items)+ ",nb_logs="+str(nb_logs)+",nb_items_groups="+str(nb_items_groups)+
                                           ",nb_users_groups="+str(nb_users_groups)+",min_user_group_proportion="+str(min_user_group_proportion)+",min_item_group_proportion="+str(min_item_group_proportion)+ ";max_noise="+str(max_noise) + ")")

        # ----- Computation

        ## --- We check if the min condition can be satisfied
        if min_user_group_proportion > 1/nb_users_groups :
            min_user_group_proportion = 1/(2*nb_users_groups)
        if min_item_group_proportion > 1/nb_items_groups :
            min_item_group_proportion = 1/(2*nb_items_groups)


        gen = np.random.default_rng(seed=0)

        # Users groups generation
        X = gen.random(size=(1, nb_users_groups))
        users_groups_prop = (1/np.sum(X)-min_user_group_proportion*nb_users_groups/np.sum(X)) * X + min_user_group_proportion

        final_users_groups_prop = [0]
        for g_prop in np.nditer(users_groups_prop):
            final_users_groups_prop.append(g_prop + final_users_groups_prop[-1])

        users_groups_prop = final_users_groups_prop

        # Items groups generation
        X = gen.random(size=(1, nb_items_groups))
        items_groups_prop = (1 / np.sum(X) - min_item_group_proportion * nb_items_groups / np.sum(
            X)) * X + min_item_group_proportion

        final_items_groups_prop = [0]
        for g_prop in np.nditer(items_groups_prop):
            final_items_groups_prop.append(g_prop + final_items_groups_prop[-1])

        items_groups_prop = final_items_groups_prop

        # User_item association
        available_items_group = list((np.arange(start=0, stop=nb_items_groups)))
        available_users_group = list((np.arange(start=0, stop=nb_users_groups)))

        gen.shuffle(available_items_group)
        gen.shuffle(available_users_group)

        A = np.zeros(shape=(nb_users_groups, nb_items_groups))

        if nb_users_groups <= nb_items_groups :
            for ug in available_users_group:
                A[ug, available_items_group.pop(0)] = 1

            if len(available_items_group) > 0:
                for ig in available_items_group:
                    A[gen.integers(0, nb_users_groups, size=1), ig] = 1
        else :
            for i,ig in enumerate(available_items_group):
                A[available_users_group.pop(0), ig] = 1

            if len(available_users_group) > 0:
                for ug in available_users_group:
                    A[ug,gen.integers(0, nb_items_groups, size=1)] = 1

        # Noise generation
        B = gen.random(size=(nb_users_groups, nb_items_groups)) * max_noise

        # Probability matrix computation
        P = np.abs(A - B)

        # response computation
        dataset = np.zeros(shape=(nb_users, nb_items))

        cat_users_num = (np.array(users_groups_prop) * nb_users).astype(int)
        cat_items_num = (np.array(items_groups_prop) * nb_items).astype(int)

        users_cat = np.zeros(shape=nb_users, dtype=int)
        items_cat = np.zeros(shape=nb_items, dtype=int)

        for i_cat in range(nb_items_groups):
            for u_cat in range(nb_users_groups):
                dataset[cat_users_num[u_cat]:cat_users_num[u_cat + 1],
                cat_items_num[i_cat]:cat_items_num[i_cat + 1]] = gen.binomial(
                    1, P[u_cat, i_cat], size=(
                    cat_users_num[u_cat + 1] - cat_users_num[u_cat], cat_items_num[i_cat + 1] - cat_items_num[i_cat]))

                users_cat[cat_users_num[u_cat]:cat_users_num[u_cat + 1]] = np.ones(
                    shape=cat_users_num[u_cat + 1] - cat_users_num[u_cat]) * u_cat

                items_cat[cat_items_num[i_cat]:cat_items_num[i_cat + 1]] = np.ones(
                    shape=cat_items_num[i_cat + 1] - cat_items_num[i_cat]) * i_cat

        # ----- Pre-processing
        self.import_dataset_from_object(np.array(dataset), dataset_name=dataset_name)
        self.vec_to_record(dataset_name=dataset_name, dtype=np.int64)

        self.datasets[dataset_name]['items_category'] = self.datasets[dataset_name].apply(
            lambda row: items_cat[int(row['item_id'])], axis=1)
        self.datasets[dataset_name]['users_category'] = self.datasets[dataset_name].apply(
            lambda row: users_cat[int(row['user_id'])], axis=1)
        if nb_logs is not None and nb_logs < self.datasets[dataset_name].shape[0] :
            self.sample(dataset_name, nb_logs=nb_logs)
        self.clean_sparse_data(dataset_name,attributes=['user_id','item_id'],t=[alpha_max+1,3], iterate = True)
        self.enc_categories_feat(dataset_name, attributes=['item_id', 'user_id'])
        self.create_ir2index(dataset_name=dataset_name)
        self.records_to_vec(dataset_name=dataset_name,init_value=-1)
        self.datasets[dataset_name].loc[:, "start_time"] = gen.random(size=(self.datasets[dataset_name].shape[0]))

        self._metadata_update(dataset_name,added_value={"rngd":{'nb_users':nb_users, 'nb_items':nb_items, 'nb_logs':nb_logs, 'nb_items_groups':nb_items_groups,
                                           'nb_users_groups':nb_users_groups, 'min_user_group_proportion':min_user_group_proportion,'min_item_group_proportion':min_item_group_proportion, 'max_noise':max_noise,'alpha_max':alpha_max}})
        return A,P
        #self._metadata_update(operation_name=dataset_name, added_value={'A':A,'B':B,'P':P}, saving=False)

    def result_landmarks(self,dataset_name,A,P) -> dict:
        """ Give landmarks on the dataset's predictions results

        Args:
            dataset_name (str): Name of the dataset to work with

        Returns:
            void
        """

        self.logger.info("result_landmarks(" + dataset_name + ")")

        if self._state_feasible('result_landmarks', dataset_name):



            # Test acc if we predict the most frequent value for the user-item log

            test_logs = self.datasets[dataset_name][
                self.datasets[dataset_name]['user_id'].isin(self.get_test_user_id(dataset_name))]

            comp = (test_logs['correct'] == A[test_logs['users_category'], test_logs['items_category']])

            acc_1 = np.sum(comp) / comp.shape[0]

            ## Acc expectancy if we randomly predict a user_item response with a the probability for the user_item group response perfectly known

            P_test = P[test_logs['users_category'], test_logs['items_category']]
            acc = np.sum(np.ones(shape=P_test.shape) - 2 * P_test * (np.ones(shape=P_test.shape)-P_test)) / test_logs.shape[0]

            return {"acc_1":acc_1,"acc":acc}
        else:
            self.logger.error("result_landmarks(" + dataset_name + END_ABORTED_MSG)


    ## Data Exploration -------------

    def display_responses_per_user(self, dataset_name: str, save: bool = True, stat='count', hue='item_distribution'):
        """ Display and save a histogram of the number of responses per user with a highlight on the support and query sets

        Requirements :
            - item_id and user_id features have been re-encoded.
            - The support and query sets have been created

        Args:
            dataset_name (str): Name of the dataset to work with
            save (bool): If true, the figure is saved

        Returns:
            void
        """

        self.logger.info("display_responses_per_user(" + dataset_name + ")")

        if self._state_feasible('display_responses_per_user', dataset_name):
            f = plt.figure(figsize=(7, 5))
            ax = f.add_subplot(1, 1, 1)

            plt.title(
                stat + " of responses (answer to 1 item)\n per users in the support set (0) and query set (1)",
                fontsize=12, y=1.3)

            subtitle = ''

            try:
                for i, m in enumerate(self.metadata[dataset_name]['distribute_items']):

                    subtitle += str(m) + ': ' + str(self.metadata[dataset_name]['distribute_items'][m])
                    if (i + 1) % 3 == 0:
                        subtitle += '\n'
                    else:
                        subtitle += ' - '
            except:
                self.logger.warning("display_responses_per_user() - distribute_items not executed before")
            try:
                for i, m in enumerate(self.metadata[dataset_name]['enc_categories_feat']):
                    subtitle += str(m) + ': ' + str(self.metadata[dataset_name]['enc_categories_feat'][m])
                    if (i + 1) % 3 == 0:
                        subtitle += '\n'
                    else:
                        subtitle += ' - '
            except:
                self.logger.warning("display_responses_per_user() - enc_categories_feat not executed before")

            plt.suptitle(subtitle, y=1.1, fontsize=10)

            sns.histplot(data=self.datasets[dataset_name], ax=ax, stat=stat, multiple="stack",
                         x="user_id", kde=False,
                         palette="pastel", hue=hue,
                         element="bars", legend=True)

            sns.axes_style("whitegrid")

            if save:
                plt.savefig(self.xp_path + "/res_p_user-" + dataset_name,
                            dpi=None,
                            facecolor='w',
                            edgecolor='w',
                            orientation='portrait',
                            format=None,
                            transparent=False,
                            bbox_inches="tight",
                            pad_inches=0.5,
                            metadata=None)
        else:
            self.logger.error("display_responses_per_user(" + dataset_name + END_ABORTED_MSG)

    def display_responses_per_item(self, dataset_name: str, save: bool = True, stat="count", hue='item_distribution'):
        """ Display and save a histogram of the number of responses per item with a highlight on the support and query sets

        Requirements :
            - item_id and user_id features have been re-encoded.
            - The support and query sets have been created

        Args:
            dataset_name (str): Name of the dataset to work with
            save (bool): If true, the figure is saved

        Returns:
            void
        """
        self.logger.info("display_responses_per_item(" + dataset_name + ")")

        if self._state_feasible('display_responses_per_item', dataset_name):

            f = plt.figure(figsize=(7, 5))
            ax = f.add_subplot(1, 1, 1)

            plt.title(stat + " of responses (answer to one item)\n per items in the support set (0) and query set "
                             "(1)",
                      fontsize=12, y=1.3)

            subtitle = ''
            try:
                for i, m in enumerate(self.metadata[dataset_name]['distribute_items']):

                    subtitle += str(m) + ': ' + str(self.metadata[dataset_name]['distribute_items'][m])
                    if (i + 1) % 3 == 0:
                        subtitle += '\n'
                    else:
                        subtitle += ' - '
            except:
                self.logger.warning("display_responses_per_user() - distribute_items not executed before")
            try:
                for i, m in enumerate(self.metadata[dataset_name]['enc_categories_feat']):
                    subtitle += str(m) + ': ' + str(self.metadata[dataset_name]['enc_categories_feat'][m])
                    if (i + 1) % 3 == 0:
                        subtitle += '\n'
                    else:
                        subtitle += ' - '
            except:
                self.logger.warning("display_responses_per_user() - enc_categories_feat not executed before")

            plt.suptitle(subtitle, y=1.1, fontsize=10)

            # sns.displot(
            #     data=self.datasets[dataset_name].groupby('item_id').count().sort_values(by='user_id', ascending=False),
            #     x="user_id", kde=True,
            #     palette="pastel",
            #     legend=True)
            sns.histplot(data=self.datasets[dataset_name], ax=ax, stat=stat, multiple="stack", x="item_id", kde=False,
                         palette="pastel", hue=hue, element="bars", legend=True)

            sns.axes_style("whitegrid")

            if save:
                plt.savefig(self.xp_path + "/res_p_item-" + dataset_name,
                            dpi=None,
                            facecolor='w',
                            edgecolor='w',
                            orientation='portrait',
                            format=None,
                            transparent=False,
                            bbox_inches="tight",
                            pad_inches=0.5,
                            metadata=None)
        else:
            self.logger.error("display_responses_per_item(" + dataset_name + END_ABORTED_MSG)

    def display_response_map(self, dataset_name: str, size_redux=True, f_w: int = 125, f_h: int = 125, max_users=20000,
                             max_items=10000, save: bool = True):
        """ Display and save a map of all the data of the *dataset_name* dataset or a random sample of it (*size_redux=True*) of size *max_users* * *max_items*

        Requirements :
            - item_id and user_id features have been re-encoded.

        Args:
            dataset_name (str): Name of the dataset to work with
            save (bool): If true, the figure is saved

        Returns:
            void
        """

        self.logger.info("display_response_map(" + dataset_name + ",size_reduc=" + str(size_redux) + ")")

        if self._state_feasible('display_response_map', dataset_name):

            users = np.arange(self.maps[dataset_name].shape[0])
            items = np.arange(self.maps[dataset_name].shape[1])

            if size_redux:
                if self.maps[dataset_name].shape[0] > max_users:
                    users = np.random.choice(users, size=max_users)

                if self.maps[dataset_name].shape[1] > max_items:
                    items = np.random.choice(items, size=max_items)

            used_map = self.maps[dataset_name][list(users), :][:, list(items)]

            # Parameters (cm)
            lr_margin = 1
            tb_margin = 1

            # Conversion
            l = lr_margin / f_w
            b = tb_margin / f_h
            w = 1 - l * 2
            h = 1 - b * 2
            cm2inch = 1 / 2.54  # inch per cm

            # specifying the width and the height of the box in inches
            fig = plt.figure(figsize=(f_w * cm2inch, f_h * cm2inch))
            ax = fig.add_axes((l, b, w, h))

            sns.heatmap(used_map, ax=ax)

            if save:
                plt.savefig(self.xp_path + "/res_matrix - " + dataset_name,
                            dpi=None,
                            facecolor='w',
                            edgecolor='w',
                            orientation='portrait',
                            format=None,
                            transparent=False,
                            bbox_inches="tight",
                            pad_inches=0.5,
                            metadata=None)
        else:
            self.logger.error(
                "display_response_map(" + dataset_name + ",size_reduc=" + str(size_redux) + END_ABORTED_MSG)

    def display_skill_map(self, dataset_name: str, size_redux=True, f_w: int = 125, f_h: int = 125, max_users=20000,
                          max_items=10000, save: bool = True):
        """ Display and save a map of all the data of the *dataset_name* dataset or a random sample of it (*size_redux=True*) of size *max_users* * *max_items*

        Requirements :
            - item_id and user_id features have been re-encoded.

        Args:
            dataset_name (str): Name of the dataset to work with
            save (bool): If true, the figure is saved

        Returns:
            void
        """

        self.logger.info("display_skill_map(" + dataset_name + ",size_reduc=" + str(size_redux) + ")")
        if self._state_feasible('display_skill_map', dataset_name):

            users = np.arange(self.skills_maps[dataset_name].shape[0])
            items = np.arange(self.skills_maps[dataset_name].shape[1])

            if size_redux:
                if self.skills_maps[dataset_name].shape[0] > max_users:
                    users = np.random.choice(users, size=max_users)

                if self.skills_maps[dataset_name].shape[1] > max_items:
                    items = np.random.choice(items, size=max_items)

            used_map = self.skills_maps[dataset_name][list(users), :][:, list(items)]

            # Parameters (cm)
            lr_margin = 1
            tb_margin = 1

            # Conversion
            l = lr_margin / f_w
            b = tb_margin / f_h
            w = 1 - l * 2
            h = 1 - b * 2
            cm2inch = 1 / 2.54  # inch per cm

            # specifying the width and the height of the box in inches
            fig = plt.figure(figsize=(f_w * cm2inch, f_h * cm2inch))
            ax = fig.add_axes((l, b, w, h))

            sns.heatmap(used_map, ax=ax)

            if save:
                plt.savefig(self.xp_path + "/skill_matrix-" + dataset_name,
                            dpi=None,
                            facecolor='w',
                            edgecolor='w',
                            orientation='portrait',
                            format=None,
                            transparent=False,
                            bbox_inches="tight",
                            pad_inches=0.5,
                            metadata=None)
        else:
            self.logger.error("display_skill_map(" + dataset_name + ",size_reduc=" + str(size_redux) + END_ABORTED_MSG)

    def plot_distribution(self, dataset_name: str, attributes: List[str], count_threshold=0, xlog=False, ylog=False,
                          f_w: int = 20, f_h: int = 20, save: bool = True):

        self.logger.info("plot_distribution(" + dataset_name + ",attributes=" + str(
            attributes) + "count_threshold=" + str(count_threshold) + ")")
        if self._state_feasible('plot_distribution', dataset_name):

            s = self.datasets[dataset_name].value_counts(attributes)

            tab = np.ndarray(shape=(s.size, 3))
            tab[:, 1] = s.to_numpy()
            tab[:, 0] = np.arange(start=0, stop=s.size)

            tab_cumul = np.ndarray(shape=(s.size, 3))
            tab_cumul[:, 1] = np.cumsum(tab[:, 1])
            tab_cumul[:, 0] = np.arange(start=0, stop=s.size)

            s2 = s[s >= count_threshold]
            thresh_tab = np.ndarray(shape=(s2.size, 3))
            thresh_tab[:, 1] = s2.to_numpy()
            thresh_tab[:, 0] = tab[0:s2.size, 0]

            thresh_tab_cumul = np.ndarray(shape=(s2.size, 3))
            thresh_tab_cumul[:, 1] = np.cumsum(thresh_tab[:, 1])
            thresh_tab_cumul[:, 0] = tab[0:s2.size, 0]

            # Parameters (cm)
            lr_margin = 1
            tb_margin = 1

            # Conversion
            l = lr_margin / f_w
            b = tb_margin / f_h
            w = 1 - l * 2
            h = 1 - b * 2
            cm2inch = 1 / 2.54  # inch per cm

            # specifying the width and the height of the box in inches
            fig = plt.figure(figsize=(f_w * cm2inch, f_h * cm2inch))
            ax = fig.add_axes((l, b, w, h))

            # Simple distribution
            color1 = 'tab:blue'
            ax.annotate("min count \nof " + str(attributes) + " : " + str(count_threshold),
                        xy=(tab[tab.shape[0] - 1, 0] * 0.85, count_threshold + tab[0, 1] * 0.01))

            ax.set_title('Distribution of logs over ' + str(attributes) + " in " + str(dataset_name))
            ax.set_xlabel(str(attributes) + " ordered by decreasing count")
            ax.set_ylabel('count', color=color1)
            ax.plot([0, tab[tab.shape[0] - 1, 0]], [count_threshold, count_threshold], color='tab:red')
            ax.tick_params(axis='y', labelcolor=color1)

            ax.plot(tab[:, 0], tab[:, 1], label="Distribution", color=color1)
            ax.fill_between(thresh_tab[:, 0], thresh_tab[:, 1], step="pre", alpha=0.4,
                            label="{:.0f}".format(thresh_tab[:, 1].sum()) + " logs : " + "{:.2f}".format(
                                thresh_tab[:, 1].sum() / tab[:, 1].sum() * 100) + "% of the logs\n" + "{:.0f}".format(
                                thresh_tab[thresh_tab.shape[0] - 1, 0]) + " " + str(
                                attributes) + " : " + "{:.2f}".format(thresh_tab[thresh_tab.shape[0] - 1, 0] / tab[
                                tab.shape[0] - 1, 0] * 100) + "% of the " + str(attributes), color=color1)

            if ylog:
                ax.set_yscale('log')
            if xlog:
                ax.set_xscale('log')

            # Cumulative distribution
            ax2 = ax.twinx()
            color = 'tab:orange'

            ax2.plot(tab_cumul[:, 0], tab_cumul[:, 1], label="Cumulated distribution", color=color)

            ax2.set_ylabel('cumulated count', color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            if ylog:
                ax2.set_yscale('log')

            fig.legend(bbox_to_anchor=(w + 0.5 * lr_margin * cm2inch, 1), loc='upper left')

            if save:
                plt.savefig(self.xp_path + "/distrib-" + str(attributes) + "-" + dataset_name,
                            dpi=None,
                            facecolor='w',
                            edgecolor='w',
                            orientation='portrait',
                            format=None,
                            transparent=False,
                            bbox_inches="tight",
                            pad_inches=0.5,
                            metadata=None)
        else:
            self.logger.error("plot_distribution(" + dataset_name + "," + str(attributes) + END_ABORTED_MSG)

    def plot_distribution_publi(self, dataset_name: str, attributes: List[str], count_threshold=0, xlog=False, ylog=False,
                          f_w: int = 20, f_h: int = 20, save: bool = True):

        self.logger.info("plot_distribution(" + dataset_name + ",attributes=" + str(
            attributes) + "count_threshold=" + str(count_threshold) + ")")
        if self._state_feasible('plot_distribution', dataset_name):

            s = self.datasets[dataset_name].value_counts(attributes)

            tab = np.ndarray(shape=(s.size, 3))
            tab[:, 1] = s.to_numpy()
            tab[:, 0] = np.arange(start=0, stop=s.size)

            tab_cumul = np.ndarray(shape=(s.size, 3))
            tab_cumul[:, 1] = np.cumsum(tab[:, 1])
            tab_cumul[:, 0] = np.arange(start=0, stop=s.size)

            s2 = s[s >= count_threshold]
            thresh_tab = np.ndarray(shape=(s2.size, 3))
            thresh_tab[:, 1] = s2.to_numpy()
            thresh_tab[:, 0] = tab[0:s2.size, 0]

            thresh_tab_cumul = np.ndarray(shape=(s2.size, 3))
            thresh_tab_cumul[:, 1] = np.cumsum(thresh_tab[:, 1])
            thresh_tab_cumul[:, 0] = tab[0:s2.size, 0]

            # Parameters (cm)
            lr_margin = 1
            tb_margin = 1

            # Conversion
            l = lr_margin / f_w
            b = tb_margin / f_h
            w = 1 - l * 2
            h = 1 - b * 2
            cm2inch = 1 / 2.54  # inch per cm

            # specifying the width and the height of the box in inches
            fig = plt.figure(figsize=(f_w * cm2inch, f_h * cm2inch))
            ax = fig.add_axes((l, b, w, h))

            # Simple distribution
            color1 = 'tab:blue'

            ax.set_title('Distribution of logs over ' + str(attributes) + " in " + str(dataset_name))
            ax.set_xlabel(str(attributes) + " ordered by decreasing count")
            ax.set_ylabel('sum(nb of logs)')
            ax.tick_params(axis='y')

            ax.plot(tab[:, 0], tab[:, 1], label="Distribution", color=color1)
            ax.fill_between(thresh_tab[:, 0], thresh_tab[:, 1], step="pre", alpha=0.4, color=color1)

            if ylog:
                ax.set_yscale('log')
            if xlog:
                ax.set_xscale('log')

            fig.legend(bbox_to_anchor=(w + 0.5 * lr_margin * cm2inch, 1), loc='upper left')

            if save:
                plt.savefig(self.xp_path + "/distrib-" + str(attributes) + "-" + dataset_name,
                            dpi=None,
                            facecolor='w',
                            edgecolor='w',
                            orientation='portrait',
                            format=None,
                            transparent=False,
                            bbox_inches="tight",
                            pad_inches=0.5,
                            metadata=None)
        else:
            self.logger.error("plot_distribution(" + dataset_name + "," + str(attributes) + END_ABORTED_MSG)

    def plot_response_prop_per_item(self, dataset_name: str, attributes: List[str], f_w: int = 20, f_h: int = 20,
                                    save: bool = True):
        self.logger.info("plot_response_prop_per_item(" + dataset_name + ",attributes=" + str(
            attributes) + ")")
        if self._state_feasible('plot_response_prop_per_item', dataset_name):

            order = self.datasets[dataset_name].value_counts(['item_id'])
            counts = self.datasets[dataset_name].value_counts(['item_id', 'correct'])

            X = np.arange(start=0, stop=order.shape[0])
            Y = np.zeros(shape=(order.shape[0], 2))

            conv = []
            items_index = order.index.to_numpy()
            for i in items_index:
                conv.append(i[0])
            for t in counts.index:
                Y[conv.index(t[0]), t[1]] = counts[t[0], t[1]]
            lr_margin = 1
            tb_margin = 1

            # specifying the width and the height of the box in inches
            l = lr_margin / f_w
            b = tb_margin / f_h
            w = 1 - l * 2
            h = 1 - b * 2
            cm2inch = 1 / 2.54  # inch per cm

            # specifying the width and the height of the box in inches
            fig = plt.figure(figsize=(f_w * cm2inch, f_h * cm2inch))
            ax = fig.add_axes((l, b, w, h))

            Y_prop = np.divide(Y[:, 1], order.to_numpy())
            Y_prop = np.sort(Y_prop)

            ax.plot(X, Y_prop, label="Distribution", color='tab:red')
            ax.set_title(
                "Proportion of correct responses per item in " + str(
                    dataset_name) )
            ax.set_ylabel('Proportion of correct responses')
            ax.set_xlabel('Items (ordered by increasing proportion)')

            if save:
                plt.savefig(self.xp_path + "/res_prop_per_items-" + str(attributes) + "-" + dataset_name,
                            dpi=None,
                            facecolor='w',
                            edgecolor='w',
                            orientation='portrait',
                            format=None,
                            transparent=False,
                            bbox_inches="tight",
                            pad_inches=0.5,
                            metadata=None)
        else:
            self.logger.error("plot_response_prop_per_item(" + dataset_name + "," + str(attributes) + END_ABORTED_MSG)

    def plot_response_prop_given_item_count(self, dataset_name: str, attributes: List[str], f_w: int = 20,
                                            f_h: int = 20, save: bool = True):

        self.logger.info("plot_response_prop_given_item_count(" + dataset_name + ",attributes=" + str(attributes) + ")")
        if self._state_feasible('plot_response_prop_given_item_count', dataset_name):

            order = self.datasets[dataset_name].value_counts(['item_id'])
            counts = self.datasets[dataset_name].value_counts(['item_id', 'correct'])

            Y = np.zeros(shape=(order.shape[0], 2))

            conv = []
            items_index = order.index.to_numpy()
            for i in items_index:
                conv.append(i[0])
            for t in counts.index:
                Y[conv.index(t[0]), t[1]] = counts[t[0], t[1]]

            lr_margin = 1
            tb_margin = 1

            # specifying the width and the height of the box in inches
            l = lr_margin / f_w
            b = tb_margin / f_h
            w = 1 - l * 2
            h = 1 - b * 2
            cm2inch = 1 / 2.54  # inch per cm

            # specifying the width and the height of the box in inches
            fig = plt.figure(figsize=(f_w * cm2inch, f_h * cm2inch))
            ax = fig.add_axes((l, b, w, h))
            np_order = order.to_numpy()
            Y_prop = np.divide(Y[:, 1], np_order)

            # Size computation of the points
            combos = list(zip(np_order, Y_prop))
            weight_counter = Counter(combos)

            weights = [weight_counter[(np_order[i], Y_prop[i])] for i, _ in enumerate(np_order)]

            # plot
            ax.scatter(np_order, Y_prop, s=weights, label="Distribution", color='tab:red')
            ax.set_title("Proportion of correct responses in " + str(dataset_name))

            if save:
                plt.savefig(self.xp_path + "/res_prop_given_item_count-" + str(attributes) + "-" + dataset_name,
                            dpi=None,
                            facecolor='w',
                            edgecolor='w',
                            orientation='portrait',
                            format=None,
                            transparent=False,
                            bbox_inches="tight",
                            pad_inches=0.5,
                            metadata=None)

        else:
            self.logger.error(
                "plot_response_prop_given_item_count(" + dataset_name + "," + str(attributes) + END_ABORTED_MSG)



    ## State handler -------------

    def end_experiment(self):
        self._write_metadata()
        self.logger.info("end_experiment()")


# FUNCTIONS -----------------------------------------------
def code2vector(x,knowledge_num):
    vector = [0] * knowledge_num
    for k in (x):
        try :
            vector[int(k) - 1] = 1
        except IndexError:
            print(knowledge_num)
            print(x)
            print(k)

    return vector

def d_response_map(dataset_matrix, size_redux=True, f_w: int = 125, f_h: int = 125, max_users=20000,
                   max_items=10000, save: bool = True, **kwargs):
    """ Display and save a map of all the data of the *dataset_name* dataset or a random sample of it (*size_redux=True*) of size *max_users* * *max_items*

    Requirements :
        - item_id and user_id features have been re-encoded.

    Args:
        dataset_name (str): Name of the dataset to work with
        save (bool): If true, the figure is saved

    Returns:
        void
    """

    users = np.arange(dataset_matrix.shape[0])
    items = np.arange(dataset_matrix.shape[1])

    if size_redux:
        if dataset_matrix.shape[0] > max_users:
            users = np.random.choice(users, size=max_users)

        if dataset_matrix.shape[1] > max_items:
            items = np.random.choice(items, size=max_items)

    used_map = dataset_matrix[list(users), :][:, list(items)]

    # Parameters (cm)
    lr_margin = 1
    tb_margin = 1

    # Conversion
    l = lr_margin / f_w
    b = tb_margin / f_h
    w = 1 - l * 2
    h = 1 - b * 2
    cm2inch = 1 / 2.54  # inch per cm

    # specifying the width and the height of the box in inches
    fig = plt.figure(figsize=(f_w * cm2inch, f_h * cm2inch))
    ax = fig.add_axes((l, b, w, h))

    sns.heatmap(used_map, ax=ax)

    if save:
        plt.savefig(kwargs['exp_path'] + "/res_matrix_2-" + kwargs['dataset_name'],
                    dpi=None,
                    facecolor='w',
                    edgecolor='w',
                    orientation='portrait',
                    format=None,
                    transparent=False,
                    bbox_inches="tight",
                    pad_inches=0.5,
                    metadata=None)

def string_to_list(s):
    return literal_eval(s)

def concat_users(x):
    return np.concatenate([x])