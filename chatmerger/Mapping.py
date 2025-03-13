import logging
import os.path
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import networkx as nx
from itertools import combinations, chain

logger = logging.getLogger('WhatsAppMerger Mapping')

class Mapping:
    def __init__(self):
        self.mapping = {}
        self.reversed_mapping = {}
        self.manual_mapping = {}

    def __repr__(self):
        return f"A mapping containing {len(self.mapping)} groups"

    @staticmethod
    def _label_match_score(author, label_message_dict):
        '''
        The overlap coefficient, or Szymkiewicz–Simpson coefficient, is a similarity measure that measures the
        overlap between two finite sets. It is related to the Jaccard index and is defined as the size of the
        intersection divided by the smaller of the size of the two sets.
        https://en.wikipedia.org/wiki/Overlap_coefficient

        :param author:
        :param label_message_dict:
        :return:
        '''
        messages = label_message_dict[author]
        len_messages = len(messages)
        scores = {}
        for author_compare, messages_compare in label_message_dict.items():
            if author == author_compare:
                continue
            intersection = messages & messages_compare
            min_len = min(len_messages, len(messages_compare))
            score = len(intersection) / min_len if min_len else 0
            scores[author_compare] = score
        return scores

    def apply_label_message_dict(self, label_message_dict, score_threshold=0.5, disable_tqdm=None):
        if disable_tqdm is None:
            disable_tqdm = logger.level > logging.INFO
        logger.info("Find pairs in text")
        all_scores = []
        self_groups = []
        for author in tqdm(label_message_dict, desc="Finding pairs in text", position=0, leave=True,
                           disable=disable_tqdm):
            scores = self._label_match_score(author=author, label_message_dict=label_message_dict)
            for s in [[author, author_match, s] for author_match, s in scores.items()]:
                all_scores.append(s)
            self_groups.append([author, author])
        df = pd.DataFrame(all_scores, columns=['Author', 'Author_match', 'score'])

        groups = [{x[0], x[1]} for x in df[df.score > score_threshold][['Author', "Author_match"]].values]
        self.add_groups(groups + self_groups)

    def update(self, new_mapping):
        self.mapping = new_mapping
        self._reverse_mapping()

    def _reverse_mapping(self):
        reversed_map = {}
        for main_label, original_labels in self.mapping.items():
            for label in original_labels:
                if label not in reversed_map:
                    reversed_map[label] = main_label
                else:
                    logger.debug(f"oh no, {label} is already mapped: {reversed_map[label]}")
        self.reversed_mapping = reversed_map

    def add_groups(self, groups):
        """
        Merge and add groups of names to the existing mapping.

        This method accepts a collection of name groups, merges them with the current groups,
        and maps each group to a primary key derived from the first element in each group.
        The merged groups are added to the existing mapping, updating any existing entries
        with additional members from the new groups.

        Args:
            groups (list[set]): A list of sets, where each set contains grouped items
                (e.g., names or identifiers). Each group will be keyed in the mapping by
                the first item in the set.
        """
        groups = self._merge_groups(groups)
        logger.info(f"Adding {len(groups)} groups to mapping")
        if not groups:
            return

        mapping = defaultdict(set)
        for map_group in groups:
            mapping[list(map_group)[0]] = mapping[list(map_group)[0]].union(map_group)

        self.update(mapping)

    def get(self, key):
        if isinstance(key, tuple):
            return self.reversed_mapping.get(key)

    def _merge_groups(self, groups: list[set]):
        old_groups = [list(g) for _, g in self.mapping.items()]
        all_groups = list(chain(*[list(combinations(og, 2)) if len(og) > 1 else list(combinations(og + og, 2)) for og in
                                  groups + old_groups]))
        all_groups = [group_member if len(group_member) > 1 else [list(group_member)[0], list(group_member)[0]] for
                      group_member in all_groups]
        start_len = len(all_groups)
        logger.info(f"merging {len(all_groups)} pairs")
        graph = nx.Graph()
        graph.add_edges_from(all_groups)
        all_groups = list(nx.connected_components(graph))
        logger.info(f"Merged {start_len} pairs to {len(all_groups)} groups")
        return all_groups


class ContactMapping(Mapping):

    def __init__(self, config_dir=None):
        super().__init__()
        if config_dir is None:
            config_dir = ""
        self.file_name = 'mapping_list_chats.xlsx'
        self.mapping_file = os.path.join(config_dir, self.file_name)
        self.get_manual_mapping()

    def get_manual_mapping(self):
        try:
            mapping_df = pd.read_excel(self.mapping_file)
            self.manual_mapping = mapping_df.set_index('Name_from')[['Name_to']].to_dict()['Name_to']
        except FileNotFoundError:
            logger.warning(
                f'\n De mapping lijst is niet gevonden in de config folder. Check of {self.mapping_file} bestaat.')

    def apply_data_holders(self, data_holders):
        for label, group in self.mapping.items():
            for group_member in list(group):
                group.remove(group_member)
                group.add((data_holders.get(group_member[0], group_member[0]), group_member[1]))

    def add_groups(self, groups):
        """
        Merge and add groups of names to the existing mapping with standardization, the name of the group is chosen
        based on the longest name in the group.

        This method accepts a collection of name groups, merges them with the current groups,
        and then maps each group to a standardized name key. Each name is processed through a
        name-weighting function to prioritize certain names, and a cleaning function to normalize
        formatting. Merged groups are added to the existing mapping.

        Args:
            groups (list[set[tuple]]): A list of groups, where each group is a set of tuples
                containing name pairs. Each name pair is processed and weighted to determine
                a primary name for that group."""

        groups = self._merge_groups(groups)
        logger.info(f"Adding {len(groups)} groups to mapping")
        if not groups:
            return

        def _name_weight(text):
            """
            Determine the sorting weight for a potential name.

            This function checks if the name is a path that exists.
            If it exists, a priority weight of 0 is returned. Otherwise, it calculates
            a sorting weight based on the length of the text, excluding digits, spaces,
            and the '+' character, thus reducing the weight of phonenumbers.

            Args:
                text (str): The name to be weighted

            Returns:
                int: Returns 0 if the name exists as a path. Otherwise, returns the length
                of the `text` string with digits, spaces, and '+' removed.
            """
            if os.path.exists(text):
                return 0
            return len(re.sub(r'[\d+\s]+', '', text))

        def _clean_name(name):
            """
            Clean and standardize a name by reducing spaces and normalizing case for specific words.

            This function performs two operations on the input name:
            1. Reduces multiple consecutive spaces to a single space.
            2. Converts specific Dutch 'tussenvoegsels' (prefixes) to lowercase, if present in the name.

            Args:
                name (str): The name string to clean and format.

            Returns:
                str: The cleaned and standardized name."""

            # reduce multiple spaces
            name = re.sub('\s+', ' ', name)

            # tussenvoegsels to lowercase
            tussenvoegsels = ["aan", "af", "bij", "de", "den", "der", "d'", "het", "'t", "in", "onder", "op", "over",
                              "'s", "'t", "te", "ten", "ter", "tot", "uit", "uijt", "van", "ver", "voor"]
            split_name = name.split(' ')
            for i in range(len(split_name)):
                if split_name[i].lower() in tussenvoegsels:
                    split_name[i] = split_name[i].lower()
            name = ' '.join(split_name)
            return name

        mapping = defaultdict(set)
        assert isinstance(list(groups[0])[0], tuple), 'groups should a set of tuples when adding groups to a Mapping'
        for map_group in groups:
            cleaned = _clean_name(max([n[1] for n in map_group], key=_name_weight))

            # if a manual mapping is provided and the cleaned name is mapped, take the mapped name.
            cleaned = self.manual_mapping.get(cleaned, cleaned)
            mapping[cleaned] = mapping[cleaned].union(map_group)

        self.update(mapping)
