import datetime
import json
import os
import re
import sys
from pathlib import Path
import logging

import pandas as pd
from tqdm import tqdm

try:
    from .WhatsAppMerger import WhatsAppMerger
    from .Parser import Parser
    from .utils import dataholder_from_filepath, get_id, slugify, get_file_hashes, error_report
except ImportError:
    from WhatsAppMerger import WhatsAppMerger
    from Parser import Parser
    from utils import dataholder_from_filepath

logger = logging.getLogger('WhatsApp Config')


def generate_config(root_folder="./", output=None, conversation_name=None):
    logger.info(f"Generating config")
    file_config = {}
    if conversation_name == None:
        conversation_name = input("Conversation name:\n")
    for chat_num, path in enumerate(Path(root_folder).glob('**/*.txt')):
        logger.debug(f"{path}, {path.name}")
        parser = Parser(str(path), autodetect=True, data_holder=f"Chat {chat_num:03}")
        file_config[str(path)] = parser.generate_config()

    config = {
        'conversation_name': conversation_name,
        'files': file_config
    }

    if os.path.isfile(output):
        resp = input(f"Config file ({output}) already exists. Do you want to overwrite? y/N\n")
        if resp.lower() != "y":
            return

    if output and output.endswith('.json'):
        with open(output, 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=4)
    return config


def load_config(config_fname):
    if config_fname.endswith('json'):
        config = load_config_from_json(config_fname)
        if mapping := config.get('mapping'):
            config['mapping'] = {data_holder: set(tuple(x) for x in name_map) for data_holder, name_map in
                                 mapping.items()}
        return config


def load_config_from_json(config_fname):
    with open(config_fname, 'r') as config_file:
        config = json.load(config_file)
    return config


def create_parser_config(parser_list: list[Parser], data_holders=None) -> dict[str, dict]:
    """
    Creates dictionaries containing the configuration for each parser in the parser_list.

    Args:
        parser_list (list[Parser]):
        data_holders (dict, optional): A mapping of filenames to data_holder names. When provided this will be used to
            configure the data_holder for the parser. When omitted then an attempt will be made to retrieve the
            data_holder form the file path using `dataholder_from_filepath`.

    Returns:
          dict: containing the configuration for each parser in the parser_list.
    """
    if data_holders is None:
        data_holders = {}
    parsers_dict = {}
    for parser in parser_list:
        date_range_format = '%d-%m-%Y'
        try:
            start_date = datetime.datetime.strptime(parser.example_dates[0], parser.datetime_format).strftime(
                date_range_format)
        except ValueError:
            start_date = None
        try:
            end_date = datetime.datetime.strptime(parser.example_dates[1], parser.datetime_format)
            end_date += datetime.timedelta(days=1)
            end_date = end_date.strftime(date_range_format)
        except ValueError:
            end_date = None
        parsers_dict[parser.fname] = dict(
            phone_os=parser.phone_os,
            datetime_format=parser.datetime_format,
            datetime_certain=parser.datetime_certain,
            example_date_1=parser.example_dates[0],
            example_date_2=parser.example_dates[1],
            start_date=start_date,
            end_date=end_date,
            load=True,
            data_holder=data_holders.get(parser.fname, dataholder_from_filepath(parser.fname)),
        )
    return parsers_dict


def create_configs(sorted_mapping, config_dir, chats, config_path="./"):
    for group_id, (group, chat_paths) in tqdm(zip(get_id(), sorted_mapping.items()),
                                                  desc='Creating configs', total=len(sorted_mapping)):
        conversation_list: list[Parser] = []
        for p in sorted(chat_paths):
            conversation_list.append(chats[p])
        merger = WhatsAppMerger(conversation_list=conversation_list, config_dir=config_dir)
        merger.author_mapping()
        topic = merger.get_topic()
        create_config_for_conversation(
            merger=merger,
            config_name=os.path.join(
                config_path,
                slugify(f'config_{group_id}_{topic}') + '.json'
            )
        )


def create_config_for_conversation(merger: WhatsAppMerger, config_name, base_config: dict = None):
        logger.info(f"Creating config {config_name}")
        data_holders = merger.get_data_holders()
        merger.author_mapping()
        phone_book = merger.create_phone_book()
        topic = merger.get_topic()
        base_config_dates = dict(start_date="01-12-2019", end_date="23-03-2024")
        if base_config is None:
            base_config = {}

        config_dict = base_config_dates | base_config | dict(
            conversation_name=f'WhatsApp conversatie {topic}',
            files={},
            mapping={
                data_holder: list(name_map)
                for data_holder, name_map
                in merger.contact_mapping.mapping.items()
            },
            phone_book=phone_book
        )

        parser_list = merger.conversation_list
        config_dict['files'] = create_parser_config(parser_list, data_holders)

        save_config(config_dict, config_name)


def regenerate_phonebook_config_for_conversation(merger: WhatsAppMerger, config_name, base_config: dict = None):
    logger.info(f"Regenerating phonebook config {config_name}")
    merger.author_mapping()
    phone_book = merger.create_phone_book()

    config_dict = base_config | dict(
        phone_book=phone_book,
        mapping={
            data_holder: list(name_map)
            for data_holder, name_map
            in merger.contact_mapping.mapping.items()
        },
    )

    config_dict['files'] = base_config["files"]

    save_config(config_dict, config_name)


def save_config(config_dict, config_name):
    """
    Saves the given configuration dictionary as a JSON file after making necessary adjustments.

    This function fixes date formats in the `config_dict`, ensures that the `mapping` field in
    the dictionary has list values, and saves the resulting dictionary to a JSON file with
    the specified `config_name` (appending the '.json' extension if not already present).

    Args:
        config_dict (dict): The configuration dictionary to be saved. It should contain a key
            'mapping' with values that need to be converted to lists before saving.
        config_name (str): The name of the file (with or without a '.json' extension) where
            the configuration will be saved.
    """
    _fix_dates_in_config(config_dict)
    fname_min_ext = os.path.splitext(config_name)[0]
    config_name = fname_min_ext + '.json'

    config_dict['mapping'] = {
            data_holder: list(name_map)
            for data_holder, name_map
            in config_dict['mapping'].items()
    }

    with open(config_name, 'w', encoding="utf-8") as f:
        logger.info(f"Writing {config_name}")
        json.dump(config_dict, f, indent=4)


def _fix_dates_in_config(config_dict):
    if 'start_date' in config_dict and isinstance(config_dict['start_date'], datetime.date):
        config_dict['start_date'] = config_dict['start_date'].strftime('%d-%m-%Y')
    if 'end_date' in config_dict and isinstance(config_dict['end_date'], datetime.date):
        config_dict['end_date'] = config_dict['end_date'].strftime('%d-%m-%Y')

    for file in config_dict["files"]:
        if 'start_date' in config_dict["files"][file] and isinstance(config_dict["files"][file]['start_date'],
                                                                     datetime.datetime):
            config_dict["files"][file]['start_date'] = config_dict["files"][file]['start_date'].strftime('%d-%m-%Y')
        if 'end_date' in config_dict["files"][file] and isinstance(config_dict["files"][file]['end_date'],
                                                                   datetime.datetime):
            config_dict["files"][file]['end_date'] = config_dict["files"][file]['end_date'].strftime('%d-%m-%Y')


def process_configs(conversation_folder,
                    reports_folder,
                    input_directory,
                    config_dir,
                    second_check_config_directory,
                    ignore_human_in_the_loop=True,
                    use_json=False
                    ):
    logger.info('Processing configs')
    file_hashes_dict = get_file_hashes(root_directory=input_directory)
    file_hashes_df = pd.DataFrame.from_dict(
        file_hashes_dict,
        orient='index',
        columns=['hash']
    ).reset_index(names='path')
    file_hashes_df = file_hashes_df[~file_hashes_df.path.str.endswith('txt')]
    file_hashes_df['path'] = file_hashes_df.path.apply(Path)
    file_hashes_df['parent'] = file_hashes_df.path.apply(lambda x: x.parent)

    hash_files_dict = {v: k for k, v in file_hashes_dict.items()} if file_hashes_dict else {}
    hash_files_df = pd.DataFrame.from_dict(
        hash_files_dict,
        orient='index',
        columns=['path']
    ).reset_index(names='hash')

    file_list = []
    for root, subdirs, files in tqdm(os.walk(second_check_config_directory)):
        for file in files:
            if use_json and file.endswith('json'):
                file_list.append([root, file])
            elif file.endswith('xlsx') or file.endswith('xls'):
                file_list.append([root, file])
    errors = {}
    with tqdm(total=len(file_list), desc='Processing') as pbar:
        for root, file in file_list:
            pbar.set_description(f'Processing config {file}')
            try:
                process_config(root, file, conversation_folder, reports_folder,
                               config_dir=config_dir,
                               ignore_human_in_the_loop=ignore_human_in_the_loop,
                               file_hashes_dict=file_hashes_dict,
                               file_hashes_df=file_hashes_df,
                               hash_files_df=hash_files_df)
            except Exception as e:
                logger.error(f'Er is iets fout gegaan terwijl {file} verkwerkt werd: {e}')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.debug(exc_type, fname, exc_tb.tb_lineno)
            finally:
                encountered_errors = []
                for handler in [x for x in logger.handlers if 'QueuingHandler' in x.__class__.__name__]:
                    encountered_errors += list(handler.message_queue.queue)
                if encountered_errors:
                    errors[file] = encountered_errors
                for handler in [x for x in logger.handlers if 'QueuingHandler' in x.__class__.__name__]:
                    handler.message_queue.queue.clear()
            pbar.update()
    error_report(errors, output_dir=reports_folder)


def process_config(root, file, conversation_folder, reports_folder, config_dir, file_hashes_dict, file_hashes_df,
                   hash_files_df):
    config = load_config(os.path.join(root, file))
    config['config_file_name'] = file
    merger = WhatsAppMerger(config=config, config_dir=config_dir, file_hashes=file_hashes_dict)
    merger.contact_mapping.update(config.get('mapping'))
    phone_book = merger.create_phone_book()
    id_ = file[7:12]
    if id_.endswith(".xlsx"):
        id_ = id_.split(".xlsx")[0]
    filename = re.sub(r'[^A-Za-z\d -]+', '', config.get('conversation_name', 'output')).strip()
    filename = f'{filename}_{id_}'
    try:
        merger.run(
            phone_book=phone_book,
            mapping=config.get("mapping")
        )
        reports_folder = os.path.join(reports_folder, filename)
        os.makedirs(
            reports_folder,
            exist_ok=True)
        os.makedirs(
            conversation_folder,
            exist_ok=True)
        merger.save(conversations_path=conversation_folder, reports_path=reports_folder, output_format='pdf,csv,xlsx',
                    filename=filename)
    except MemoryError:
        logger.error(f'Memory Error processing {file}')
    except Exception as e:
        logger.exception(f'Er is iets fout gegaan terwijl {file} verkwerkt werd: {e}')

    merger.generate_conversation_report(output_dir=reports_folder,
                                        filename=filename,
                                        file_hashes_df=file_hashes_df,
                                        hash_files_df=hash_files_df,
                                        gdpr_proof=False)
    merger.generate_conversation_report(output_dir=reports_folder,
                                        filename=f'{id_}_verantwoording',
                                        file_hashes_df=file_hashes_df,
                                        hash_files_df=hash_files_df,
                                        gdpr_proof=True)
