import json
import os
import sys
from pathlib import Path

from chatmerger import WhatsAppMerger
from chatmerger.utils_config import create_configs, regenerate_phonebook_config_for_conversation, process_config, \
    error_report,  load_config, create_parser_config
from chatmerger.utils import get_chats_from_path, get_chat_mapping, get_file_hashes, get_chats_from_config, QueuingHandler
import pandas as pd
from tqdm import tqdm

import logging
import queue

logger = logging.getLogger("WhatsApp Demo")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

message_queue = queue.Queue()
queue_handler = QueuingHandler(message_queue=message_queue, level=logging.DEBUG)
queue_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
queue_handler.setLevel('WARNING')
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("WhatsApp")]

for temp_logger in loggers:
    temp_logger.setLevel(logging.WARNING)
    temp_logger.addHandler(stream_handler)
    temp_logger.addHandler(queue_handler)


input_dir = r'Path\to\Demo Data\Chats'
output_dir = r'Path\to\Demo Data\Output'

output_configs_path = os.path.join(output_dir, 'Configs')


# Step one: parse all chats
chats = get_chats_from_path(input_dir)
chats_dict = create_parser_config(list(chats.values()))
with open(os.path.join(output_configs_path, 'all_chat_settings.json'), 'w') as f:
    json.dump(chats_dict, f, indent=2)

print(f'Step One: Found {len(chats)} chats')
input('Step Two: Check all_chat_settings.json. Press [Enter] when done')

chats = get_chats_from_config(os.path.join(output_configs_path, 'all_chat_settings.json'))
chat_mapping = get_chat_mapping(chats)

print(f"Step Three: Found {len(chat_mapping)} conversations")
create_configs(sorted_mapping=chat_mapping,
               config_dir=output_configs_path,
               chats=chats,
               config_path=output_configs_path)

file_hashes_dict = get_file_hashes(root_directory=input_dir)

input('Step Four: Check configs for conversation composition. Press [Enter] when done')

file_list = os.walk(output_configs_path)

for root, _, files in file_list:
    for file in files:
        if not file.endswith('json') or file == 'all_chat_settings.json':
            continue
        config = load_config(os.path.join(root, file))
        merger = WhatsAppMerger(config=config, config_dir=output_configs_path)
        regenerate_phonebook_config_for_conversation(merger=merger,
                                                     config_name=os.path.join(output_configs_path, file),
                                                     base_config=config)

print("Step Five: Regenerated the configs")
input("Step Six: Check participant mapping in the configs. Press [Enter] when done")

file_hashes_dict = get_file_hashes(root_directory=input_dir)
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
for root, subdirs, files in tqdm(os.walk(output_configs_path)):
    for file in files:
        if file != "all_chat_settings.json" and file.endswith('json'):
            file_list.append([root, file])

errors = {}
with tqdm(total=len(file_list), desc='Processing') as pbar:
    for root, config_file_name in file_list:
        pbar.set_description(f'Processing config {config_file_name}')
        try:
            process_config(root=root,
                           file=config_file_name,
                           conversation_folder=os.path.join(output_dir, 'Conversations'),
                           reports_folder=os.path.join(output_dir, 'Reports'),
                           config_dir=output_configs_path,
                           file_hashes_dict=file_hashes_dict,
                           file_hashes_df=file_hashes_df,
                           hash_files_df=hash_files_df)
        except Exception as e:
            print(f'Er is iets fout gegaan terwijl {config_file_name} verkwerkt werd: {e}')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        finally:
            encountered_errors = []
            for handler in [x for x in logger.handlers if 'QueuingHandler' in x.__class__.__name__]:
                encountered_errors += list(handler.message_queue.queue)
            if encountered_errors:
                errors[config_file_name] = encountered_errors
            for handler in [x for x in logger.handlers if 'QueuingHandler' in x.__class__.__name__]:
                handler.message_queue.queue.clear()
error_report(errors, output_dir=os.path.join(output_dir, 'Reports'))
print("Step Seven: Created conversations and reports")

