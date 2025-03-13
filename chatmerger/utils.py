import json
import math
import re
import secrets

import numpy as np
import unicodedata
import zipfile
from zipfile import BadZipFile

from fpdf import FPDF, FontFace
from tqdm import tqdm
from pathlib import Path
import logging
import os
import hashlib

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


try:
    from .Parser import Parser, IphoneParser, AndroidParser, MESSAGES_DELETED, MESSAGES_OMITTED
    from .Mapping import Mapping
except ImportError:
    from Parser import Parser, IphoneParser, AndroidParser, MESSAGES_DELETED, MESSAGES_OMITTED
    from Mapping import Mapping

logger = logging.getLogger('WhatsApp Misc')


class ContinueHashingException(Exception):
    pass

def create_gantt_plot(df, start_date, end_date, fname='gantt'):
    fig, ax = plt.subplots()
    yticks = []
    plt.axvline(start_date, color='black')
    plt.axvline(end_date, color='black')
    for i, row in df.iterrows():
        timeline = [
            (row['eerste datum'], pd.Timedelta(days=row.pre_selection)),
            (row['start selectie'], pd.Timedelta(days=row.selection)),
            (row['eind selectie'], pd.Timedelta(days=row.post_selection))
        ]
        yticks.append(i + 1 + 0.25)
        ax.broken_barh(xranges=timeline, yrange=(i + 1, 0.5), facecolors=('tab:red', 'tab:green', 'tab:red'))

    ax.set_yticks(yticks)
    ax.set_yticklabels(df['data_holder'])
    date_format = mdates.DateFormatter('%d/%m/%Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=90)
    plt.savefig(f'{fname}.png', bbox_inches='tight')


def zip_output(output_name):
    with zipfile.ZipFile(f'{output_name}.zip', 'w',
                         compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for path in tqdm(Path(output_name).glob('**/*'),
                         desc='Compressing output', disable=logger.level > logging.INFO):
            if all(a == b for a, b in zip(Path(output_name).parts, path.parts)):
                arcname = os.path.join(*path.parts[len(Path(output_name).parts):])
                zf.write(path, arcname=arcname)


def get_file_hashes(root_directory='./', append=False) -> dict:
    def get_digest_from_file(file):
        h = hashlib.sha256()
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
        return h.hexdigest()

    def get_digest_from_path(file_path):
        with open(file_path, 'rb') as file:
            h = get_digest_from_file(file)
        return h

    def get_file_hashes_from_zip(zip_path, seen=None) -> dict:
        file_hashes = {}
        if seen==None:
            seen = set()
        try:
            with zipfile.ZipFile(zip_path, "r") as unzipped_file:
                if not [x.filename for x in unzipped_file.filelist if x.filename.endswith('.txt')]:
                    logger.debug(f'{zip_path} does not contain a .txt file. Therefore we do not need to hash any files'
                                 f' in this zip')
                    return file_hashes
                for zipped_file in unzipped_file.filelist:
                    fpath = str(Path(zip_path, zipped_file.filename))
                    if not zipped_file.filename.endswith(".txt") and not zipped_file.filename.endswith(".zip") and fpath not in seen:
                        with unzipped_file.open(zipped_file.filename, 'r') as whatsapp_file:
                            h = get_digest_from_file(whatsapp_file)
                            file_hashes[fpath] = h
        except zipfile.BadZipFile:
            logger.warning(f'{zip_path} can not be extracted. Skipping this zip file')
        return file_hashes

    logger.info(f"Getting file hashes for files in '{root_directory}'")
    try:
        with open(os.path.join(root_directory, 'file_hashes.json'), 'rb') as f:
            logger.info("Getting the hashes from json")
            file_hashes = json.load(f)
            if append:
                raise ContinueHashingException
            logger.info("File found, returning results.")
            return file_hashes
    except FileNotFoundError:
        file_hashes = {}
        logger.info("File not found, generating hashes.")
    except ContinueHashingException:
        logger.info("File found, generating new hashes for new files.")

    all_paths = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            logger.debug(os.path.join(dirpath, filename))
            all_paths.append(os.path.join(dirpath, filename))

    for fpath in tqdm(all_paths, desc='Hashing files', position=0, leave=True, disable=logger.level > logging.INFO):
        if fpath.endswith('zip'):
            file_hashes.update(get_file_hashes_from_zip(fpath, seen=set(file_hashes.keys())))
        else:
            if str(Path(fpath)) not in file_hashes:
                file_hashes[str(Path(fpath))] = get_digest_from_path(fpath)
    with open(os.path.join(root_directory, 'file_hashes.json'), 'w') as f:
        logger.info("Storing the hashes in json")
        json.dump(file_hashes, f)
    return file_hashes


def get_conversation_list(config: dict) -> list[Parser]:
    conversation_list = []
    for fname, file_config in config.items():
        if file_config.get("load", True):
            phone_os = file_config.get("phone_os")

            wa_parser = Parser('')
            if phone_os == "android":
                wa_parser = AndroidParser(fname,
                                          datetime_format=file_config.get('datetime_format'),
                                          data_holder=file_config.get("data_holder"))
            elif phone_os == "iphone":
                wa_parser = IphoneParser(fname,
                                         datetime_format=file_config.get('datetime_format'),
                                         data_holder=file_config.get("data_holder"))
            wa_parser.parse()
            conversation_list.append(wa_parser)
    return conversation_list


def get_chats_from_path(data_path: str) -> dict[str, Parser]:
    """
    Retrieve and parse chat files from a specified directory.

    This function scans the provided directory for chat files in `.txt` or `.zip` formats,
    initializes a `Parser` object for each valid file, and parses its contents. Each parsed
    chat is then stored in a dictionary with the filename as the key.

    Args:
        data_path (str): Path to the directory containing chat files. The function
            recursively searches through all subdirectories for `.txt` or `.zip` files.

    Returns:
        dict: A dictionary where each key is a file path, and each value is a `Parser`
        object containing the parsed chat data.
    """
    logger.info('Retrieving chats')

    file_list = []
    for root, subdirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".txt") or file.endswith('.zip'):
                file_list.append([root, file])

    chat_list = {}
    with tqdm(total=len(file_list), desc='Creating parsers') as pbar:
        for root, file in file_list:
            pbar.set_postfix(file=file)
            try:
                parser = Parser(fname=f'{root}/{file}', autodetect=True)
                parser.parse()
                chat_list[parser.fname] = parser
            except ValueError as e:
                logger.error(f'Could not parse {file}: {str(e)}')
            except BadZipFile as e:
                logger.error(f'Could not parse {file} because of a bad zip file: {str(e)}')
            pbar.update()
    return chat_list


def get_chats_from_config(config_path: str, base_chat_list: dict[str, Parser] = None) -> dict[str, Parser]:
    """
    Retrieve and parse chat configurations from a JSON file, using pre-existing parsers
    from `base_chat_list` where available.

    This function reads a JSON configuration file containing chat settings and
    initializes a `Parser` object for each chat configuration if it is not already
    in `base_chat_list`. Parsed chats are stored in a dictionary using the filename
    as the key, and existing entries from `base_chat_list` are reused directly.


    Args:
       config_path (str): Path to the JSON configuration file. This file should
           contain a dictionary where keys represent paths to individual chat
           files, and values are dictionaries of chat settings. Each chat
           setting dictionary may include the following optional keys:
               - 'datetime_format': Format string for parsing dates in the chat.
               - 'data_holder': Custom data holder type for the parsed chat data.
       base_chat_list (dict[str, Parser]): Dictionary of pre-parsed `Parser` objects,
           keyed by file path. These will be reused if they exist for any chat paths
           in the configuration.

    Returns:
       dict: A dictionary where each key is a chat file path, and each value
       is a `Parser` object that contains the parsed data for the respective chat.
   """
    logger.info('Retrieving chats')

    if base_chat_list is None:
        chat_list = {}
    else:
        chat_list = base_chat_list.copy()

    with open(config_path) as f:
        all_chats = json.load(f)
        for chat_path, chat_config in tqdm(all_chats.items(), desc='Parsing chats'):

            parser = Parser(fname=chat_path,
                            autodetect=True,
                            datetime_format=chat_config.get('datetime_format'),
                            data_holder=chat_config.get('data_holder'))

            if chat_path in chat_list and parser == chat_list[chat_path]:
                if chat_list[chat_path].data_holder != parser.data_holder:
                    chat_list[chat_path].data_holder = parser.data_holder
                logger.info(f'Skipping already parsed chat: {chat_path}')
                continue
            parser.parse()
            chat_list[chat_path] = parser

    return chat_list


def get_chat_mapping(chats, combine_date_message=True):
    """
    This function groups the chats in conversations.
    """
    chat_message_dict = {}
    for chat in tqdm(chats.values(), desc='creating chat message dict'):
        df = chat.to_dataframe()
        mask = (~df.message.isin(MESSAGES_DELETED) & ~df.message.isin(MESSAGES_OMITTED) & (df.media == '') & (
                df.sender != '') & (df.message.str.len() >= 30))
        if combine_date_message:
            combined = df['date'].dt.strftime('%d-%m-%Y %H:%M') + df['message']
        else:
            combined = df['message']
        chat_message_dict[chat.fname] = set(
            m for m in combined[mask].drop_duplicates()
        )
    chat_mapping = Mapping()
    chat_mapping.apply_label_message_dict(
        label_message_dict=chat_message_dict,
        score_threshold=0.1,
        disable_tqdm=False
    )
    sorted_mapping = {k: chat_mapping.mapping[k] for k in sorted(chat_mapping.mapping.keys())}
    return sorted_mapping


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def dataholder_from_filepath(filepath: str) -> str:
    """Extracts the data_holder from the directory structure from a file path.

    Depending on the file extension, this function returns a different
    directory from the path hierarchy:

    - For `.zip` files, it returns the second-to-last directory in the path.
    - For `.txt` files, it returns the third-to-last directory in the path.
    - For other file extensions, it defaults to returning the third-to-last directory.

    Args:
        filepath (str): The file path to process.

    Returns:
        str: The extracted directory name from the file path.

    Examples:
        >>> dataholder_from_filepath('/Chat Database/2024-12-19 141221/Chats 1/Alice Wonderland/Export 1.zip')
        'Alice Wonderland'
        >>> dataholder_from_filepath('/Chat Database/2024-12-19 141221/Chats 1/Alice Wonderland/Export 1/WhatsApp Chat with Test export.txt')
        'Alice Wonderland'
    """
    if filepath.endswith('.zip'):
        return Path(filepath).parts[-2]
    elif filepath.endswith('.txt'):
        return Path(filepath).parts[-3]

    return Path(filepath).parts[-3]


class QueuingHandler(logging.Handler):
    """A thread safe logging.Handler that writes messages into a queue object.

       Designed to work with LoggingWidget so log messages from multiple
       threads can be shown together in a single ttk.Frame.

       The standard logging.QueueHandler/logging.QueueListener can not be used
       for this because the QueueListener runs in a private thread, not the
       main thread.

       Warning:  If multiple threads are writing into this Handler, all threads
       must be joined before calling logging.shutdown() or any other log
       destinations will be corrupted.
    """

    def __init__(self, *args, message_queue, **kwargs):
        """Initialize by copying the queue and sending everything else to superclass."""
        logging.Handler.__init__(self, *args, **kwargs)
        self.message_queue = message_queue

    def emit(self, record):
        """Add the formatted log message (sans newlines) to the queue."""
        self.message_queue.put(self.format(record).rstrip('\n'))



def get_id():
    """A generator function that returns unique IDs. This generator function allows the caller to generate a continuous
     stream of unique IDs without the need to manually keep track of which IDs have been generated.

    Yields:
        str: A unique ID generated using the `secrets` module.

    Raises:
        None

    """
    seen = set()
    i = 0
    conts = 0
    while True:
        id_ = secrets.token_urlsafe(3)
        if id_ in seen:
            conts += 1
            if conts > 20:
                print('Kon geen id genereren binnen 20 pogingen')
                break
            continue
        conts = 0
        seen.add(id_)
        i+=1
        yield f'c{id_:0>4}'


class TableRow:
    """
    TableRow
    A helper class to sort messages
    """

    def __init__(self, **kwargs):
        self.data = kwargs

    def __eq__(self, o):
        """
        Compares two TableRows for equality.
        TableRows are equal when the dates are equal AND when all the original orders are equal.
        """
        if self.data.get('date') != o.data.get('date'):
            return False
        l_self = [v for k, v in self.data.items() if 'original_order' in k]
        l_other = [v for k, v in o.data.items() if 'original_order' in k]
        if l_self == l_other:
            return True
        return False

    def __le__(self, o):
        """
        Compares two TableRows for less then or equal.
        If the date is smaller, the row is smaller
        If the dates are equal the original order must be checked.
        """
        if self.data.get('date') < o.data.get('date'):
            return True
        elif self.data.get('date') == o.data.get('date'):
            self_orders = [v for k, v in self.data.items() if 'original_order' in k]
            other_orders = [v for k, v in o.data.items() if 'original_order' in k]
            gt = 0
            le = 0
            for so, oo in zip(self_orders, other_orders):
                if not math.isnan(so) and not math.isnan(oo):
                    if so > oo:
                        gt += 1
                    else:
                        le += 1
            if not le and not gt:
                if np.nanmax(np.array(self_orders)) <= np.nanmax(np.array(other_orders)):
                    return True
                else:
                    return False
            elif le >= gt:
                return True
            else:
                return False
        else:
            return False

    def __gt__(self, o):
        if self.data.get('date') > o.data.get('date'):
            return True
        elif self.data.get('date') == o.data.get('date'):
            self_orders = [v for k, v in self.data.items() if 'original_order' in k]
            other_orders = [v for k, v in o.data.items() if 'original_order' in k]
            gt = 0
            le = 0
            for so, oo in zip(self_orders, other_orders):
                if not math.isnan(so) and not math.isnan(oo):
                    if so > oo:
                        gt += 1
                    else:
                        le += 1

            if not le and not gt:
                if np.nanmax(np.array(self_orders)) > np.nanmax(np.array(other_orders)):
                    return True
                else:
                    return False
            elif gt > le:
                return True
            else:
                return False
        else:
            return False

    def __repr__(self):
        return self.data.__repr__()


class PDF(FPDF):
    def __init__(self, parser: Parser = None, title=""):
        super().__init__()
        self.add_font('NotoSansEmojiMerged', '',
                      os.path.join(os.path.dirname(__file__), 'font/NotoSansEmoji-Regular_FontLabMerged.ttf'))
        # Setting font: NotoSansEmoji
        self.set_font("NotoSansEmojiMerged", size=10)
        self.title = title

    def header(self):
        # Rendering logo:
        self.set_font("times", size=10)
        # Moving cursor to the right:
        self.cell(80)
        # Printing title:
        self.cell(30, 10, self.title.encode('utf-8', 'ignore').decode('ascii', 'ignore'), align="C")
        # Performing a line break:
        self.ln(20)

    def footer(self):
        # Position cursor at 1.5 cm from bottom:
        self.set_y(-15)
        # Setting font: helvetica italic 8
        self.set_font("helvetica", "I", 8)
        # Printing page number:
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def print_chapter(self, txt):
        self.add_page()
        self.multi_cell(0, 5, txt, align='L')


def error_report(errors, output_dir):
    template = f'''
        <section> 
        <h2>Files with errors</h2>
        <p>The following files encountered errors while creating the conversations:</p>
        <ul><li>
        {"</li><li>".join(errors.keys())}
        </li></ul>
        </section>
        <section>
        <h2>Errors by file</h2>
    '''
    for file, file_errors in errors.items():
        template += f'<h3>{file}</h3>'
        template += r'<ul>'+'\n'
        template += ''.join(map(lambda x: f'<li>{x}</li>\n', file_errors))
        template += r'</ul>'+'\n'
    template += '</section>'

    tag_styles = {
        "h1": FontFace(color="#000", size_pt=28),
        "h2": FontFace(color="#000", size_pt=24),
        "h3": FontFace(color="#000", size_pt=18),
    }
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('helvetica', size=12)
    pdf.write_html(text=template,
            tag_styles=tag_styles)
    pdf.output(os.path.join(output_dir, f"errors.pdf"))
    print('Errors:', os.path.join(output_dir, f"errors.pdf"))
