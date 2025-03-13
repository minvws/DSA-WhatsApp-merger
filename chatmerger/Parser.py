import zipfile

from dateutil.parser import ParserError
import os

from tqdm import tqdm
from datetime import datetime
from dateutil import parser
import re

import logging
import pandas as pd

tqdm.pandas()

logger = logging.getLogger('WhatsAppParser')

MESSAGES_DELETED = [
    'U hebt dit bericht verwijderd.',
    'Dit bericht is verwijderd.',
    'Dit bericht is verwijderd',
    'You deleted this message',
    'This message was deleted',
    'Gemist spraakgesprek'
]

MESSAGES_OMITTED = [
    "<Media weggelaten>",
    "<Media omitted>",
    "afbeelding weggelaten",
    'video weggelaten',
]


class Parser:
    DATETIME_PATTERN = re.compile(
        r"(?P<dmy1>\d{1,4})(?P<sep>[-\/])(?P<dm2>\d{1,2})[-\/](?P<dmy3>\d{1,4})"
        r"(?P<dtsep>,?\s?)(?P<h>\d{1,2}):(?P<m>\d{1,2}):?(?P<s>\d{0,2})\s?(?P<ampm>PM|AM)?"
    )

    def __new__(cls, fname=None, autodetect=False, datetime_format="", data_holder="", datetime_certain=True, **kwargs):
        if autodetect:
            if datetime_format:
                phone_os = cls.detect_os(fname)
                datetime_certain = True
            else:
                phone_os, datetime_format, datetime_certain = cls.detect_os_and_datetime_format(fname)

            if phone_os == "iphone":
                return IphoneParser(fname=fname, datetime_format=datetime_format,
                                    datetime_certain=datetime_certain, data_holder=data_holder)
            elif phone_os == "android":
                return AndroidParser(fname=fname, datetime_format=datetime_format,
                                     datetime_certain=datetime_certain, data_holder=data_holder)
            else:
                raise ValueError(f"unknown os type: {phone_os}")
        elif cls.__name__ == Parser.__name__ and cls.__module__ == Parser.__module__:
            if phone_os := kwargs.get('phone_os'):
                if phone_os == "iphone":
                    return IphoneParser(fname=fname, autodetect=autodetect, datetime_format=datetime_format,
                                        data_holder=data_holder, datetime_certain=datetime_certain, **kwargs)
                elif phone_os == "android":
                    return AndroidParser(fname=fname, autodetect=autodetect, datetime_format=datetime_format,
                                         data_holder=data_holder, datetime_certain=datetime_certain, **kwargs)
                else:
                    raise ValueError(f"unknown os type: {phone_os}")

        self = object.__new__(cls)
        self._df = pd.DataFrame()
        self.fname = fname
        self.message_list = kwargs.get('message_list', [])
        self.datetime_format = datetime_format
        self.datetime_certain = datetime_certain
        self.data_holder = data_holder if data_holder else fname
        self.example_dates = kwargs.get('example_dates', [None, None])
        return self

    @property
    def df(self):
        if not len(self._df):
            self._df = self.to_dataframe()
        return self._df

    def generate_config(self):
        if self.example_dates[0] is None:
            self.parse()
        return dict(
            phone_os=self.phone_os,
            datetime_format=self.datetime_format,
            example_date_1=self.example_dates[0],
            example_date_2=self.example_dates[1],
            load=True,
            data_holder=self.data_holder,
        )

    @staticmethod
    def detect_os_and_datetime_format(fname):
        lines = Parser._get_file_contents(fname)
        if lines is None:
            return None, None
        phone_os = Parser._detect_os(lines)

        datetimes = []
        for line in lines:
            if phone_os == 'iphone':
                match = IphoneParser.datetime_pattern.match(line)
                if match and Parser.DATETIME_PATTERN.match(match.group(1)):
                    datetimes.append(match.group(1))
            elif phone_os == 'android':
                date_time_raw, _, remaining_line = line.partition(' - ')
                if remaining_line and Parser.DATETIME_PATTERN.match(date_time_raw):
                    datetimes.append(date_time_raw)
            else:
                return None, None
        datetime_format, datetime_certain = Parser._detect_datetime_format(datetimes)
        logger.info(f"{fname} is exported from {phone_os} with datetime format {datetime_format}")
        return phone_os, datetime_format, datetime_certain

    @staticmethod
    def _detect_datetime_format(datetimes, dayfirst=None):
        def only_year(options):
            return all(item in ['y', 'Y'] for item in options)

        def fix_year(o):
            """
            When a position can only be a year, check the number of digits and set accordingly
            """

            for pos, option in o.items():
                if only_year(option):
                    o[pos] = {"Y"} if (df[pos].str.len() == 4).any() else {'y'}

            return o

        def use_certainties(uncertain_options):
            n_certain_prev = 0
            certain = False
            uncertain_options = fix_year(uncertain_options)
            while len(certain_positions := [x for x in uncertain_options
                                            if (len(uncertain_options[x]) == 1) or only_year(
                    uncertain_options[x])]) != n_certain_prev:
                certain = True
                for certain_pos in certain_positions:
                    certain_value = uncertain_options[certain_pos]
                    for pos in uncertain_options:
                        if pos != certain_pos:
                            uncertain_options[pos] -= certain_value
                n_certain_prev = len(certain_positions)
            return fix_year(uncertain_options), certain

        def date_format_done(options):
            if len([x for x in options if len(options[x]) == 1]) == 3:
                pos1 = options.get('dmy1').pop()
                pos2 = options.get('dm2').pop()
                pos3 = options.get('dmy3').pop()
                return pos1, pos2, pos3

        def detect_date_format(df, dayfirst):
            options = {
                'dmy1': set('dmyY'),
                'dm2': set('dm'),
                'dmy3': set('dmyY')
            }
            # When a position contains a number higher than 12, it can not be a month
            for pos, can_be_month in (df[['dmy1', 'dm2', 'dmy3']].max().astype(int) <= 12).items():
                if not can_be_month:
                    options[pos].discard('m')

            # When a position contains a string that is exactly 4 characters, it must be a year (Y)
            if (contains_year := df[['dmy1', 'dm2', 'dmy3']].apply(lambda x: (x.str.len() == 4).any())).any():
                for pos, is_year in contains_year.items():
                    if not is_year:
                        options[pos] -= set('Yy')
                    if is_year:
                        options[pos] = {"Y"}

            # When a position has only one possibility, remove that possibility from the other positions
            options, date_certain = use_certainties(options)

            if positions := date_format_done(options):
                return positions, date_certain

            if dayfirst is None:
                dayfirst = len(df.dmy1.value_counts()) >= len(df.dm2.value_counts())

            datetime_certain = False

            if len(options['dmy1']) > 1:
                if dayfirst:
                    options['dmy1'] = {"d"}
                else:
                    options['dmy1'] = {"m"}

            options, _ = use_certainties(options)
            if positions := date_format_done(options):
                return positions, datetime_certain

            if len(options['dm2']) > 1 or len(options['dmy3']) > 1:
                # By fixing the first position to either day or month there are no other options left so this code
                # should be unreachable
                raise Exception("How did this happen?")

        groupdicts = []
        for dt in datetimes:
            groupdicts.append(Parser.DATETIME_PATTERN.match(dt).groupdict())
        df = pd.DataFrame(groupdicts)

        sep = df.sep.values[0]

        (pos1, pos2, pos3), datetime_certain = detect_date_format(df, dayfirst)

        dtsep = df.dtsep.values[0]

        datetime_format = f'%{pos1}{sep}%{pos2}{sep}%{pos3}{dtsep}%H:%M'

        if (df.s.str.len() > 0).any():
            datetime_format += ":%S"

        return datetime_format, datetime_certain

    @staticmethod
    def detect_os(fname):
        lines = Parser._get_file_contents(fname)
        if lines is None:
            return None, None
        phone_os = Parser._detect_os(lines)
        return phone_os

    @staticmethod
    def _detect_os(lines):
        if lines[0][0] == '[':
            phone_os = 'iphone'
        elif "-" in lines[0][:18] and lines[0][0].isnumeric():
            phone_os = 'android'
        else:
            phone_os = 'unknown'
        return phone_os

    def is_date(self, possible_date):
        if self.datetime_format:
            try:
                return datetime.strptime(possible_date, self.datetime_format).replace(second=0)
            except ValueError:
                return None
        else:
            try:
                return parser.parse(possible_date, dayfirst=True).replace(second=0)
            except ParserError:
                return None

    def get_media(self, message):
        pattern = re.compile(
            r"<bijgevoegd: (.*\.[a-zA-Z\d]{1,5})>|"
            r"(.*) \(file attached\)|"
            r"(.*) \(bestand bijgevoegd\)|"
            r"<Media omitted>|"
            r"<Media weggelaten>"
        )
        match = pattern.search(message)
        if match:
            if result := "".join([str(x) for x in match.groups() if x]).strip():
                if self.fname.endswith('.zip'):
                    media_folder = self.fname
                else:
                    media_folder = os.path.dirname(self.fname)
                return os.path.join(media_folder, result)
            else:
                return "<Media weggelaten>"
        return ""

    def add_message(self, **kwargs):
        if 'end-to-end' in kwargs.get('message', ''):
            return

        self.message_list.append(kwargs)

    def __bool__(self):
        if self.fname.endswith(".txt"):
            return True
        elif self.fname.endswith('.zip'):
            if 'iMessage' in self.fname:
                logger.info(f"Zip is probably iMessage '{self.fname}'. This file will be skipped")
                return False
            with zipfile.ZipFile(self.fname, "r") as unzipped_file:
                for zipped_file in unzipped_file.filelist:
                    if zipped_file.filename.endswith(".txt"):
                        return True

    @staticmethod
    def _get_file_contents(fname):
        if fname.endswith(".txt"):
            with open(fname, 'r', encoding='utf-8') as whatsapp_file:
                lines = []
                for count, line in enumerate(whatsapp_file.readlines()):
                    line = re.sub('[\u200e\u202a\xa0\u202c\ufeff]', '', line)
                    if len(lines) == 0 and not line.strip():
                        continue
                    lines.append(line)
                return lines
        elif fname.endswith('.zip.zip'):
            fname = fname.split(".zip")[0] + ".zip"
            with zipfile.ZipFile(fname, "r") as unzipped_file:
                for zipped_file in unzipped_file.filelist:
                    if zipped_file.filename.endswith(".txt"):
                        with unzipped_file.open(zipped_file.filename, 'r') as whatsapp_file:

                            lines = []
                            for count, line in enumerate(whatsapp_file.readlines()):
                                line = line.decode("utf-8")
                                line = re.sub('[\u200e\u202a\xa0\u202c\ufeff]', '', line)
                                if len(lines) == 0 and not line.strip():
                                    continue
                                lines.append(line)
                            return lines

        elif fname.endswith('.zip'):
            with zipfile.ZipFile(fname, "r") as unzipped_file:
                for zipped_file in unzipped_file.filelist:
                    if zipped_file.filename.endswith(".txt"):
                        with unzipped_file.open(zipped_file.filename, 'r') as whatsapp_file:

                            lines = []
                            for count, line in enumerate(whatsapp_file.readlines()):
                                line = line.decode("utf-8")
                                line = re.sub('[\u200e\u202a\xa0\u202c\ufeff]', '', line)
                                if len(lines) == 0 and not line.strip():
                                    continue
                                lines.append(line)
                            return lines

    def parse(self):
        logger.info(f"parsing {self.fname}...")

        name = message = ""
        message_date_time = None
        message_count = 0
        last_date_time_raw = ''
        for line in tqdm(file_contents := self._get_file_contents(self.fname), position=0,
                         leave=True, desc=f'parsing {self.fname}', delay=10):
            date_time_raw, date_time, remaining_line = self.split_line_by_date(line)
            if date_time_raw:
                last_date_time_raw = date_time_raw
            if date_time:
                if message:
                    media = self.get_media(message)
                    self.add_message(
                        date=message_date_time,
                        sender=name.strip(),
                        message=message.strip(),
                        media=media,
                        file=self.fname,
                        original_order=message_count
                    )
                    message = ""
                    message_count += 1
                if ":" in remaining_line:
                    message_date_time = date_time
                    name, _, message = remaining_line.partition(":")
                else:
                    self.add_message(
                        date=date_time,
                        message=remaining_line.strip(),
                        file=self.fname,
                        original_order=message_count
                    )
                    message_count += 1
            else:
                message += line
        if message:
            media = self.get_media(message)
            self.add_message(
                date=message_date_time,
                sender=name.strip(),
                message=message.strip(),
                media=media,
                file=self.fname,
                original_order=message_count
            )
        self.example_dates = (self.split_line_by_date(file_contents[0])[0],
                              last_date_time_raw)
        logger.info(f"Parsed {self.fname}. Found {len(self.message_list)} messages.")
        return self

    def to_dataframe(self, add_merged_column=False, add_data_holder=True):
        df = pd.DataFrame(self.message_list)
        df = df.fillna("")
        if add_data_holder:
            df['data_holder'] = self.data_holder
        if add_merged_column:
            df['merged'] = False
        if 'media' not in df:
            df['media'] = ''
        if 'sender' not in df:
            df['sender'] = ''
        self._df = df
        return self._df

    def to_dict(self):
        message_list = []
        for message in self.message_list:
            message['date'] = message['date'].strftime('%Y-%m-%d %H:%M:%S')
            message_list.append(message)
        d = {
            'data_holder': self.data_holder,
            'datetime_certain': self.datetime_certain,
            'datetime_format': self.datetime_format,
            'example_dates': self.example_dates,
            'fname': self.fname,
            'message_list': message_list,
            'phone_os': self.phone_os,
        }
        return d

    @classmethod
    def from_dict(cls, parser_dict):
        if message_list := parser_dict.get('message_list'):
            for message in message_list:
                message['date'] = datetime.strptime(message['date'], '%Y-%m-%d %H:%M:%S')
        return cls.__new__(cls, **parser_dict)


    def get_datetime_range(self):
        if not len(self._df):
            self.to_dataframe()

        if 'date' not in self._df:
            return None, None

        min_date = self._df[self._df.sender != ''].date.min().date()
        max_date = (self._df[self._df.sender != ''].date.max() + pd.DateOffset(1)).date()
        return min_date, max_date

    def split_line_by_date(self, line):
        original_date = ''
        date_time = ""
        remaining_line = line
        return original_date, date_time, remaining_line

    def __str__(self):
        return f"A {self.__class__.__name__} object for {self.fname}, parsed {len(self.message_list)} messages"

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.fname}')"

    def __len__(self):
        return len(self.message_list)

    def __eq__(self, other):
        return (self.fname == other.fname and
                self.datetime_format == other.datetime_format)


class AndroidParser(Parser):
    phone_os = "android"

    def split_line_by_date(self, line):
        date_time_raw, _, remaining_line = line.partition(' - ')
        date_time = self.is_date(date_time_raw)
        return date_time_raw, date_time, remaining_line


class IphoneParser(Parser):
    datetime_pattern = re.compile(r"^\[(\d{2}-\d{2}-\d{4},? \d{2}:\d{2}:\d{2})]")
    phone_os = "iphone"

    def split_line_by_date(self, line):
        match = self.datetime_pattern.match(line)
        date_time_raw = match.group(1) if match else ''
        remaining_line: str = line[22:]
        date_time = self.is_date(date_time_raw) if match else ""
        return date_time_raw, date_time, remaining_line
