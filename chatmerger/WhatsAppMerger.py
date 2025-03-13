import os
import re
import shutil
import warnings
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import collections.abc
from statistics import mode

import pandas as pd
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
from fpdf import FontFace
from tqdm import tqdm, trange

import logging

try:
    from .Mapping import ContactMapping
    from .Parser import MESSAGES_DELETED, MESSAGES_OMITTED, Parser
    from .utils import zip_output, create_gantt_plot, get_conversation_list, dataholder_from_filepath, TableRow, PDF
except ImportError:
    from Mapping import ContactMapping
    from Parser import MESSAGES_DELETED, MESSAGES_OMITTED, Parser
    from utils import zip_output, create_gantt_plot, get_conversation_list, dataholder_from_filepath, TableRow, PDF

tqdm.pandas()

logger = logging.getLogger('WhatsAppMerger')

VERANTWOORDING_TEMPLATE_PART_1 = """
  <h1>Verantwoordingsdocument samenvoeging WhatsApp gesprekken: %(conversation_name)s</h1>
  %(error_html)s
  <section>
    <h2>Introductie</h2>
    <p>Dit is een automatisch gegenereerd document en geeft context voor een automatisch ontdubbeld chatgesprek.
Het gesprek is ontdubbeld op basis verschillende ge&euml;xporteerde Whatsapp export-bestanden. Deze bestanden
zijn verkregen van de telefoon(s) van één of meerdere <em>datahouders</em>.
    </p>
  </section>
  <section>
    <h2>Disclaimer</h2>
      <p>Tijdens het exporteren van chatberichten uit telefoons gaat bepaalde informatie over de data wel en niet mee.
De volgende informatie gaat wel mee bij het exporteren:</p>
    <ul>
      <li>Mediabestand(en);</li>
      <li>Locatie(s);</li>
      <li>Contacten;</li>
      <li>Peilingen (ge&iuml;mplementeerd in WhatsApp in november 2022).</li>
    </ul>
    <p>De volgende informatie gaat niet mee bij het exporteren:</p>
    <ul>
        <li>De verwijzing dat een bericht is doorgestuurd;</li>
        <li>De verwijzing dat een bericht het antwoord is op een eerder bericht;</li>
        <li>Met ster gemarkeerde berichten;</li>
        <li>Emoji reactie(s) op een specifiek bericht (gemplementeerd in WhatsApp in mei 2022);</li>
        <li>De groepsbeschrijving;</li>
        <li>De groepsafbeelding of individuele profielfoto's.</li>
    </ul>
</section>
<section>
    <h2>Samenvatting</h2>
    <p>Voor het chatgesprek '%(conversation_name)s' zijn %(n_conversations)s ge&euml;xporteerde bestanden samengevoegd.
Deze bestanden bevatten gezamenlijk %(orig_len)s chatberichten. Dit aantal is gereduceerd tot  %(merged_len)s berichten. Een
reductie van %(reduction)s.</p><p>
Door periode selectie is het aantal bijlagen gereduceert van %(n_bijlagen_orig)s naar %(n_bijlagen_date_range)s. Dit is 
door ontdubbelen verder gereduceert naar %(n_bijlagen_merged)s.</p>
</section>
"""

VERANTWOORDING_TEMPLATE_PART_2 = """
<section>
    <h2>Bronvermelding</h2>
    <p>Er zijn %(n_conversations)s uitgelezen bestanden, elke chat wordt aangeduid met een label. In onderstaande tabel
is te zien welke labels bij welk export bestand horen.
%(chat_map)s</p>
</section>
<section>
    <h2>Bronsamenvatting</h2>
    <p>Onderstaande tabel toont het aantal berichten per chat. Daarnaast laat het zien hoeveel van deze berichten zijn
 verwijderd, media bevatten, of weggelaten media bevatten.</p>
%(sources_table)s
<p>
Per chat is ook te zien welke periode beschikbaar is in de export. Voor zowel de eerste datum als de laatste datum 
is gekeken naar berichten die door een persoon zijn verstuurd. Hierbij is het aanmaken en/of verlaten van een groep dus 
niet meegenomen.</p>
%(chat_periode)s</section>"""
VERANTWOORDING_TEMPLATE_PART_3 = """
<img chatmerger="gantt.png" width="400">
<section>
    <h2>Bijlagen openen</h2>
    <p>In het pdf-bestand van de conversatie zit een ZIP bijlage met daarin alle media bestanden die in het geprek zijn verstuurd.
Deze bestanden zijn uit de zip te pakken met het programma <a href="https://www.7-zip.org/">7-ZIP</a>. Open 7-ZIP en navigeer naar 
het pdf-bestand document dat uitgepakt dient te worden. Druk vervolgens op de 'Uitpakken' knop en kies een 
locatie. Klik op 'OK'. Klik na het uitpakken op 
'Afsluiten'.</p>
</section>
<section>
    <h2>Bijlagen referentie</h2>
    <p>De bijgeleverde media bestanden zijn bij alle ontvangers hetzelfde bestand, ze zijn oorspronkelijk wel anders genoemd.
Daarom zit in de conversatie maar &eacute;&eacute;n kopie van elk bestand. Onderstaande lijst toont welke bestanden
identiek zijn. De bestanden zijn gesorteerd op de datum dat ze in het gesprek zijn verzonden. Dik gedrukte regels geven 
aan welk bestand gekozen is om dezelfde bestanden te representeren. Schuin gedrukte regels geven aan welke bestanden
gelijk zijn aan het verstuurde bestand, maar zich in een ander gesprek bevinden.</p>

%(bijlagen_html_list)s
</section>
"""


class WhatsAppMerger:
    TEMPLATES = [
        {
            'regex': "(?P<admin>.*) (heeft|hebt) deze groep (aan)?gemaakt",
            'template': "{admin} heeft deze groep aangemaakt"
        },
        {
            'regex': "(?P<admin>.*) (heeft|hebt)( de)? groep ['“\"]?(?P<to_topic>.*)['”\"]? (aan)?gemaakt",
            'template': "{admin} heeft de groep '{to_topic}' aangemaakt"
        },
        {
            'regex': "(?P<admin>.*) created group ['“\"]?(?P<to_topic>.*)['”\"]?",
            'template': "{admin} heeft de groep '{to_topic}' aangemaakt"
        },
        {
            'regex': "(?P<admin>.*) (heeft|hebt) (?P<name>.*) toegevoegd",
            'template': "{admin} heeft {name} toegevoegd"
        },
        {
            'regex': "(?P<name>.*) neemt deel via (?P<admin>.*) uitnodiging",
            'template': "{admin} heeft {name} toegevoegd"
        },
        {
            'regex': "(?P<admin>.*) added (?P<name>.*)",
            'template': "{admin} heeft {name} toegevoegd"
        },
        {
            'regex': "(?P<admin>.*) (heeft|hebt) (?P<name>(?!dit bericht).*) verwijderd",
            'template': "{admin} heeft {name} verwijderd"
        },
        {
            'regex': "(?P<name>.*) (bent|is) toegevoegd",
            'template': "{name} is toegevoegd"
        },
        {
            'regex': "(?P<name>.*) is deelnemer geworden via de uitnodigingslink van deze groep",
            'template': "{name} is toegevoegd"
        },
        {
            'regex': "(?P<name>.*) (heeft|hebt) de groep verlaten",
            'template': "{name} heeft de groep verlaten"
        },
        {
            'regex': "(?P<name>.*) left",
            'template': "{name} heeft de groep verlaten"
        },
        {
            'regex': "(?P<name>.*) is verwijderd",
            'template': "{name} is verwijderd"
        },
        {
            'regex': "(?P<admin>.*) (heeft|hebt) de groepsafbeelding gewijzigd",
            'template': "{admin} heeft de groepsafbeelding gewijzigd"
        },
        {
            'regex': "(?P<admin>.*) bent nu (een )?beheerder",
            'template': "{admin} is nu een beheerder",
            'use_for_mapping': False
        },
        {
            'regex': "(?P<admin>.*) (heeft|hebt) het (groeps)?onderwerp gewijzigd naar ['“\"]?(?P<to_topic>.*)['”\"]?",
            'template': "{admin} heeft het onderwerp gewijzigd naar '{to_topic}'"
        },
        {
            'regex': "(?P<admin>.*) (heeft|hebt) het (groeps)?onderwerp gewijzigd van ['“\"]?(?P<from_topic>.*)['”\"]? naar ['“\"]?(?P<to_topic>.*)['”\"]?",
            'template': "{admin} heeft het onderwerp gewijzigd van '{from_topic}' naar '{to_topic}'"
        },
        {
            'regex': "(?P<admin>.*) (heeft|hebt) het (groeps)?onderwerp gewijzigd",
            'template': "{admin} heeft het onderwerp gewijzigd"
        },
        {
            'regex': "(?P<name>.*) joined using (?P<admin>.*) invite",
            'template': "{admin} heeft {name} toegevoegd"
        },
        {
            'regex': r"(?P<name>.*) heeft berichten met vervaldatum ingeschakeld\. (Alle n|N)ieuwe berichten "
                     r"verdwijnen( na)? (?P<duration>[0-9]{1,2} (dagen|uren))( nadat ze zijn verzonden)? uit "
                     r"deze chat\.( Tik om dit te wijzigen)?",
            'template': "{name} heeft berichten met vervaldatum ingeschakeld. Alle nieuwe berichten verdwijnen "
                        "{duration} nadat ze zijn verzonden uit deze chat.",
            'use_for_mapping': False
        },
        {
            'regex': r"(?P<name>.*) heeft berichten met vervaldatum uitgeschakeld\.( Tik om dit te wijzigen)?",
            'template': "{name} heeft berichten met vervaldatum uitgeschakeld.",
            'use_for_mapping': False
        },
        {
            'regex': r"(?P<name>.*) heeft een nieuw telefoonnummer\. "
                     r"Tik om een bericht te sturen of om het nieuwe nummer toe te voegen\.",
            'template': "{name} heeft een nieuw telefoonnummer."
                        " Tik om een bericht te sturen of om het nieuwe nummer toe te voegen.",
            'use_for_mapping': False
        },

    ]

    def __init__(self, config=None, config_dir=None, conversation_list=None, file_hashes=None,
                 conversation_name_prefix="WhatsApp conversatie "):
        if conversation_list is not None:
            self.conversation_list = conversation_list
        elif config is not None:
            self.conversation_list: list[Parser] = get_conversation_list(config['files'])
        if not len(self.conversation_list):
            raise ValueError('There must be parsers in the conversation list to merge.')
        self.config = config if config else {}
        self.config_dir = config_dir
        self.concat_df = pd.concat(c.to_dataframe() for c in self.conversation_list)
        self.file_hashes = file_hashes if file_hashes else {}
        self.hash_file = {v: k for k, v in self.file_hashes.items()} if self.file_hashes else {}
        self.conversation_name_prefix = conversation_name_prefix

        self.merged_df: pd.DataFrame = pd.DataFrame()
        self.contact_mapping: ContactMapping = ContactMapping(self.config_dir)
        self.topics: dict = {}
        self.topic = ""
        self.chat_participants = None
        self.statistics = defaultdict(dict)
        l = [x.data_holder for x in self.conversation_list]
        if len(l) > len(set(l)):
            # one or more dataholders have multiple version of this conversation. The get_data_holders
            # method will make them all unique. This is neccecary to differentiate between them later on.
            self.get_data_holders()

    def author_mapping(self, text_threshold=0.5):
        self.author_mapping_by_text(score_threshold=text_threshold)
        self.parse_announcements()
        logger.info(f"Mapped authors for {self.topic}")
        return self.contact_mapping.mapping

    @staticmethod
    def _replace_self(row):
        self_list = ['u', 'uw', 'you', 'your']
        for col in ['name', 'admin']:
            if col in row and isinstance(row[col], str) and row[col].lower() in self_list:
                row[col] = row.data_holder
        return row

    def parse_announcements(self):
        logger.info("Parsing announcements")
        df = self._extract_variables_from_announcements()
        if len(df) == 0:
            return

        # Save topics
        trailing_quotes = '"”'
        if not df[~df.to_topic.isna()].empty:
            df_topic = df[~df.to_topic.isna()].sort_values('date')[['date', 'to_topic']]
            df_topic['to_topic'] = df_topic.to_topic.str.strip(trailing_quotes)
            df_topic = df_topic.drop_duplicates()
            self.topics = {x['date'].strftime('%Y-%m-%d %H:%M'): x['to_topic'] for x in
                           df_topic.to_dict(orient='records')}
            if temp := df_topic.to_topic.tolist():
                self.topic = temp[-1]

        # Group contact labels
        groups = []
        for date, date_df in tqdm(
                df[df.use_for_mapping].groupby(['date', 'template']),
                desc="Grouping contact labels", disable=logger.level > logging.INFO):
            for col in ['name', 'admin']:
                if col in date_df:
                    if len(date_df) == len(date_df.data_holder.unique()):
                        map_group = list(tuple(x) for x in date_df[['data_holder', col]].to_records(index=False) if
                                         isinstance(x[1], str))
                        if map_group:
                            groups.append(map_group)
        self.contact_mapping.add_groups(groups)

    def get_data_holders(self) -> dict:
        data_holders = {}
        name_set = set()
        for i, parser in enumerate(self.conversation_list):
            data_holder_counter = ''
            filepath = parser.fname
            original_data_holder = parser.data_holder
            if filepath != original_data_holder:
                data_holder = original_data_holder
            elif (match := dataholder_from_filepath(filepath)) != filepath:
                data_holder = match
            elif original_data_holder.startswith('Chat '):
                data_holder = self.contact_mapping.reversed_mapping.get((original_data_holder, original_data_holder),
                                                                        f'Chat {i:02d}')
            else:
                data_holder = original_data_holder
            while data_holder + data_holder_counter in name_set:
                data_holder_counter = f' {int(data_holder_counter) + 1}' if data_holder_counter else ' 1'
            data_holder += data_holder_counter
            name_set.add(data_holder)
            data_holders[filepath] = data_holder
            parser.data_holder = data_holder
        self.contact_mapping.apply_data_holders(data_holders)
        self.concat_df['data_holder'] = self.concat_df.apply(
            lambda x: data_holders.get(x.data_holder, data_holders.get(x.file, x.data_holder)), axis=1)
        return data_holders

    def get_topic(self, base_config=None, regenerate_config=None):
        if regenerate_config:
            self.topic = base_config["conversation_name"].removeprefix(self.conversation_name_prefix)
        elif len(self.contact_mapping.mapping) == 2:
            self.topic = " - ".join(self.contact_mapping.mapping.keys())
        return self.topic

    def get_conversation_name(self, gdpr_proof=False):
        if not gdpr_proof:
            name = ": ".join([self.config.get("config_file_name")[7:12], self.config.get('conversation_name')])
            return name
        return f'{self.conversation_name_prefix.strip()} {self.config.get("config_file_name")[7:12]}'

    def _extract_variables_from_announcements(self):
        df = self.concat_df.set_index(['data_holder'], append=True)
        variables = ['admin', 'name', 'to_topic', 'from_topic', 'duration']
        df = df.reindex(
            columns=list(df.columns) + variables + ['template', 'use_for_mapping'])
        df['template'] = ''

        pattern = re.compile(r'\{([^{}]*)\}')
        for template in self.TEMPLATES:
            matches = df[(df.template == '') & (df.sender == '')].message.str.extract(template['regex'])
            matches.loc[
                ~matches[pattern.findall(template['template'])].isna().any(axis=1), ['template', "use_for_mapping"]
            ] = [template['template'], template.get("use_for_mapping", True)]
            df.update(matches)
            df = df.replace(b'', np.nan)
        df['use_for_mapping'].fillna(False, inplace=True)
        df.reset_index(level='data_holder', inplace=True)
        df[variables] = df[variables].apply(lambda x: x.str.strip() if x.dtype=='object' else x)
        df = df.apply(self._replace_self, axis=1)
        return df

    def author_mapping_by_text(self, score_threshold: float = 0.5, min_message_length: int = 10) -> None:
        author_message_dict = {}
        for data_holder, conv_df in tqdm(self.concat_df.groupby('data_holder'), desc="Mapping authors by text",
                                         disable=logger.level > logging.INFO):
            messages_by_author = conv_df[conv_df["sender"] != ""].groupby(by="sender")
            for author, df in messages_by_author:
                mask = (~df.message.isin(MESSAGES_DELETED) & (df.media == '') & (df.sender != ''))
                author_message_dict[(data_holder, author)] = set(
                    m for m in df[mask].message.drop_duplicates() if len(m) >= min_message_length)
        self.contact_mapping.apply_label_message_dict(label_message_dict=author_message_dict,
                                                      score_threshold=score_threshold)

    def create_phone_book(self):
        phone_book = {}
        for author, original_labels in self.contact_mapping.mapping.items():
            for i, label in original_labels:
                label = str(label).replace(" ", "")
                if label.strip("+").isnumeric():
                    if label not in phone_book:
                        phone_book[label] = author
                    elif phone_book[label] != author:
                        logger.debug(f"{label} already in phonebook: {phone_book[label]}")
        return phone_book

    def apply_mapping(self, phone_book=None):
        logger.info('Applying mapping...')
        if phone_book is None:
            phone_book = {}

        df = self._extract_variables_from_announcements()

        # Apply contact mapping
        dfs = []
        for data_holder, chat_df in df.groupby("data_holder"):
            self._apply_mapping(reversed_map=self.contact_mapping.reversed_mapping, phone_book=phone_book, df=chat_df,
                                column=['sender', 'name', 'admin'])
            dfs.append(chat_df)
        df = pd.concat(dfs)

        # Rewrite messages
        def apply_template(values, template):
            return template.format(**values.to_dict())

        dfs = []
        df['template'] = df['template'].fillna('')
        for template, template_df in df.groupby('template'):
            if template:
                template_df['message'] = template_df.apply(apply_template, args=[template], axis=1)
            dfs.append(template_df)

        df = pd.concat(dfs)
        complete_df = self.concat_df.set_index('data_holder', append=True)
        df2 = df.set_index('data_holder', append=True)
        complete_df.loc[df2.index, 'message'] = df2.message
        complete_df.loc[df2.index, 'sender'] = df2.sender
        self.concat_df = complete_df.reset_index(level='data_holder')

    @staticmethod
    def _apply_mapping(*, reversed_map, phone_book, df, column):
        data_holder = df.data_holder.values[0]
        if not isinstance(column, list):
            column = [column]
        for col in column:
            if col in df:
                for author in df[col].unique():
                    if author and isinstance(author, str):
                        if (label := reversed_map.get((data_holder, author))) is None:
                            label = phone_book.get(author.replace(" ", ""), author)
                        df.loc[df[col] == author, col] = label

    def deduplicate_attachments(self):
        def inner_func(x):
            x = str(Path(x))
            return self.hash_file.get(self.file_hashes.get(x), x)

        def get_hashes(x):
            x = str(Path(x))
            return self.file_hashes.get(x)

        logger.info(f"Deduplicating attachments...")
        mask = (self.concat_df.media != '') & (self.concat_df.media != '<Media weggelaten>')
        mapped_media = self.concat_df[mask]['media'].apply(inner_func)

        self.concat_df.loc[mask, 'media'] = mapped_media
        self.concat_df.loc[mask, 'message'] = mapped_media.apply(lambda x: f'<bijgevoegd: {Path(x).name}>')
        self.concat_df.loc[(self.concat_df.media == '<Media weggelaten>'), 'message'] = '<Media weggelaten>'

        # toevoegen van kolom met hash code
        self.concat_df.loc[mask, 'hash'] = self.concat_df[mask]['media'].apply(get_hashes)

        # check of er messages zijn met verschillende hashes
        df_double = self.concat_df.loc[mask].groupby(by=["message"])["hash"].nunique().reset_index()
        df_double = df_double.rename(columns={"hash": "aantal_hashes"})
        # merge zodat aantal hashes per message is toegevoegd
        self.concat_df = self.concat_df.merge(df_double,
                                              how="left",
                                              on="message")

        # maak nieuwe kolom aan die wordt gebruikt om media weg te schrijven als meerdere hashes zijn
        self.concat_df["media_to"] = ""

        for duplicate_number, (_, df_hash) in enumerate(
                self.concat_df[self.concat_df["aantal_hashes"] > 1].groupby('hash')
        ):
            duplicate_number += 1
            message_from_first_row = df_hash['message'].values[0]
            new_message = (message_from_first_row.split(".")[0] + "_" +
                           str(duplicate_number) + "." + message_from_first_row.split(".")[1])
            df_hash['message'] = new_message

            media_from_first_row = df_hash['media'].values[0]
            df_hash['media_to'] = (media_from_first_row.split(media_from_first_row.split("\\")[-1])[0] +
                                   new_message[13:-1])
            self.concat_df.loc[df_hash.index, ['message', 'media_to', "hash"]] = df_hash[
                ['message', 'media_to', "hash"]]

        self.concat_df = self.concat_df.drop(columns=["aantal_hashes"])

    def _get_chat_config_by_data_holder(self, data_holder):
        files_config = self.config.get('files', {})
        for path, file_config in files_config.items():
            if file_config.get('data_holder') == data_holder:
                return file_config
        return {}

    def merge(self):
        logger.info(f"Merging {len(self.concat_df.data_holder.unique())} chats...")
        if 'media' not in self.concat_df:
            self.concat_df['media'] = ''
        self.concat_df['message-media'] = self.concat_df[['message', 'media']] \
                                              .replace('', float('nan'))[['message', 'media']] \
                                              .ffill(axis=1).iloc[:, 1]

        res = pd.DataFrame()
        prev_name = ''
        for i, (data_holder, g) in enumerate(self.concat_df.groupby('data_holder', sort=False)):
            file_config = self._get_chat_config_by_data_holder(data_holder)
            n_media_before = len(g[(g.media != '') & (g.media != '<Media weggelaten>')])
            n_message_before = len(g[g.media == ''])
            if start_date := file_config.get('start_date'):
                if isinstance(start_date, str):
                    mask = (g.date > pd.to_datetime(start_date, format='%d-%m-%Y'))
                else:
                    mask = g.date > start_date
                g = g[mask]
            if end_date := file_config.get('end_date'):
                if isinstance(end_date, str):
                    mask = (g.date < pd.to_datetime(end_date, format='%d-%m-%Y'))
                else:
                    mask = g.date < end_date
                g = g[mask]

            n_media_after = len(g[(g.media != '') & (g.media != '<Media weggelaten>')])
            n_message_after = len(g[g.media == ''])

            self.statistics['data_holders'][data_holder] = dict(
                n_media_before=n_media_before,
                n_message_before=n_message_before,
                n_media_after=n_media_after,
                n_message_after=n_message_after
            )
            if i == 0 or len(res) == 0:
                res = g[['date', 'sender', 'message-media', 'original_order']]
            else:
                res = res.merge(
                    g[['date', 'sender', 'message-media', 'original_order']],
                    how='outer',
                    on=['date', 'sender', 'message-media'],
                    suffixes=['_' + prev_name, '_' + data_holder]
                )
            duplicated = res.duplicated(subset=['date', 'sender', 'message-media'])
            if sum(duplicated):
                res = res.drop(res[duplicated].index)
            prev_name = data_holder

        tr_list = [TableRow(**x) for x in res.to_dict(orient='records')]
        tr_list.sort()
        sorted_result = pd.DataFrame([x.data for x in tr_list])

        cols = [c for c in sorted_result.columns if c.startswith('original_order')]
        for col in cols:
            data_holder = col[14:].strip("_")
            # TODO PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling
            #  `frame.insert` many times, which has poor performance.  Consider joining all columns at once
            #  using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
            sorted_result['in_order_' + data_holder] = (sorted_result[col] - sorted_result[col].shift(1)).fillna(0) >= 0

        def fix_orders(res):
            cols = [x for x in res.columns if x.startswith('original_order')]
            for i in trange(len(res), desc='Fixing the order', disable=logger.level > logging.INFO):
                row = res.iloc[i]
                max_order = row[cols].max()
                for col in cols:
                    if (diff := (max_order - row[col])) > 0:
                        res.loc[res[col] >= row[col], col] += diff

        fix_orders(sorted_result)

        tr_list = [TableRow(**x) for x in sorted_result.to_dict(orient='records')]
        tr_list.sort()
        sorted_result = pd.DataFrame([x.data for x in tr_list])
        if len(sorted_result) == 0:
            return pd.DataFrame(columns=['date', 'sender', 'message-media', 'sources', 'unknown_order'])

        for col in [c for c in sorted_result.columns if c.startswith('original_order')]:
            data_holder = col[14:].strip("_")
            sorted_result['in_order_' + data_holder] = \
                (
                        (sorted_result[col] - sorted_result[col].shift(1)).fillna(0) >= 0
                ) & (
                        (sorted_result[col] - sorted_result[col].shift(-1)).fillna(0) <= 0
                )

        sorted_result['any_out_of_order'] = ~sorted_result[
            [x for x in sorted_result.columns if x.startswith('in_order')]].all(
            axis=1)

        def get_order_number(row):
            filtered = [x[1] for x in zip(
                row[row.index.str.startswith('in_order')],
                row[row.index.str.startswith('original_order')]
            ) if x[0] and not np.isnan(x[1])]
            if not len(filtered):
                filtered = [x[1] for x in zip(
                    row[row.index.str.startswith('in_order')],
                    row[row.index.str.startswith('original_order')]
                ) if not np.isnan(x[1])]
            return mode(filtered)

        def get_sources(row):
            return ','.join([x[15:] for x in row.index if x.startswith('original_order') and not np.isnan(row[x])])

        sorted_result['order'] = sorted_result.apply(get_order_number, axis=1)

        sorted_result['no_unique_order'] = False
        unknowns = sorted_result.order.value_counts()
        sorted_result.loc[sorted_result.order.isin(unknowns[unknowns > 1].index), 'no_unique_order'] = True

        sorted_result['unknown_order'] = sorted_result[['any_out_of_order', 'no_unique_order']].any(axis=1)

        sorted_result['sources'] = sorted_result.apply(get_sources, axis=1)

        merged_df = sorted_result[
            ['date', 'sender', 'message-media', 'sources', 'unknown_order']
        ].merge(self.concat_df[['message', 'media', 'message-media', "media_to", 'hash']], on='message-media',
                how='left') \
            .drop_duplicates(subset=['date', 'sender', 'message-media', 'sources', 'unknown_order'])[
            ['date', 'sender', 'message', 'media', 'sources', 'unknown_order', "media_to", "hash"]
        ]

        def to_datetime(string):
            try:
                return pd.to_datetime(string, format='%d-%m-%Y')
            except ValueError:
                return pd.to_datetime(string, format='%Y-%m-%d %H:%M:%S')

        self.merged_df = merged_df[(merged_df.date > to_datetime(self.config.get('start_date'))) &
                                   (merged_df.date <= to_datetime(self.config.get('end_date')))]

        self.calculate_statistics()

    def calculate_statistics(self):
        if not len(self.merged_df):
            logging.error('There is no merged result for this conversation')

        def deep_update(dictionary, update):
            for k, v in update.items():
                if isinstance(v, collections.abc.Mapping):
                    dictionary[k] = deep_update(dictionary.get(k, {}), v)
                else:
                    dictionary[k] = v
            return dictionary


        all_df = self.get_original_data()
        all_df = all_df[
            (self.concat_df.sender != '') & ~self.concat_df.sender.isin(
                self.topics.values())]
        all_df['media weggelaten'] = all_df['media'] == '<Media weggelaten>'
        all_df['media'] = ((all_df['media'] != '') & (all_df['media'] != '<Media weggelaten>'))
        all_df['verwijderde berichten'] = all_df['message'].isin(MESSAGES_DELETED)

        # sources_table = self.create_sources_table(all_df)

        overzicht = all_df.groupby("data_holder").agg({
            'verwijderde berichten': 'sum',
            'media': 'sum',
            'media weggelaten': 'sum',
            'date': ['min', 'max']
        }).reset_index()
        overzicht.columns = [' '.join(col).strip() for col in overzicht.columns.values]
        overzicht = overzicht.rename(columns={
            'verwijderde berichten sum': 'num_deleted_messages',
            'media weggelaten sum': 'num_deleted_attachements',
            'media sum': 'num_attachments',
            'date min': 'first_date',
            'date max': 'last_date'
        })
        selection_dict = {
            v.get('data_holder'):
                {'start_selection': v.get('start_date'), 'end_selection': v.get('end_date')}
            for k, v in
            self.config['files'].items()
        }

        overzicht = overzicht.merge(
            pd.DataFrame.from_dict(selection_dict, orient='index'),
            left_on='data_holder',
            right_index=True
        ).set_index('data_holder')
        overzicht[['first_date', 'last_date']] = overzicht[['first_date', 'last_date']].apply(lambda x: x.dt.strftime('%d-%m-%Y'))
        self.statistics["data_holders"] = deep_update(self.statistics.get("data_holders"), overzicht.to_dict(orient='index'))
        # self.statistics.get("data_holders").update(overzicht.to_dict(orient='index'))
        self.statistics['all'] = {
            'num_chats': len(self.conversation_list),
            'num_messages_before': sum(len(x) for x in self.conversation_list),
            'num_messages_after': len(self.merged_df),
        }

    def to_txt(self, skip_omitted=False):
        if skip_omitted:
            df = self.merged_df[
                ~self.merged_df.message.isin(MESSAGES_DELETED) | ~self.merged_df.message.isin(MESSAGES_OMITTED)]
        else:
            df = self.merged_df
        lines = df.date.dt.strftime('%d-%m-%Y %H:%M').apply(lambda x: x + " - ") + \
                df.sender.astype(str).apply(lambda x: x + ": " if x else x) + \
                df.message
        return "\n".join(lines)

    def write_txt(self, output, skip_omitted=False):
        with open(output, 'w', encoding='utf-8') as output_file:
            output_file.write(self.to_txt(skip_omitted=skip_omitted))

    def run(self, phone_book=None, mapping=None):
        if mapping is None:
            self.author_mapping()
        else:
            self.contact_mapping.update(mapping)

        if phone_book is None:
            phone_book = self.create_phone_book()

        self.apply_mapping(phone_book=phone_book)
        self.deduplicate_attachments()
        self.merge()

    def save(self, conversations_path: str, reports_path: str, output_format='txt',
             filename='output') -> None:

        if isinstance(output_format, str):
            output_format = [x.strip() for x in output_format.strip().split(',')]
        if not isinstance(output_format, list):
            raise ValueError(f"Parameter 'output_format' should be str or list, but is: {type(output_format)}")

        output_conversation_file = os.path.join(conversations_path, filename)
        output_report_file = os.path.join(reports_path, filename)

        if set(output_format).intersection(['xlsx', 'xls']):
            logger.info(f"Creating Excel file for merged chat: {output_report_file}.xlsx")

            self.merged_df.applymap(
                lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x
            ).to_excel(f'{output_report_file}.xlsx', index=False)
        if 'csv' in output_format:
            logger.info(f"Creating CSV file for merged chat: {output_report_file}.csv")
            self.merged_df.to_csv(f'{output_report_file}.csv', index=False)
        if 'txt' in output_format:
            logger.info(f"Creating TXT file for merged chat: {output_conversation_file}.txt")
            self.write_txt(f'{output_conversation_file}.txt')
        if 'pdf' in output_format or 'zip' in output_format:
            temp_media_path = os.path.join(self.config_dir, 'temp')
            self.save_media(temp_media_path)
            with zipfile.ZipFile('bijlagen.zip',
                                 mode="w",
                                 compression=zipfile.ZIP_DEFLATED) as new_zip:
                for path in tqdm(Path(os.path.join(temp_media_path, 'media')).glob('**/*'),
                                 desc='Creating temporary zipfile', disable=logger.level > logging.INFO):
                    new_zip.write(path, arcname=path.name)
        if 'pdf' in output_format or 'pdf_no_media' in output_format:
            pdf_path = f'{output_conversation_file}.pdf'
            pdf = PDF()
            pdf.print_chapter(self.to_txt())
            pdf.output(pdf_path)

            reader = PdfReader(pdf_path)
            pages = reader.pages

            pdfwriter = PdfWriter()
            for page in pages:
                pdfwriter.add_page(page)

            if 'pdf_no_media' in output_format:
                with open(f'{output_conversation_file}_no_media.pdf', 'wb') as pdf_file:
                    pdfwriter.write(pdf_file)

            if 'pdf' in output_format:
                with open("bijlagen.zip", 'rb') as attachment:
                    pdfwriter.add_attachment("bijlagen.zip", attachment.read())

                with open(pdf_path, 'wb') as pdf_file:
                    pdfwriter.write(pdf_file)
        if 'zip' in output_format:
            shutil.copyfile('bijlagen.zip', os.path.join(conversations_path, 'bijlagen.zip'))
        if 'pdf' in output_format or 'pdf_no_media' in output_format:
            shutil.rmtree(temp_media_path)
            os.remove("bijlagen.zip")

    def save_media(self, output):
        logger.info(f"Saving media")
        media_path = os.path.join(output, 'media')
        if not os.path.exists(media_path):
            os.makedirs(media_path)

        mask = ((self.merged_df.media != "") & (self.merged_df.media != "<Media weggelaten>"))
        zips = defaultdict(list)
        dict_exceptions = {}

        for i, row in tqdm(self.merged_df[mask].iterrows(), desc="Copying media files",
                           disable=logger.level > logging.INFO):
            if row.media and os.path.isfile(row.media):
                if row.media_to != "":
                    shutil.copyfile(row.media, os.path.join(media_path, os.path.basename(row.media_to)))
                    # print the warning maar 1x
                    if "1" in row.media_to.split("_")[-1]:
                        logger.warning(
                            f'There were two attachments with the same name: "{row.media}" but a different hash.'
                            f' A postfix was added to the filename.')
                else:
                    shutil.copyfile(row.media, os.path.join(media_path, os.path.basename(row.media)))
            elif '.zip' in row.media:
                if row.media_to != "":
                    zipfile_path, _, file = row.media_to.partition('.zip')
                    dict_exceptions[row.media_to.split("\\")[-1]] = row.media.split("\\")[-1]
                    # print the warning maar 1x
                    if "1" in row.media_to.split("_")[-1]:
                        logger.warning(
                            f'There were two attachments with the same name: "{row.media}" but a different hash.'
                            f' A postfix was added to the filename.')
                elif '.zip.zip' in row.media:
                    zipfile_path, _, file = row.media.partition('.zip.zip')
                    # zipfile_path = zipfile_path+".zip"
                else:
                    zipfile_path, _, file = row.media.partition('.zip')
                zipfile_path += '.zip'
                zips[zipfile_path].append(file[1:])
            else:
                logger.error(f'Could not find \'{row.media}\'.')
        if zips:
            for zipfile_path, files in zips.items():
                try:
                    with zipfile.ZipFile(zipfile_path, 'r') as zf:
                        for media_file in files:
                            new_file_path = os.path.join(media_path, os.path.basename(media_file))
                            try:
                                if media_file in dict_exceptions.keys():
                                    with zf.open(dict_exceptions[media_file], 'r') as mf, open(new_file_path,
                                                                                               'wb') as nf:
                                        shutil.copyfileobj(mf, nf)
                                else:
                                    with zf.open(media_file, 'r') as mf, open(new_file_path, 'wb') as nf:
                                        shutil.copyfileobj(mf, nf)

                            except KeyError:
                                if media_file.endswith(".vcf"):
                                    list_files = zf.namelist()
                                    matches = []
                                    for list_file in filter(lambda x: media_file[:-4] in x, list_files):
                                        matches.append(list_file)
                                    if len(matches) == 1:
                                        logger.warning(f'\'{list_file}\' was copied to the conversation,'
                                                       f'\'{media_file}\', is referenced in the export but was not found'
                                                       f' among the attachments.')
                                        media_file = list_file
                                        with zf.open(matches[0], 'r') as mf, open(new_file_path, 'wb') as nf:
                                            shutil.copyfileobj(mf, nf)
                                    elif len(matches) > 1:
                                        logger.warning(f'There are multiple vcf files that contain '
                                                       f'\'{media_file[:-4]}\'.Take a look at these files.')
                                    else:
                                        logger.warning(f'No matches were found for the file \'{media_file}\'')
                                else:
                                    logger.error(f'Something went wrong moving \'{media_file}\' in \'{zipfile_path}\'.')

                except FileNotFoundError:
                    logger.error(f'Could not find \'{zipfile_path}\'.')
                except PermissionError:
                    logger.error(f'No permission to open \'{zipfile_path}\'.')

    def __len__(self):
        return len(self.merged_df)

    def __str__(self):
        return f'WhatsAppMerger with {len(self.conversation_list)} parsers'

    def calculate_anonymized_chat_participants(self):
        data_holder = {x: f'Functionaris {i + 1}' for i, x in enumerate(self.concat_df.data_holder.unique())}
        self.chat_participants = {}
        i = 1
        for chat_participant in self.concat_df.sender.unique():
            if chat_participant not in data_holder:
                self.chat_participants[chat_participant] = f'Deelnemer {i}'
                i += 1
        self.chat_participants.update(data_holder)

    def get_original_data(self, gdpr_proof=False):
        if not gdpr_proof:
            return self.concat_df.copy()

        logger.info('Creating gdpr proof concat_df')
        if not self.chat_participants:
            self.calculate_anonymized_chat_participants()
        return self._anonymize_dataframe(self.concat_df)
        # return self.concat_df.replace({'data_holder': self.chat_participants, 'sender': self.chat_participants})

    def get_merged_data(self, gdpr_proof=False):
        if not len(self.merged_df):
            logging.error('There is no merged result for this conversation')

        if not gdpr_proof:
            return self.merged_df.copy()

        return self._anonymize_dataframe(self.merged_df)

    def _anonymize_dataframe(self, df):
        logger.info('Creating gdpr proof merged_df')
        if not self.chat_participants:
            self.calculate_anonymized_chat_participants()

        file_locations = {
            re.compile(str(Path(k)).replace('\\', r'\\')): self.chat_participants.get(v.get('data_holder'), 'Onbekend')
            for k, v in self.config['files'].items()
        }
        if 'media' in df and len(df.loc[df.media != '', 'media']) != 0:
            pattern = re.compile(r'\\(?:.(?!\\))+(\.[a-z0-9]+)$')
            df.loc[df.media != '', 'media'] = df.loc[df.media != ''].reset_index().apply(
                lambda x: pd.Series(
                    [pattern.sub(f'\\\\bijlage_{x.name}\\1', x.media), x['index']]
                ), axis=1
            ).set_index(1)[0]
            df.media.fillna('', inplace=True)

        return df.replace(
            {'data_holder': self.chat_participants, 'sender': self.chat_participants}
        ).replace(
            {'media': file_locations}, regex=True
        ).replace(
            {'media': {r'^E:.*\\': r'Ander gesprek\\'}}, regex=True
        )

    def generate_conversation_report(self, *args, **kwargs):
        verantwoordings_document = VerantwoordingsDocument(self)
        verantwoordings_document.generate_conversation_report(*args, **kwargs)


class VerantwoordingsDocument:
    def __init__(self, whatsapp_merger: WhatsAppMerger):
        self.whatsapp_merger: WhatsAppMerger = whatsapp_merger

    def plot_gantt(self, df, fname='gantt'):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df['selection'] = (df['eind selectie'] - df['start selectie']).dt.days
            df['post_selection'] = (df['laatste datum'] - df['eind selectie']).dt.days
            df['pre_selection'] = (df['start selectie'] - df['eerste datum']).dt.days
            start_date = start_date if isinstance((start_date := self.whatsapp_merger.config.get('start_date')),
                                                  datetime) else datetime.strptime(start_date, '%d-%m-%Y')
            end_date = end_date if isinstance((end_date := self.whatsapp_merger.config.get('end_date')),
                                              datetime) else datetime.strptime(end_date, '%d-%m-%Y')
            create_gantt_plot(df,
                              start_date=start_date,
                              end_date=end_date,
                              fname=fname)

    def generate_conversation_report(self,
                                     file_hashes_df,
                                     hash_files_df,
                                     output_dir='',
                                     filename="verantwoording",
                                     gdpr_proof=False):
        start_time = datetime.now()
        logger.info(f"Generating report...")

        errors = []
        for handler in [x for x in logger.handlers if 'QueuingHandler' in x.__class__.__name__]:
            errors += handler.message_queue.queue

        error_html = ''
        if errors:
            error_html = r'''<section>
            <h2>Errors</h2>
            Tijdens het verwerken van de configuratie hebben de volgende errors plaatsgevonden. Kijk of dit aan de 
            configuratie ligt, of dat het terug gekoppeld moet worden aan de ontwikkelaar.
            <ul><li>
            '''
            error_html += '</li><li>'.join(errors) + "</li></ul></section>"
        gdpr_text = ''
        if gdpr_proof:
            gdpr_text = 'Dit is een versie van het verantwoordingsdocument dat geen gevoelige informatie bevat in het ' \
                        'kader van de AVG'

        with ((pd.option_context("max_colwidth", 1000))):
            all_df = self.whatsapp_merger.get_original_data(gdpr_proof=gdpr_proof)
            all_df = all_df[
                (self.whatsapp_merger.concat_df.sender != '') & ~self.whatsapp_merger.concat_df.sender.isin(
                    self.whatsapp_merger.topics.values())]
            all_df['media weggelaten'] = all_df['media'] == '<Media weggelaten>'
            all_df['media'] = ((all_df['media'] != '') & (all_df['media'] != '<Media weggelaten>'))
            all_df['verwijderde berichten'] = all_df['message'].isin(MESSAGES_DELETED)

            sources_table = self.create_sources_table(all_df)

            overzicht = all_df.groupby("data_holder").agg({
                'message': 'count',
                'verwijderde berichten': 'sum',
                'media': 'sum',
                'media weggelaten': 'sum',
                'date': ['min', 'max']
            }) \
                .reset_index()
            overzicht.columns = [' '.join(col).strip() for col in overzicht.columns.values]
            overzicht.rename(columns={
                'message_count': 'aantal berichten',
                'verwijderde berichten sum': 'aantal verwijderde berichten',
                'media sum': 'aantal bijlagen',
                'date min': 'eerste datum',
                'date max': 'laatste datum'
            }, inplace=True)
            if gdpr_proof:
                selection_dict = {
                    self.whatsapp_merger.chat_participants.get(v.get('data_holder')):
                        {'start selectie': v.get('start_date'), 'eind selectie': v.get('end_date')}
                    for k, v in
                    self.whatsapp_merger.config['files'].items()
                }
            else:
                selection_dict = {
                    v.get('data_holder'):
                        {'start selectie': v.get('start_date'), 'eind selectie': v.get('end_date')}
                    for k, v in
                    self.whatsapp_merger.config['files'].items()
                }

            overzicht = overzicht.merge(
                pd.DataFrame.from_dict(selection_dict, orient='index'),
                left_on='data_holder',
                right_index=True
            )
            cols = ['start selectie', 'eind selectie']
            overzicht[cols] = overzicht[cols].apply(pd.to_datetime, errors='coerce', dayfirst=True)
            cols = ['data_holder', 'eerste datum', 'start selectie', 'eind selectie', 'laatste datum']
            chat_periode = overzicht[cols]
            if any(chat_periode['eind selectie'].isna()):
                chat_periode.loc[chat_periode['eind selectie'].isna(), 'eind selectie'] = \
                    chat_periode[chat_periode['eind selectie'].isna()]['laatste datum']
            chat_periode = chat_periode.assign(**{
                col: chat_periode[col].apply(lambda x: x.strftime('%d-%m-%Y'))
                for col in cols[1:]
            }).rename(columns={'data_holder': 'Datahouder'})
            chat_periode = chat_periode.style.hide(axis="index").to_html()

            if gdpr_proof:
                chat_map = '\n\n\\noindent Deze informatie is niet beschikbaar in het geanonimiseerde verantwoordingsdocument.'
            else:
                chat_map = all_df[['data_holder', 'file']].drop_duplicates().sort_values('data_holder').rename(
                    columns={
                        'data_holder': 'Datahouder'
                    }
                )

                chat_map = chat_map.style.hide(
                    axis="index").to_html()

        bijlagen_html_list, n_bijlagen_merged, n_bijlagen_orig = self.create_attachement_info(file_hashes_df,
                                                                                              hash_files_df,
                                                                                              gdpr_proof=gdpr_proof)

        try:
            self.plot_gantt(df=overzicht)
        except Exception as e:
            logger.error('something went wrong creating the gantt chart', exc_info=e)
        conversation_name = self.whatsapp_merger.get_conversation_name(gdpr_proof=gdpr_proof).replace("_", "\_")
        variable_text = dict(
            datum=f'{datetime.now():%d-%m-%Y}',
            conversation_name=conversation_name,
            error_html=error_html,
            gdpr=gdpr_text,
            n_conversations=self.whatsapp_merger.statistics.get('all').get('num_chats'),
            orig_len=self.whatsapp_merger.statistics.get('all').get('num_messages_before'),
            merged_len=self.whatsapp_merger.statistics.get('all').get('num_messages_before'),
            reduction=f"{(1 - (self.whatsapp_merger.statistics.get('all').get('num_messages_before') / self.whatsapp_merger.statistics.get('all').get('num_messages_before'))):.2%}",
            chat_map=chat_map,
            chat_periode=chat_periode,
            sources_table=sources_table,
            bijlagen_html_list=bijlagen_html_list,
            n_bijlagen_orig=n_bijlagen_orig,
            n_bijlagen_date_range=str(
                pd.DataFrame(self.whatsapp_merger.statistics['data_holders']).T.n_media_after.sum()),
            n_bijlagen_merged=n_bijlagen_merged,
        )

        pdf = PDF(title=conversation_name)
        pdf.add_page()
        pdf.set_font("Times", size=12)

        tag_styles = {
            "h1": FontFace(color="#000", size_pt=28),
            "h2": FontFace(color="#000", size_pt=24),
        }
        pdf.write_html(
            (VERANTWOORDING_TEMPLATE_PART_1 % variable_text).encode('ascii', 'ignore').decode('ascii'),
            tag_styles=tag_styles
        )
        pdf.add_page()
        pdf.write_html(
            (VERANTWOORDING_TEMPLATE_PART_2 % variable_text).encode('ascii', 'ignore').decode('ascii'),
            tag_styles=tag_styles
        )
        pdf.add_page()
        pdf.write_html(
            (VERANTWOORDING_TEMPLATE_PART_3 % variable_text).encode('ascii', 'ignore').decode('ascii'),
            tag_styles=tag_styles
        )

        pdf.output(os.path.join(output_dir, f"{filename}.pdf"))

        os.remove("gantt.png")
        logging.info(f"generated report in {(datetime.now() - start_time).__str__()}")

    def create_attachement_info(self, file_hashes_df, hash_files_df, gdpr_proof=False):
        if len(self.whatsapp_merger.merged_df) < 1:
            return '', 0, 0
        files = self.whatsapp_merger.get_merged_data(gdpr_proof=gdpr_proof)[
            (self.whatsapp_merger.merged_df['media'] != '') &
            (self.whatsapp_merger.merged_df['media'] != '<Media weggelaten>')
            ][['date', 'media']]
        files['media'] = files.media.apply(Path)
        files['date'] = files.date.apply(lambda x: x.date())
        files['fname'] = files.media.apply(lambda x: Path(x).name)

        if gdpr_proof:
            bijlagen_html_list, n_bijlagen_merged, n_bijlagen_orig = self._file_list_gdpr_proof(files)

        else:
            bijlagen_html_list, n_bijlagen_merged, n_bijlagen_orig = self._file_list_normal(file_hashes_df, files,
                                                                                            hash_files_df)
        return bijlagen_html_list, n_bijlagen_merged, n_bijlagen_orig

    def _file_list_gdpr_proof(self, files):
        if len(files):
            bijlagen_html_list = ""
            for date, date_df in files.groupby('date'):
                bijlagen_html_list += r"<li>" + str(date) + "</li>"
                file_list = ""
                for file, file_df in date_df.groupby('fname'):
                    file_list += r"<li>" + str(file) + "</li>"
                file_list = f"""<ul>
                                            {file_list}
                                        </ul>
                                        """
                bijlagen_html_list += file_list
        else:
            bijlagen_html_list = ''
        n_bijlagen_orig = f'~{self.whatsapp_merger.concat_df.media.astype(bool).sum()}'
        n_bijlagen_merged = f'~{files.media.astype(bool).sum()}'
        return bijlagen_html_list, n_bijlagen_merged, n_bijlagen_orig

    def _file_list_normal(self, file_hashes_df, files, hash_files_df):
        included_directories = self.calculate_included_directories()
        file_table_df = self.create_file_table(included_directories, file_hashes_df, hash_files_df)
        files = file_table_df.merge(files,
                                    left_on='path_other',
                                    right_on='media',
                                    how='left')
        if len(files):
            bijlagen_html_list = ""
            for date, date_df in files.groupby('date'):
                bijlagen_html_list += r"<li>" + str(date) + "</li>\n"
                file_list = ""
                for file, file_df in date_df.groupby('fname'):
                    file_list += r"<li>" + str(file) + "</li>\n"
                    mapped_files = ""
                    for _, (file_path, chosen_location, external_location) in file_df[
                        ['path_self', 'chosen_location', 'external_location']
                    ].iterrows():
                        list_item_path = str(file_path)
                        if chosen_location:
                            list_item_path = r"<b>" + list_item_path + "</b>"
                        if external_location:
                            list_item_path = r"<em>" + list_item_path + "</em>"
                        mapped_files += r"<li>" + list_item_path + "</li>\n"

                    mapped_files = f"""<ul>
                                {mapped_files}
                            </ul>
                            """
                    file_list += mapped_files
                file_list = f"""<ul>
                            {file_list}
                        </ul>
                        """
                bijlagen_html_list += file_list

            bijlagen_html_list = f"""<ul>
                        {bijlagen_html_list}
                    </ul>
                    """
            n_bijlagen_orig = len(files['path_self'][~files.external_location].unique())
            n_bijlagen_merged = files[files.chosen_location].media.count()
        else:
            bijlagen_html_list = 'Er zijn geen bijlagen in deze conversatie'
            n_bijlagen_orig = 0
            n_bijlagen_merged = 0
        return bijlagen_html_list, n_bijlagen_merged, n_bijlagen_orig

    def create_sources_table(self, all_df):
        sources_table = all_df.groupby("data_holder").agg(
            {'message': 'count',
             'verwijderde berichten': 'sum',
             'media': 'sum',
             'media weggelaten': 'sum',
             }
        ).reset_index().rename(columns={'data_holder': 'Datahouder',
                                        'message': 'aantal berichten',
                                        'media': 'aantal media'})
        sources_table = (sources_table.style.hide(axis="index").to_html())

        return sources_table

    def create_file_table(self, included_directories, file_hashes_df, hash_files_df) -> pd.DataFrame:
        logger.debug('creating file table')

        in_directory = file_hashes_df[
            file_hashes_df.parent.isin(included_directories)
        ]
        in_directory = file_hashes_df[file_hashes_df.hash.isin(in_directory.hash)]
        in_directory = in_directory.merge(
            hash_files_df,
            on='hash',
            suffixes=('_self', '_other')
        )
        in_directory['path_other'] = in_directory.path_other.apply(Path)
        in_directory['chosen_location'] = in_directory.path_self == in_directory.path_other
        in_directory['external_location'] = in_directory.path_self.apply(
            lambda x: Path(x).parent not in included_directories)
        return in_directory

    def create_file_table_old(self, included_directories, input_dir) -> pd.DataFrame:
        logger.debug('creating file table')
        file_table = {}
        for file, value in self.whatsapp_merger.file_hashes.items():
            if not file.endswith('txt'):
                other_path = self.whatsapp_merger.hash_file.get(value, file)
                other_path_parent = Path(other_path).parent
                dirs = {Path(file).parent, other_path_parent}
                if '.zip' in other_path:
                    dirs.add(other_path_parent.parent)
                if set(dirs).intersection(included_directories):
                    file_table[file.replace(input_dir, '')] = other_path
        file_table_df = pd.DataFrame(
            file_table.items(),
            columns=['original_location', 'chosen_location']
        )
        return file_table_df

    def calculate_included_directories(self):
        included_directories = set()
        for chat_path, settings in self.whatsapp_merger.config.get('files', {}).items():
            if settings['load']:
                if chat_path.endswith('.zip'):
                    included_directories.add(Path(chat_path))
                else:
                    included_directories.add(Path(chat_path).parent)
        return included_directories
