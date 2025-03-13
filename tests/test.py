import datetime

import pandas as pd
import pytest

from Mapping import Mapping
from WhatsAppMerger import WhatsAppMerger, ContactMapping, TableRow
from Parser import Parser, AndroidParser, IphoneParser
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


@pytest.fixture
def simple_conversation_list():
    conversation_df_list = [
        pd.DataFrame(
            [
                {
                    'date': pd.Timestamp('2022-05-02 16:21:00'),
                    'message': 'Announcement',
                    'file': 'fake/path.txt',
                    'original_order': 0,
                    'sender': '', 'media': '', 'data_holder': 'chat01'
                },
                {
                    'date': pd.Timestamp('2022-05-02 16:22:00'),
                    'message': 'dit is een bericht',
                    'file': 'fake/path.txt',
                    'original_order': 1,
                    'sender': 'persoon1', 'media': '', 'data_holder': 'chat01'
                },
                {
                    'date': pd.Timestamp('2022-05-02 16:23:00'),
                    'message': 'een reactie op persoon1\nover twee regels',
                    'file': 'fake/path.txt',
                    'original_order': 2,
                    'sender': 'persoon2', 'media': '', 'data_holder': 'chat01'},
                {
                    'date': pd.Timestamp('2022-05-02 16:24:00'),
                    'message': 'een reactie op persoon1\nover twee regels: met dubbele punt',
                    'file': 'fake/path.txt',
                    'original_order': 3,
                    'sender': 'persoon2', 'media': '', 'data_holder': 'chat01'
                }
            ]
        ),
        pd.DataFrame(
            [{'date': pd.Timestamp('2022-05-02 16:21:00'), 'message': 'Announcement', 'file': 'fake/path.txt',
              'original_order': 0, 'sender': '', 'media': '', 'data_holder': 'chat02'},
             {'date': pd.Timestamp('2022-05-02 16:22:00'), 'message': 'dit is een bericht', 'file': 'fake/path.txt',
              'original_order': 1, 'sender': 'c1', 'media': '', 'data_holder': 'chat02'},
             {'date': pd.Timestamp('2022-05-02 16:23:00'), 'message': 'een reactie op persoon1\nover twee regels',
              'file': 'fake/path.txt', 'original_order': 2, 'sender': "+123456789", 'media': '',
              'data_holder': 'chat02'},
             {'date': pd.Timestamp('2022-05-02 16:24:00'),
              'message': 'een reactie op persoon1\nover twee regels: met dubbele punt',
              'file': 'fake/path.txt', 'original_order': 3, 'sender': "+123456789",
              'media': '', 'data_holder': 'chat02'},
             {'date': pd.Timestamp('2022-05-02 16:25:00'),
              'message': 'Extra bericht in chat 2',
              'file': 'fake/path.txt', 'original_order': 4, 'sender': "+123456789",
              'media': '', 'data_holder': 'chat02'}]
        )
    ]
    conversation_list = []
    for df, data_holder in zip(conversation_df_list, ['chat01', 'chat02']):
        p = Parser()
        df['hash'] = ''
        df['media_to'] = ''
        p.message_list = df.to_dict('records')
        p.data_holder = data_holder
        conversation_list.append(p)
    return conversation_list


@pytest.fixture
def conversation_list_with_announcements(simple_conversation_list):
    addition1 = pd.DataFrame(
        [
            {
                'date': pd.Timestamp('2022-05-02 16:25:00'),
                'message': 'U hebt persoon3 toegevoegd',
                'file': 'fake/path.txt',
                'original_order': 0,
                'sender': '', 'media': '', 'data_holder': 'chat01'
            },
        ]
    )
    simple_conversation_list[0] = pd.concat([simple_conversation_list[0], addition1])

    addition2 = pd.DataFrame(
        [
            {
                'date': pd.Timestamp('2022-05-02 16:25:00'),
                'message': '+123456789 heeft +987654321 toegevoegd',
                'file': 'fake/path.txt',
                'original_order': 0,
                'sender': '', 'media': '', 'data_holder': 'chat02'
            },
        ]
    )
    simple_conversation_list[1] = pd.concat([simple_conversation_list[1], addition2])
    return simple_conversation_list


@pytest.fixture
def simple_mapping():
    return {
        'persoon1': {('chat01', 'persoon1'), ('chat02', 'c1')},
        'persoon2': {('chat01', 'persoon2'), ('chat02', '+123456789')}
    }


@pytest.fixture
def simple_author_message_dict():
    return {
        ('chat01', ''): {'announcement'},
        ('chat01', 'persoon1'): {'dit is een bericht'},
        ('chat01', 'persoon2'): {'een reactie op persoon1'},
        ('chat02', ''): {'announcement'},
        ('chat02', 'c1'): {'dit is een bericht'},
        ('chat02', '+123456789'): {'een reactie op persoon1'}
    }


class TestParser:
    def test_is_date_with_datetime_format(self):
        parser = Parser(fname='',
                        datetime_format='%d-%m-%Y %H:%M:%S')
        assert parser.is_date('02-05-2022 16:21:32')
        assert parser.is_date('01-02-2022 00:00:0')
        assert parser.is_date('No date') is None

    def test_is_date_without_datetime_format(self):
        parser = Parser(fname='')
        assert parser.is_date('02-05-2022 16:21:32')
        assert parser.is_date('01-02-2022 00:00:0')
        assert parser.is_date('No date') is None

    def test_get_media(self):
        media_messages = [
            ['IMG-20220502-WA0003.jpg (file attached)\nDit is een foto', 'IMG-20220502-WA0003.jpg'],
            ['IMG-20220502-WA0005.jpg (file attached)', 'IMG-20220502-WA0005.jpg'],
            ['Manifesto.pdf (file attached)\nManifesto.pdf', 'Manifesto.pdf'],
            ['<Media omitted>\nManifesto.pdf', '<Media weggelaten>'],
            ['<Media omitted>', '<Media weggelaten>'],
            ['<Media weggelaten>', '<Media weggelaten>'],
            ['IMG-20220502-WA0001.jpg (bestand bijgevoegd)\nDit is een foto', 'IMG-20220502-WA0001.jpg'],
            ['IMG-20220502-WA0005.jpg (bestand bijgevoegd)', 'IMG-20220502-WA0005.jpg'],
            ['DOC-20220502-WA0006. (bestand bijgevoegd)\nManifesto.pdf', 'DOC-20220502-WA0006.'],
            ['<bijgevoegd: 00000014-PHOTO-2022-05-02-16-17-01_fout.jpg>', '00000014-PHOTO-2022-05-02-16-17-01_fout.jpg'],
            ["Manifesto.pdf • 116 pagina's <bijgevoegd: 00000017-Manifesto.pdf>", '00000017-Manifesto.pdf'],
            ["Dit bericht bevat geen media", "", ]
        ]

        parser = Parser(fname='')
        for message in media_messages:
            assert parser.get_media(message[0]) == message[1]
        assert True

    def test_add_message(self):
        parser = Parser(fname='')
        parser.add_message(message='Message', sender='sender', datetime='datetime')
        assert parser.message_list == [{'message': 'Message', 'sender': 'sender', 'datetime': 'datetime'}]

        parser = Parser(fname='')
        parser.add_message(**{'message': 'Message', 'sender': 'sender', 'datetime': 'datetime'})
        assert parser.message_list == [{'message': 'Message', 'sender': 'sender', 'datetime': 'datetime'}]

        parser = Parser(fname='')
        for i in range(10):
            parser.add_message(message=f'Message{i}', sender=f'sender{i}', datetime=f'datetime{i}')
        assert len(parser.message_list) == 10

    def test_basic_parse_android(self, monkeypatch):
        def mock_get_file_contents(*args, **kwargs):
            messages = """
5/2/22, 16:21 - Announcement
5/2/22, 16:22 - persoon1: dit is een bericht
5/2/22, 16:23 - persoon2: een reactie op persoon1
over twee regels
5/2/22, 16:24 - persoon2: een reactie op persoon1
over twee regels: met dubbele punt
""".strip().split("\n", )
            # add the new_line to each line, as readlines() does this as well.
            return [m + "\n" for m in messages]

        monkeypatch.setattr(Parser, "_get_file_contents", mock_get_file_contents)

        parser = AndroidParser(fname='fake/path.txt', datetime_format='%m/%d/%y, %H:%M', data_holder='chat01')
        parser.parse()
        assert parser.message_list == [
            {'date': datetime.datetime(2022, 5, 2, 16, 21), 'message': 'Announcement', 'file': 'fake/path.txt',
             'original_order': 0},
            {'date': datetime.datetime(2022, 5, 2, 16, 22), 'sender': 'persoon1', 'message': 'dit is een bericht',
             'media': '', 'file': 'fake/path.txt', 'original_order': 1},
            {'date': datetime.datetime(2022, 5, 2, 16, 23), 'sender': 'persoon2',
             'message': 'een reactie op persoon1\nover twee regels', 'media': '', 'file': 'fake/path.txt',
             'original_order': 2},
            {'date': datetime.datetime(2022, 5, 2, 16, 24), 'sender': 'persoon2',
             'message': 'een reactie op persoon1\nover twee regels: met dubbele punt',
             'media': '', 'file': 'fake/path.txt', 'original_order': 3}
        ]

        assert parser.message_list[0]['message'] == "Announcement"
        assert parser.message_list[0]['date'] == datetime.datetime(year=2022, month=5, day=2, hour=16, minute=21)

    def test_basic_parser_iphone(self, monkeypatch):
        def mock_get_file_contents(*args, **kwargs):
            messages = """
[02-05-2022 16:21:32] Announcement
[02-05-2022 16:22:32] c1: dit is een bericht
[02-05-2022 16:23:32] +123456789: een reactie op persoon1
over twee regels
[02-05-2022 16:24:32] +123456789: een reactie op persoon1
over twee regels: met dubbele punt
        """.strip().split("\n", )
            # add the new_line to each line, as readlines() does this as well.
            return [m + "\n" for m in messages]

        monkeypatch.setattr(Parser, "_get_file_contents", mock_get_file_contents)

        parser = IphoneParser(fname='fake/path.txt', datetime_format="%d-%m-%Y %H:%M:%S", data_holder='chat02')
        parser.parse()
        assert parser.message_list == [
            {'date': datetime.datetime(2022, 5, 2, 16, 21), 'message': 'Announcement', 'file': 'fake/path.txt',
             'original_order': 0},
            {'date': datetime.datetime(2022, 5, 2, 16, 22), 'sender': 'c1', 'message': 'dit is een bericht',
             'media': '', 'file': 'fake/path.txt', 'original_order': 1},
            {'date': datetime.datetime(2022, 5, 2, 16, 23), 'sender': "+123456789",
             'message': 'een reactie op persoon1\nover twee regels', 'media': '', 'file': 'fake/path.txt',
             'original_order': 2},
            {'date': datetime.datetime(2022, 5, 2, 16, 24), 'sender': "+123456789",
             'message': 'een reactie op persoon1\nover twee regels: met dubbele punt',
             'media': '', 'file': 'fake/path.txt', 'original_order': 3}
        ]
        assert [m.get('message') for m in parser.message_list] == [
            "Announcement", "dit is een bericht", "een reactie op persoon1\nover twee regels",
            "een reactie op persoon1\nover twee regels: met dubbele punt"
        ]


class TestWhatsAppMerger:
    def test_author_match_score(self, simple_author_message_dict):
        author = ('chat01', 'persoon1')
        scores = Mapping()._label_match_score(author=author,
                                              label_message_dict=simple_author_message_dict)
        assert scores[('chat02', '')] == 0
        assert scores[('chat02', 'c1')] == 1
        assert scores[('chat02', '+123456789')] == 0

    def test_merge_groups(self):
        groups = [
            {1, 2, 3},
            {2, 4},
            {5, 6},
            {7, 8},
            {6, 7}
        ]
        cm = ContactMapping()
        res_groups = cm._merge_groups(groups)
        assert res_groups == [{1, 2, 3, 4}, {5, 6, 7, 8}]

    def test_author_mapping(self, simple_conversation_list, simple_mapping):
        merger = WhatsAppMerger(conversation_list=simple_conversation_list)
        merger.author_mapping()
        for key, value in simple_mapping.items():
            assert merger.contact_mapping.mapping.get(key, key) == value

    def test_reverse_mapping(self, simple_mapping):
        cm = ContactMapping()
        cm.mapping = simple_mapping
        cm._reverse_mapping()
        assert cm.reversed_mapping == {
            ('chat01', 'persoon1'): 'persoon1',
            ('chat01', 'persoon2'): 'persoon2',
            ('chat02', '+123456789'): 'persoon2',
            ('chat02', 'c1'): 'persoon1'
        }

    def test_apply_mapping(self, simple_conversation_list, simple_mapping):
        merger = WhatsAppMerger(conversation_list=simple_conversation_list)
        merger.contact_mapping.mapping = simple_mapping
        merger.contact_mapping._reverse_mapping()
        merger.apply_mapping()

        expected_result = [
            pd.DataFrame(
                [
                    ['chat01', '', 'Announcement', ''],
                    ['chat01', 'persoon1', 'dit is een bericht', ''],
                    ['chat01', 'persoon2', 'een reactie op persoon1\nover twee regels', ''],
                    ['chat01', 'persoon2', 'een reactie op persoon1\nover twee regels: met dubbele punt', '']

                ], columns=['data_holder', 'sender', 'message', 'media']
            ),
            pd.DataFrame(
                [
                    ['chat02', '', 'Announcement', ''],
                    ['chat02', 'persoon1', 'dit is een bericht', ''],
                    ['chat02', 'persoon2', 'een reactie op persoon1\nover twee regels', ''],
                    ['chat02', 'persoon2', 'een reactie op persoon1\nover twee regels: met dubbele punt', ''],
                    ['chat02', 'persoon2', 'Extra bericht in chat 2', '']
                ], columns=['data_holder', 'sender', 'message', 'media']
            )
        ]
        assert (pd.concat(expected_result) == merger.concat_df[
            ['data_holder', 'sender', 'message', 'media']]).all().all()

    def test_merge(self, simple_conversation_list, simple_mapping):
        merger = WhatsAppMerger(
            conversation_list=simple_conversation_list,
            config={
                'start_date': datetime.datetime(2022, 5, 1),
                'end_date': datetime.datetime(2022, 6, 1),
            }
        )

        merger.contact_mapping.mapping = simple_mapping
        merger.contact_mapping._reverse_mapping()
        merger.apply_mapping()

        merger.merge()

        assert list(merger.merged_df.sender.values) == ['', 'persoon1', 'persoon2', 'persoon2', 'persoon2']
        assert list(merger.merged_df.sources.values) == ['chat01,chat02', 'chat01,chat02', 'chat01,chat02', 'chat01,chat02', 'chat02']

class TestTableRow:
    def test_eq_one(self):
        r1 = TableRow(**dict(
            date=1,
            original_order_x=1,
            original_order_y=2
        ))
        assert (r1 == r1)

    def test_le_one(self):
        r1 = TableRow(**dict(
            date=1,
            original_order_x=1,
            original_order_y=2
        ))
        r2 = TableRow(**dict(
            date=2,
            original_order_x=1,
            original_order_y=2
        ))
        assert r1 < r2

    def test_le_two(self):
        r1 = TableRow(**dict(
            date=1,
            original_order_x=1,
            original_order_y=2
        ))
        r2 = TableRow(**dict(
            date=1,
            original_order_x=1,
            original_order_y=3
        ))
        assert r1 <= r2
        assert not r2 < r1

    def test_gt_one(self):
        r1 = TableRow(**dict(
            date=1,
            original_order_x=1,
            original_order_y=2
        ))
        r2 = TableRow(**dict(
            date=1,
            original_order_x=2,
            original_order_y=3
        ))
        assert r2 > r1
        assert not r1 > r2
