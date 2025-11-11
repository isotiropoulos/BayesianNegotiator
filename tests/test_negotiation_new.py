
import unittest
from typing import  Any
from src.bambon.negotiator import Bambon
import json
import pandas as pd
from .utils_test import *


def find(d, op):
    for i in d['permission'][0]['constraint']:
        if i['leftOperand'] == op:
            return f'{i["operator"]} {i["rightOperand"]}'

class TestCatalog(unittest.TestCase):

    def test_init(self):

        records = []

        init_provider, hard_provider, init_consumer, hard_consumer = fetch_data('agreement')
        consumer = Bambon("consumer", 1, init_offer=init_consumer, hard_policy=hard_consumer, max_rounds=60)
        provider = Bambon("provider", 1, init_offer=init_provider, hard_policy=hard_provider, max_rounds=60)

        turn = 'provider'
        if turn == 'provider':
            received = init_consumer
        else:
            received = init_provider

        cnt = 1
        while True:
            # for item in received['permission'][0]['constraint']:
            #     print(turn, cnt, item['leftOperand'], '\n============================')
            #     print(f'Recieved: {item["operator"]} {item["rightOperand"]}')
            #     print(f'PrIni: {find(init_provider, item['leftOperand'])}')
            #     print(f'PrHrd: {find(hard_provider, item['leftOperand'])}')
            #     print(f'CoIni: {find(init_consumer, item['leftOperand'])}')
            #     print(f'CoHrd: {find(hard_consumer, item['leftOperand'])}')
            #     print('\n\n')

            # print(turn, 'received offer', received)
            # print(turn, 'recieved these violating the hard constraints:')
            if turn=='provider':
                stop, replied = provider.observe(received)
                add_block(
                    records, turn, provider.rounds_seen,
                    received['permission'][0]['constraint'],
                    replied['permission'][0]['constraint'],
                )
                received = replied; replied = None

                turn = 'consumer'
            else:

                stop, replied = consumer.observe(received)
                add_block(
                    records, turn, consumer.rounds_seen,
                    received['permission'][0]['constraint'],
                    replied['permission'][0]['constraint'],
                )
                received = replied; replied = None

                turn = 'provider'

            if stop:
                print(f'\n\nAgreement on {provider.rounds_seen, consumer.rounds_seen}\n---------\n', received, '\n', init_provider, '\n',init_consumer)

                for item in received['permission'][0]['constraint']:
                    print('\n\n')
                    print(item['leftOperand'],'\n============================')
                    print(f'Final: {item["operator"]} {item["rightOperand"]}')
                    print(f'PrIni: {find(init_provider, item['leftOperand'])}')
                    print(f'PrHrd: {find(hard_provider, item['leftOperand'])}')
                    print(f'CoIni: {find(init_consumer, item['leftOperand'])}')
                    print(f'CoHrd: {find(hard_consumer, item['leftOperand'])}')


                break

            cnt += 1
