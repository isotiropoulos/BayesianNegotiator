
import unittest
from typing import  Any
from src.bambonOld.negotiator import Bambon
import json
import pandas as pd
from .utils_test import *

class TestCatalog(unittest.TestCase):

    def test_one_negotiation(self):

        records = []

        init_provider, hard_provider, init_consumer, hard_consumer = fetch_data('agreement')
        print('\nInit Provider', init_provider)
        print('\nInit Consumer', init_consumer)
        print('\nHard Provider', hard_provider)
        print('\nHard Consumer', hard_provider)
        print('\nConsumer starts by sending init consumer as counter offer to provider')
        consumer = Bambon("consumer", 1, init_offer=init_consumer, hard_policy=hard_consumer, max_rounds=10)
        provider = Bambon("provider", 1, init_offer=init_provider, hard_policy=hard_provider, max_rounds=10)

        turn = 'provider'

        if turn == 'provider':
            received = init_consumer
        else:
            received = init_provider


        while True:
            print(turn, 'recieved', received['permission'][0]['constraint'])


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
                break

        save_log_text(records, "./logs/provider.log",actor_filter="provider")
        save_log_text(records, "./logs/consumer.log",actor_filter="consumer")
        save_log_text(records, "./logs/all.log")

    def test_one_negotiation2(self):
        def find(d, op):
            for i in d['permission'][0]['constraint']:
                if i['leftOperand'] == op:
                    return f'{i["operator"]} {i["rightOperand"]}'
        records = []

        init_provider, hard_provider, init_consumer, hard_consumer = fetch_data('agreement')

        provider = Bambon("provider", 1, init_offer=init_provider, hard_policy=hard_provider)
        consumer = Bambon("consumer", 1, init_offer=init_consumer, hard_policy=hard_consumer)

        turn = 'provider'

        if turn == 'provider':
            received = init_consumer
        else:
            received = init_provider

        # print(init_provider['permission'][0]['constraint'])
        # print(init_consumer['permission'][0]['constraint'])
        cnt = 1
        while True:
            # print(turn, received['permission'][0]['constraint'])
            for item in received['permission'][0]['constraint']:
                print(turn, cnt, item['leftOperand'], '\n============================')
                print(f'Recieved: {item["operator"]} {item["rightOperand"]}')
                print(f'PrIni: {find(init_provider, item['leftOperand'])}')
                print(f'PrHrd: {find(hard_provider, item['leftOperand'])}')
                print(f'CoIni: {find(init_consumer, item['leftOperand'])}')
                print(f'CoHrd: {find(hard_consumer, item['leftOperand'])}')
                print('\n\n')

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
                    print(item['leftOperand'],'\n============================')
                    print(f'Final: {item["operator"]} {item["rightOperand"]}')
                    print(f'PrIni: {find(init_provider, item['leftOperand'])}')
                    print(f'PrHrd: {find(hard_provider, item['leftOperand'])}')
                    print(f'CoIni: {find(init_consumer, item['leftOperand'])}')
                    print(f'CoHrd: {find(hard_consumer, item['leftOperand'])}')
                    print('\n\n')

            cnt += 1

    def test_many_negotiation(self):
        import os
        print('\n')

        num = 1000
        for case in os.listdir('./tests/data'):
            if 'case' not in case:
                continue
            print(case,'\n============')
            with open(f'./tests/data/{case}/README.md', 'r') as f:
                lines = f.readlines()
                print(lines[2].replace('**',"")[:-1])
                print(lines[4].replace('**',"")[:-1], '\n------------------\n')

            records = []
            init_provider, hard_provider, init_consumer, hard_consumer = fetch_data(case)

            cnt = 0
            for i in range(num):
                current = {
                    'iteration': i+1,
                    'provider_rounds': None,
                    'consumer_rounds': None,
                    'result': None
                }
                provider = Bambon("provider", 1, init_offer=init_provider, hard_policy=hard_provider)
                consumer = Bambon("consumer", 1, init_offer=init_consumer, hard_policy=hard_consumer)

                turn = 'consumer'

                if turn == 'provider':
                    received = init_consumer
                else:
                    received = init_provider

                try:
                    while True:
                        if turn=='provider':
                            stop, replied = provider.observe(received)
                            received = replied
                            turn = 'consumer'
                        else:
                            stop, replied = consumer.observe(received)
                            received = replied
                            turn = 'provider'

                        if stop:
                            current['provider_rounds'] = provider.rounds_seen
                            current['consumer_rounds'] = consumer.rounds_seen
                            current['result'] = replied
                            break
                except Exception as e:
                    # print(i, e)
                    current['provider_rounds'] = provider.rounds_seen
                    current['consumer_rounds'] = consumer.rounds_seen
                    current['result'] = str(e)
                else:
                    cnt += 1

                records.append(current)
                import time
                # time.sleep(3)

            print(cnt,'/ 1000', round(cnt/1000,5))
            with open(f'{case}.txt', 'w') as f:
                f.write(str(records))
