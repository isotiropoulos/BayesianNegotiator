
import unittest
import random
from time import perf_counter
from src.bambon.models.models import NegotiationInfeasible
from src.bambon.negotiator import Bambon
from .utils_test import *


def find(d, op):
    for i in d['permission'][0]['constraint']:
        if i['leftOperand'] == op:
            return f'{i["operator"]} {i["rightOperand"]}'


PROFILES = {
    "balanced": {
        "lr": 0.55,               # faster adaptation → larger early spread
        "inertia": 0.35,          # less attached to own offer early
        "tau_in": 0.45,          # accepts lower overlap early → more diversity
        "tau_out": 0.7,
        "max_set_k": 3,

        "no_agreement_until": 3,
        "hopeless_window": 10,
        "jaccard_snap_threshold": 0.5,

        "max_rounds": 40,         # provider never drags negotiation very long
        "tau_cooling_threshold": 0.7, # 15/40,      # cools a bit earlier → earlier convergence
        "tau_in_cooled": 0.70,    # demands high overlap → pulls runs together
        "tau_out_cooled": 0.65,

        "lr_min": 0.25,
        "lr_max": 1.0,
        "vol_window": 5,
        "max_param_delta": 0.18,  # allow relatively big jumps early
        "hopeless_severity": 0.95,
        "hopeless_patience": 8,
        "hopeless_context": 0.15,
    },
    "aggressive": {
        "lr": 0.35,               # still slower than balanced, but not tiny
        "inertia": 0.85,          # very attached to own offer early
        "tau_in": 0.75,          # demands high overlap early
        "max_set_k": 3,

        "no_agreement_until": 5,
        "hopeless_window": 12,
        "jaccard_snap_threshold": 0.72,

        "max_rounds": 80,         # allows longer negotiations, but not extreme
        "tau_cooling_threshold": 30/80,      # cools clearly later than provider
        "tau_in_cooled": 0.78,   # requires very high overlap → strong snap
        "tau_out_cooled": 0.90,  # requires very high overlap → strong snap

        "lr_min": 0.20,
        "lr_max": 0.8,
        "vol_window": 5,
        "max_param_delta": 0.12,  # a bit more generous than before → some spread
        "hopeless_severity": 0.90,
        "hopeless_patience": 10,
        "hopeless_context": 0.20,
    },
    "exploratory": {
        "lr": 0.85,               # very fast belief adaptation
        "inertia": 0.20,          # weak attachment to own offers
        "tau_in": 0.35,          # accepts weak overlap early
        "tau_out":0.4,
        "max_set_k": 4,

        "no_agreement_until": 2,
        "hopeless_window": 8,
        "jaccard_snap_threshold": None,

        "max_rounds": 50,
        "tau_cooling_threshold": 0.5, #20/50,      # cools early
        "tau_in_cooled": 0.65,
        "tau_out_cooled": 0.6,

        "lr_min": 0.30,
        "lr_max": 1.2,
        "vol_window": 4,
        "max_param_delta": 0.25,  # large exploratory jumps
        "hopeless_severity": 0.90,
        "hopeless_patience": 6,
        "hopeless_context": 0.10,
    }
}


class TestCatalog(unittest.TestCase):

    def test_for_paper(self):
        iterations = 10
        results = []
        init_provider, hard_provider, init_consumer, hard_consumer = fetch_data('full')

        aggressive = PROFILES['balanced']
        balanced = PROFILES['exploratory']
        aggressive_kwargs = {k: v for k, v in aggressive.items() if k != "lr"}
        balanced_kwargs = {k: v for k, v in balanced.items() if k != "lr"}

        for it in range(iterations):
            random.seed()
            provider = Bambon(
                name=f"Provider",
                lr=balanced.get("lr", 1.0),
                init_offer=init_provider,
                hard_policy=hard_provider,
                **balanced_kwargs,
            )

            consumer = Bambon(
                name=f"Consumer",
                lr=aggressive.get("lr", 1.0),
                init_offer=init_consumer,
                hard_policy=hard_consumer,
                **aggressive_kwargs,
            )

            cnt = 1
            accepted = 0

            data = {'data': {issue.operand: [] for issue in provider.beliefs[('permission', 'use')].issue_models},
                    'durations': {'provider': [], 'consumer': []},
                    'rounds': []}

            turn = 'provider'
            if turn == 'provider':
                received = init_consumer
                for item in init_provider['permission'][0]['constraint']:
                    data['data'][item['leftOperand'].replace('odrl:', '')].append(item["rightOperand"])
                for item in init_consumer['permission'][0]['constraint']:
                    data['data'][item['leftOperand'].replace('odrl:', '')].append(item["rightOperand"])
            else:
                received = init_provider
                for item in init_consumer['permission'][0]['constraint']:
                    data['data'][item['leftOperand'].replace('odrl:', '')].append(item["rightOperand"])
                for item in init_provider['permission'][0]['constraint']:
                    data['data'][item['leftOperand'].replace('odrl:', '')].append(item["rightOperand"])

            data['rounds'] = [-1,0]

            while True:
                data['rounds'].append(cnt)
                key = turn
                start = perf_counter()
                try:
                    if turn=='provider':
                        accept, replied = provider.observe(received)
                        received = replied; replied = None
                        turn = 'consumer'
                    else:
                        accept, replied = consumer.observe(received)
                        received = replied; replied = None
                        turn = 'provider'
                except NegotiationInfeasible:
                    break

                end = perf_counter()
                data['durations'][key].append(end-start)
                for item in received['permission'][0]['constraint']:
                    data['data'][item['leftOperand'].replace('odrl:','')].append(item["rightOperand"])

                if accept :
                    accepted += 1
                else:
                    accepted = 0
                if accepted == 2:
                    results.append(data)
                    break
                cnt += 1
        print(len(results)/iterations*100, '%')
        import json
        with open(r'tests\out_test\resnew.json', 'w') as f:
            f.write(json.dumps(results))

    @staticmethod
    def _negotiate(init_provider, hard_provider, init_consumer, hard_consumer,
                   prov_profile, cons_profile):
        """Run a single bilateral negotiation to completion.

        Returns the agreed offer (dict) when both parties signal acceptance on two
        consecutive turns, or None if the negotiation is deemed infeasible.
        """
        import copy
        prov_kwargs = {k: v for k, v in prov_profile.items() if k != "lr"}
        cons_kwargs = {k: v for k, v in cons_profile.items() if k != "lr"}

        provider = Bambon(name="Provider", lr=prov_profile.get("lr", 1.0),
                          init_offer=copy.deepcopy(init_provider), hard_policy=hard_provider,
                          **prov_kwargs)
        consumer = Bambon(name="Consumer", lr=cons_profile.get("lr", 1.0),
                          init_offer=copy.deepcopy(init_consumer), hard_policy=hard_consumer,
                          **cons_kwargs)

        received = copy.deepcopy(init_consumer)
        turn = "provider"
        accepted = 0
        final = None
        while True:
            try:
                if turn == "provider":
                    accept, replied = provider.observe(received)
                    turn = "consumer"
                else:
                    accept, replied = consumer.observe(received)
                    turn = "provider"
                received = replied
            except NegotiationInfeasible:
                return None
            if accept:
                accepted += 1
                final = received
            else:
                accepted = 0
            if accepted == 2:
                return final

    def test_full_multi_rule(self):
        """Permissions, obligations and prohibitions (with multiple actions) all converge."""
        init_provider, hard_provider, init_consumer, hard_consumer = fetch_data('full')
        prov = PROFILES['exploratory']
        cons = PROFILES['balanced']

        iterations = 40
        agreements = 0
        sample = None
        for _ in range(iterations):
            random.seed()
            res = self._negotiate(init_provider, hard_provider, init_consumer, hard_consumer, prov, cons)
            if res:
                agreements += 1
                sample = res

        # The negotiation must conclude in a meaningful fraction of runs (it previously
        # never concluded for multi-action / multi-rule offers).
        self.assertGreater(agreements, 0, "no agreement ever reached for full multi-rule offer")

        # The resulting agreement must carry every negotiated rule type and action.
        self.assertEqual(sample['@type'], 'odrl:Agreement')
        perm_actions = {b['action'] for b in sample.get('permission', [])}
        obl_actions = {b['action'] for b in sample.get('obligation', [])}
        prohib_actions = {b['action'] for b in sample.get('prohibition', [])}
        self.assertEqual(perm_actions, {'use', 'transform'})
        self.assertEqual(obl_actions, {'anonymize'})
        self.assertEqual(prohib_actions, {'redistribute'})
