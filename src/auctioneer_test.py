from auctioneer import Auctioneer
import unittest
import numpy as np
import random


class AuctioneerTest(unittest.TestCase):

    def setUp(self):
        # TODO Initialize common variables among all tests
        pass

    def test_constructor_is_initializing_correct_values(self):
        random.seed(0)
        auctioneer = Auctioneer(2, 3, 5, 3, False)

        self.assertEqual(range(2), auctioneer.m_item_types)
        self.assertEqual(3, auctioneer.k_sellers)
        self.assertEqual(5, auctioneer.n_buyers)
        self.assertEqual(3, auctioneer.r_rounds)

        self.assertEqual(100, auctioneer.max_starting_price)
        self.assertEqual(0.1, auctioneer.penalty_factor)

        self.assertEqual([False, False, False, False, False], auctioneer.buyers_already_won)
        self.assertTrue(auctioneer.increase_bidding_factor.all() in range(1, 2))
        # self.assertTrue(auctioneer.decrease_bidding_factor.all() in float(0, 1))

        self.assertEqual(0, auctioneer.market_price.all())
        self.assertEqual(0, auctioneer.buyers_profits.all())
        self.assertEqual(0, auctioneer.sellers_profits.all())

        self.assertEqual(0, len(auctioneer.history))

    def test_real_case_scenario(self):
        starting_prices = [
            [40, 50, 20]
        ]

        auctioneer = Auctioneer(starting_prices=starting_prices, M_types=2, K_sellers=3, N_buyers=5, R_rounds=2,
                                level_comm_flag=False)
        auctioneer.increase_bidding_factor = [2, 3, 4, 5, 6]
        auctioneer.decrease_bidding_factor = [0.6, 0.5, 0.4, 0.3, 0.2]
        auctioneer.sellers_types = [1, 1, 0]
        # auctioneer.bidding_factor = np.array([
        #     # Buyer 0
        #     [
        #
        #         [1, 2, 3],  # Type 0
        #         [4, 5, 6]   # Type 1
        #     ],
        #     # Buyer 1
        #     [
        #         [1.68791717, 1.43217411, 1.1566692],
        #         [1.20532547, 1.05372195, 1.19885528]
        #     ],
        #     # Buyer 2
        #     [
        #         [1.71709178, 1.83604667, 1.4957177],
        #         [1.50015315, 1.77615324, 1.00780864]
        #     ],
        #     # Buyer 3
        #     [
        #         [1.62403167, 1.51698165, 1.74709901],
        #         [1.84536679, 1.29700791, 1.08997174]
        #     ],
        #     # Buyer 4
        #     [
        #         [1.81391097, 1.2531242, 1.01217679],
        #         [1.15969576, 1.55215565, 1.34450197]
        #     ]
        # ])

        auctioneer.start_auction()

        self.assertEqual([], auctioneer.market_price)
