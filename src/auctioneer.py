import numpy as np
import random


class Auctioneer:

    def __init__(self, starting_prices=[], M=3, K=4, N=10, R=3, level_comm_flag=False):

        self.m_item_types = range(M)
        self.k_sellers = K
        self.n_buyers = N
        self.r_rounds = R

        self.max_starting_price = 100
        self.penalty_factor = 0.1

        # If level commitment is activated sellers cannot cancel a won auction
        self.level_commitment_activated = level_comm_flag
        self.buyers_already_won = self.initialize_buyers_flag()
        self.buyers_history = {}
        if level_comm_flag:
            self.buyers_history = self.initialize_buyers_history()
        self.sellers_types = [random.sample(self.m_item_types, 1)[0] for seller in range(self.k_sellers)]

        self.bidding_factor = np.random.uniform(1, 2, size=(self.n_buyers, len(self.m_item_types), self.k_sellers))

        self.increase_bidding_factor = np.random.uniform(1, 2, size=self.n_buyers)
        self.decrease_bidding_factor = np.random.uniform(0, 1, size=self.n_buyers)

        self.market_price = np.zeros((self.r_rounds, self.k_sellers))
        self.buyers_profits = np.zeros((self.r_rounds, self.n_buyers))
        self.sellers_profits = np.zeros((self.r_rounds, self.k_sellers))

        self.history = {}

        self.starting_prices = starting_prices

    def start_auction(self):
        # TODO fill market price, buyers and sellers profit matrix
        for auction_round in range(self.r_rounds):
            self.buyers_already_won = self.initialize_buyers_flag()
            if self.level_commitment_activated:
                self.buyers_history = self.initialize_buyers_history()

            for seller in range(self.k_sellers):
                buyers_bid, item, n_buyer_auction, starting_price, total_bid = self.initialize_auction_parameters(auction_round, seller)
                for buyer in range(self.n_buyers):
                    if self.buyers_already_won[buyer] and not self.level_commitment_activated:
                        continue
                    n_buyer_auction += 1
                    bid = self.calculate_bid(buyer, item, seller, starting_price)
                    buyers_bid[buyer] = bid
                    total_bid += bid

                market_price = total_bid / n_buyer_auction
                winner, price_to_pay = self.choose_winner(buyers_bid, market_price)

                self.buyers_already_won[winner] = True

                if self.level_commitment_activated:
                    self.store_buyer_history(buyer=winner, profit=(market_price - price_to_pay),
                                             price_paid=price_to_pay)

                self.update_alphas(winner, seller, item)
                self.market_price[auction_round, seller] = market_price
                self.buyers_profits[auction_round, winner] += market_price - price_to_pay
                self.sellers_profits[auction_round, seller] += price_to_pay
                # self.history[auction_round] = {seller, [winner, price]}

    def initialize_auction_parameters(self, auction_round, seller):
        try:
            starting_price = self.starting_prices[auction_round][seller]
        except IndexError:
            starting_price = self.calculate_starting_price()
        n_buyer_auction = 0
        total_bid = 0
        buyers_bid = {}
        item = self.sellers_types[seller]
        return buyers_bid, item, n_buyer_auction, starting_price, total_bid

    def calculate_starting_price(self):
        return random.random() * self.max_starting_price

    def calculate_bid(self, buyer_id, item_type, seller_id, starting_price):
        bid = self.bidding_factor[buyer_id, item_type, seller_id] * starting_price
        if not self.level_commitment_activated \
                or not self.buyers_already_won[buyer_id]:
            # If the buyer flag is not ON it means the buyer hasn't win an auction in this round yet
            return bid

        previous_profit, penalty = self.get_information_buyers_history(buyer_id)
        return max(bid, starting_price + previous_profit + penalty)

    # def update_alpha(self, winner_id, type, seller_id):

    def choose_winner(self, bids, market_price):
        # TODO dealing with two people with the same bid as winning bid
        valid_bids = []
        for bid in bids.values():

            if bid > market_price:
                continue

            valid_bids.append(bid)

        valid_bids = sorted(valid_bids, reverse=True)

        winner_id = [key for key in bids.keys() if bids[key] == valid_bids[0]][0]
        try:
            price_to_pay = valid_bids[1]
        except IndexError:
            price_to_pay = valid_bids[0]

        return winner_id, price_to_pay

    def update_alphas(self, winner, seller, item):
        for buyer in range(self.n_buyers):
            if buyer == winner:
                self.bidding_factor[buyer, item, seller] *= self.decrease_bidding_factor[buyer]
            else:
                self.bidding_factor[buyer, item, seller] *= self.increase_bidding_factor[buyer]

    def initialize_buyers_flag(self):
        return [False for buyer in range(self.n_buyers)]

    def initialize_buyers_history(self):
        """
        The history of the buyers saves the information needed to calculate the bid when level commitment is allowed.
        This informations is:
            - profit made, stored in position 0
            - penalty to pay, stored in position 1
        """
        return [[] for buyer in range(self.n_buyers)]

    def get_information_buyers_history(self, buyer):
        profit_made = self.buyers_history[buyer][0]
        penalty = self.buyers_history[buyer][1]
        return profit_made, penalty

    def store_buyer_history(self, buyer, profit, price_paid):
        self.buyers_history[buyer] = [profit, self.penalty_factor * price_paid]

    def print_outcome(self):
        # TODO Implement print_outcome
        print("The market price history is:")
        print(self.market_price)
        print("The buyers profits are:")
        print(self.buyers_profits)
        print("The sellers profits are:")
        print(self.sellers_profits)


if __name__ == '__main__':
    auctioneer = Auctioneer(level_comm_flag=False)
    auctioneer.start_auction()
    auctioneer.print_outcome()
