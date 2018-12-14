import numpy as np
import random
from auction import Auction
from prettytable import PrettyTable
import matplotlib.pyplot as plt


class Auctioneer:

    def __init__(self, penalty_factor, bidding_factor_strategy = [], use_seller = True, starting_prices = [], M_types = 3, K_sellers = 4,
                 N_buyers = 10, R_rounds = 3, level_comm_flag = False, debug = True):
        """
        :param bidding_factor_strategy: array with the bidding factor strategy of each buyer
        :param starting_prices: Debug purposes, starting prices can be forced this way.
        :param M_types: Number of types of items
        :param K_sellers: Number of sellers
        :param N_buyers: Number of buyers
        :param R_rounds: Number of rounds
        :param level_comm_flag: Flag to say if level commitment is allowed or not
        """
        self.debug = debug
        if len(bidding_factor_strategy) == 0:
            # If the strategy is not passed, it is set to default 0
            # bidding_factor_strategy = [np.random.randint(0, 2, 1) for n in range(N_buyers)]
            bidding_factor_strategy = [0 for n in range(N_buyers)]

        self.m_item_types = range(M_types)
        self.k_sellers = K_sellers
        self.n_buyers = N_buyers
        self.r_rounds = R_rounds

        self.max_starting_price = 100
        self.penalty_factor = penalty_factor

        # If level commitment is activated sellers cannot cancel a won auction
        self.level_commitment_activated = level_comm_flag
        self.buyers_already_won = self.initialize_buyers_flag()
        self.auctions_history = []

        # Assign a type of item to each seller randomly
        self.sellers_types = [random.sample(self.m_item_types, 1)[0] for seller in range(self.k_sellers)]

        # Assign second dimension of alpha following the input flag
        if use_seller:
            self.second_dimension = self.k_sellers
        else: self.second_dimension = self.m_item_types

        self.bidding_factor_strategy = bidding_factor_strategy
        self.bidding_factor = self.calculate_bidding_factor()

        self.increase_bidding_factor = np.random.uniform(1, 2, size = self.n_buyers)
        self.decrease_bidding_factor = np.random.uniform(0, 1, size = self.n_buyers)

        self.market_price = np.zeros((self.r_rounds, self.k_sellers))
        self.buyers_profits = np.zeros((self.r_rounds, self.n_buyers))
        self.cumulative_buyers_profits = np.zeros((self.n_buyers, self.r_rounds))
        self.cumulative_sellers_profits = np.zeros((self.k_sellers, self.r_rounds))
        self.sellers_profits = np.zeros((self.r_rounds, self.k_sellers))

        self.starting_prices = self.calculate_starting_prices(starting_prices)
        self.print_alphas()

    def calculate_bid(self, buyer_id, item_type, seller_id, starting_price, auction_round):

        if self.second_dimension == self.k_sellers:
            second_dimension = seller_id
        elif self.second_dimension == self.m_item_types:
            second_dimension = item_type

        bid = self.bidding_factor[buyer_id][second_dimension] * starting_price

        if not self.level_commitment_activated \
                or not self.buyers_already_won[buyer_id]:
            # If the buyer flag is not ON it means the buyer hasn't win an auction in this round yet
            return bid
        auction, seller = self.get_auction_with_winner(buyer_id, auction_round)
        previous_profit, market_price = auction.winner_profit, auction.market_price
        penalty = self.calculate_fee(market_price - previous_profit)

        return max(bid, starting_price + previous_profit + penalty)

    def calculate_bidding_factor(self):
        """
        Bidding factor strategies:
            0 - Depends only on the seller using proposed one by assignment
            1 - Depends only on the kind of item using proposed one by assignment
            2 - Depends on the kind of item, but has a max value to avoid price explosion.
                If alpha bigger than 2, decrease it using decrease factor.
            3 - Depends on the kind of item, but checks the market price to see if previous
                alpha update was helpful or not
        :return: bidding factor
        """
        bidding_factor = []
        for buyer in range(self.n_buyers):

            bidding_factor.append(
                np.random.uniform(1, 2, self.second_dimension)
            )

        return bidding_factor

    def calculate_starting_prices(self, starting_prices):
        if len(starting_prices) > 0:
            return starting_prices

        prices = []
        for seller in range(self.k_sellers):
            prices.append(random.random() * self.max_starting_price)
        return prices

    def calculate_fee(self, price_paid):
        return self.penalty_factor * price_paid

    def choose_item_to_keep(self, auction, market_price, price_to_pay, winner, seller, auction_round):
        previous_auction, previous_seller = self.get_auction_with_winner(winner, auction_round)
        previous_winner_profit = previous_auction.winner_profit
        previous_fee = self.calculate_fee(previous_auction.price_paid)
        new_profit = market_price - price_to_pay
        new_fee = self.calculate_fee(price_to_pay)
        if new_profit - previous_fee > previous_winner_profit - new_fee:
            # It is profitable to keep the new item, pay fee to previous seller
            previous_auction.return_item(previous_fee,
                                         kept_item_profit = new_profit,
                                         kept_item_fee = new_fee,
                                         seller_item_kept = seller,
                                         kept_item_price = price_to_pay)
        else:
            auction.return_item(new_fee,
                                kept_item_profit = previous_winner_profit,
                                kept_item_fee = previous_fee,
                                seller_item_kept = previous_seller,
                                kept_item_price = previous_auction.price_paid)

    def choose_winner(self, bids, market_price):
        # TODO dealing with two people with the same bid as winning bid
        valid_bids = []
        for bid in bids.values():

            if bid > market_price:
                continue

            valid_bids.append(bid)

        if len(valid_bids) == 0:
            valid_bids.append(next(iter(bids.values())))

        valid_bids = sorted(valid_bids, reverse = True)

        winner_id = [key for key in bids.keys() if bids[key] == valid_bids[0]][0]
        try:
            price_to_pay = valid_bids[1]
        except IndexError:
            price_to_pay = valid_bids[0]

        return winner_id, price_to_pay

    def get_alphas(self, seller, item):
        alphas = []
        for buyer in range(self.n_buyers):
            if self.bidding_factor_strategy[buyer] == 0:
                second_dimension = seller
            else:
                second_dimension = item

            alphas.append(self.bidding_factor[buyer][second_dimension])
        return alphas

    def get_auction_with_winner(self, winner, auction_round):
        seller = 0
        for auction in self.auctions_history[auction_round]:
            if winner == auction.winner:
                return auction, seller
            seller += 1
        assert 0 == 1

    def initialize_auction_parameters(self, seller):
        starting_price = self.starting_prices[seller]
        n_buyer_auction = 0
        total_bid = 0
        buyers_bid = {}
        item = self.sellers_types[seller]
        return buyers_bid, item, n_buyer_auction, starting_price, total_bid

    def initialize_buyers_flag(self):
        return [False for buyer in range(self.n_buyers)]

    def print_alphas(self, extra_debug = False):
        if not self.debug and not extra_debug:
            return

        buyer = 0
        alphas_table = PrettyTable()

        if self.second_dimension == self.k_sellers:
            alphas_table.field_names = ["S-0"] + ["S" + str(seller) for seller in range(self.k_sellers)]
        elif self.second_dimension == self.m_item_types:
            alphas_table.field_names = ["S-1"] + ["Type " + str(item_type) for item_type in self.m_item_types]

        for strategy in self.bidding_factor_strategy:
            alphas_table.add_row(["B" + str(buyer)] + ['%.2f' % elem for elem in self.bidding_factor[buyer]])
            str_0 = True
            buyer += 1

        print(alphas_table)

    def print_factors(self, extra_debug = False):
        if not self.debug and not extra_debug:
            return
        initial_table = PrettyTable()
        initial_table.field_names = [""] + ["B" + str(buyer) for buyer in range(self.n_buyers)]
        initial_table.add_row(["Increasing factor"] + ['%.2f' % elem for elem in self.increase_bidding_factor])
        initial_table.add_row(["Decreasing factor"] + ['%.2f' % elem for elem in self.decrease_bidding_factor])
        print(initial_table)

    def print_round(self, round_number, extra_debug = False):
        if not self.debug and not extra_debug:
            return
        print()
        print("Round", round_number, "history")
        seller = 0
        for auction in self.auctions_history[round_number]:
            auction.print_auction(seller)
            seller += 1
        print()
        print("------------------------------------------------------")

    def update_alphas(self, winner, seller, item, bids):

        if self.second_dimension == self.k_sellers:
            second_dimension = seller
        elif self.second_dimension == self.m_item_types:
            second_dimension = item

        new_alphas = []
        for buyer in range(self.n_buyers):

            # Strategy 0 - Depends only on the seller using proposed one by assignment
            if self.bidding_factor_strategy[buyer] == 0:

                if buyer == winner:

                    self.bidding_factor[buyer][second_dimension] *= self.decrease_bidding_factor[buyer]

                elif self.buyers_already_won[buyer] and not self.level_commitment_activated:

                    self.bidding_factor[buyer][second_dimension] = self.bidding_factor[buyer][second_dimension]

                else:

                    self.bidding_factor[buyer][second_dimension] *= self.increase_bidding_factor[buyer]

                new_alphas.append(self.bidding_factor[buyer][second_dimension])

            # Strategy 1 - Depends only on the kind of item using proposed one by assignment
            elif self.bidding_factor_strategy[buyer] == 1:

                if buyer == winner:

                    self.bidding_factor[buyer][second_dimension] *= self.decrease_bidding_factor[buyer]

                elif self.buyers_already_won[buyer] and not self.level_commitment_activated:

                    self.bidding_factor[buyer][second_dimension] = self.bidding_factor[buyer][second_dimension]

                else:

                    self.bidding_factor[buyer][second_dimension] *= self.increase_bidding_factor[buyer]

                new_alphas.append(self.bidding_factor[buyer][second_dimension])

            # Strategy 2 - Depends on the kind of item, but has a max value to avoid price explosion.
            # If alpha bigger than 2, decrease it using decrease factor.
            elif self.bidding_factor_strategy[buyer] == 2:

                # if buyer == winner:

                # Do not update

                if buyer != winner and self.bidding_factor[buyer][second_dimension] < 2:

                    self.bidding_factor[buyer][second_dimension] *= self.increase_bidding_factor[buyer]

                elif self.buyers_already_won[buyer] and not self.level_commitment_activated:

                    self.bidding_factor[buyer][second_dimension] = self.bidding_factor[buyer][second_dimension]

                elif buyer != winner and self.bidding_factor[buyer][second_dimension] > 2:

                    self.bidding_factor[buyer][second_dimension] *= self.decrease_bidding_factor[buyer]

                new_alphas.append(self.bidding_factor[buyer][second_dimension])

            # Strategy 3 - Depends on the kind of item, but checks the market price
            # to see if previous alpha update was helpful or not
            elif self.bidding_factor_strategy[buyer] == 3:

                if buyer == winner:

                    self.bidding_factor[buyer][second_dimension] *= self.decrease_bidding_factor[buyer]

                elif self.buyers_already_won[buyer] and not self.level_commitment_activated:

                    self.bidding_factor[buyer][second_dimension] = self.bidding_factor[buyer][second_dimension]

                else:

                    if bids[buyer] > np.mean(list(bids.values())):

                        self.bidding_factor[buyer][second_dimension] *= self.decrease_bidding_factor[buyer]
                    else:
                        self.bidding_factor[buyer][second_dimension] *= self.increase_bidding_factor[buyer]

                new_alphas.append(self.bidding_factor[buyer][second_dimension])

            # If the bidding factor is less than 1, replace it with 1
            if self.bidding_factor[buyer][second_dimension] < 1:
                self.bidding_factor[buyer][second_dimension] = 1

            # Strategy 4 - Fully random each time
            # to see if previous alpha update was helpful or not

        return new_alphas

    def update_profits(self, auction_round):
        seller = 0
        for auction in self.auctions_history[auction_round]:
            self.buyers_profits[auction_round, auction.winner] += auction.winner_profit
            self.sellers_profits[auction_round, seller] += auction.seller_profit
            seller += 1

        for buyer in range(self.n_buyers):
            self.cumulative_buyers_profits[buyer][auction_round] = self.cumulative_buyers_profits[
                                                                       buyer, auction_round - 1] + self.buyers_profits[
                                                                       auction_round, buyer]
        for seller in range(self.k_sellers):
            self.cumulative_sellers_profits[seller][auction_round] = self.cumulative_sellers_profits[
                                                                         seller, auction_round - 1] + \
                                                                     self.sellers_profits[auction_round, seller]

    def start_auction(self):
        self.print_factors()
        for auction_round in range(self.r_rounds):
            self.buyers_already_won = self.initialize_buyers_flag()
            # if self.level_commitment_activated:
            self.auctions_history.append([])
            for seller in range(self.k_sellers):
                buyers_bid, item, n_buyer_auction, starting_price, total_bid = self.initialize_auction_parameters(
                    seller)
                for buyer in range(self.n_buyers):
                    if self.buyers_already_won[buyer] and not self.level_commitment_activated:
                        continue
                    n_buyer_auction += 1
                    bid = self.calculate_bid(buyer, item, seller, starting_price, auction_round)
                    buyers_bid[buyer] = bid
                    total_bid += bid

                market_price = total_bid / n_buyer_auction
                winner, price_to_pay = self.choose_winner(buyers_bid, market_price)
                auction = self.store_auction_history(winner = winner,
                                                     price_paid = price_to_pay,
                                                     starting_price = starting_price,
                                                     market_price = market_price,
                                                     bid_history = buyers_bid,
                                                     previous_alphas = self.get_alphas(seller, item),
                                                     auction_round = auction_round,
                                                     item_kind = item)

                if self.level_commitment_activated and self.buyers_already_won[winner]:
                    # The buyer already won an auction in this round so he has to choose which one to return
                    self.choose_item_to_keep(auction, market_price, price_to_pay, winner, seller, auction_round)

                self.market_price[auction_round, seller] = market_price
                new_alphas = self.update_alphas(winner, seller, item, buyers_bid)
                auction.set_new_alphas(new_alphas)
                self.buyers_already_won[winner] = True
            self.update_profits(auction_round)
            self.print_round(auction_round)

    def store_auction_history(self, starting_price, market_price, winner, price_paid, bid_history, previous_alphas,
                              auction_round, item_kind):
        auction = Auction(starting_price, market_price, price_paid, winner, bid_history, previous_alphas, item_kind)
        self.auctions_history[auction_round].append(auction)
        return auction

    def plot_statistics(self):
        market_prices = np.zeros((self.r_rounds, self.k_sellers))

        for n, auctions_round in enumerate(self.auctions_history):
            for seller in range(self.k_sellers):
                market_prices[n, seller] = auctions_round[seller].market_price

        # Plot price history
        for seller in range(self.k_sellers):
            plt.plot(market_prices[:, seller], label = "Seller " + str(seller))
        plt.title('Price history across all rounds for each seller')
        plt.ylabel('Price')
        plt.xlabel('Auctions')
        plt.legend()
        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        # Plot seller profits
        plt.figure()
        for seller in range(self.k_sellers):
            plt.plot(self.cumulative_sellers_profits[seller], label = "Seller " + str(seller))
        plt.title('Seller cumulative profits across all auctions')
        plt.ylabel('Seller profits')
        plt.xlabel('Rounds')
        plt.legend()
        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        # Plot Buyers profits
        plt.figure()
        for buyer in range(self.n_buyers):
            plt.plot(self.cumulative_buyers_profits[buyer], label = "Buyer " + str(buyer))
        plt.title('Buyer cumulative profits across all auctions')
        plt.ylabel('Buyer profits')
        plt.xlabel('Rounds')
        plt.legend()
        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        plt.show()


if __name__ == '__main__':
    buyers = 5
    auctioneer = Auctioneer(0.1,
                            bidding_factor_strategy = [2 for n in range(buyers)],
                            M_types = 2,
                            K_sellers = 3,
                            N_buyers = buyers,
                            R_rounds = 50,
                            level_comm_flag = False,
                            debug = True)
    auctioneer.start_auction()
    auctioneer.plot_statistics()
    print("\nBidding factors when the simulation is finished")
    auctioneer.print_alphas()
