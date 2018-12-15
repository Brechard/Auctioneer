import random

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

from auction import Auction


class Auctioneer:

    def __init__(self, penalty_factor=0.1, bidding_factor_strategy=[], use_seller=True, starting_prices=[], M_types=3,
                 K_sellers=4, N_buyers=10, R_rounds=3, level_comm_flag=False, debug=True, universal_maximum_price=100):
        """
        :param penalty_factor: Multiplier for fee calculationz
        :param bidding_factor_strategy: Array with the bidding factor strategy of each buyer
        :param use_seller: Flag to use seller or item as second dimension for alpha
        :param starting_prices: Debug purposes, starting prices can be forced this way.
        :param M_types: Number of types of items
        :param K_sellers: Number of sellers
        :param N_buyers: Number of buyers
        :param R_rounds: Number of rounds
        :param level_comm_flag: Flag to say if level commitment is allowed or not
        :param debug: Flag for debug prints
        :param universal_maximum_price: Max initial starting price
        """
        self.debug = debug
        if len(bidding_factor_strategy) == 0:
            # If the strategy is not passed, it is set to default 0
            # bidding_factor_strategy = [np.random.randint(2, 4, 1) for n in range(N_buyers)]
            bidding_factor_strategy = [2 for n in range(N_buyers)]
        else:
            for s in bidding_factor_strategy:
                if s not in [1, 2, 3, 4]:
                    print("Error in the strategy input")
                    return

        self.m_item_types = range(M_types)
        self.k_sellers = K_sellers
        self.n_buyers = N_buyers
        self.r_rounds = R_rounds

        self.max_starting_price = universal_maximum_price
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
        else:
            self.second_dimension = M_types

        self.bidding_factor_strategy = bidding_factor_strategy
        self.bidding_factor = self.calculate_bidding_factor()

        self.increase_bidding_factor = np.random.uniform(1, 1.5, size=self.n_buyers)
        self.decrease_bidding_factor = np.random.uniform(0.3, 0.8, size=self.n_buyers)

        # Ceiling threshold for strategy 2
        self.ceiling = 2

        self.market_price = np.zeros((self.r_rounds, self.k_sellers))
        self.buyers_profits = np.zeros((self.r_rounds, self.n_buyers))
        self.cumulative_buyers_profits = np.zeros((self.n_buyers, self.r_rounds))
        self.cumulative_sellers_profits = np.zeros((self.k_sellers, self.r_rounds))
        self.sellers_profits = np.zeros((self.r_rounds, self.k_sellers))

        self.starting_prices = self.calculate_starting_prices(starting_prices)
        self.print_alphas()

        self.times_items_returned = 0
        self.times_bad_trade = 0

    def calculate_bid(self, buyer_id, item_type, seller_id, starting_price, auction_round):
        """
        Calculate the bid for a specific buyer considering his bidding strategy
        :param buyer_id: id of the buyer to calculate the bid from
        :param item_type: kind of item that is being auction
        :param seller_id: id of the seller that is auctioning
        :param starting_price: starting price of the item that is being auctioned
        :param auction_round: round of the auction
        :return: bid of the buyer
        """

        second_dimension = seller_id
        if self.second_dimension == len(self.m_item_types):
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
            1 - When an auction is won, the bidding factor is multiplied by the increasing factor and when lost by
                the decreasing factor
            2 - Depends on the kind of item, but has a max value to avoid price explosion.
                If alpha bigger than 2, decrease it using decrease factor.
            3 - Depends on the kind of item, if the bid is higher than market price, bidding factor is multiplied by
                the decreasing factor while if it is lower multiply by the increasing factor.
        """

        bidding_factor = []
        for buyer in range(self.n_buyers):
            bidding_factor.append(
                np.random.uniform(1, 2, self.second_dimension)
            )

        return bidding_factor

    def calculate_starting_prices(self, starting_prices):
        """
        Calculate the starting prices of the sellers. If the input parameter is empty they will be empty otherwise they
        will be the same as the input parameter, this is only for debug purposes.
        :param starting_prices: DEBUG purposes. Set with the desired initial prices. If empty calculate them randomly.
        :return: the starting prices for the auctions
        """
        if len(starting_prices) > 0:
            return starting_prices

        prices = []
        for seller in range(self.k_sellers):
            prices.append(random.random() * self.max_starting_price)
        return prices

    def calculate_fee(self, price_paid):
        # Calculate the fee to pay for an item if it is cancelled
        return self.penalty_factor * price_paid

    def choose_item_to_keep(self, auction, market_price, price_to_pay, winner, seller, auction_round):
        """
        When an buyers wins a second item in a round one of the items has to be returned. The agent is rational and
        therefore will always keep the item with higher return considering the fee to pay for the returned item.
        :param auction: auction object with the information of the auction that made the buyer win the new item
        :param market_price: market price of the item just won
        :param price_to_pay: price paid for the new item
        :param winner: id of the buyer
        :param seller: id of the seller
        :param auction_round: round of the auction
        """

        self.times_items_returned += 1
        previous_auction, previous_seller = self.get_auction_with_winner(winner, auction_round)
        previous_winner_profit = previous_auction.winner_profit
        previous_fee = self.calculate_fee(previous_auction.price_paid)
        new_profit = market_price - price_to_pay
        new_fee = self.calculate_fee(price_to_pay)

        if new_profit - previous_fee > previous_winner_profit - new_fee:
            # It is profitable to keep the new item, pay fee to previous seller
            previous_auction.return_item(previous_fee,
                                         kept_item_profit=new_profit,
                                         kept_item_fee=new_fee,
                                         seller_item_kept=seller,
                                         kept_item_price=price_to_pay)

            if new_profit - previous_fee < 0:
                self.times_bad_trade += 1
        else:
            auction.return_item(new_fee,
                                kept_item_profit=previous_winner_profit,
                                kept_item_fee=previous_fee,
                                seller_item_kept=previous_seller,
                                kept_item_price=previous_auction.price_paid)

            if previous_winner_profit - new_fee < 0:
                self.times_bad_trade += 1

    def choose_winner(self, bids, market_price):
        """
        Chooose the winner of an auction.
        :param bids: map with the bids made by the buyers. Key is the id of the buyer and Value the bid
        :param market_price: market price of the item to sell
        :return: id of the buyer that wins the item, price to pay by the winner
        """
        valid_bids = []
        for bid in bids.values():

            if bid > market_price:
                continue

            valid_bids.append(bid)

        if len(valid_bids) == 0:
            valid_bids.append(next(iter(bids.values())))

        valid_bids = sorted(valid_bids, reverse=True)

        winner_id = [key for key in bids.keys() if bids[key] == valid_bids[0]][0]
        try:
            price_to_pay = valid_bids[1]
        except IndexError:
            price_to_pay = valid_bids[0]

        return winner_id, price_to_pay

    def get_alphas(self, seller, item):
        """
        Get the bidding factors
        :param seller: id of the seller
        :param item: kind of item
        :return: bidding factors
        """
        second_dimension = seller
        if self.second_dimension == len(self.m_item_types):
            second_dimension = item

        alphas = []
        for buyer in range(self.n_buyers):
            alphas.append(self.bidding_factor[buyer][second_dimension])
        return alphas

    def get_auction_with_winner(self, winner, auction_round):
        """
        Retrieve the auction object of a previous auction with the winner. Used when level commitment is activated and
        a buyer wins a second time.
        :param winner: id of the winner
        :param auction_round: round of the auction
        :return: auction object, seller id of the auction
        """
        seller = 0
        for auction in self.auctions_history[auction_round]:
            if winner == auction.winner:
                return auction, seller
            seller += 1
        assert 0 == 1

    def initialize_auction_parameters(self, seller):
        # Initialize all the parameters needed for an auction
        starting_price = self.starting_prices[seller]
        n_buyer_auction = 0
        total_bid = 0
        buyers_bid = {}
        item = self.sellers_types[seller]
        return buyers_bid, item, n_buyer_auction, starting_price, total_bid

    def initialize_buyers_flag(self):
        # Initialize the list with the flags that indicates if a buyer has already won an auction in the round
        return [False for buyer in range(self.n_buyers)]

    def print_alphas(self, extra_debug=False):
        """
        Print the values of the bidding factors.
        :param extra_debug: Even if in the parent object debug is set to false, it is possible that this printing is
        required. With this input parameter this is possible.
        """
        if not self.debug and not extra_debug:
            return

        buyer = 0
        alphas_table = PrettyTable()

        if self.second_dimension == self.k_sellers:
            alphas_table.field_names = ["S-0"] + ["S" + str(seller) for seller in range(self.k_sellers)]
        elif self.second_dimension == len(self.m_item_types):
            alphas_table.field_names = ["S-1"] + ["Type " + str(item_type) for item_type in self.m_item_types]

        for strategy in self.bidding_factor_strategy:
            alphas_table.add_row(["B" + str(buyer)] + ['%.2f' % elem for elem in self.bidding_factor[buyer]])
            str_0 = True
            buyer += 1

        print(alphas_table)

    def print_factors(self, extra_debug=False):
        """
        Print the increasing and decreasing factors for every buyer.
        :param extra_debug: Even if in the parent object debug is set to false, it is possible that this printing is
        required. With this input parameter this is possible.
        """
        if not self.debug and not extra_debug:
            return
        initial_table = PrettyTable()
        initial_table.field_names = [""] + ["B" + str(buyer) for buyer in range(self.n_buyers)]
        initial_table.add_row(["Increasing factor"] + ['%.2f' % elem for elem in self.increase_bidding_factor])
        initial_table.add_row(["Decreasing factor"] + ['%.2f' % elem for elem in self.decrease_bidding_factor])
        print(initial_table)

    def print_round(self, round_number, extra_debug=False):
        """
        Print the information of all the auctions in a round
        :param round_number: round of auction
        :param extra_debug: Even if in the parent object debug is set to false, it is possible that this printing is
        required. With this input parameter this is possible.
\        """
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
        """
        Update the bidding factor depending on the strategies of each buyer
        :param winner: id of the winner of the auction
        :param seller: seller of the item of the auction
        :param item: kind of items that the seller auctions
        :param bids: dictionary with the bids of the buyers, key is the id of the buyer and the value is the bid
        :return: new alphas after updating
        """

        second_dimension = seller
        if self.second_dimension == len(self.m_item_types):
            second_dimension = item

        new_alphas = []
        for buyer in range(self.n_buyers):
            if self.bidding_factor_strategy[buyer] == 1:
                if buyer == winner:
                    self.bidding_factor[buyer][second_dimension] *= self.decrease_bidding_factor[buyer]

                elif self.buyers_already_won[buyer] and not self.level_commitment_activated:
                    self.bidding_factor[buyer][second_dimension] = self.bidding_factor[buyer][second_dimension]

                else:
                    self.bidding_factor[buyer][second_dimension] *= self.increase_bidding_factor[buyer]

                new_alphas.append(self.bidding_factor[buyer][second_dimension])

            # Strategy 2 - Depends on the kind of item, but has a max value to avoid price explosion.
            # If alpha bigger than ceiling, decrease it using decrease factor.
            elif self.bidding_factor_strategy[buyer] == 2:
                # if buyer == winner:
                    # Do not update

                if buyer != winner and self.bidding_factor[buyer][second_dimension] < self.ceiling:
                    self.bidding_factor[buyer][second_dimension] *= self.increase_bidding_factor[buyer]

                elif self.buyers_already_won[buyer] and not self.level_commitment_activated:
                    continue

                elif buyer != winner and self.bidding_factor[buyer][second_dimension] > self.ceiling:
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

            # Strategy 4 - Fully random each time
            # to see if previous alpha update was helpful or not
            elif self.bidding_factor_strategy[buyer] == 4:
                self.bidding_factor[buyer][second_dimension] = np.random.uniform(1, 2)

            new_alphas.append(self.bidding_factor[buyer][second_dimension])

            # If the bidding factor is less than 1, replace it with the increasing factor
            if self.bidding_factor[buyer][second_dimension] < 1:
                self.bidding_factor[buyer][second_dimension] = self.increase_bidding_factor[buyer]

        return new_alphas

    def update_profits(self, auction_round):
        """
        Update the profit of every buyer and seller after a round is finished
        :param auction_round: number of round
        """
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
        """
        Main method of the program, runs the actual simulation
        """
        self.print_factors()
        for auction_round in range(self.r_rounds):
            self.buyers_already_won = self.initialize_buyers_flag()
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
                auction = self.store_auction_history(winner=winner,
                                                     price_paid=price_to_pay,
                                                     starting_price=starting_price,
                                                     market_price=market_price,
                                                     bid_history=buyers_bid,
                                                     previous_alphas=self.get_alphas(seller, item),
                                                     auction_round=auction_round,
                                                     item_kind=item)

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
        """
        Store the information of an auction in an auction object and store it in the auctions history
        :param starting_price: Starting price of the auction
        :param market_price: market price of the item
        :param winner: id of the buyer that wins the auction
        :param price_paid: price that the buyer pays for the item
        :param bid_history: dictionary with the bid of the buyers
        :param previous_alphas: bidding factor before the auction
        :param auction_round: round that this auction took place in
        :param item_kind: kind of item that is sold
        :return: auction object with all the information
        """
        auction = Auction(starting_price, market_price, price_paid, winner, bid_history, previous_alphas, item_kind)
        self.auctions_history[auction_round].append(auction)
        return auction

    def plot_statistics(self):
        """
        Plot the statistics of the history of the prices, the profit of the buyers and the sellers
\       """
        market_prices = np.zeros((self.r_rounds, self.k_sellers))

        for n, auctions_round in enumerate(self.auctions_history):
            for seller in range(self.k_sellers):
                market_prices[n, seller] = auctions_round[seller].market_price

        # Plot price history
        for seller in range(self.k_sellers):
            if self.bidding_factor_strategy[0] == 1:
                plt.semilogy(market_prices[:, seller], label="Seller " + str(seller))
            else:
                plt.plot(market_prices[:, seller], label="Seller " + str(seller))

        plt.title('Price history across all rounds for each seller')
        plt.ylabel('Price')
        plt.xlabel('Auctions')
        plt.legend()

        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        # Plot seller profits
        plt.figure()
        for seller in range(self.k_sellers):
            if self.bidding_factor_strategy[0] == 1:
                plt.semilogy(self.cumulative_sellers_profits[seller], label="Seller " + str(seller))
            else:
                plt.plot(self.cumulative_sellers_profits[seller], label="Seller " + str(seller))

        plt.title('Seller cumulative profits across all auctions')
        plt.ylabel('Seller profits')
        plt.xlabel('Rounds')
        plt.legend()

        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        # Plot Buyers profits
        plt.figure()
        for buyer in range(self.n_buyers):
            if self.bidding_factor_strategy[0] == 1:
                plt.semilogy(self.cumulative_buyers_profits[buyer], label="Buyer " + str(buyer))
            else:
                plt.plot(self.cumulative_buyers_profits[buyer], label="Buyer " + str(buyer))
        plt.title('Buyer cumulative profits across all auctions')
        plt.ylabel('Buyer profits')
        plt.xlabel('Rounds')
        plt.legend()

        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        plt.show()


if __name__ == '__main__':
    buyers = 10
    strategy = [1 for n in range(buyers)]
    # strategy[0] = 4
    auctioneer = Auctioneer(0.1,
                            bidding_factor_strategy=strategy,
                            M_types=3,
                            K_sellers=4,
                            N_buyers=buyers,
                            R_rounds=100,
                            level_comm_flag=False,
                            use_seller=False,
                            debug=True)
    auctioneer.start_auction()
    auctioneer.plot_statistics()
    print("\nBidding factors when the simulation is finished")
    auctioneer.print_alphas()
