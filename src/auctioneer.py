import numpy as np
import random
from prettytable import PrettyTable


class Auctioneer:

    def __init__(self, bidding_factor_strategy=[], starting_prices=[], M_types=3, K_sellers=4,
                 N_buyers=10, R_rounds=3, level_comm_flag=False):
        """
        :param bidding_factor_strategy: array with the bidding factor strategy of each buyer
        :param starting_prices: Debug purposes, starting prices can be forced this way.
        :param M_types: Number of types of items
        :param K_sellers: Number of sellers
        :param N_buyers: Number of buyers
        :param R_rounds: Number of rounds
        :param level_comm_flag: Flag to say if level commitment is allowed or not
        """
        if len(bidding_factor_strategy) == 0:
            # If the strategy is not passed, it is set to default 0
            bidding_factor_strategy = [0 for n in range(N_buyers)]

        self.m_item_types = range(M_types)
        self.k_sellers = K_sellers
        self.n_buyers = N_buyers
        self.r_rounds = R_rounds

        self.max_starting_price = 100
        self.penalty_factor = 0.1

        # If level commitment is activated sellers cannot cancel a won auction
        self.level_commitment_activated = level_comm_flag
        self.buyers_already_won = self.initialize_buyers_flag()
        self.auctions_history = {}
        if level_comm_flag:
            self.auctions_history = self.initialize_auction_history()

        # Assign a type of item to each seller randomly
        self.sellers_types = [random.sample(self.m_item_types, 1)[0] for seller in range(self.k_sellers)]

        self.bidding_factor_strategy = bidding_factor_strategy
        self.bidding_factor = self.calculate_bidding_factor()

        self.increase_bidding_factor = np.random.uniform(1, 2, size=self.n_buyers)
        self.decrease_bidding_factor = np.random.uniform(0, 1, size=self.n_buyers)

        self.market_price = np.zeros((self.r_rounds, self.k_sellers))
        self.buyers_profits = np.zeros((self.r_rounds, self.n_buyers))
        self.sellers_profits = np.zeros((self.r_rounds, self.k_sellers))

        self.history = {}

        self.starting_prices = self.calculate_starting_prices(starting_prices)

    def calculate_bid(self, buyer_id, item_type, seller_id, starting_price):
        if self.bidding_factor_strategy[buyer_id] == 1:
            bid = self.bidding_factor[buyer_id][item_type] * starting_price
        else:
            bid = self.bidding_factor[buyer_id][seller_id] * starting_price

        if not self.level_commitment_activated \
                or not self.buyers_already_won[buyer_id]:
            # If the buyer flag is not ON it means the buyer hasn't win an auction in this round yet
            return bid
        auction, seller = self.get_auction_with_winner(buyer_id)
        previous_profit, market_price = auction.winner_profit, auction.market_price
        penalty = self.calculate_fee(market_price - previous_profit)

        return max(bid, starting_price + previous_profit + penalty)

    def calculate_bidding_factor(self):
        """
        Bidding factor strategies:
            0 - Depends only on the seller
            1 - Depends only on the kind of item
        :return:
        """
        bidding_factor = []
        for buyer in range(self.n_buyers):
            if self.bidding_factor_strategy[buyer] == 1:
                bidding_factor.append(
                    np.random.uniform(1, 2, self.m_item_types)
                )
            else:
                bidding_factor.append(
                    np.random.uniform(1, 2, self.k_sellers)
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

    def choose_item_to_keep(self, auction, market_price, price_to_pay, winner, seller):
        previous_auction, previous_seller = self.get_auction_with_winner(winner)
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
        else:
            auction.return_item(new_fee,
                                kept_item_profit=previous_winner_profit,
                                kept_item_fee=previous_fee,
                                seller_item_kept=previous_seller,
                                kept_item_price=previous_auction.price_paid)

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

    def get_alphas(self, seller, item):
        alphas = []
        for buyer in range(self.n_buyers):
            if self.bidding_factor_strategy[buyer] == 1:
                second_dimension = item
            else:
                second_dimension = seller

            alphas.append(self.bidding_factor[buyer][second_dimension])
        return alphas

    def get_auction_with_winner(self, winner):
        seller = 0
        for auction in self.auctions_history:
            if winner == auction.winner:
                return auction, seller
            seller += 1
        assert 0 == 1

    def initialize_auction_history(self):
        """
        List with the history of auctions, it will be filled with auctions objects
        """
        return []

    def initialize_auction_parameters(self, seller):
        starting_price = self.starting_prices[seller]
        n_buyer_auction = 0
        total_bid = 0
        buyers_bid = {}
        item = self.sellers_types[seller]
        return buyers_bid, item, n_buyer_auction, starting_price, total_bid

    def initialize_buyers_flag(self):
        return [False for buyer in range(self.n_buyers)]

    def update_alphas(self, winner, seller, item):
        new_alphas = []
        for buyer in range(self.n_buyers):
            if self.bidding_factor_strategy[buyer] == 1:
                second_dimension = item
            else:
                second_dimension = seller

            if buyer == winner:
                self.bidding_factor[buyer][second_dimension] *= self.decrease_bidding_factor[buyer]
            else:
                self.bidding_factor[buyer][second_dimension] *= self.increase_bidding_factor[buyer]

            new_alphas.append(self.bidding_factor[buyer][second_dimension])

        return new_alphas

    def update_profits(self, auction_round):
        seller = 0
        for auction in self.auctions_history:
            self.buyers_profits[auction_round, auction.winner] += auction.winner_profit
            self.sellers_profits[auction_round, seller] += auction.seller_profit
            seller += 1

    def print_outcome(self):
        # TODO Implement print_outcome
        print("The market price history is:")
        print(self.market_price)
        print("The buyers profits are:")
        print(self.buyers_profits)
        print("The sellers profits are:")
        print(self.sellers_profits)

    def print_factors(self):
        initial_table = PrettyTable()
        initial_table.field_names = [""] + ["B" + str(buyer) for buyer in range(self.n_buyers)]
        initial_table.add_row(["Increasing factor"] + ['%.2f' % elem for elem in self.increase_bidding_factor])
        initial_table.add_row(["Decreasing factor"] + ['%.2f' % elem for elem in self.decrease_bidding_factor])
        print(initial_table)

    def print_round(self, round_number):
        print()
        print("Round", round_number, "history")
        seller = 0
        for auction in self.auctions_history:
            auction.print_auction(seller)
            # print()
            # print("Seller", seller, "sells item", self.sellers_types[seller])
            # print("Market price", round(auction.market_price, 4))
            # print("Winner is", auction.winner, "with a profit of", round(auction.winner_profit, 4))
            # print("Seller profit:", round(auction.seller_profit, 4))
            # if auction.item_returned:
            #     print("The item was returned")
            seller += 1
        print()
        print("------------------------------------------------------")

    def start_auction(self):
        self.print_factors()
        for auction_round in range(self.r_rounds):
            self.buyers_already_won = self.initialize_buyers_flag()
            if self.level_commitment_activated:
                self.auctions_history = self.initialize_auction_history()

            for seller in range(self.k_sellers):
                buyers_bid, item, n_buyer_auction, starting_price, total_bid = self.initialize_auction_parameters(
                    seller)
                for buyer in range(self.n_buyers):
                    if self.buyers_already_won[buyer] and not self.level_commitment_activated:
                        continue
                    n_buyer_auction += 1
                    bid = self.calculate_bid(buyer, item, seller, starting_price)
                    buyers_bid[buyer] = bid
                    total_bid += bid

                market_price = total_bid / n_buyer_auction
                winner, price_to_pay = self.choose_winner(buyers_bid, market_price)
                auction = self.store_auction_history(winner=winner,
                                                     price_paid=price_to_pay,
                                                     starting_price=starting_price,
                                                     market_price=market_price,
                                                     bid_history=buyers_bid,
                                                     previous_alphas=self.get_alphas(seller, item))

                if self.level_commitment_activated and self.buyers_already_won[winner]:
                    # The buyer already won an auction in this round so he has to choose which one to return
                    self.choose_item_to_keep(auction, market_price, price_to_pay, winner, seller)

                self.buyers_already_won[winner] = True
                new_alphas = self.update_alphas(winner, seller, item)
                auction.set_new_alphas(new_alphas)
                self.market_price[auction_round, seller] = market_price
            self.update_profits(auction_round)
            self.print_round(auction_round)

    def store_auction_history(self, starting_price, market_price, winner, price_paid, bid_history, previous_alphas):
        auction = Auction(starting_price, market_price, price_paid, winner, bid_history, previous_alphas)
        self.auctions_history.append(auction)
        return auction


class Auction:

    def __init__(self, starting_price, market_price, price_paid, winner, bid_history, previous_alphas):
        self.starting_price = starting_price
        self.market_price = market_price
        self.price_paid = price_paid
        self.winner = winner
        self.seller_profit = price_paid
        self.winner_profit = self.market_price - self.price_paid
        self.item_returned = False

        # Debug purposes
        self.bid_history = ['%.2f' % elem for elem in bid_history.values()]
        self.previous_alphas = ['%.2f' % elem for elem in previous_alphas]
        self.new_alphas = []
        self.kept_item_profit = None
        self.kept_item_fee = None
        self.seller_item_kept = None
        self.original_info = None
        self.kept_item_price = None

    def return_item(self, fee, kept_item_profit, kept_item_fee, seller_item_kept, kept_item_price):
        self.original_info = [self.winner_profit, fee, self.seller_profit]
        self.seller_profit = fee
        self.winner_profit = - fee
        self.item_returned = True
        self.kept_item_profit = kept_item_profit
        self.kept_item_fee = kept_item_fee
        self.kept_item_price = kept_item_price
        self.seller_item_kept = seller_item_kept

    def set_new_alphas(self, new_alphas):
        self.new_alphas = ['%.2f' % elem for elem in new_alphas]
        self.factor = ['%.2f' % (float(new_alpha) / float(old_alpha)) for new_alpha, old_alpha in
                       zip(new_alphas, self.previous_alphas)]

    def print_auction(self, n):
        # Printing buyer info
        buyer_info = PrettyTable()
        field_names = ["Auction #" + str(n)]
        for buyer in range(len(self.bid_history)):
            heading = "B" + str(buyer)
            if buyer == self.winner:
                heading = heading + " - W"
            field_names.append(heading)
        # field_names.extend(["Market Price", "Winner", "Price to pay"])
        buyer_info.field_names = field_names
        buyer_info.add_row(["Old Alpha"] + self.previous_alphas)
        buyer_info.add_row(["Bids"] + self.bid_history)
        buyer_info.add_row(["New Alpha"] + self.new_alphas)
        buyer_info.add_row(["Multiplier"] + self.factor)

        print(buyer_info)

        # Printing market prices info
        auction_info = PrettyTable()

        field_names = ["Starting price", "Market Price", "Winner", "Price to pay", "Buyer profit", "Seller profit"]
        auction_info.field_names = field_names
        row = [self.starting_price, self.market_price, self.winner, self.price_paid, self.winner_profit,
               self.seller_profit]
        row = ['%.2f' % elem for elem in row]
        row[2] = self.winner
        auction_info.add_row(row)
        print(auction_info)
        print()

        # Printing return info
        if self.item_returned:
            return_info = PrettyTable()
            field_names = ["Buyer profit for discarded item", "Buyer fee for canceling this item",
                           "Profit of seller before cancel", "Buyer profit for kept item",
                           "Buyer fee if canceling kept item", "Seller of the kept item",
                           "Final Profit (profit - fee paid)"]
            return_info.field_names = field_names
            row = [self.original_info[0], self.original_info[1],
                   self.original_info[2], self.kept_item_profit,
                   self.kept_item_fee, self.seller_item_kept,
                   self.kept_item_profit - self.original_info[1]]
            row = ['%.2f' % elem for elem in row]
            row[5] = self.seller_item_kept
            return_info.add_row(row)
            print(return_info)
            print()
        print()

    def round_dict_values(self, dict):
        for dict_value in dict:
            for k, v in dict_value.items():
                dict_value[k] = round(v, 2)
        return dict


if __name__ == '__main__':
    auctioneer = Auctioneer(level_comm_flag=True)
    auctioneer.start_auction()
    auctioneer.print_outcome()
