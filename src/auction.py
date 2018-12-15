from prettytable import PrettyTable


class Auction:

    def __init__(self, starting_price, market_price, price_paid, winner, bid_history, previous_alphas, item_kind):
        """
        Object that contains all the useful information of an auction
        :param starting_price: Starting price of the auction
        :param market_price: Market price of the item sold
        :param price_paid: Price to pay for the item
        :param winner: id of the winner of the auction
        :param bid_history: Dictionary with the bids from the buyers that participated in the auction
        :param previous_alphas: Values of the bidding factor applied to calculate the bids of this auction
        :param item_kind: Kind of item sold in this auction
        """
        self.starting_price = starting_price
        self.market_price = market_price
        self.price_paid = price_paid
        self.winner = winner
        self.seller_profit = price_paid
        self.winner_profit = self.market_price - self.price_paid
        self.item_kind = item_kind
        self.item_returned = False

        # Debug purposes
        # ['%.2f' % elem for elem in bid_history.values()]
        self.bid_history = bid_history
        self.previous_alphas = previous_alphas
        self.new_alphas = []
        self.kept_item_profit = None
        self.kept_item_fee = None
        self.seller_item_kept = None
        self.original_info = None
        self.kept_item_price = None

    def return_item(self, fee, kept_item_profit, kept_item_fee, seller_item_kept, kept_item_price):
        """
        Method used when the item from this auction is returned
        :param fee: fee to pay for returning the item
        :param kept_item_profit: Profit of the item the buyer has decided to keep
        :param kept_item_fee: Fee that the buyer would have payed for canceling the item that is kept
        :param seller_item_kept: Seller of the item kept
        :param kept_item_price: price paid for the item kept
        """
        self.original_info = [self.winner_profit, fee, self.seller_profit]
        self.seller_profit = fee
        self.winner_profit = - fee
        self.item_returned = True
        self.kept_item_profit = kept_item_profit
        self.kept_item_fee = kept_item_fee
        self.kept_item_price = kept_item_price
        self.seller_item_kept = seller_item_kept

    def set_new_alphas(self, new_alphas):
        """
        Save the values of the bidding factors after this auction. Debug and printing purposes mainly.
        :param new_alphas: Values of the new bidding factors
        """
        self.new_alphas = ['%.4f' % elem for elem in new_alphas]
        self.factor = ['%.4f' % (float(new_alpha) / float(old_alpha)) for new_alpha, old_alpha in
                       zip(new_alphas, self.previous_alphas)]

    def print_auction(self, n):
        """
        Print the all the information of the auction using tables
        :param n: Number of auction (seller id)
        """
        self.previous_alphas = ['%.4f' % elem for elem in self.previous_alphas]
        # Printing buyer info
        buyer_info = PrettyTable()
        field_names = ["Auction #" + str(n)]
        old_alphas = ["Old Alpha"]
        new_alphas = ["New Alpha"]
        multiplier = ["Multiplier"]
        bids = ["Bids"]
        for buyer, bid in self.bid_history.items():
            heading = "B" + str(buyer)
            if buyer == self.winner:
                heading = heading + " - W"
            field_names.append(heading)
            old_alphas.append(self.previous_alphas[buyer])
            new_alphas.append(self.new_alphas[buyer])
            multiplier.append(self.factor[buyer])
            bids.append(round(bid, 2))

        buyer_info.field_names = field_names
        buyer_info.add_row(old_alphas)
        buyer_info.add_row(bids)
        buyer_info.add_row(new_alphas)
        buyer_info.add_row(multiplier)

        print(buyer_info)

        # Printing market prices info
        auction_info = PrettyTable()

        field_names = ["Starting price", "Market Price", "Winner", "Price to pay", "Buyer profit", "Seller profit",
                       "Item kind"]
        auction_info.field_names = field_names
        row = [self.starting_price, self.market_price, self.winner, self.price_paid, self.winner_profit,
               self.seller_profit]
        row = ['%.2f' % elem for elem in row]
        row[2] = self.winner
        row.append(self.item_kind)
        auction_info.add_row(row)
        print(auction_info)

        # Printing return info
        if self.item_returned:
            return_info = PrettyTable()
            field_names = ["Buyer profit for discarded item", "Buyer fee for canceling this item",
                           "Profit of seller before cancel", "Buyer profit for kept item",
                           "Buyer fee if canceling kept item", "Seller of the kept item",
                           "Final Profit (profit - fee paid)", "Profit if kept this item"]
            return_info.field_names = field_names
            row = [self.original_info[0], self.original_info[1],
                   self.original_info[2], self.kept_item_profit,
                   self.kept_item_fee, self.seller_item_kept,
                   self.kept_item_profit - self.original_info[1],
                   self.original_info[0] - self.kept_item_fee]
            row = ['%.2f' % elem for elem in row]
            row[5] = self.seller_item_kept
            return_info.add_row(row)
            print(return_info)
        print()
        print()

    def round_dict_values(self, dict):
        """
        Round the values of a dictionary to the second decimal
        :param dict: dictionary to round
        :return: dictionary with the new values
        """
        for dict_value in dict:
            for k, v in dict_value.items():
                dict_value[k] = round(v, 2)
        return dict
