import numpy as np
class Auction:

    def __init__(self, M, K, N, R, level_comm_flag):

        self.m_item_types = M
        self.k_sellers = K
        self.n_buyers = N
        self.r_rounds = R

        self.max_starting_price = 100

        self.bidding_factor = np.random.uniform(1, 2, size=(self.n_buyers, self.m_item_types, self.k_sellers))

        self.market_price = np.zeros(self.r_rounds, self.k_sellers)
        self.buyers_profits = np.zeros(self.r_rounds, self.n_buyers)
        self.sellers_profits = np.zeros(self.r_rounds, self.k_sellers)

    def calculate_bid(self, buyer_id, item_type, seller_id, starting_price):

        bid = self.bidding_factor[buyer_id, item_type, seller_id] * starting_price

        return bid


    def choose_winner(self, bids, market_price):

        valid_bids = []
        for bid in bids.value():

            if bid > market_price :
                continue

            if bid <= market_price :
                valid_bids.append(bid)


        valid_bids = sorted(valid_bids, reverse=True)

        winner_id = [key for key in bids.keys() if bids[key] == valid_bids[0]]
        price_to_pay = valid_bids[1]

        return (winner_id, price_to_pay)




    """
    Initialize:
    Number of Items, Sellers, Buyers
    """
    def initialize_variables(self):
        # TODO Implement initialize_variables
        pass

    """
    Initialize alpha values for each buyer and seller and item
    """
    def initialize_alpha_values(self):
        # TODO Implement initialize_alpha_values
        pass

    """
    Initialize delta values
    """
    def initialize_delta_values(self):
        # TODO Implement initialize_delta_values
        pass

    """
    Perform simulation
    """
    def run_simulation(self):
        # Perform initializations
        self.initialize_variables()
        self.initialize_alpha_values()
        self.initialize_delta_values()

        # TODO Implement rest of the simulation
        pass

    def print_outcome(self):
        # TODO Implement print_outcome
        pass


if __name__ == '__main__':
    auctioneer = Auctioneer()
    auctioneer.run_simulation()
    auctioneer.print_outcome()
