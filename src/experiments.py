from auctioneer import Auctioneer
import matplotlib.pyplot as plt
import numpy as np
import statistics

n_buyers = 4
k_sellers = 2


def create_auctioneer(strategy=0):
    return Auctioneer(penalty_factor=0.1,
                      bidding_factor_strategy=[strategy for n in range(n_buyers)],
                      M_types=3,
                      K_sellers=k_sellers,
                      N_buyers=n_buyers,
                      R_rounds=100,
                      level_comm_flag=True,
                      debug=False)


def calculate_avg_difference(initial_price, market_price):
    total_difference = 0
    for item in range(len(initial_price)):
        total_difference += market_price[item] - initial_price[item]

    return total_difference / len(initial_price)


def effect_inc_decr_bid_factors():
    i_range = 100
    d_range = 100
    differences = np.zeros((i_range, d_range))
    for increasing_delta in range(i_range):
        for decreasing_delta in range(1, d_range):
            auctioneer = create_auctioneer(2)
            auctioneer.increase_bidding_factor = [1 + increasing_delta / i_range for n in range(n_buyers)]
            auctioneer.decrease_bidding_factor = [0 + decreasing_delta / d_range for n in range(n_buyers)]
            auctioneer.start_auction()
            differences[increasing_delta, decreasing_delta] = min(300,
                                                                  calculate_avg_difference(auctioneer.starting_prices,
                                                                                           auctioneer.market_price[
                                                                                               auctioneer.r_rounds - 1]))
    fig, ax = plt.subplots()
    i_factors = [1 + n / i_range for n in range(i_range)]
    d_factors = [0 + n / d_range for n in range(d_range)]
    im = ax.pcolormesh(d_factors, i_factors, differences[:, :])
    ax.set_xlabel("Decreasing factor")
    ax.set_ylabel("Increasing factor")
    ax.set_title(
        "Increase/Decrease bidding factor effect for " + str(n_buyers) + " buyers and " + str(k_sellers) + " sellers")

    fig.colorbar(im)
    plt.show()


def check_bias():
    max_profit = np.zeros(n_buyers)
    for n in range(10000):
        auctioneer = create_auctioneer(2)
        auctioneer.bidding_factor = []
        for buyer in range(n_buyers):
            # 3 because strategy 2 depends on the item type
            if buyer == 2:
                bid_fact = np.random.uniform(1.1, 1.3, 3)
            else:
                bid_fact = np.random.uniform(1, 1.001, 3)

            auctioneer.bidding_factor.append(bid_fact)

        auctioneer.increase_bidding_factor = [1.2 for n in range(n_buyers)]
        auctioneer.decrease_bidding_factor = [0.8 for n in range(n_buyers)]
        auctioneer.start_auction()
        buyers_prof = auctioneer.cumulative_buyers_profits[:, auctioneer.r_rounds - 1]

        for buyer in range(n_buyers):
            if buyers_prof[buyer] == max(buyers_prof):
                max_profit[buyer] += 1

    [print("The buyer", buyer, "was the one with more profit", max_profit[buyer], "times") for buyer in range(n_buyers)]


check_bias()

# effect_inc_decr_bid_factors()
