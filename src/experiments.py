from auctioneer import Auctioneer
import matplotlib.pyplot as plt
import numpy as np

n_buyers = 5
k_sellers = 5


def create_auctioneer(strategy=0):
    return Auctioneer(penalty_factor=0.1,
                      bidding_factor_strategy=[strategy for n in range(n_buyers)],
                      M_types=3,
                      K_sellers=k_sellers,
                      N_buyers=n_buyers,
                      R_rounds=100,
                      level_comm_flag=False,
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


effect_inc_decr_bid_factors()
