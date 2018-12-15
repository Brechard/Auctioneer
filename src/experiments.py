import matplotlib.pyplot as plt
import numpy as np

from auctioneer import Auctioneer

n_buyers = 4
k_sellers = 2
rounds = 100


def create_auctioneer(strategy=0, penalty_factor=0.1, level_flag=True):
    return Auctioneer(penalty_factor=penalty_factor,
                      bidding_factor_strategy=[strategy for n in range(n_buyers)],
                      M_types=3,
                      K_sellers=k_sellers,
                      N_buyers=n_buyers,
                      R_rounds=rounds,
                      level_comm_flag=level_flag,
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

"""
Experiment 2
Motivation
We dont know if the prices will still be stable if we change the ceiling for the strategy
Setting up
We change the ceiling for N buyers and K sellers and see how the average price differs in the last round
Result
To be discussed
"""

def check_price_stability_varying_ceiling():

    iterations = 500

    ceilings = [top for top in np.arange(1.5, 5, 0.1)]
    # avg_market_prices = []
    # diff_marketprice_start_prices = []
    mrkt_price_starting_price_ratio = []

    for iter in range(len(ceilings)):

        auctioneer = Auctioneer(K_sellers=1, N_buyers=10, R_rounds=100, debug=False)
        auctioneer.ceiling = ceilings[iter]

        auctioneer.start_auction()

        # avg_market_prices.append(np.mean(auctioneer.market_price))
        # diff_marketprice_start_prices.append(calculate_avg_difference(auctioneer.starting_prices, auctioneer.market_price))
        mrkt_price_starting_price_ratio.append(auctioneer.market_price[auctioneer.r_rounds - 1] / auctioneer.starting_prices[auctioneer.k_sellers - 1])

    # plt.plot(ceilings, diff_marketprice_start_prices)
    plt.plot(ceilings, mrkt_price_starting_price_ratio)
    plt.xlabel("Ceiling")
    plt.ylabel("Ratio between market price and starting price of last round")
    plt.show()






def check_bias(times=1000):
    max_profit = np.zeros(n_buyers)
    for n in range(times):
        auctioneer = Auctioneer(bidding_factor_strategy=2, R_rounds=100, )
        auctioneer.bidding_factor = []
        for buyer in range(n_buyers):
            bid_fact = np.random.uniform(1, 1.001, 3)
            auctioneer.bidding_factor.append(bid_fact)

        auctioneer.increase_bidding_factor = [1.2 for n in range(n_buyers)]
        auctioneer.decrease_bidding_factor = [0.8 for n in range(n_buyers)]
        auctioneer.start_auction()
        buyers_prof = auctioneer.cumulative_buyers_profits[:, auctioneer.r_rounds - 1]

        for buyer in range(n_buyers):
            if buyers_prof[buyer] == max(buyers_prof):
                max_profit[buyer] += 1

    [print("Buyer", buyer, "was the one with more profit", max_profit[buyer], "times") for buyer in range(n_buyers)]


def check_penalty_factor_effect(strategy=2):
    differences = []
    times_items_returned = []
    buyers_profits = []
    sellers_profits = []
    penalty_factors = []
    bad_trades = []
    for n in range(200):
        penalty_factor = n / 400
        times_for_avg = 30
        penalty_factors.append(penalty_factor)

        diffs = []
        times_returned = []
        buyers_profit = []
        sellers_profit = []
        n_bad_trade = []

        for t in range(times_for_avg):
            auctioneer = create_auctioneer(strategy=strategy,
                                           penalty_factor=penalty_factor,
                                           level_flag=True)
            auctioneer.start_auction()

            diffs.append(calculate_avg_difference(auctioneer.starting_prices,
                                              auctioneer.market_price[auctioneer.r_rounds - 1]))
            times_returned.append(auctioneer.times_items_returned / (rounds * k_sellers))

            buyers_profit.append(np.average(auctioneer.cumulative_buyers_profits[:, rounds - 1]))
            sellers_profit.append(np.average(auctioneer.cumulative_sellers_profits[:, rounds - 1]))

            if auctioneer.times_items_returned == 0:
                n_bad_trade.append(0)
            else:
                n_bad_trade.append(auctioneer.times_bad_trade / auctioneer.times_items_returned)

        differences.append(min(300, np.mean(diffs)))
        times_items_returned.append(np.mean(times_returned))
        buyers_profits.append(np.mean(buyers_profit))
        sellers_profits.append(np.mean(sellers_profit))
        bad_trades.append(np.mean(n_bad_trade))

    plt.plot(penalty_factors, differences)
    plt.xlabel("Penalty factor")
    plt.ylabel("Difference between market price and initial price")

    plt.figure()
    plt.plot(penalty_factors, times_items_returned)
    plt.xlabel("Penalty factor")
    plt.ylabel("Percentage of number of items cancelled")

    plt.figure()
    plt.plot(penalty_factors, buyers_profits)
    plt.xlabel("Penalty factor")
    plt.ylabel("Average profit of buyers")

    plt.figure()
    plt.plot(penalty_factors, sellers_profits)
    plt.xlabel("Penalty factor")
    plt.ylabel("Average profit of sellers")

    plt.figure()
    plt.plot(penalty_factors, bad_trades)
    plt.xlabel("Penalty factor")
    plt.ylabel("Percentage of bad trades for buyers")

    plt.show()


# check_penalty_factor_effect(2)
# check_penalty_factor_effect(3)

# check_bias()
# effect_inc_decr_bid_factors()

check_price_stability_varying_ceiling()