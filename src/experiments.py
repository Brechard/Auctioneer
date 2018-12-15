import copy

import matplotlib.pyplot as plt
import numpy as np

from auctioneer import Auctioneer

n_buyers = 4
k_sellers = 2
rounds = 100


def create_auctioneer(strategy=0, penalty_factor=0.1, level_flag=True, types=3, sellers=k_sellers, buyers=n_buyers):
    return Auctioneer(penalty_factor=penalty_factor,
                      bidding_factor_strategy=[strategy for n in range(buyers)],
                      M_types=types,
                      K_sellers=sellers,
                      N_buyers=buyers,
                      R_rounds=rounds,
                      level_comm_flag=level_flag,
                      debug=False)


def calculate_avg_difference(initial_price, market_price):
    total_difference = 0
    for item in range(len(initial_price)):
        total_difference += market_price[item] - initial_price[item]

    return total_difference / len(initial_price)


def effect_inc_decr_bid_factors(strategy=2):
    i_range = 100
    d_range = 100
    differences = np.zeros((i_range, d_range))
    for increasing_delta in range(i_range):
        for decreasing_delta in range(1, d_range):
            auctioneer = create_auctioneer(strategy)
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
    # ax.set_title(
    #     "Increase/Decrease bidding factor effect for " + str(n_buyers) + " buyers and " + str(k_sellers) + " sellers")

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
        auctioneer = Auctioneer(K_sellers=3, N_buyers=10, R_rounds=100, debug=False,
                                bidding_factor_strategy=[3 for x in range(10)])
        auctioneer.ceiling = ceilings[iter]

        auctioneer.start_auction()

        # avg_market_prices.append(np.mean(auctioneer.market_price))
        # diff_marketprice_start_prices.append(calculate_avg_difference(auctioneer.starting_prices, auctioneer.market_price))
        mrkt_price_starting_price_ratio.append(
            np.mean(auctioneer.market_price[auctioneer.r_rounds - 1]) /
            np.mean(auctioneer.starting_prices))

    # plt.plot(ceilings, diff_marketprice_start_prices)
    plt.plot(ceilings, mrkt_price_starting_price_ratio)
    plt.xlabel("Ceiling")
    plt.ylabel("Ratio between market price and starting price of last round")
    plt.legend()

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
    level_flag = True
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
                                           level_flag=level_flag)
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

    print_graphs(penalty_factors, "Penalty factor", level_flag, bad_trades, buyers_profits, differences,
                 sellers_profits, times_items_returned)


def print_graphs(x_values, x_title, level_commitment, bad_trades=[], buyers_profits=[], differences=[],
                 sellers_profits=[],
                 times_items_returned=[],
                 distances=False):
    if len(differences) > 0:
        plt.plot(x_values, differences)
        plt.xlabel(x_title)
        if distances:
            plt.ylabel("Distance between market price and initial price from average")
        else:
            plt.ylabel("Difference between market price and initial price")

    if len(times_items_returned) > 0 and level_commitment:
        plt.figure()
        plt.plot(x_values, times_items_returned)
        plt.xlabel(x_title)
        plt.ylabel("Percentage of number of items cancelled")

    if len(buyers_profits) > 0:
        plt.figure()
        plt.plot(x_values, buyers_profits)
        plt.xlabel(x_title)
        plt.ylabel("Average profit of buyers")

    if len(sellers_profits) > 0:
        plt.figure()
        plt.plot(x_values, sellers_profits)
        plt.xlabel(x_title)
        plt.ylabel("Average profit of sellers")

    if len(bad_trades) > 0 and level_commitment:
        plt.figure()
        plt.plot(x_values, bad_trades)
        plt.xlabel(x_title)
        plt.ylabel("Percentage of bad trades for buyers")

    plt.show()


def get_distances_from_mean(array, distance):
    if distance:
        return [n / np.mean(array) for n in array]
    else:
        return [n - np.mean(array) for n in array]


def buyers_effect(strategy, level_flag):
    k_sellers = 2
    differences = []
    times_items_returned = []
    buyers_profits = []
    sellers_profits = []
    bad_trades = []
    x_values = []

    for buyers in range(k_sellers, 16):
        times_for_avg = 100
        penalty_factor = 0.1
        x_values.append(buyers)
        diffs = []
        times_returned = []
        buyers_profit = []
        sellers_profit = []
        n_bad_trade = []
        for t in range(times_for_avg):
            auctioneer = create_auctioneer(strategy=strategy,
                                           penalty_factor=penalty_factor,
                                           level_flag=level_flag)
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

    print_graphs(x_values=x_values,
                 x_title="Buyers",
                 level_commitment=level_flag,
                 bad_trades=bad_trades,
                 buyers_profits=buyers_profits,
                 # differences=get_distances_from_mean(differences, True),
                 differences=differences,
                 sellers_profits=sellers_profits,
                 times_items_returned=times_items_returned)
    # times_items_returned=times_items_returned,
    # distances=True)


def check_bias_v2(times=1000, buyers=10):
    starting_winner_is_final_winner = 0
    winners = []
    first_bidding_factor = []
    for n in range(times):
        auctioneer = create_auctioneer(strategy=3, types=1, buyers=buyers, sellers=1, level_flag=False)

        # auctioneer.increase_bidding_factor = [1.2 for n in range(buyers)]
        # auctioneer.decrease_bidding_factor = [0.8 for n in range(buyers)]

        first_winner, alpha = calculate_best_alpha(auctioneer)
        previous_alpha = round(alpha[0], 2)
        firsts_alphas = copy.deepcopy(auctioneer.bidding_factor)

        auctioneer.start_auction()

        buyers_prof = auctioneer.cumulative_buyers_profits[:, auctioneer.r_rounds - 1]

        final_winner = np.argmax(buyers_prof)
        winners.append(final_winner)
        first_bidding_factor.append(round(firsts_alphas[final_winner][0], 2))

        if final_winner == first_winner:
            starting_winner_is_final_winner += 1

    percentage = 0
    if starting_winner_is_final_winner != 0:
        percentage = 100 * starting_winner_is_final_winner / times

    plt.hist(first_bidding_factor)
    plt.xlabel("Alpha")
    plt.ylabel("Times won")
    plt.title("Initial alphas used by the users that won the simulation")
    plt.show()

    print("After", times, "simulations, the initial winner is the final winner", starting_winner_is_final_winner,
          "times. Considering there are", buyers, "buyers, it should be close to", 100 / buyers, "% and it its",
          percentage, "%")


def calculate_best_alpha(auctioneer):
    best_below_avg = 0
    best_below_avg_pos = 0
    avg = np.mean(auctioneer.bidding_factor)
    for pos, factor in enumerate(auctioneer.bidding_factor):
        if avg > factor > best_below_avg:
            best_below_avg = factor
            best_below_avg_pos = pos

    return best_below_avg_pos, best_below_avg


# check_penalty_factor_effect(2)
# check_penalty_factor_effect(3)

# check_bias()
# effect_inc_decr_bid_factors(2)
# buyers_effect(2, False)
# check_bias_v2(1000)
# effect_inc_decr_bid_factors()

check_price_stability_varying_ceiling()
