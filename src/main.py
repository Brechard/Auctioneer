from auctioneer import Auctioneer


# Validate that the input is numeric before accepting it
def request_integer_input(string):
    cool = False
    x = input(string)
    while not cool:
        try:
            int(x)
            cool = True
        except ValueError:
            x = input(string)

    return int(x)


def request_float_input(string):
    cool = False
    x = input(string)
    while not cool:
        try:
            float(x)
            cool = True
        except ValueError:
            x = input(string)

    return float(x)


def request_boolean_input(string):
    cool = False
    x = input(string)
    while not cool:
        if x == 'y' or x == 'n':
            cool = True
        else:
            x = input(string)

    if x == 'y': return True
    if x == 'n': return False


# Request input from user
number_of_product_types = request_integer_input("Number of product types: ")
number_of_sellers = request_integer_input("Number of sellers: ")
number_of_buyers = request_integer_input("Number of buyers: ")
number_of_rounds = request_integer_input("Number of auction rounds: ")
universal_maximum_price = request_integer_input("Universal maximum price: ")
strat = "Bidding factor strategies: " \
        "\n\t0 - Depends only on the seller using proposed one by assignment" \
        "\n\t1 - Depends only on the kind of item using proposed one by assignment" \
        "\n\t2 - Depends on the kind of item, but has a max value to avoid price explosion." \
        "If alpha bigger than 2, decrease it using decrease factor." \
        "\n\t3 - Depends on the kind of item, if the bid is higher than market price, bidding factor is multiplied by " \
        "the decreasing factor while if it is lower multiply by the increasing factor.\n"
strategy = request_integer_input(strat)
level_commitment_activated = request_boolean_input("Should use level commitment? y/n ")
if level_commitment_activated:
    penalty_factor = request_float_input("Penalty factor: ")
else:
    penalty_factor = 0

alpha = 0
while alpha != 1 and alpha != 2:
    alpha = request_integer_input("Bidding factor depends on the sellers (1) or types of items (2): ")

# Execute with parameters
auctioneer = Auctioneer(penalty_factor=penalty_factor,
                        bidding_factor_strategy=[strategy for n in range(number_of_buyers)],
                        use_seller=(alpha == 1),
                        M_types=number_of_product_types,
                        K_sellers=number_of_sellers,
                        N_buyers=number_of_buyers,
                        R_rounds=number_of_rounds,
                        level_comm_flag=level_commitment_activated,
                        debug=True)
auctioneer.start_auction()
auctioneer.plot_statistics()
