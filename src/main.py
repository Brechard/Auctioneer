from auctioneer import Auctioneer


# Validate that the input is numeric before accepting it
def request_numeric_input(string):
    cool = False
    x = input(string)
    while not cool:
        try:
            int(x)
            cool = True
        except ValueError:
            x = input(string)

    return int(x)


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
number_of_product_types = request_numeric_input("Number of product types: ")
number_of_sellers = request_numeric_input("Number of sellers: ")
number_of_buyers = request_numeric_input("Number of buyers: ")
number_of_rounds = request_numeric_input("Number of rounds: ")
level_commitment_activated = request_boolean_input("Should use level commitment? y/n ")

# Execute with parameters
auctioneer = Auctioneer(0.1,
                        [],
                        [],
                        number_of_product_types,
                        number_of_sellers,
                        number_of_buyers,
                        number_of_rounds,
                        level_commitment_activated)
auctioneer.start_auction()
auctioneer.print_outcome()
auctioneer.plot_statistics()
