class Auctioneer:

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
