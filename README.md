# Auctioneer
Program that simulates a modified second-price sealed-bid (Vickrey) auction scenario, both "pure" and level commitment version. 

As input it accepts:
  - the number of:
    * product types
    * sellers
    * buyers
    * rounds
 - universal maximum price
 - bidding factor strategies:
    * t1: When an auction is won, the bidding factor is multiplied by the increasing factor and when lost by the decreasing factor
    * t2: Depends on the kind of item, but has a max value to avoid price explosion. 
          If alpha bigger than 2, decrease it using decrease factor.
    * t3: Depends on the kind of item, if the bid is higher than market price, bidding factor is multiplied by the decreasing factor 
          while if it is lower multiply by the increasing factor.
  - level commitment flag
  - penalty factor (only when level commitment is possible)
  - bidding factor dependance (sellers or types of items)
  - debug flag (if activated, it prints the table with the information at every round)

The output of the program is: 
  - Statistics of market price 
  - Sellers profit 
  - Buyers profit
