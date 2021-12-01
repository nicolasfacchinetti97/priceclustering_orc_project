class item:
    """ class to represent an item with its price and demand """
    def __init__(self, price, demand):
        self.price = price
        self.demand = demand
    
    def __str__(self):
        return(f'price {self.price}, demand {self.demand}')