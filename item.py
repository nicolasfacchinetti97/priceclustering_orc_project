class item:
    """ class to represent an item with its price and demand """
    def __init__(self, price, demand, cumdemand):
        self.price = int(price)
        self.demand = int(demand)
        self.cumdemand = int(cumdemand)
    
    def __str__(self):
        return(f'price {self.price}, demand {self.demand}, cum demand {self.cumdemand}')
