from item import item

class instance:
    """class to represent an instance of the problem, which is a list of item """
    def __init__(self, path):
        self.items = instance.read_file(path)
        self.N = len(self.items)
        
    def __str__(self):
        string = "Items list:\n"
        for counter, item in enumerate(self.items):
            string += f"\tItem {counter}: {item}\n"
        return string

    @staticmethod
    def read_file(filename):
        """
        Read and parse the passed filename instance.
        For each row is crated an item object which is apppended to a list that is finally returned
        Before return is checked that the number of ridden rows is equal to the first row of the file which indicate the number of items
        """
        with open(filename, 'r') as f:
            # read number of items
            num_items = int(f.readline())
            # skip empty line
            f.readline()
            # read instances
            items = []
            counter = 0
            cum_demand = 0
            for line in f:
                counter += 1
                price, demand = line.split()               # instance are in format "price demand"
                cum_demand += int(demand)
                new_item = item(price, demand, cum_demand)          # use item object
                items.append(new_item)

            # check that the number of read items is equal to the first row of instance file
            if counter != num_items:
                raise ValueError(f"Read {counter} items while excepting {num_items} rows")
            
            return items

    def get_item(self, index):
        return self.items[index]

    def __str__(self):
        strings = [f'price {i.price}, demand {i.demand}, cum demand {i.cumdemand}\n' for i in self.items]
        return(' '.join(map(str, strings)))