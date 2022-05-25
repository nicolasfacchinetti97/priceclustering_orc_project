# A dynamic programming approach for a price clustering problem
The purpose of this project is to implement and conduct an experimental campaign on three dynamic programming algorithms that respectively solve three variants of a price clustering problem with constraints on the order of the elements inserted in the clusters and on their price, in particular:

1. With order and no price constraint, [order_no_price_constraint.py](order_no_price_constraint.py)
2. With order and price constraint, [order_price_constraint.py](order_price_constraint.py)
3. With price and no order constraint, [no_order_price_constraint.py](no_order_price_constraint.py)   


The scripts assume that a .ini configuration file is passed as first argument when they are launched. For reference, look at the [config.ini](config.ini) file for the various parameters and their meanings.   

[main.py](main.py) is used to conduct the experimental campaign by launching the three variants with different configuration files and collecting execution time of each.