import os
import datetime

c1 = "order_no_price_constraint.py"
c2 = "order_price_constraint.py"
c3 = "no_order_price_constraint.py"

i100_1 = "test_algo/100/100_10.ini"
i100_2 = "test_algo/100/100_15.ini"
i100_3 = "test_algo/100/100_20.ini"

instances = [[c1,i100_1], [c1,i100_2], [c1, i100_3]]
for c, i in instances:
    print(f'Case {c} instance {i}')
    start = datetime.datetime.now()
    os.system(f"python {c} {i}")
    end = datetime.datetime.now()
    comp = end-start
    print(f"\nSecond elapsed: {comp.total_seconds()}")