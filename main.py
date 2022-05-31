import os
import datetime

c1 = "order_no_price_constraint.py"
c2 = "order_price_constraint.py"
c3 = "no_order_price_constraint.py"
c = [c1, c2, c3]

inst = "test_algo/{i}/{i}_{c}.ini"

i = [inst.format(i=100, c=c) for c in [10,15,20]]
i += [inst.format(i=200, c=c) for c in [10,15,20]]
i += [inst.format(i=400, c=c) for c in [20,30,40]]
i += [inst.format(i=800, c=c) for c in [20,30,40]]

instances = [(version , instance) for version in c for instance in i ]
res = {}
for c, i in instances:
    print(f'Case {c} instance {i}')
    start = datetime.datetime.now()
    os.system(f"python {c} {i}")
    end = datetime.datetime.now()
    comp = end-start
    res[(c,i)] = comp
    print(f"\nSecond elapsed: {comp.total_seconds()}\n")