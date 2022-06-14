import os
import datetime

file_res = 'res.txt'

c1 = "order_no_price_constraint.py"
c2 = "order_price_constraint.py"
c3 = "no_order_price_constraint.py"
c = [c1, c2]

inst = "test_algo/{i}/{i}_{c}.ini"

i = [inst.format(i=100, c=c) for c in [10,15,20]]
i += [inst.format(i=200, c=c) for c in [10,15,20]]
i += [inst.format(i=400, c=c) for c in [20,30,40]]
i += [inst.format(i=800, c=c) for c in [20,30,40]]

instances = [(version , instance) for version in c for instance in i ]
res = {}
for c, i in instances:
    case_name = f'Case {c} instance {i}'
    print(case_name)
    start = datetime.datetime.now()
    os.system(f"python {c} {i}")
    end = datetime.datetime.now()
    comp = end-start
    res[(c,i)] = comp
    result  = f"\nSecond elapsed: {comp.total_seconds()}\n"
    print(result)
    with open(file_res, 'a') as f:
        f.write(case_name + result)