class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
        self.ratio = value / weight

def fractional_knapsack(items, capacity):
    items.sort(key=lambda x: x.ratio, reverse=True)
    
    total_value = 0
    for i in items:
        if capacity >= i.weight:
            capacity -= i.weight
            total_value += i.value
        else:
            fraction = capacity / i.weight
            total_value += i.value * fraction
            break
    return total_value

def read_items_from_file(filename):
    items = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        capacity = int(lines[0].strip())
        for line in lines[1:]:
            weight, value = map(int, line.strip().split())
            items.append(Item(weight, value))
    return capacity, items

filename = "E:\\SI\\lab5SI\\Data2.txt"

capacity, items = read_items_from_file(filename)
print(fractional_knapsack(items, capacity))
