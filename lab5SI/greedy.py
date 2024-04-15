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

# Test the function
items = [Item(20, 100), Item(30, 120), Item(10, 60)]
capacity = 50
print(fractional_knapsack(items, capacity))