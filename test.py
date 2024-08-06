import pandas as pd

class Practice:
    def __init__(self, value, ini_cap=1000):
        self.value = value
        self.ini_cap=ini_cap
    def getData(self, value=1000):
        self.ini_cap = value

test = Practice(1, 2)
test.getData()
print(test.value)
test.getData(ini_cap=5)
print(test.value)