class Creature:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def info(self):
        print(f"{self.name} - {self.age}")

    
    def __del__(self):
        print("poka(")


class JSJ:
    def __init__(self):
        self.name = "fdsaf"




bob = Creature("bob", 21)
print(bob.name)
angela = Creature("angela", 58)

print(isinstance(bob, Creature))
bob.info()

angela.info()
















sym = "@!$$^&&*()"
a = input("напиши имя")

for i in a:
    if i in sym:
        print("такое имя нельзя")
    else:
        print("""все
               верно""")