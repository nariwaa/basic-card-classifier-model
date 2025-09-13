class MySweetRobot:
    def __init__(self, name, color, battery_level):
        self.name = name
        self.color = color
        self.battery_level = battery_level
        self.is_on = False
        print(f"heya! {self.name} the {self.color} robot has been built! o/")

    def turn_on(self):
        if not self.is_on:
            self.is_on = True
            print(f"{self.name} is now ON! beep boop! -w-")
        else:
            print(f"{self.name} is already on, silly! hehe.")

    def charge(self, amount):
        self.battery_level += amount
        if self.battery_level > 1:
            self.battery_level = 1
        print(f"{self.name} charged! current battery: {self.battery_level}")

    def greet(self, other_name):
        if self.is_on:
            print(f"hello, {other_name}! i am {self.name}! (-w-)")
        else:
            print(f"zzzz... {self.name} is off and can't greet. ^_^")

bot1 = MySweetRobot("bob", "blue", 0.9)
bot2 = MySweetRobot("annae", "pink", 0.75)
bot3 = MySweetRobot("loan", "orange", 0.272)

print(bot1.name)

bot3.turn_on()
bot2.charge(1)
print(bot2.battery_level)

bot2.greet("aili")
bot2.turn_on()
bot2.greet("aili")
