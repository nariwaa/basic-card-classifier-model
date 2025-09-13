class MySweetRobot:
    def __init__(self, name, color, battery_level):
        # 'self' refers to the specific robot object we're currently building!
        self.name = name # this robot's name
        self.color = color # this robot's color
        self.battery_level = battery_level # this robot's battery
        self.is_on = False # all new robots start off!
        print(f"heya! {self.name} the {self.color} robot has been built! o/")

beep = MySweetRobot("bob", "blue", 0.9)
