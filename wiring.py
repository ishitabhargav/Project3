import random
import numpy as np


class Wiring:
    def __init__(self):
        imageLength = 20
        self.vector = np.zeros(imageLength ** 2)
        # red = 1, blue = 2, yellow = 3, green = 4
        self.colors = [1, 2, 3, 4]
        self.isDangerous = 0 # false
        # if < 0.5, pick row first, otherwise pick col first
        rowOrCol = random.random()
        print(rowOrCol)

        if rowOrCol < 0.5:
            # 1. pick first row to color in
            randRow1 = np.random.randint(0, 20)
            print("picking first row: " + str(randRow1))
            randColor1 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if randColor1 == 1:  # if we selected red
                self.isDangerous = 1 # true
            for count in range(randRow1 * 20, randRow1 * 20 + 20):
                self.vector[count] = randColor1

            # 2. pick first column to color in
            randCol1 = np.random.randint(0, 20)
            randColor2 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not self.isDangerous and randColor1 != 3 and randColor2 == 1:
                self.isDangerous = 1 # true
            for count in range(randCol1, len(self.vector), 20):
                self.vector[count] = randColor2

            # 3. pick second row to color in
            randRow2 = np.random.randint(0, 20)
            while randRow2 == randRow1:
                randRow2 = np.random.randint(0, 20)
            randColor3 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not self.isDangerous and randColor1 != 3 and randColor2 != 3 and randColor3 == 1:
                self.isDangerous = 1
            for count in range(randRow2 * 20, randRow2 * 20 + 20):
                self.vector[count] = randColor3

            # 4. pick second column to color in
            randCol2 = np.random.randint(0, 20)
            while randCol2 == randCol1:
                randCol2 = np.random.randint(0, 20)
            randColor4 = self.colors.pop(np.random.randint(0, len(self.colors)))
            for count in range(randCol2, len(self.vector), 20):
                self.vector[count] = randColor4

        else:
            # 1. pick first column to color in
            randCol1 = np.random.randint(0, 20)
            print("picking first col: " + str(randCol1))
            randColor1 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if randColor1 == 1:  # if we selected red
                self.isDangerous = 1
            for count in range(randCol1, len(self.vector), 20):
                self.vector[count] = randColor1

            # 2. pick first row to color in
            randRow1 = np.random.randint(0, 20)
            randColor2 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not self.isDangerous and randColor1 != 3 and randColor2 == 1:
                self.isDangerous = 1
            for count in range(randRow1 * 20, randRow1 * 20 + 20):
                self.vector[count] = randColor2

            # 3. pick second column to color in
            randCol2 = np.random.randint(0, 20)
            while randCol2 == randCol1:
                randCol2 = np.random.randint(0, 20)
            randColor3 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not self.isDangerous and randColor1 != 3 and randColor2 != 3 and randColor3 == 1:
                self.isDangerous = 1
            for count in range(randCol2, len(self.vector), 20):
                self.vector[count] = randColor3

            # 4. pick second row to color in
            randRow2 = np.random.randint(0, 20)
            while randRow2 == randRow1:
                randRow2 = np.random.randint(0, 20)
            randColor4 = self.colors.pop(np.random.randint(0, len(self.colors)))
            for count in range(randRow2 * 20, randRow2 * 20 + 20):
                self.vector[count] = randColor4


image = Wiring()

for count in range(400):
    print(int(image.vector[count]), end=" ")
    if (count + 1) % 20 == 0:
        print("")
print(image.isDangerous)
