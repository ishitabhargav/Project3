import random
import numpy as np


class DangerousWiring:
    def __init__(self):
        image_length = 20
        self.vector = np.zeros(image_length ** 2 * 4 + 1)  # each pixel represented by 4 digits
        self.vector[0] = 1  # first index of entire vector/image should be 1
        red = [1, 0, 0, 0]
        blue = [0, 1, 0, 0]
        yellow = [0, 0, 1, 0]
        green = [0, 0, 0, 1]
        self.colors = [red, blue, green]
        added_red = False
        # if < 0.5, pick row first, otherwise pick col first
        row_or_col_first = random.random()
        print(row_or_col_first)

        if row_or_col_first < 0.5:
            # 1. pick first row to color in
            rand_row_1 = np.random.randint(0, image_length)
            print("picking first row: " + str(rand_row_1))
            rand_color_1 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if rand_color_1 == red:  # if we selected red
                self.colors.append(yellow)
                added_red = True
            for count in range(rand_row_1 * (image_length * 4) + 1,
                               rand_row_1 * (image_length * 4) + (image_length * 4) + 1, 4):
                self.vector[count] = rand_color_1[0]
                self.vector[count + 1] = rand_color_1[1]
                self.vector[count + 2] = rand_color_1[2]
                self.vector[count + 3] = rand_color_1[3]

            # 2. pick first column to color in
            rand_col_1 = np.random.randint(0, image_length)
            rand_color_2 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not added_red and rand_color_2 == red:
                self.colors.append(yellow)
                added_red = True
            for count in range(rand_col_1 + 1, len(self.vector), image_length * 4):
                self.vector[count] = rand_color_2[0]
                self.vector[count + 1] = rand_color_2[1]
                self.vector[count + 2] = rand_color_2[2]
                self.vector[count + 3] = rand_color_2[3]

            # 3. pick second row to color in
            rand_row_2 = np.random.randint(0, image_length)
            while rand_row_2 == rand_row_1:
                rand_row_2 = np.random.randint(0, image_length)
            rand_color_3 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not added_red and rand_color_3 == red:
                self.colors.append(yellow)
                added_red = True
            for count in range(rand_row_2 * (image_length * 4) + 1,
                               rand_row_2 * (image_length * 4) + (image_length * 4) + 1, 4):
                self.vector[count] = rand_color_3[0]
                self.vector[count + 1] = rand_color_3[1]
                self.vector[count + 2] = rand_color_3[2]
                self.vector[count + 3] = rand_color_3[3]

            # 4. pick second column to color in
            rand_col_2 = np.random.randint(0, image_length)
            while rand_col_2 == rand_col_1:
                rand_col_2 = np.random.randint(0, image_length)
            rand_color_4 = self.colors.pop(np.random.randint(0, len(self.colors)))
            for count in range(rand_col_2 + 1, len(self.vector), image_length * 4):
                self.vector[count] = rand_color_4[0]
                self.vector[count + 1] = rand_color_4[1]
                self.vector[count + 2] = rand_color_4[2]
                self.vector[count + 3] = rand_color_4[3]

        else:
            # 1. pick first column to color in
            rand_col_1 = np.random.randint(0, image_length)
            print("picking first col: " + str(rand_col_1))
            rand_color_1 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if rand_color_1 == red:
                self.colors.append(yellow)
                added_red = True
            for count in range(rand_col_1 + 1, len(self.vector), image_length * 4):
                self.vector[count] = rand_color_1[0]
                self.vector[count + 1] = rand_color_1[1]
                self.vector[count + 2] = rand_color_1[2]
                self.vector[count + 3] = rand_color_1[3]

            # 2. pick first row to color in
            rand_row_1 = np.random.randint(0, image_length)
            rand_color_2 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not added_red and rand_color_2 == red:
                self.colors.append(yellow)
                added_red = True
            for count in range(rand_row_1 * (image_length * 4) + 1,
                               rand_row_1 * (image_length * 4) + (image_length * 4) + 1, 4):
                self.vector[count] = rand_color_2[0]
                self.vector[count + 1] = rand_color_2[1]
                self.vector[count + 2] = rand_color_2[2]
                self.vector[count + 3] = rand_color_2[3]

            # 3. pick second column to color in
            rand_col_2 = np.random.randint(0, image_length)
            while rand_col_2 == rand_col_1:
                rand_col_2 = np.random.randint(0, image_length)
            rand_color_3 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not added_red and rand_color_3 == red:
                self.colors.append(yellow)
                added_red = True
            for count in range(rand_col_2 + 1, len(self.vector), image_length * 4):
                self.vector[count] = rand_color_3[0]
                self.vector[count + 1] = rand_color_3[1]
                self.vector[count + 2] = rand_color_3[2]
                self.vector[count + 3] = rand_color_3[3]

            # 4. pick second row to color in
            rand_row_2 = np.random.randint(0, image_length)
            while rand_row_2 == rand_row_1:
                rand_row_2 = np.random.randint(0, image_length)
            rand_color_4 = self.colors.pop(np.random.randint(0, len(self.colors)))
            for count in range(rand_row_2 * (image_length * 4) + 1,
                               rand_row_2 * (image_length * 4) + (image_length * 4) + 1, 4):
                self.vector[count] = rand_color_4[0]
                self.vector[count + 1] = rand_color_4[1]
                self.vector[count + 2] = rand_color_4[2]
                self.vector[count + 3] = rand_color_4[3]


image = DangerousWiring()

for count in range(1, len(image.vector)):
    print(int(image.vector[count]), end=" ")
    if count % 80 == 0:
        print("")
