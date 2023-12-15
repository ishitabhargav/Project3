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
        self.wire_to_cut = None
        added_red = False
        # if < 0.5, pick row first, otherwise pick col first
        row_or_col_first = random.random()
        #print(row_or_col_first)

        if row_or_col_first < 0.5:
            # 1. pick first row to color in
            rand_row_1 = np.random.randint(0, image_length)
            #print("picking first row: " + str(rand_row_1))
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
            for count in range(rand_col_1*4 + 1, len(self.vector), image_length * 4):
                self.vector[count] = rand_color_2[0]
                self.vector[count + 1] = rand_color_2[1]
                self.vector[count + 2] = rand_color_2[2]
                self.vector[count + 3] = rand_color_2[3]

            # 3. pick second row to color in
            rand_row_2 = np.random.randint(0, image_length)
            while rand_row_2 == rand_row_1:
                rand_row_2 = np.random.randint(0, image_length)
            rand_color_3 = self.colors.pop(np.random.randint(0, len(self.colors)))
            self.wire_to_cut = rand_color_3
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
            for count in range(rand_col_2*4 + 1, len(self.vector), image_length * 4):
                self.vector[count] = rand_color_4[0]
                self.vector[count + 1] = rand_color_4[1]
                self.vector[count + 2] = rand_color_4[2]
                self.vector[count + 3] = rand_color_4[3]

        else:
            # 1. pick first column to color in
            rand_col_1 = np.random.randint(0, image_length)
            #print("picking first col: " + str(rand_col_1))
            rand_color_1 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if rand_color_1 == red:
                self.colors.append(yellow)
                added_red = True
            for count in range(rand_col_1*4 + 1, len(self.vector), image_length * 4):
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
            self.wire_to_cut = rand_color_3
            if not added_red and rand_color_3 == red:
                self.colors.append(yellow)
                added_red = True
            for count in range(rand_col_2*4 + 1, len(self.vector), image_length * 4):
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

        # add information about the intersection of the colors. calculate sums of 2x2 areas and append them
        sum_list = []
        for big in range(0, len(self.vector) - 1, image_length * 8):
            for little in range(1, image_length * 4, 8):
                index_top_left = big + little
                index_top_right = index_top_left + 4
                index_bottom_left = index_top_left + image_length*4
                index_bottom_right = index_bottom_left + 4
                top_left = [self.vector[index_top_left], self.vector[index_top_left+1], self.vector[index_top_left+2], self.vector[index_top_left+3]]
                top_right = [self.vector[index_top_right], self.vector[index_top_right+1], self.vector[index_top_right+2], self.vector[index_top_right+3]]
                bottom_left = [self.vector[index_bottom_left], self.vector[index_bottom_left+1], self.vector[index_bottom_left+2], self.vector[index_bottom_left+3]]
                bottom_right = [self.vector[index_bottom_right], self.vector[index_bottom_right+1], self.vector[index_bottom_right+2], self.vector[index_bottom_right+3]]
                sum_square = []
                for i in range(4):
                    sum_square.append(top_left[i] + top_right[i] + bottom_left[i] + bottom_right[i])
                sum_list.append(sum_square)
        # self.vector = np.append(self.vector, quad_features2)
        for item in sum_list: # add each list of sums for each 2x2 window
            self.vector = np.append(self.vector, item)
        # print('s')


image = DangerousWiring()

'''for count in range(1, len(image.vector)):
    print(int(image.vector[count]), end=" ")
    if count % 80 == 0:
        print("")'''


'''image = WiringQuadFeatures()
num = len(image.vector)
for count in range(1, 1601):
    print(int(image.vector[count]), end=" ")
    if count % 80 == 0:
        print("")
for count in range(1601, len(image.vector)):
    print(int(image.vector[count]), end=" ")
print("")
print("classified as: " + str(image.is_dangerous))
print("length of input vector: " + str(num))'''