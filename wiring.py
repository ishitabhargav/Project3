import random
import numpy as np


class Wiring:
    def __init__(self):
        image_length = 20
        self.vector = np.zeros(image_length ** 2 * 4 + 1)  # each pixel represented by 4 digits
        self.vector[0] = 1  # first index of entire vector/image should be 1
        red = [1, 0, 0, 0]
        blue = [0, 1, 0, 0]
        yellow = [0, 0, 1, 0]
        green = [0, 0, 0, 1]
        self.colors = [red, blue, yellow, green]

        self.is_dangerous = 0  # false
        # if < 0.5, pick row first, otherwise pick col first
        row_or_col_first = random.random()
        # print(row_or_col_first)

        if row_or_col_first < 0.5:
            # 1. pick first row to color in
            rand_row_1 = np.random.randint(0, image_length)
            #print("picking first row: " + str(rand_row_1))
            rand_color_1 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if rand_color_1 == red:
                self.is_dangerous = 1  # true
            for count in range(rand_row_1 * (image_length * 4) + 1,
                               rand_row_1 * (image_length * 4) + (image_length * 4) + 1, 4):
                self.vector[count] = rand_color_1[0]
                self.vector[count + 1] = rand_color_1[1]
                self.vector[count + 2] = rand_color_1[2]
                self.vector[count + 3] = rand_color_1[3]

            # 2. pick first column to color in
            rand_col_1 = np.random.randint(0, image_length)
            #print("picking first col: " + str(rand_col_1 * 4))
            rand_color_2 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not self.is_dangerous and rand_color_1 != yellow and rand_color_2 == red:
                self.is_dangerous = 1  # true
            for count in range(rand_col_1 * 4 + 1, len(self.vector), image_length * 4):
                self.vector[count] = rand_color_2[0]
                self.vector[count + 1] = rand_color_2[1]
                self.vector[count + 2] = rand_color_2[2]
                self.vector[count + 3] = rand_color_2[3]

            # 3. pick second row to color in
            rand_row_2 = np.random.randint(0, image_length)
            while rand_row_2 == rand_row_1:
                rand_row_2 = np.random.randint(0, image_length)
            #print("picking second row: " + str(rand_row_2))
            rand_color_3 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not self.is_dangerous and rand_color_1 != yellow and rand_color_2 != yellow and rand_color_3 == red:
                self.is_dangerous = 1
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
            #print("picking second col: " + str(rand_col_2 * 4))
            rand_color_4 = self.colors.pop(np.random.randint(0, len(self.colors)))
            for count in range(rand_col_2 * 4 + 1, len(self.vector), image_length * 4):
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
                self.is_dangerous = 1
            for count in range(rand_col_1 * 4 + 1, len(self.vector), image_length * 4):
                self.vector[count] = rand_color_1[0]
                self.vector[count + 1] = rand_color_1[1]
                self.vector[count + 2] = rand_color_1[2]
                self.vector[count + 3] = rand_color_1[3]

            # 2. pick first row to color in
            rand_row_1 = np.random.randint(0, image_length)
            rand_color_2 = self.colors.pop(np.random.randint(0, len(self.colors)))
            if not self.is_dangerous and rand_color_1 != yellow and rand_color_2 == red:
                self.is_dangerous = 1
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
            if not self.is_dangerous and rand_color_1 != yellow and rand_color_2 != yellow and rand_color_3 == red:
                self.is_dangerous = 1
            for count in range(rand_col_2 * 4 + 1, len(self.vector), image_length * 4):
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

        # add quadratic features to the input vector by multiplying each neighboring pair of pixels together
        '''quadratic_features = []
        for count in range(1, len(self.vector), 4):
            current = [self.vector[count], self.vector[count+1], self.vector[count+2], self.vector[count+3]]
            if count - 4 > 0 and (count - 1) % 80 != 0: # if there is a pixel to its left
                left = [self.vector[count-4], self.vector[count-3], self.vector[count-2], self.vector[count-1]]
                dot_product = np.dot(current, left)
                quadratic_features.append(dot_product)
            if count + 4 < len(self.vector) and (count + 3) % 80 != 0: # if there is a pixel to its right
                right = [self.vector[count+4], self.vector[count+5], self.vector[count+6], self.vector[count+7]]
                dot_product = np.dot(current, right)
                quadratic_features.append(dot_product)
            bottom_index = count + (image_length*4)
            if bottom_index < len(self.vector): # if there is a pixel directly below it
                bottom = [self.vector[bottom_index], self.vector[bottom_index+1], self.vector[bottom_index+2], self.vector[bottom_index+3]]
                dot_product = np.dot(current, bottom)
                quadratic_features.append(dot_product)
            top_index = count - (image_length*4)
            if top_index > 0: # if there is a pixel directly above it
                top = [self.vector[top_index], self.vector[top_index+1], self.vector[top_index+2], self.vector[top_index+3]]
                dot_product = np.dot(current, top)
                quadratic_features.append(dot_product)
        self.vector = np.append(self.vector, quadratic_features)'''


        # add quadratic features by taking the dot product of pairs of neighboring pixels and appending the dot product
        # of those 2 dot products
        # calculate dot product of groups of 4 in rows
        quad_features2 = []
        for count in range(1, len(self.vector), 16):
            first = [self.vector[count], self.vector[count+1], self.vector[count+2], self.vector[count+3]]
            second = [self.vector[count+4], self.vector[count+5], self.vector[count+6], self.vector[count+7]]
            third = [self.vector[count+8], self.vector[count+9], self.vector[count+10], self.vector[count+11]]
            fourth = [self.vector[count + 12], self.vector[count + 13], self.vector[count + 14], self.vector[count + 15]]
            dot_prod1 = np.dot(first, second)
            dot_prod2 = np.dot(third, fourth)
            dot_prod3 = np.dot(dot_prod1, dot_prod2)
            quad_features2.append(dot_prod3)

        # calculate dot product of groups of 4 in columns
        for big in range(0, len(self.vector) - 1, image_length*16):
            for little in range(1, image_length*4, 4):
                first = [self.vector[little+big], self.vector[little+big+1], self.vector[little+big+2], self.vector[little+big+3]]
                second_index = little + big + image_length*4 # 81
                second = [self.vector[second_index], self.vector[second_index+1], self.vector[second_index+2], self.vector[second_index+3]]
                third_index = little + big + image_length*8 # 161
                third = [self.vector[third_index], self.vector[third_index+1], self.vector[third_index+2], self.vector[third_index+3]]
                fourth_index = little + big + image_length*12 # 241
                fourth = [self.vector[fourth_index], self.vector[fourth_index+1], self.vector[fourth_index+2], self.vector[fourth_index+3]]
                dot_prod_pair1 = np.dot(first, second)
                dot_prod_pair2 = np.dot(third, fourth)
                dot_prod = np.dot(dot_prod_pair1, dot_prod_pair2)
                quad_features2.append(dot_prod)
        self.vector = np.append(self.vector, quad_features2)


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
