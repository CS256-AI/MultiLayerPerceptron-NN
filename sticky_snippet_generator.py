import random
import numpy as np


class Data:
    def __init__(self):
        self.items = ['A', 'B', 'C', 'D']
        self.item_len = 40
        self.sticky_dict = {'A': ['C'],
                            'B': ['D'],
                            'C': ['A'],
                            'D': ['B']}

    def is_stick_palindrome(self, input):
        length = len(input)
        if length % 2 == 1:
            return False
        return self.check_stickiness(input[:length/2], input[length-1:length/2-1:-1])

    def check_stickiness(self, input1, input2):
        k = 0
        if len(input1) != len(input2):
            return k
        for i in range(len(input1)):
            if input1[i] not in self.items:
                print('Invalid data : Data should consist of "ABCD"')
            if input2[i] not in self.sticky_dict[input1[i]]:  # Checks whether reverse of second half is sticky
                return k
            k += 1
        return k

    def gen_stick_palindrome(self):
        data = self.item_len * [None]
        for i in range(self.item_len/2):
            item = random.choice(self.items)
            item_stick = random.choice(self.sticky_dict[item])
            # print('item : {}'.format(item))
            # print('item : {}'.format(item_stick))
            data[i] = item
            data[self.item_len - 1 - i] = item_stick
        return data

    def mutate_stick_pal(self, stick_pal, mutation_rate, from_ends):
        result = []
        len_pal = len(stick_pal)
        for i in range(len_pal):
            selected_char = stick_pal[i]
            print('(i,selected_char): {}'.format((i, selected_char)))
            if i in range(from_ends) or i in range(len_pal - from_ends, len_pal):
                prob = [(mutation_rate / (len(self.items) - 1)) if char != selected_char else 1 - mutation_rate for char
                        in self.items]
            else:
                prob = [(1.0 / (len(self.items) - 1)) if char != selected_char else 0.0 for char in self.items]
            print(prob)
            result.append(np.random.choice(a=self.items, p=prob))
        return ''.join(result)

    def gen_data(self, num_snippets, mutation_rate, from_ends, output_file):
        data = []
        with open(output_file, 'w') as f:
            for i in range(num_snippets):
                stick_pal = self.gen_stick_palindrome()
                result = self.mutate_stick_pal(stick_pal, mutation_rate, from_ends)
                if i != num_snippets - 1:
                    result = result + "\n"
                data.append(result)
                print(data)
                print(len(data))
            f.writelines(data)

data_gen = Data()
# print(data.check_stickiness('AABDC', 'CCDBA'))
# print(data.is_stick_palindrome('AABDCDBDCC'))
# print(data_gen.gen_stick_palindrome())
print(data_gen.gen_data(num_snippets = 1000, mutation_rate = 0.6, from_ends = 4, output_file = 'data.txt'))
