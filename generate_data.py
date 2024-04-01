import random
import numpy as np

def generate_data(n, lo=0, hi=100):
    pairs = np.random.randint(lo, hi, (n, 2))
    sums = pairs.sum(1)
    return pairs, sums


if __name__ == "__main__":
    # general
    pairs, sums = generate_data(int(5e6), 0, 10000)
    pairs4, sums4 = generate_data(int(1e6), 0, 1000)

    # specifically targetting smaller numbers
    pairs2, sums2 = generate_data(int(1e4), 0, 100)
    pairs3, sums3 = generate_data(20, 0, 10)

    pairs = np.concatenate((pairs, pairs2, pairs3, pairs4))
    sums = np.concatenate((sums, sums2, sums3, sums4))
    # pairs, sums = generate_data(int(1e4), 0, 1000)

    text = "\n"
    for (first, second), out in zip(pairs, sums):
        # reverse the string so the model learns to do addition 1 decimal place at a time
        # this mirrors how we do it starting at lowest decimal place then moving higher
        text += f"{first} + {second} = {str(out)[::-1]}\n"

    with open('data.txt', "w") as file:
        file.write(text)
    # with open('small_data.txt', "w") as file:
    #     file.write(text)
