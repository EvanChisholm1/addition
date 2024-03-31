import random
import numpy as np

def generate_data(n, lo=0, hi=100):
    pairs = np.random.randint(lo, hi, (n, 2))
    sums = pairs.sum(1)
    return pairs, sums

if __name__ == "__main__":
    pairs, sums = generate_data(int(1e7), 0, 1000)

    text = "\n"
    for (first, second), out in zip(pairs, sums):
        text += f"{first} + {second} = {out}\n"

    with open('data.txt', "w") as file:
        file.write(text)
