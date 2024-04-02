from generate_data import generate_data


pairs, sums = generate_data(int(1e5), 0, 100)

text = "\n"
for (first, second), out in zip(pairs, sums):
    # reverse the string so the model learns to do addition 1 decimal place at a time
    # this mirrors how we do it starting at lowest decimal place then moving higher
    text += f"{first} + {second} = {str(out)[::-1]}\n"

with open('xs_data.txt', "w") as file:
    file.write(text)

