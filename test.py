import os

f = open("output.txt", "a")
for elem in os.listdir("data/eval"):
    try:
        print(int(elem))
        f.write(elem + "\n")
    except:
        print("and exception", elem)
