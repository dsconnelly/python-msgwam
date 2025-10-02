import sys

def main(fname):
    with open(fname) as f:
        lines = f.readlines()

    n = 1
    for line in lines:
        if '[' not in line:
            continue

        i = line.index('[')
        j = line.index(']')

        line = line[(i + 1):j]
        n = n * len(line.split(','))

    print(n - 1)

if __name__ == '__main__':
    main(sys.argv[1])