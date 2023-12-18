import sys

def get_parameters():

    try:
        arr = [float(i) for i in sys.argv[1:]]
        if (len(arr) == 3 and arr[0] != 0):
            return arr
        else:
            print("Error")
    except:
        print("Error")

    ans = 0
    while ans == 0:
        try:
            print("Enter 3 real numbers:")
            arr = list(map(float, input().split()))
            if len(arr) == 3 and arr[0] != 0:
                return arr
            else:
                print("Error")
                ans = 0
        except:
            print("Error")
            ans = 0

def solve(arr):
    sols = []
    d = float(arr[1]**2 - 4 * arr[0] * arr[2])
    if d == 0:
        x = -arr[1] / (2 * arr[0])
        sols.append(x)
    elif d > 0:
        x1 = (-arr[1] + d**0.5) / (2 * arr[0])
        x2 = (-arr[1] - d**0.5) / (2 * arr[0])
        sols.append(x1)
        sols.append(x2)

    return sols




def main():
    arr = get_parameters()
    sols = solve(arr)
    if (len(sols) == 2):
        print(f"X1 = {sols[0]}; X2 = {sols[1]}")
    elif (len(sols) == 1):
        print(f"X = {sols[0]}")
    else:
        print("No roots")




if '__main__' == __name__:
    main()