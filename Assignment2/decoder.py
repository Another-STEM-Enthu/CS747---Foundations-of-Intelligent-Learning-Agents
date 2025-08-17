import numpy as np 
import argparse
import math

parser = argparse.ArgumentParser()

def decode(opponentlines, policylines):
    V = np.zeros(len(opponentlines)-1, dtype = float)
    A = np.zeros(len(opponentlines)-1, dtype = int)


    for idx in range(len(policylines)-2):
        arr = policylines[idx].split()
        # possession = idx // 4096 + 1
        # idx %= 4096
        # r = idx // 256 + 1
        # idx %= 256
        # b2 = idx // 16 + 1
        # b1 = idx % 16  + 1     
        # b1 = "{:02d}".format(b1)
        # b2 = "{:02d}".format(b2)
        # r = "{:02d}".format(r)
            # state = str(b1)+str(b2)+str(r)+str(possession)
        V[idx] = float(arr[0])
        A[idx] = int(arr[1])
    # print(np.max(V))

    file = open("D:/CS747/Assignment2/policy.txt", "w")

    for line in opponentlines:
        state = (line.split())[0]
        if state == "state":
            continue
        b1 = int(state[:2])
        b2 = int(state[2:4])
        r = int(state[4:6])
        possession = int(state[6])        
        state_idx = (b1-1) + (b2-1)*16 + (r-1)*256 + 4096*(possession-1)
        file.write(str(state)+" "+str(A[state_idx])+" "+str(V[state_idx])+"\n")
        print(str(state)+" "+str(A[state_idx])+" "+str(V[state_idx]))

    file.close()

if __name__ == "__main__":
    parser.add_argument("--opponent",type=str,default="D:/CS747/Assignment2/data/football/test-1.txt")
    parser.add_argument("--value-policy",type=str,default="D:/CS747/Assignment2/value.txt")
    args = parser.parse_args()

    file = open(args.opponent,"r")
    opponentlines = file.readlines()    
    file.close()

    file = open(args.value_policy,"r")
    policylines = file.readlines() 
    file.close()

    decode(opponentlines, policylines)