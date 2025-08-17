import numpy as np 
import argparse
import math

parser = argparse.ArgumentParser()

def numbertoxy(b):
    b += 1
    x = (b-1) % 4
    y = 0
    if x == 3:
        y = b//4 - 1
    else:
        y = b//4
    return x,y 

def detect_diagonal(b1,b2,r):
    xb1, yb1 = numbertoxy(b1)
    xb2, yb2 = numbertoxy(b2)
    xr, yr = numbertoxy(r)

    #remember to subtract if sending to numbertoxy

    if ((yb1 - yb2) == (xb1 - xb2)) and ((yb1 - yr) == (xb1 - xr) and not(b2==b1) ):
        if ((b2<=r)and(r<=b1) or (b1<=r)and(r<=b2)):
            return True
    if ((yb1 - yb2) == -1*(xb1 - xb2)) and ((yb1 - yr) == -1*(xb1 - xr) and not(b2==b1) ):
        if ((b2<=r)and(r<=b1) or (b1<=r)and(r<=b2)):
            return True
    if r==b1 and r==b2:
            return True
    return False

def is_tackle(b1old, b2old, rold, b1, b2, r, possession):
    p_tackle = 1.0

    if ((possession-1)*b2 + (2- possession)*b1) == r:
        p_tackle = 0.5
    
    if (rold == ((possession-1)*b2 + (2- possession)*b1)) and (r == ((possession-1)*b2old + (2- possession)*b1old)):
        p_tackle = 0.5

    return p_tackle

def find_p(b1,b2,r, q, possession):
    xb1, yb1 = numbertoxy(b1)
    xb2, yb2 = numbertoxy(b2)
    xr, yr = numbertoxy(r)

    x_possess, y_possess = xb1, yb1
    if possession == 2:
        x_possess, y_possess = xb2, yb2

    p_pass = q - 0.1*max(abs(xb1-xb2),abs(yb1-yb2))
    if detect_diagonal(b1,b2,r):
        p_pass *= 0.50
        # if r>=0 and r<=15 and (r==b1 and r ==b2):
        # if b1+1 == 1 and r+1 == 7 and b2+1 == 12:
            # print(b1+1,r+1,b2+1)
    elif (yb1 == yr) and (yb2 == yr) and ((xb1 <= xr <= xb2) or (xb2 <= xr <= xb1)):
        p_pass *= 0.50
        # print(b1+1,r+1,b2+1)
    elif (xb1 == xr) and (xb2 == xr)and ((yb1 <= yr <= yb2) or (yb2 <= yr <= yb1)):
        p_pass *= 0.50
        # print(b1+1,r+1,b2+1)


    p_shoot = q - 0.2*(3.00 - x_possess)
    if (xr == 3 and yr == 1) or (xr == 3 and yr == 2):
        p_shoot *= 0.5
    return p_pass, p_shoot



def generate_write(state, file, b1,b2,r,possession, action_player, which_player, action_opponent, p, q, p_opponent):
    xb1, yb1 = numbertoxy(b1)
    xb2, yb2 = numbertoxy(b2)
    xr, yr = numbertoxy(r)
    b1old, b2old, rold = b1, b2, r
    
    p_sas = 0

    if action_opponent == 0:
        xr -= 1
    elif action_opponent == 1:
        xr += 1
    elif action_opponent == 2:
        yr -= 1
    else:
        yr += 1
    r = xr + 4*yr

    #player actions:
    action = 0
    reward = 0
    if which_player == "NP":
        possession = (2-possession)*2 + (possession - 1)*1
        p *= 0.50

    if action_player == "L": #left 
        if (possession - 1)*xb2 + (2 - possession)*xb1 - 1 >= 0:
            xb2 -= (possession - 1)
            xb1 -= (2 - possession)
            action = 0*(2-possession)+4*(possession-1)
            p_tackle = 1.00
            if which_player == "P":
                p_tackle = is_tackle(b1old, b2old, rold, b1 - (2 - possession),b2 - (possession - 1),r,possession)
            p_sas = (1-2.00*p)*p_tackle

    elif action_player == "R": #right 
        if (possession - 1)*xb2 + (2 - possession)*xb1 + 1 <= 3:
            xb2 += (possession - 1)
            xb1 += (2 - possession)
            action = 1*(2-possession)+5*(possession-1)
            p_tackle = 1.00
            if which_player == "P":
                p_tackle = is_tackle(b1old, b2old, rold, b1 + (2 - possession),b2 + (possession - 1),r,possession)
            p_sas = (1-2.00*p)*p_tackle

    elif action_player == "U": #up 
        if (possession - 1)*yb2 + (2 - possession)*yb1 - 1 >= 0:
            yb2 -= (possession - 1)
            yb1 -= (2 - possession)
            action = 2*(2-possession)+6*(possession-1)
            p_tackle = 1.00
            if which_player == "P":
                p_tackle = is_tackle(b1old, b2old, rold, b1 - 4*(2 - possession),b2 - 4*(possession - 1),r,possession)            
            p_sas = (1-2.00*p)*p_tackle

    elif action_player == "D": #down 
        if (possession - 1)*yb2 + (2 - possession)*yb1 + 1 <= 3:
            yb2 += (possession - 1)
            yb1 += (2 - possession)
            action = 3*(2-possession)+7*(possession-1)
            p_tackle = 1.00
            if which_player == "P":
                p_tackle = is_tackle(b1old, b2old, rold, b1 + 4*(2 - possession),b2 + 4*(possession - 1),r,possession)
            p_sas = (1-2.00*p)*p_tackle
    
    #need to revert the changes made in possession because we initially changed it to use it for appropriate movement, but no real change in possession has occured
    if which_player == "NP":
        possession = (2-possession)*2 + (possession - 1)*1   
    
    b2 = xb2 + 4*yb2
    b1 = xb1 + 4*yb1
    
    state_ = (b1) + (b2)*16 + (r)*256 + 4096*(possession-1)
    
    p_pass, p_shoot = find_p(b1,b2,r,q,possession)
    #writing shoot and pass after calculating the state based on position, because position doesn't change in these actions
    if action_player == "S":#shoot
        action = 9
        state_ = 8193
        p_sas = p_shoot
        reward = 1

    elif action_player == "P":#pass
        action = 8
        possession = (2-possession)*2 + (possession - 1)*1
        state_ = (b1) + (b2)*16 + (r)*256 + 4096*(possession-1)
        p_sas = p_pass

    p_sas = p_sas*p_opponent
    if not(p_sas < 1e-15):
        file.write("transition "+str(int(state))+" "+ str(action)+ " "+ str(int(state_))+" " + str(reward)+ " "+ str(p_sas)+"\n")
        file.write("transition "+str(int(state))+" "+ str(action)+ " "+ str(8192)+" " + str(0)+ " "+ str(1-p_sas)+"\n") 
        # print("transition "+str(int(state))+" "+ str(action)+ " "+ str(int(state_))+" " + str(reward)+ " "+ str(p_sas))
        # print("transition "+str(int(state))+" "+ str(action)+ " "+ str(8192)+" " + str(0)+ " "+ str(1-p_sas)) 



def encode(filename, p, q):
    file = open(filename,'r')
    lines = file.readlines()
    file.close()


    file = open("D:/CS747/Assignment2/football_mdp.txt", "w") 
    #8192 is lose, 8193 is win
    file.write("numStates 8194\n")
    file.write("numActions 10\n")
    file.write("end 8192 8193\n")
    print("numStates 8194")
    print("numActions 10")
    print("end 8192 8193")

    #encoding now
    for line in lines:
        vals = line.split()
        if vals[0] == 'state':
            continue

        state = vals[0]
        b1 = int(state[:2])
        b2 = int(state[2:4])
        r = int(state[4:6])
        #The b1,b2,r here need to be subtracted by 1 to be processed further
        possession = int(state[6])        
        state = (b1-1) + (b2-1)*16 + (r-1)*256 + 4096*(possession-1)

        L,R,U,D = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
        
        #move possessor, move un-possessor, pass, shoot
        for opponent_action, opponent_action_prob in enumerate([L,R,U,D]):
            #moving 
            for player in ["P","NP"]:
                for direction in ["L","R","U","D"]:
                    generate_write(state, file, b1-1, b2-1, r-1, possession, direction, player, opponent_action, p, q, opponent_action_prob)

            #pass/shoot
            for action in ["S","P"]:
                generate_write(state, file, b1-1, b2-1, r-1, possession, action, "P", opponent_action, p, q, opponent_action_prob)
            


    file.write("mdptype episodic\n")
    file.write("discount 1.0")
    print("mdptype episodic")
    print("discount 1.0")
    file.close()

            



if __name__ == "__main__":
    parser.add_argument("--opponent",type=str,default="D:/CS747/Assignment2/data/football/test-1.txt")
    parser.add_argument("--p",type=float,default=0.1)
    parser.add_argument("--q",type=float,default=0.7)

    args = parser.parse_args()

    encode(args.opponent, args.p, args.q)