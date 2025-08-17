import numpy as np
import argparse
from pulp import *   #lp solver

parser = argparse.ArgumentParser()

def solve(filename, algo, policyfile):
    file = open(filename)
    lines = file.readlines()
    file.close()

    policy_lines = ""
    
    if not(policyfile == "not"):
        file = open(policyfile)
        policy_lines = file.readlines()
        file.close()

    # N_S = int( re.findall(r'\d+', lines[0])[0] ) 
    # N_A = int( re.findall(r'\d+', lines[1])[0] )
    # end = ( re.findall(r'\d+', lines[2]) )
    # gamma = float( (re.findall( r'[+-]?\d+\.*\d*', lines[-1]))[0] ) 

    # tpm = np.zeros((N_S, N_A, N_S))
    # rewards = np.zeros((N_S, N_A, N_S))


    # N_defs = len(lines) - 5
    # for i in range(3,N_defs+3) :
    #     arr = re.findall( r'[+-]?\d+\.*\d*', lines[i])
    #     a = int(arr[0])
    #     b = int(arr[1])
    #     c = int(arr[2])
    #     tpm[a, b, c] = float(arr[4])        #tpm[s][action][s']
    #     rewards[a, b, c] = float(arr[3])       

    N_S = int(lines[0].split()[-1])
    N_A = int(lines[1].split()[-1])
    end = lines[2].split()[1:]
    gamma = float(lines[-1].split()[-1])

    tpm = np.zeros((N_S, N_A, N_S))
    rewards = np.zeros((N_S, N_A, N_S))

        

    N_defs = len(lines) - 5
    for i in range(3,N_defs+3) :
        arr = lines[i].split()
        a = int(arr[1])
        b = int(arr[2])
        c = int(arr[3])
        tpm[a, b, c] = float(arr[5])        #tpm[s][action][s']
        rewards[a, b, c] = float(arr[4])  
        
    V = np.zeros(N_S)     #value function 
    A = np.ones(N_S, dtype = int)     #policy
    a = np.zeros(N_S)     # temporary variable for policy

    if len(policy_lines) > 0:
        for i in range(len(policy_lines)):
            A[i] = int(policy_lines[i])

        delta_pe = 0.1
        while(delta_pe>1e-11):
            sum_diff = 0
            vn = np.zeros(N_S)
            for p in range(0,N_S):
                vn[p] = V[p]

            for s in range(0, N_S): 
                V[s] = np.sum( tpm[s, A[s], :]*(rewards[s, A[s], :] + gamma*vn) )    

            delta_pe = np.sqrt(np.sum(np.square(vn - V))/N_S)
            
        for p in range(0,N_S):  #results
            print(V[p],"\t",A[p])


    if(algo == "vi" and len(policy_lines) < 1):
        delta = 0.1
        while(delta>1e-9):
            sum_diff = 0
            vn = np.zeros(N_S)
            for p in range(0,N_S):
                vn[p] = V[p]

            for s in range(0, N_S): 
                sum_fordiff_a = np.zeros(N_A)
                for action in range(0, N_A):
                    sum_fordiff_a[action] = np.sum( tpm[s, action, :]*(rewards[s, action, :] + gamma*vn) )    
                max_a = np.argmax(sum_fordiff_a)
                A[s] = max_a
                V[s] = np.max(sum_fordiff_a)

            # print(delta)

            delta = np.sqrt(np.sum(np.square(vn - V))/N_S)
            
        for p in range(0,N_S):  #results
            print(V[p],"\t",A[p])




    elif(algo == "hpi" and len(policy_lines) < 1):
        
        #policy improvement:
        policy_stable = False

        while(not(policy_stable)):
            #policy evaluation:
            delta_pe = 0.1
            while(delta_pe>1e-11):
                sum_diff = 0
                vn = np.zeros(N_S)
                for p in range(0,N_S):
                    vn[p] = V[p]

                for s in range(0, N_S): 
                    V[s] = np.sum( tpm[s, A[s], :]*(rewards[s, A[s], :] + gamma*vn) )    

                delta_pe = np.sqrt(np.sum(np.square(vn - V))/N_S)



            policy_stable = True

            for s in range(0, N_S): 
                a[s] = A[s]     #a = pi(s)
                sum_fordiff_a = np.zeros(N_A)
                for action in range(0, N_A):
                    sum_fordiff_a[action] = np.sum( tpm[s, action, :]*(rewards[s, action, :] + gamma*vn) )    
                max_a = np.argmax(sum_fordiff_a)
                A[s] = max_a

                if(a[s] != A[s]):
                    policy_stable = False



        for p in range(0,N_S):  #results
            print(V[p],"\t",A[p])




    elif(algo == "lp" and len(policy_lines) < 1):
        #string encoding for defining variables in PuLP
        S = [str(i) for i in range(N_S)]
        A = [str(i) for i in range(N_S)]

        #numerical encoding of states to access tpm, rewards
        S1 = np.arange(0,N_S, 1)
        A1 = np.arange(0,N_A, 1)

        #disabling PuLP messages
        LpSolverDefault.msg = False

        problem = LpProblem("mdp", LpMaximize)

        #defining the variable
        V = {}
        for s in S:
            V[s] = pulp.LpVariable(f"V{s}", lowBound = None)
        # print(V['1'])

        #the objective to be maximised, as formulated in class
        problem += (-lpSum(V[s] for s in S))

        #setting the Bellman constraints of linearity
        #V'(s) >= B*(V(s)), here V'(s) is the space that doesn't have a corresponding policy, so we approach the optimal policy
        #from this space, which shares a common point with the space in which each point has a corresponding policy. 
        #This common point is V_optimal
        for s in S1:
            for a in A1:
                problem += lpSum([p * (rewards[s, a, s_] + gamma * V[str(s_)]) for s_, p in enumerate(tpm[s, a])]) <= V[str(s)] 
        problem.solve()

        #extracting optimal V into a variable named 'V'
        V_optimal = np.zeros(N_S)
        for s in S:
            V_optimal[int(s)] = V[s].varValue
            if V[s].varValue == -0.0:
                V_optimal[int(s)] = 0.0
        V = V_optimal

        #finding the an optimal policy for the optimal value function
        for s in range(0, N_S): 
            sum_fordiff_a = np.zeros(N_A)
            for action in range(0, N_A):
                sum_fordiff_a[action] = np.sum( tpm[s, action, :]*(rewards[s, action, :] + gamma*V) )    
            A[s] = np.argmax(sum_fordiff_a)

        for p in range(0,N_S):  #results
            print(V[p],"\t",A[p])



    os.chdir("D:/CS747/Assignment2/")
    policyfile = open("value.txt", "w")
    for p in range(0,N_S): 
        policyfile.write(str(V[p])+"\t"+str(A[p])+str('\n'))
    policyfile.close()







if __name__ == "__main__":
    parser.add_argument("--mdp",type=str,default="D:/CS747/Assignment2/data/mdp/episodic-mdp-10-5.txt")
    parser.add_argument("--algorithm",type=str,default="vi")
    parser.add_argument("--policy",type=str,default="not")

    args = parser.parse_args()

    solve(args.mdp, args.algorithm, args.policy)

