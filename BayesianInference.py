
def getPosterior(priorOfA, priorOfB, likelihood):
	
##################################################
#		Your code here
##################################################
    
    marginalOfA = {}
    marginalOfB = {}
    
    # compute marginal of A
    total = 0
    for event_A in priorOfA:
        prob_b = 0
        for event_B in priorOfB:
            prob_b = prob_b + likelihood[event_A, event_B]
        
        marginalOfA[event_A] = priorOfA[event_A] * prob_b
        total = total + marginalOfA[event_A]
    
    # normalize to 1
    for event_A in marginalOfA:
        marginalOfA[event_A] = marginalOfA[event_A]/total
    
    # compute marginal of B
    total = 0
    for event_B in priorOfB:
        prob_a = 0
        for event_A in priorOfA:
            prob_a = prob_a + likelihood[event_A, event_B]
      
        marginalOfB[event_B] = priorOfB[event_B] * prob_a
        total = total + marginalOfB[event_B]
    
    # normalize to 1
    for event_B in marginalOfB:
        marginalOfB[event_B] = marginalOfB[event_B]/total
                
    return([marginalOfA, marginalOfB])



def main():
    exampleOnePriorofA = {'a0': .5, 'a1': .5}
    exampleOnePriorofB = {'b0': .25, 'b1': .75}
    exampleOneLikelihood = {('a0', 'b0'): 0.42, ('a0', 'b1'): 0.12, ('a1', 'b0'): 0.07, ('a1', 'b1'): 0.02}
    print(getPosterior(exampleOnePriorofA, exampleOnePriorofB, exampleOneLikelihood))

    exampleTwoPriorofA = {'red': 1/10 , 'blue': 4/10, 'green': 2/10, 'purple': 3/10}
    exampleTwoPriorofB = {'x': 1/5, 'y': 2/5, 'z': 2/5}
    exampleTwoLikelihood = {('red', 'x'): 0.2, ('red', 'y'): 0.3, ('red', 'z'): 0.4, ('blue', 'x'): 0.08, ('blue', 'y'): 0.12, ('blue', 'z'): 0.16, ('green', 'x'): 0.24, ('green', 'y'): 0.36, ('green', 'z'): 0.48, ('purple', 'x'): 0.32, ('purple', 'y'): 0.48, ('purple', 'z'): 0.64}
    print(getPosterior(exampleTwoPriorofA, exampleTwoPriorofB, exampleTwoLikelihood))


if __name__ == '__main__':
    main()

    

