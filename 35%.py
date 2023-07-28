from random import randint

def p243():

    #if a number x < n is relatively prime to n, then x/n is resilliant
    #the number is NOT relatively prime to n, then x/n is NOT resilliant

    #therefore R(n) = e(n)/(n-1)
    #   where e(n) is euler's totient
    #recall that if n = p1^k1*p2^k2*...*pm^km, then e(n) = p1^(k1-1)(p1-1)*...*pm^(km-1)(pm-1)
    #so the ideal d likely has many small factors.
    #   if d prime, then R(n) = 1
    #   e.g., if d = 2813 = 97*29, then R(n) = 98*28/2813 = 2744/2813 which is about 0.95, way larger than our intended answer of about 0.16
    #   12 = 4*3 = 2*2*3 -> e(12) = 2*1*2 = 4 -> R(n) = 4/11

    # was able to surmise the answer from here, no code needed
    answer = 892371480 # = 2^3*3*5*7*11*13*17*19*23

def p84():

    #idea: set up all the rules and run n turns. then take (number of times visiting square x)/n for an estimated probability of landing on x
    #that gets more accurate as n increases

    #1) squares listed 0-39, so add the roll mod 40
    #2) i am gonna choose to select a random card from the deck instead of cycling through a chosen shuffle (this may prove incorrect)
    #3) JAIL = 10, GJ2 = 30, CC = 2, 17, 33, CH = 7, 22, 36
    #4) CC outcomes: GO, JAIL (1/16 chance each)
    #5) CH outcomes: GO, JAIL, 11, 24, 39, 5 (1/16 chance each)
    #              OR advance to next railroad: 7->15 , 22->25, 36->5 (2/16 chance)
    #              OR advance to next utility: 7->12 , 22->28, 36->12 (1/16 chance)
    #              OR go back 3 squares (1/16 chance)

    # lowest of (12, 28) lower than current square (1/16 chance)

    sides = 6
    cts = [0]*40
    n = 10**6
    sq = 0

    JAIL = 10
    CCs = [2,17,33]
    CHs = [7,22,36]
    GO = 0

    for i in range(n):

        roll = randint(1, sides)
        sq = (sq + roll)%40

        if sq == 30: sq = JAIL
        elif sq in CCs:
            card = randint(1, 16)
            if card == 1: sq = GO
            if card == 2: sq = JAIL
        elif sq in CHs:
            card = randint(1,16)
            if card == 1: sq = GO
            if card == 2: sq = JAIL
            if card == 3: sq = 11
            if card == 4: sq = 24
            if card == 5: sq = 39
            if card == 6: sq = 5
            if card == 7: sq -= 3
            if card == 8:
                if sq == 7: sq = 12
                if sq == 22: sq = 28
                if sq == 36: sq = 12
            if card == 9 or card == 10:
                if sq == 7: sq = 15
                if sq == 22: sq = 25
                if sq == 36: sq = 5

        cts[sq] += 1

    
    for i in range(40):
        print(i, ": ", 100*cts[i]/n, "%")

if __name__ == '__main__':
    p84()