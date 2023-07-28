import numpy as np
import math
import time

def p13():

    numlen = 50

    #50 digit numbers
    sums = np.zeros(numlen + 2)

    with open('C:/Users/whous/Euler/p13.txt') as file:
        for line in file:
            for i in range(numlen):
                sums[numlen - i - 1] += int(line[i])

    #now have sums of all the numbers
    #make bigger array for final number
    print(sums)
    #fin = np.zeros(52)
    for i in range(numlen):
        n = sums[i]
        hundreds = n // 100
        tens = (n - 100*hundreds) // 10
        ones = n - 100*hundreds - 10*tens

        sums[i] = ones
        sums[i+1] += tens
        sums[i+2] += hundreds
    
    print(sums)
    print(sums[-10:])

def p15():
    #calculate 40 choose 20
    n = 20

    #0->39
    num = 1
    for i in range(2*n):
        num *= (i+1)
    denom = 1
    for i in range(n):
        denom *= (i+1)
    denom *= denom

    print(num/denom)
    
def p16():
    ans = 0
    for c in str(2**1000):
        ans += int(c)
    print(ans)

def p17():
    #one, two, three, four, five, six, seven, eight, nine
    #twenty, thirty, forty, fifty, sixty, seventy, eighty, ninety
    #hundred
    #thousand
    #ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen

    #first 99: 9*(sum of one thru nine) + 10*(sum of twenty thru ninety) + (sum of eleven thru nineteen) + ten
    # (n-1)00 to (n)99: first 99 + (100)*(n hundred) => 10*first 99 + 90*(one to nine) + 900*(hundred) + 891*(and)

    #all: first 99 + (100 to 199) + ... + (900 to 999) + (one thousand
    
    #most importantly, find value of first 99
    oneToNine = 3 + 3 + 5 + 4 + 4 + 3 + 5 + 5 + 4
    tenToNineteen = 3 + 6 + 6 + 8 + 8 + 7 + 7 + 9 + 8 + 8
    twentyToNinety = 6 + 6 + 5 + 5 + 5 + 7 + 6 + 6

    hundred = 7
    onethousand = 11
    ampersand = 3

    #one to nine repeats nine times
    #each twenty to ninety repeats 10 times
    # special case ten to nineteen happens once
    first99 = 9*oneToNine + tenToNineteen + 10*twentyToNinety

    #first 99 repeats 10 times
    #each 'hundred' is prefixed with each one through nine 100 times
    #   'hundred' appears one hundred times per one through nine, for 900 total times
    #   'hundred' appears without 'and' nine times
    ans = 10*first99 + 100*oneToNine + 900*hundred + onethousand + 891*ampersand

    print(ans)

def p18():
    #idea: picking a direction eliminates an entire straight-line path of values from ever being touched
    #   pick route that eliminates smallest values?
    #   this would be a greedy algorithm
    #Doesn't work -- counterexample:
    #        0
    #    1        2
    #  2     2       0
    #
    # optimal path is 4, this algorithm picks 3
    #seems a greedy algorithm is unlikely to work
    #
    #Divide & counquer algorithm?

    #triangle data structure >:)
    triangle = []

    with open('C:/Users/whous/Euler/p67.txt') as file:
        for line in file:
            for n in line.split():
                triangle.append(int(n))
    
    depth = 99
    sums = []

    for i in range(depth + 2):
        sums.append(sum(range(i)))

    #need to store results of best(i,j) somewhere, so that we only need to make 5050 (sum of 1 to 100) calls, as opposed to going all the way down the
    #   triangle every time, which is likely equivalent to just checking every path
    
    #new triangle data type containing all best values
    optimals = []
    for i in range(sum(range(depth + 2))):
        optimals.append(-1)

    def best(i,j):

        inx = sums[i+1] + j
 
        if i == depth:
            optimals[inx] = triangle[inx]
            return optimals[inx]
        if optimals[inx] != -1:
            return optimals[inx]
        
        optimals[inx] = max(best(i+1,j), best(i+1,j+1)) + triangle[inx]
        return optimals[inx]
    
    print(best(0,0))

def p19():
    #idea: set up a loop that increments through the date range (1 Jan 1901 to 31 Dec 2000) and has flags for when either
    #   the index is a it is sunday or it is the first of the month. If both flags are true, add one to counter
    count = 0

    wkday = 1 #so if wkday%7 = 0, then it is sunday.
    mthday = 1 #so if mthday = 1, then it is the first of the month

    month = 0
    year = 1900 #use this to keep track of leap years. If year%4 = 0 (and year NOT = 1900), then it is a leap year and set months[1] = 29
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    while year < 2001:
        #if it is sunday and the first of the month (and year is not 1900)
        if wkday%7 == 0 and mthday == 1 and year > 1900:
            count += 1
        
        wkday += 1
        #now the harder part, incrementing the date(s)
        mthday += 1

        #if we've reached the end of the month
        if mthday > months[month]:

            #go back to first of the month
            mthday = 1

            #if we've reached the end of the year, reset year
            if month == 11:
                month = 0
                year += 1

                #check for leap year. don't need to check if 1900, since we start off w february = 28
                if year%4 == 0:
                    months[1] = 29
                else:
                    months[1] = 28
            else: #otherwise, we aren't at the end of the year, so just increment month
                month += 1

    print(count)


def p20():
    ans = 0
    for c in str(math.factorial(100)):
        ans += int(c)
    print(ans)

def p21():
    #euler's totient e(n) -- numbers less than n which have gcd 1 with n. So these cannot be divisors of n
    #however, other numbers may not divide n. e.g., gcd(6,9) = 3 but 6 does not divide 9.

    #finding proper divisors --
    #   naive: check if n%a == 0. (division remainder is 0)
    #   do this for each number < n, then compute sum
    #do this for all n < 10000, now have all d(n).
    #finding amicable numbers --
    #   look up d(n), and check if d(d(n)) = n. If so, we've found amicable pair.
    #       (naive method: now add (1/2)(n + d(n))) to total because looping through everything double count)
    #
    #Runtime: O(n) to check all proper divisors, O(n) total checks -> O(n^2) to calculate all d(n)
    #           O(n) to look up all amicable numbers
    #Naive method gives reasonable runtime, try to implement this

    #this may be asking for all nubers < 10000, not just ones for which the pair is under 10000.
    #Alternate method:
    #   given n, calculate d(n). Then calculate d(d(n)). if n = d(d(n)), add n only.
    #   this gave the same answer, unfortunately . . .

    max = 10000

    def d(n):
        sum = 0
        for a in range(n):
            if a != 0 and n%a == 0:
                sum += a
        return sum

    total = 0
    for n in range(max):
        dn = d(n)
        ddn = d(dn)
        #a != b
        if ddn == n and dn != n:
            total += n

    print(total)

def p22():

    #parse file
    with open('C:/Users/whous/Euler/p22.txt') as file:

        for line in file: #only one line i think
            names = line.replace('\"', '').split(',')

    alph = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,
            'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26}
    
    #sort list. 'name1 < name2' compares alphabetically in python
    names.sort()

    #calculate score
    ans = 0
    for i in range(len(names)):
        aValue = 0
        for c in names[i]:
            aValue += alph[c]
        ans += aValue*(i+1)

    print(ans)


def p23():

    #recall: d(n) finds sum of proper divisors
    def abdundance(n):
        count = 0
        a = 2
        while a*a <= n:
            if n%a == 0:
                count += a
                if a < n/a:
                    count += n/a
                if count > n:
                    return True 
            a += 1
        return False

    abc = []
    min = 12
    max = 28123
    #this takes a long time
    for n in range(min, max - min):
        if abdundance(n):
            abc.append(n)

    print('abdnce done')
    
    cache = []
    for i in range(max):
        cache.append(0)

    for i in range(len(abc)):
        for j in range(len(abc)):

            sm = abc[i] + abc[j]
            if sm < max and cache[sm] == 0:
                cache[sm] = sm

    #now we have sorted list of abundant numbers
    print(sum(range(max)) - sum(cache))

def p24():
    #3*9! = 1,088,640 so first digit must be 2. Bc 1,000,000 is between 2*9! and 3*9!
    #   Now we have a digit starting w 1 and then permutations of 023456789
    #2*9! = 725760
    #next we want to find largest number x of [023456789] st 2*9! + x*8! is less than 1,000,000
    #after that, want to find largest y in remaining numbers st 2*9! + x*8! is less than 1,000,000
    # et cetera

    perm = []
    permlength = 10
    permindex = 1000000

    #array of numbers to choose from
    nums = []
    for x in range(permlength):
        nums.append(1)
    
    i = 9
    runningsum = 0
    while i > -1:
        mn = permindex
        toUse = -1

        #cycle thru all remaining numbers
        for j in range(permlength):
            if nums[j]:
                
                adj = sum(nums[:j])

                ps = permindex - (runningsum + math.factorial(i)*(adj))
                if ps > 0 and ps < mn:
                    toUse = j
                    mn = ps


        runningsum += math.factorial(i)*(sum(nums[:toUse]))
        perm.append(toUse)
        nums[toUse] = 0
        i -= 1

    print(perm)

def p25():
    # n has 1000 digits -> floor(n*10^(-1000)) > 0
    i = 2
    #only need last two fib. numbers to get the next
    
    Fminus1 = 1
    Fminus2 = 1
    F = 0

    digs = 1000

    while math.floor( F / (10**(digs - 1)) ) == 0:

        i += 1
        F = Fminus1 + Fminus2

        #move back F, Fminus1
        temp = Fminus1
        Fminus1 = F
        Fminus2 = temp

    print(i)

def p27():

    #fast prime check based on wikipedia article
    #   can be improved using a cache of primes, may need in the future
    #let n be integer >= 2
    def fast_primecheck(n):
        if n%2 == 0 or n%3 == 0 or n <= 1:
            return False
        k = 1
        while (6*k - 1)**2 <= n:
            if n%(6*k - 1) == 0 or (n%(6*k + 1) == 0 and (6*k + 1)**2 <= n):
                return False
            k += 1
        return True
    
    #do the actual problem:
    #   first, b must be prime bc/ the thing must work for n = 0
    #   second, it must be true that 1 + a + b is prime, since it must work for n = 1 (since it must work for at least 40)
    #       b prime -> (b+1) even -> a must be odd and nonzero
    #so primitive method: given |b| < 1000 prime, test all |a| < 1000 odd and nonzero.
    #   so we have ~ 1000*2*(number of primes < 1000). which is less than 200,000 checks
    #   each check will probably involve at most 80 (i assume) checks

    #find all primes < 1000
    primecache = []
    for n in range (3,1000, 2):
        if fast_primecheck(n):
            primecache.append(n)

    mx = 0
    argmax = (0,0)

    for b in primecache:
        for a in range(-999, 999, 2):
            for sign in [-1,1]:

                #test the quadratic
                n = 0
                while fast_primecheck(n**2 + a*n + sign*b):
                    n += 1
                n -= 1
                if n > mx:
                    mx = n
                    argmax = (a,sign*b)
        
    print("n= ", mx)
    print("(a,b)= ", argmax)
    print(argmax[0]*argmax[1])
    #doesn't take long at all!

def p28():

    n = 1001

    sum = 1
    k = 1
    #for every odd number
    for i in range(3, n+1, 2):
        for j in range(4):
            k += (i - 1)
            sum += k

    print(sum)

def p29():

    s = set()
    n = 100

    for a in range(2, n+1):
        for b in range(2, n+1):
            s.add(a**b)

    print(len(s))

def p30():

    # 999,999 gives a 6 digit number 354294. 9,999,999 gives a 6 digit number also, so we only need to test 6 digit numbers and below.
    pow = 5
    ell = []

    for n in range(2, 354294):
        en = str(n)
        test = 0
        for i in range(len(en)):
            test += int(en[i])**5
        if n == test:
            ell.append(n)

    print(ell)
    print(sum(ell))

def p31():

    # a + 2b + 5c + 10d + 20e + 50f + 100g + 200h = 200
    
    sum = 200
    count = 0
    for a in range(sum + 1):
        for b in range(math.floor((sum - a)/2) + 1):
            for c in range(math.floor((sum - a - 2*b)/5) + 1):
                for d in range(math.floor((sum - a - 2*b - 5*c)/10) + 1):
                    for e in range(math.floor((sum - a - 2*b - 5*c - 10*d)/20) + 1):
                        for f in range(math.floor((sum - a - 2*b - 5*c - 10*d - 20*e)/50) + 1):
                            for g in range(math.floor((sum - a - 2*b - 5*c - 10*d - 20*e - 50*f)/100) + 1):
                                for h in range(math.floor((sum - a - 2*b - 5*c - 10*d - 20*e - 50*f - 100*g)/200) + 1):
                                    if a + 2*b + 5*c + 10*d + 20*e + 50*f + 100*g + 200*h == 200:
                                        if a < 1 and b < 1:
                                            print([a,b,c,d,e,f,g,h])
                                        count += 1
                                    
    print(count)

def p32():

    ans = set()
    
    #number of total digits must = 9
    #   possibilities: 4*1 -> 4
    #                  3*2 -> 4
    # not possibilities: 2*2 -> 3 or 4, 3*3 -> 5 or higher
    #so only need to multiply all 3 dig numbers w/ 2 dig numbers and all  1 digit numbers w/ all 4 digit numbers

    #start w/ 4/1. can only choose each number one time. So esentially want to test all permutations of 5-element subset of 1,...,9
    #   in fact, once we have each such permutation, we can test for both 4/1 and 3/2.

    # 1. get all 5-element subsets of 1,...,9
    # 2. get all permutations of each
    # 3. test 4/1 and 3/2 for each

    onenine = [1,2,3,4,5,6,7,8,9]
    s = onenine.copy()
    for a in s:
        for b in s:
            if b != a:
                for c in s:
                    if c not in [a,b]:
                        for d in s:
                            if d not in [a,b,c]:
                                for e in s:
                                    if e not in [a,b,c,d]:
                                        
                                        #print([a,b,c,d,e])

                                        n1 = str(int(str(a) + str(b) + str(c))*int(str(d) + str(e)))
                                        n2 = str(int(str(a) + str(b) + str(c) + str(d))*e)

                                            
                                        for n in [n1, n2]:
                                            if len(n) == 4:
                                                test = [a,b,c,d,e,int(n[0]),int(n[1]),int(n[2]),int(n[3])]
                                                test.sort()
                                                if test == onenine:
                                                    ans.add(n)
    
    print(sum([int(i) for i in ans]))

def p33():
    
    weird = []

    for a in range(10, 100):
        for b in range(10, 100):

            #check for a = b and check trivial case
            if a != b and (str(a)[1] != '0' or str(b)[1] != '0'):

                match = False
                minx = []
                for i in [0,1]:
                    for j in [0,1]:
                        if str(a)[i] == str(b)[j]:

                            if int(str(b)[[1,0][j]]) > 0 and a/b == int(str(a)[[1,0][i]])/int(str(b)[[1,0][j]]):
                                weird.append([a,b])
                                break
                    

    print(weird)

    #solns:
    #   16/64 = 1/4
    #   19/95 = 1/5
    #   26/65 = 2/5
    #   49/98 = 1/2

def p34():

    ans = []
    n = 10
    while len(str(n))*math.factorial(9) >= n:
        
        #print(n)
        if sum([math.factorial(int(x)) for x in str(n)]) == n:
            ans.append(n)
            #print(len(ans))
    
        n += 1

    print(ans)
    print(sum(ans))

#Methods for calculating primes:
def fast_primecheck(n):      
        if n%2 == 0 or n%3 == 0 or n <= 1:
            return False
        k = 1
        while (6*k - 1)**2 <= n:
            if n%(6*k - 1) == 0 or (n%(6*k + 1) == 0 and (6*k + 1)**2 <= n):
                return False
            k += 1
        return True

def faster_primecheck(n, minicache):
    for p in minicache:
        if n%p == 0:
            return False
        return fast_primecheck(n)

def get_primecache(max):
    primecache = [2,3]
    for n in range (3,max,2):
        if fast_primecheck(n):
            primecache.append(n) 
    return primecache

def get_primecache_with_minicache(max, minicache):
    primecache = minicache
    for n in range(minicache[len(minicache)-1],max,2):
        if faster_primecheck(n, minicache):
            primecache.append(n)
    return primecache


def p35():

    #time this one -- gotta optimize prime stuff
    start_time = time.time()

    #get primes 1 thru 1,000,000
    max = 1000000
    primecache = get_primecache_with_minicache(max, get_primecache(500))
    n = len(primecache)

    #for each prime, find 'rotations' and then do binary search of the cache for them
    #   implementing the search for practice

    circs = []
    for p in primecache:

        #if all numbers are equal, add the prime here to avoid double counting
        #add in 1-digit primes
        if p < 10:
            circs.append(p)
        else:
            #get all potential primes a
            strp = str(p)
            m = len(strp)
            miniprimes = 0
            for i in range(1,m):

                a = int("".join([strp[j%m] for j in range(i,m+i)]))
                
                l = 0
                r = n-1
                while l <= r:
                    mid = (l+r)//2
                    if primecache[mid] > a:
                        r = mid-1
                    elif primecache[mid] < a:
                        l = mid+1
                    else:
                        miniprimes += 1
                        break
            if miniprimes == m-1:
                circs.append(p)


    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    #print(circed_primes)
    print(circs)
    print(len(circs))

def p36():
    #1. find all palindormic base 10 numbers, 2. test for palindromic base 2?

    def is_pdrome(n):
        s = str(n)
        m = len(str(n))
        #3 -> 3/2 = 1.5 -> 2
        #so get 0,1:
        #   check s[0] == s[2]
        #   check s[1] == s[1]
        for i in range(math.ceil(m/2)):
            if s[i] != s[m-1 - i]:
                return False
        return True
    
    ans = 0
    for n in range(10**6):
        if is_pdrome(n) and is_pdrome(bin(n)[2:]):
            ans += n
    print(ans)

#Binary search, accepting a sorted list l and target T; returns whether target item was found
def bin_search(ls, T):
    l = 0
    r = len(ls)-1
    while l <= r:
        m = (l+r)//2
        if ls[m] > T:
            r = m-1
        elif ls[m] < T:
            l = m+1
        else:
            return m
    return 0


def p37():
    #all digits must be odd except the first
    #the first and last digits must be prime (first digit can be two, while the last cannot)
    #
    # 23 is the smallest example -- 2 and 3 are both prime
    # 3797 the given example
    # guessing that the truncatable primes don't get much higher than this -- going to start w/ a cache of primes under 10,000
    start_time = time.time()

    pc = get_primecache_with_minicache(1000000, get_primecache(100))

    count = 0
    i = 5 #go past 2,3,5,7
    trunks = []
    while count < 11 and i < len(pc):

        s = str(pc[i])
        flag = True
        for j in range(1,len(s)):
            if not (bin_search(pc, int(s[j:])) and bin_search(pc, int(s[:(len(s) - j)]))):
                flag = False
                break

        if flag:
            count += 1
            trunks.append(int(s))    

        i += 1
    
    print("Process finished --- %s seconds ---" % (time.time() - start_time))

    print(trunks)
    print(count)
    print(sum(trunks))

# ---- From here on out, no more casting strings to ints and vice versa. It is much faster to use arthimetic operations ----

def p38():
    
    #every permutation of 3-digit subsets of [1,2,...,8] generates a new 'a' value
    # generate 'a' value starting at 876 and going down.... if any one fulfills the rule, we are done

    def soln():
        for i in range(8, -1, -1):
            for j in range(8, -1, -1):

                if i != j:
                    for k in range(8, -1, -1):

                        if k != j and k != i:
                            #check result
                            #1. generate number of the form 9ijk and multiply by 2
                            a = 9*1000 + i*100 + j*10 + k
                            b = 2*a
                            #2. extract numbers from a
                            nums = [i,j,k]
                            while b > 0:
                                nums.append(b%10)
                                b = b//10
                            #3. check if we have a numbers 1 thru 8
                            nums.sort()
                            if nums == [1,2,3,4,5,6,7,8]:
                                return [a, 2*a]

    print(soln())

def p816():

    s0 = 290797
    m = 50515093
    k = 2*10**6
    
    b = 20

    s = [s0]
    lastvalue = s0

    #only calculate the even elements
    for i in range(1,2*k - 1): #find the next 2k-2 terms, for a total of 2k-1  
        value = ((lastvalue*lastvalue)%m)
        if i%2 == 0:
            s.append(value)
        lastvalue = value
    s.sort()

    start_time = time.time()
    min = float('inf')
    n = len(s)

    pts = [[],[]]
    for i in range(0,n):

        #only check values within b of s:
        for sign in [-1,1]:
            j = 1
            i2 = i + sign*j
            while i2 >= 0 and i2 < n and abs(s[i] - s[i2]) <= b:

                x2 = ((s[i]*s[i])%m)
                y2 = ((s[i2]*s[i2])%m)
                
                if abs(x2 - y2) <= b:
                    f = (s[i] - s[i2])**2 + (x2 - y2)**2
                    if f < min:
                        pts[0] = [s[i], x2]
                        pts[1] = [s[i2], y2]
                        min = f
                j += 1
                i2 = i + sign*j

    print("Minimization time --- %s seconds ---" % (time.time() - start_time))

    #possible problems:
    #   1. the points im finding are wrong
    #       dont belong to the sequence, i've misrepresented the answer
    #       there are two closer points that i'm somehow missing
    #   2. im calculating the distance wrong
    #   3. i copy pasted the number from or something
    print(pts)
    print(min**0.5)

def p39():

    start_time = time.time()

    argmax = 120
    mx = 3
    #in retrospect, would probably be easier to find all a^2 + b^2 = c^2 solns and checked a + b + c = p


    for p in range(4, 1000 + 1):

        sols = 0
        for a in range(1, p//3 + 1):
            for b in range(a, p - a + 1):
                c = p - a - b
                if a**2 + b**2 == c**2:
                    sols += 1

        if sols >= mx:
            argmax = p
            mx = sols

    print("soln time --- %s seconds ---" % (time.time() - start_time))
    print(argmax)
    print(sols)      

def p40():

    soln = 1
    i = 1
    k = 0
    n = 1

    while i <= 10**6:

        a = n
        num = []
        while a > 0:
            num.append(a%10)
            a = a // 10
        for j in range(1,len(num)+1):
            if i%10**k == 0:
                soln*=num[-j]
                k += 1
            i += 1
        n += 1

    print(soln)

def p42():
    #import file
    with open('C:/Users/whous/Euler/p42.txt') as file:
        for line in file:
            names = line.split("\",\"")
            names[0] = names[0][1:]
            names[len(names)-1] = names[len(names)-1][:(len(names[len(names)-1])-1)]
    
    alph = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,
            'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26}
    tris = [1]
    count = 0
    for nm in names:

        val = 0
        for c in nm:
            val += alph[c]

        while val > tris[len(tris)-1]:
            n = len(tris)+1
            tris.append(0.5*n*(n+1))

        if bin_search(tris, val) > -1:
            count += 1

    print(count)
    print(tris)

def p43():

    #note that:
    #   d4 must be 0,2,4,6, or 8
    #   d6 must be 0 or 5

    def has_unique_digits(n):
        digs = set()
        numDigs = 0
        while n > 0:
            digs.add(n%10)
            n = n//10
            numDigs += 1
        if len(digs) == numDigs: return True
        return False

    # 1) get a number between 99 and 1000 that is divisble by 17 and has unique digits
    #17*6 = 102
    i = 6
    """
    while 17*i < 1000:
        if has_unique_digits(17*i):

            last = 17*i
            # 2) choose the next number -- must be divisible by 13...
            for d7 in range(10):
                next = d7*100 + last//10
                if not next%13:
                    if has_unique_digits(1000*d7 + 17*i):
                        
                        # choose d6 -- must be divisible by 11...
                        last = next
                        for d6 in [0,5]:
                            next = d6*100 + last//10
                            if not next%11:
                                if has_unique_digits(d6*10**4 + d7*10**3 + 17*i):

                                    print(d6*10**4 + d7*10**3 + 17*i)

        i += 1
    """
    """
    for i in [0,1,4,3]:
        for j in [0,1,4,3]:
            n = i*100 + j*10 + 9
            if has_unique_digits(n) and not n%3:
                print(n)
    """
    sum = 0
    for i in [1,4]:
        for j in [1,4]:
            if i+j ==5:
                for k in [6357289, 60357289, 30952867]:
                    
                    n = k + j*10**8 + i*10**9
                    print(n)
                    sum += n
    print(sum)
                
        #looks like we can solve the rest with pen & paper, since there are only 3 options left

def p44():

    lim = 3*10**3
    P = [n*(3*n - 1)/2 for n in range(1, lim+1)]
    D = []

    for i in P:
        for j in P:
            if i != j and bin_search(P, abs(i-j)) and bin_search(P, i + j):
                D.append(abs(i-j))
    
    print(P[:10])
    print(min(D))

def p45():

    #Tn = n(n+1)/2 = (1/2)(n^2 + n)
    #Pn = n(3n-1)/2 = (1/2)(3n^2 - n)
    #Hn = n(2n-1) = 2n^2 - n

    #Let n > 1
    #   n^2 + n < 3n^2 - n  <=> 2n < 2n^2  <=> 1 < n. So Tn < Pn for all n
    #   3n^2 - n < 4n^2 - 2n  <=>  n < n^2 .... So Pn < Hn for all n
    #hence Tn < Pn < Hn.
    #   => find Hn => check all Tn/Pn less than it
    #
    #Given ex: T285 = P165 = H143 = 40755
    #   so dont need check any less than these
    #method: find Tn, check all Pn with lower indices. If found, check Hn with lower indices than that found for Pn
    
    lim = 10**5
    T = [n*(n+1)/2 for n in range(1, lim+1)] #286 apparently all H are T tho
    P = [n*(3*n - 1)/2 for n in range(1, lim+1)] #166
    H = [n*(2*n-1) for n in range(1, lim+1)] #144


    flag = True
    ans = -1
    i = 166

    while flag and i < len(P):
        if bin_search(H[144:i], P[i]) > -1:
            flag = False
            ans = P[i]
        i += 1
    print(ans)

def p46():

    #It was proposed by Chistrian Goldbach that every odd composite number can be written as the sum of a prime and twice a square
    #   9 = 7 + 2*1^2       15 = 7 + 2*2^2      21 = 3 + 2*3^2      25 = 7 + 2*3^2      27 = 19 + 2*2^2     33 = 31 + 2*1^2
    #Turns out the conjecture was false. What is the smallest odd composite that can't be written this way?

    #get all odd composite numbers up to some (even) limit
    """
    lim = 100
    print(list(set([2*i + 1 for i in range(3, int(lim/2))]).difference(set(get_primecache(lim)))))
    print([2*n**2 for n in range(1, math.ceil(math.sqrt(lim/2)))]) #2*n^2 < lim -> n^2 < lim/2 -> n < sqrt(lim/2)
    print(get_primecache(lim))
    """

    #naive method: get composite odd # -> check all combinations of prime/2*square integer less than the #
    #observations:
    #   will never add 2 as the prime, else even + even = odd #
    #   there are fewer squared integers in a range then there are primes
    #       let sq = 2*n^2 for some n, and let o be an odd composite. Then o = p + sq <-> o - sq = p
    #       faster to, given o, check if o - n is prime for all n < o
    #       how many primechecks per n?
    #           idk but def fewer than # primes and there are ~log(n) primes under n
    #       forgot how expensive my prime check is. if using the cache it's log(n). getting the cache is somewhat expensive though

    start_time = time.time()

    lim = 10**4
    cache = get_primecache(lim)
    oddc = list(set([2*i + 1 for i in range(3, int(lim/2))]).difference(set(cache)))
    sq = [2*n**2 for n in range(1, math.ceil(math.sqrt(lim/2)))]
    ans = -1

    for o in oddc:

        flag = True
        for s in sq[:o]:
            if fast_primecheck(o-s):
                flag = False
                break
        if flag:
            ans = o
            break

    print("end time --- %s seconds ---" % (time.time() - start_time))  
    print(ans)

    #easier than expected -- ans = 5777

def p47():
    
    #The first two consecutive numbers to have two distinct prime factors are:
    #   14 = 2*7        15 = 3*5
    #The first three consecutive numbers to have three distinct prime factors are:
    #   644 = (2^2)*7*23      645 = 3*5*43        646 = 2*17*19
    #Find the first four consecutive integers to have four distinct prime factors each. What is the first of these numbers?

    #prime factorization algorithm? don't have that yet
    #i guess 2^2 is a different 'prime factor' from 2
    #nto sure if these numbers are allowed to have more than 4 factors. maybe the lowest one doesnt'

    start_time = time.time()

    #need a cache regardless
    lim = 2*10**5
    cache = get_primecache(lim)

    def get_num_pf(x):

        n = 0
        a = x
        while a > 1:

            for i in range(len(cache)):

                p = cache[i]
                if not a%p:
                    n += 1
                    a = a / p

                    while not a%p:
                        a = a / p
                    break
        return n

    x = 644
    lim = cache[len(cache) - 1]*4
    while x < lim:

        #check if number is prime
        if not bin_search(cache, x):
            
            n = x
            #print(x)

            for i in range(4):
                if get_num_pf(x + i) < 4:
                    
                    #print('miss:', x)
                    x += i + 1
                    break
            
            if n == x:
                print('HIT')
                print(n)
                break
        else:
            x += 1


    print(bin_search(cache, 409))

    print("end time --- %s seconds ---" % (time.time() - start_time)) 

def p48():

    #only need 'last' (least significant) digits of each number
    #   ex) 11^11 % 10^10
    lim = 10
    sum = 0
    for i in range(1,lim+1):
        sum += i**i
    print(sum)

def p50():
    start_time = time.time()

    # prime cache 100,000 long
    cache = get_primecache_with_minicache(100000, get_primecache(2000))

    #find most consecutive primes < 1,000,000 and work backwards
    max = 1000
    i = 0
    j = 0
    soln_found = False

    #while not soln_found:

    while sum(cache[i:j]) < max:
        j += 1

    t = sum(cache[i:j])
    print(t)
    print(fast_primecheck(t))  

    print("end time --- %s seconds ---" % (time.time() - start_time)) 
    

if __name__ == '__main__':
    p48()
