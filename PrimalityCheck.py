import random

""" 
The Miller–Rabin primality test or Rabin–Miller primality test is a probabilistic primality test: 
an algorithm which determines whether a given number is likely to be prime, 
similar to the Fermat primality test and the Solovay–Strassen primality test.
"""
def isPrimeCheck_RabinMiller(n, k):
    # Return True if n is a prime number.
    # Return False if n is a composite number.
    # The function uses the Miller-Rabin primality test.
    # The parameter k is the number of iterations.
    # The function has a small probability of returning a false positive.
    
    # Handle small values of n
    if n <= 3:
        return n > 1

    # Test if n is an even number.
    if n % 2 == 0:
        return False

    # Write n as (2^r)*d + 1.
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Perform the Miller-Rabin primality test.
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

# list of prime and composite numbers
# https://t5k.org/curios/page.php/10089886811898868001.html

k = 10 
test_numbers = [2131131137, # prime number
				94371217349, # prime number
				99999999907, # prime number
				13433444444444444443343344447, # prime number
				10089886811898868001, # prime number
				2, 
				3, 
				4, # composite number
				5, 
				7, 
				11, 
				13]

for i in test_numbers:
    if isPrimeCheck_RabinMiller(i, k):
        print(i, "is a prime number according to the Miller-Rabin test.")
    else:
        print(i, "is a composite number according to the Miller-Rabin test.")



def isPrimeCheckSchoolVersion(n): 

	# Corner case 
	if n <= 1: 
		return False

	# Check from 2 to n-1 
	for i in range(2, n): 
		if n % i == 0: 
			return False

	return True

# The test using the school version takes a long time to run.

# for i in test_numbers:
#     if isPrimeCheckSchoolVersion(i):
#         print(i, "is a prime number according to the school version.")
#     else:
#         print(i, "is a composite number according to the school version.")
		
# Iterative Function to calculate 
# (a^n)%p in O(logy) 
def power(a, n, p):
	
	# Initialize result 
	res = 1
	
	# Update 'a' if 'a' >= p 
	a = a % p 
	
	while n > 0:
		
		# If n is odd, multiply 
		# 'a' with result 
		if n % 2:
			res = (res * a) % p
			n = n - 1
		else:
			a = (a ** 2) % p
			
			# n must be even now 
			n = n // 2
			
	return res % p
	
# If n is prime, then always returns true,
# If n is composite than returns false with
# high probability Higher value of k increases
# probability of correct result
def isPrimeFermat(n, k):
	
	# Corner cases
	if n == 1 or n == 4:
		return False
	elif n == 2 or n == 3:
		return True
	
	# Try k times 
	else:
		for i in range(k):
			
			# Pick a random number 
			# in [2..n-2]	 
			# Above corner cases make 
			# sure that n > 4 
			a = random.randint(2, n - 2)
			
			# Fermat's little theorem 
			if power(a, n - 1, n) != 1:
				return False
				
	return True
			

for i in test_numbers:
    if isPrimeFermat(i, k):
        print(i, "is a prime number according to Fermat's test.")
    else:
        print(i, "is a composite number according to Fermat's test.")

# modulo function to perform binary 
# exponentiation 
def modulo(base, exponent, mod): 
	x = 1; 
	y = base; 
	while (exponent > 0): 
		if (exponent % 2 == 1): 
			x = (x * y) % mod; 

		y = (y * y) % mod; 
		exponent = exponent // 2; 

	return x % mod; 

# To calculate Jacobian symbol of a 
# given number 
def calculateJacobian(a, n): 

	if (a == 0): 
		return 0;# (0/n) = 0 

	ans = 1; 
	if (a < 0): 
		
		# (a/n) = (-a/n)*(-1/n) 
		a = -a; 
		if (n % 4 == 3): 
		
			# (-1/n) = -1 if n = 3 (mod 4) 
			ans = -ans; 

	if (a == 1): 
		return ans; # (1/n) = 1 

	while (a): 
		if (a < 0): 
			
			# (a/n) = (-a/n)*(-1/n) 
			a = -a; 
			if (n % 4 == 3): 
				
				# (-1/n) = -1 if n = 3 (mod 4) 
				ans = -ans; 

		while (a % 2 == 0): 
			a = a // 2; 
			if (n % 8 == 3 or n % 8 == 5): 
				ans = -ans; 

		# swap 
		a, n = n, a; 

		if (a % 4 == 3 and n % 4 == 3): 
			ans = -ans; 
		a = a % n; 

		if (a > n // 2): 
			a = a - n; 

	if (n == 1): 
		return ans; 

	return 0; 

# To perform the Solovay- Strassen 
# Primality Test 
def solovoyStrassen(p, iterations): 

	if (p < 2): 
		return False; 
	if (p != 2 and p % 2 == 0): 
		return False; 

	for i in range(iterations): 
		
		# Generate a random number a 
		a = random.randrange(p - 1) + 1; 
		jacobian = (p + calculateJacobian(a, p)) % p; 
		mod = modulo(a, (p - 1) / 2, p); 

		if (jacobian == 0 or mod != jacobian): 
			return False; 

	return True; 


iterations = 50; 

for i in test_numbers:
    if solovoyStrassen(i, iterations):
        print(i, "is a prime number according to Solovay-Strassen test.")
    else:
        print(i, "is a composite number according to Solovay-Strassen test.")

"""
Output: 

2131131137 is a prime number according to the Miller-Rabin test.
94371217349 is a prime number according to the Miller-Rabin test.
99999999907 is a prime number according to the Miller-Rabin test.
13433444444444444443343344447 is a composite number according to the Miller-Rabin test.
10089886811898868001 is a prime number according to the Miller-Rabin test.
2 is a prime number according to the Miller-Rabin test.
3 is a prime number according to the Miller-Rabin test.
4 is a composite number according to the Miller-Rabin test.
5 is a prime number according to the Miller-Rabin test.
7 is a prime number according to the Miller-Rabin test.
11 is a prime number according to the Miller-Rabin test.
13 is a prime number according to the Miller-Rabin test.
2131131137 is a prime number according to Fermat's test.
94371217349 is a prime number according to Fermat's test.
99999999907 is a prime number according to Fermat's test.
13433444444444444443343344447 is a composite number according to Fermat's test.
10089886811898868001 is a prime number according to Fermat's test.
2 is a prime number according to Fermat's test.
3 is a prime number according to Fermat's test.
4 is a composite number according to Fermat's test.
5 is a prime number according to Fermat's test.
7 is a prime number according to Fermat's test.
11 is a prime number according to Fermat's test.
13 is a prime number according to Fermat's test.
2131131137 is a prime number according to Solovay-Strassen test.
94371217349 is a prime number according to Solovay-Strassen test.
99999999907 is a prime number according to Solovay-Strassen test.
13433444444444444443343344447 is a composite number according to Solovay-Strassen test.
10089886811898868001 is a composite number according to Solovay-Strassen test.
2 is a prime number according to Solovay-Strassen test.
3 is a prime number according to Solovay-Strassen test.
4 is a composite number according to Solovay-Strassen test.
5 is a prime number according to Solovay-Strassen test.
7 is a prime number according to Solovay-Strassen test.
11 is a prime number according to Solovay-Strassen test.
13 is a prime number according to Solovay-Strassen test.

""" 

