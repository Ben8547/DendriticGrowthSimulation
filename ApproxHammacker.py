"""
Since the Hamaker potential has a singularity at d = 2R; and the Taylor expansion idea did not work particularly well, we instead aim to sample the curve from 2R to 10 R
and construct an interpolating curve between the points so that we can perhaps more accurately reconstruct the curve.
We sample the curve with more density towards the singularity so that the probability density function is 
a beta distrobution shifted to the interval (2R, 10R); more density near the singularity.
"""

#from scipy.special import gamma , betainc


'''def beta(x,beg,end): # alpha = 0.5, beta = 1
    delta = (end-beg)
    u = (x-beg)/delta
    return (u)**-0.5  * (1-u)**0 * gamma(1.5) / gamma(0.5) / gamma(1.)'''

# Now we need to approximate the CDF for this function somehow ; the CDF comes out to be the regularized incomplete beta function which scipy has integrated (betainc) when we have not shifted the distrobution.
# instead what we can do is map x into [0,1] and then retun that value of CDF
'''def Beta_CDF(x,beg,end):
    u = (x-beg)/(end-beg)
    return betainc(0.5,1,u) # returns of probability begin x or less in the interval [beg, end].'''


# Skip the above; just use scipy to sample from the beta distrobution directly.

from scipy.stats import beta

from defineParameters import params

from numpy import fill_diagonal

R = params["Average_particle_Radius"]

n = params['Hamaker_Sample_Size']

def Hamaker(d,A):
    return A/6 * ( (4.*R**2.*d)/(d**2. -4.*R**2.)**2 + (4.*R**2. * d)/(d**4.) + (2.*d)/(d**2.-4.*R**2.) - 2./(d)  )

def gen_interp_poly(A):
    random_samples = beta.rvs(0.5,1.,size = n) # n samples of the beta distrobution from 0 to 1.
    random_samples = random_samples * 8*R + 2*R # translate into the interval [2R,10R].
    Hamacker_Samples = Hamaker(random_samples,A)

    #print(random_samples)

    diff = random_samples[:, None] - random_samples[None, :] # compute the denominators in advance since division by zero causes issues; also saves time since these will never change
    fill_diagonal(diff, 1.0) # avoid zeros on diagonal
    D = diff.prod(axis=1)
    #print(D)

    def out(x):

        diff_x = x[None, :] - random_samples[:, None]
        prod = diff_x.prod(axis=0) / diff_x # this is the vector of indicator terms for each sample 

        return (Hamacker_Samples / D) @ prod # Matrix vector multiplication does the summation - thus we find the hamacker potential for each x simultaneously to an approximation

    return out

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy import linspace,min,max
    A = 1.
    approx = gen_interp_poly(A)
    domain = linspace(2.5*R,10*R,100)
    plt.plot(domain, approx(domain),label="Sample")
    actuals = Hamaker(domain,A)
    plt.plot(domain,actuals,label="actual")
    plt.ylim(1.1*min(actuals),1.1*max(actuals))
    plt.show()



# this is a complete failure - the samples oscilate far too much - we would need to perhaps interpolate with some other basis or use a different interpolating alogrithm
