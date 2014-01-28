#!/usr/bin/env python3
# coding: utf-8

import argparse
import bs4
import getpass
import http.cookiejar
import itertools
import keyring
import logging
import math
import png
import random
import re
import time
import urllib.error
import urllib.parse as ulp
import urllib.request as ulr




baseURL = "http://projecteuler.net/"


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

# def displayCaptcha(fileishThing):
#     from PIL import Image
#     captcha = Image.open(fileishThing)
#     captcha.show()

def problemURL(N):
    return ulp.urljoin(baseURL,"problem=" + str(N))

def loginURL():
    return ulp.urljoin(baseURL,"login")

# fix
def getURL(someurl):
    req = ulr.Request(someurl)
    try:
        response = ulr.urlopen(req)
    except urllib.error.URLError as e:
        if hasattr(e, 'reason'):
            print('We failed to reach a server.')
            print('Reason: ', e.reason)
        elif hasattr(e, 'code'):
            print('The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
    else:
        return response

# note: we store both the PE username and password in the keyring 
# (as if they were passwords), using the login username as the user
def getUserPass(resetData = False):
    sysUsername = getpass.getuser()
    username = keyring.get_password('ProjectEulerSubmitUser', sysUsername)
    password = keyring.get_password('ProjectEulerSubmitPass', sysUsername)

    if username is None or resetData:
        username = input("Project Euler Username:\n")
        # store the password
        keyring.set_password('ProjectEulerSubmitUser', sysUsername, username)

    if password is None or resetData:
        password = getpass.getpass("Project Euler Password:\n")
        # store the password
        keyring.set_password('ProjectEulerSubmitPass', sysUsername, password)

    # note: the login = "Login" part seems required by Project Euler
    return dict(username = username, password = password, login = "Login")


#  r=png.Reader(file=urllib.urlopen('http://www.schaik.com/pngsuite/basn0g02.png'))
# d = dict(username="fabianlischka",password="eulerul8u")

def bsLoggedInName(b):
    if "Logged in as" in str(b.find(id="info_panel").div):
        return b.find(id="info_panel").div.strong.get_text()
    # else None

def bsAlreadyCompleted(b):
    if len(b.find_all("span",text=re.compile("Completed on"))):
        return b.find_all("span",text=re.compile("Completed on"))[0].text
    # else None

def bsBadCaptcha(b):
    if "not valid" in str(b.find("div",id="message")):
        return True

def bsSolutionCorrect(b):
    if b.find("img",alt="Correct") is not None:
        return True

def bsSolutionWrong(b):
    if b.find("img",alt="Wrong") is not None:
        return True

def bsTooManySubmissions(b):
    if "You are not permitted" in str(b.find("div",id="message")):
        return True

def getOpenerWithCookieJar():
    cj = http.cookiejar.CookieJar()
    opener = ulr.build_opener(ulr.HTTPCookieProcessor(cj))
    return opener

def eulerLogin(opener, retryCount = 3):
    # acquire cookie & login
    logging.debug("Trying to open login page to get cookie...")
    request = ulr.Request(loginURL())
    r = opener.open(request) # to pick up the cookie
    loginData = ulp.urlencode(getUserPass())
    loginData = loginData.encode('utf-8')
    request.add_header("Content-Type","application/x-www-form-urlencoded;charset=utf-8")
    logging.debug("Sending login data...")
    r = opener.open(request,loginData)
    b1 = bs4.BeautifulSoup(r)
    loginResult = bsLoggedInName(b1)
    if loginResult is None:
        if retryCount > 0:
            # failed, retry a few times
            logging.warning("Login failed, retrying... {}".format(retryCount))
            return eulerLogin(opener, retryCount = retryCount - 1)
        else:
            logging.error("Login failed, getting new credentials.")
            # failed, get new username/password
            getUserPass(resetData = True)
            return eulerLogin(opener)
    else:
        logging.info("Login succeeded, logged in as {}".format(loginResult))
        return True

def solveCaptcha(opener, b2):
    # solve captcha
    captcha_elem = b2.find_all("img", attrs={"name":"captcha"})
    captcha_path = ulp.urljoin(baseURL,captcha_elem[0]["src"])

    logging.debug("CAPTCHA path: {}".format(captcha_path))
    # captcha_cached = io.BytesIO(opener.open(captcha_path).read())
    # displayCaptcha(captcha_cached)
    # captcha_cached.seek(0) # rewind
    # captcha_pic = png.Reader(file=captcha_cached)

    captcha_pic = png.Reader(opener.open(captcha_path))
    pixelsBW = turnImgToBW(captcha_pic)
    cuts = findCuts(pixelsBW)
    digitPics = truncateAndInvertDigits(pixelsBW,cuts)
    guess = "".join([str(guessDigit(digitPic)) for digitPic in digitPics])
    return guess

def submitGuess(opener,problemNumber,soln,guess):
    guessPostData = dict(confirm = str(guess))
    guessPostData["guess_"+str(problemNumber)] = str(soln)
    # print(guessPostData)
    guessPostData = ulp.urlencode(guessPostData)
    guessPostData = guessPostData.encode('utf-8')
    ## TO POST solution: data is: guess_301=987654321&confirm=12345
    logging.debug(str(guessPostData))
    request = ulr.Request(problemURL(problemNumber))
    request.add_header("Content-Type","application/x-www-form-urlencoded;charset=utf-8")
    r = opener.open(request,guessPostData)
    return bs4.BeautifulSoup(r)


def submit(problemNumber,soln):
    logging.info("Submitting. Problem: {}, Soln: {}".format(problemNumber,soln))

    opener = getOpenerWithCookieJar()
    eulerLogin(opener)
    
    retryCount = 5
    while retryCount > 0:
        retryCount -= 1
        r = opener.open(problemURL(problemNumber))
        bsProblem = bs4.BeautifulSoup(r)
        if bsAlreadyCompleted(bsProblem) is not None:
            logging.warning("Already {}".format(bsAlreadyCompleted(bsProblem)))
            return 1
        guess = solveCaptcha(opener,bsProblem)
        logging.info("CAPTCHA Guess: {}".format(guess))
        logging.debug("Submitting solution...")
        bsResult = submitGuess(opener,problemNumber,soln,guess)
        # logging.debug(bsResult.prettify)

        # could be bad captcha (->retry)
        if bsBadCaptcha(bsResult):
            logging.info("CAPTCHA wrong. Attempts remaining: {}".format(retryCount))
            time.sleep(2**(3-retryCount))
        elif bsSolutionWrong(bsResult):
            logging.warning("Sorry, but the answer you gave appears to be incorrect.")
            return 2
        elif bsSolutionCorrect(bsResult):
            logging.info("Congratulations, the answer you gave is correct!")
            return 0
        elif bsTooManySubmissions(bsResult):
            logging.warning("Too many submissions. Waiting 31 seconds.")
            time.sleep(31)
        else:
            logging.error("Unknown return result:")
            print(bsResult.prettify())
            return 4

    logging.warning("Could not solve CAPTCHA, sorry.")
    return 3

        # parseResult: could be: good, captchaBad, solnBad, unknown


def findCuts(pixelsBW, numCuts = 5):
    M = len(pixelsBW)
    N = len(pixelsBW[0])
    colSum = [sum(col)/M for col in zip(*pixelsBW)]

    for i in range(N):
        if colSum[i] < 0.999:
            iLeft = i
            # print("yay")
            break
    for i in range(N-1,0,-1):
        if colSum[i] < 0.999:
            iRight = i+1
            break
    potential = colSum[iLeft:iRight]

    g = 200
    means = [i/numCuts for i in range(1,numCuts)] # [0.2,0.4,0.6,0.8]
    std = 0.3 / numCuts
    lowest = 999999999
    repellMin = sum((x-y)**(-2) for x,y in pairwise([0]+means+[1])) 
    for r in range(1000):
        particleXs = sorted([0.01,0.99]+[random.gauss(m,std) for m in means])[1:-1]
        coords = [math.floor(x*len(potential)) for x in particleXs]
        repell = sum((x-y)**(-2) for x,y in pairwise([0]+particleXs+[1])) - repellMin
        gravity = sum(1-potential[c] for c in coords) * g
        totalEnergy = repell + gravity
        if totalEnergy < lowest:
            lowest = totalEnergy
            best = coords
            # print(particleXs,totalEnergy,repell, gravity)

    cuts = [iLeft + c for c in best]
    return cuts


def moment(pixels,p,q):
    m = sum( sum(x**p * y**q * pixel for y, pixel in enumerate(row)) for x,row in enumerate(pixels))
    # print(p,q,m)
    return m

def cmoment(pixels,p,q,xbar,ybar):
    return sum( sum((x-xbar)**p * (y-ybar)**q * pixel for y, pixel in enumerate(row)) for x,row in enumerate(pixels))

def moments(pixels):
    # see http://en.wikipedia.org/wiki/Image_moment
    M00 = moment(pixels,0,0) # = µ00 (note: µ = asci 181, option-m on the Mac, not small greek letter mu)
    xbar = moment(pixels,1,0) / M00
    ybar = moment(pixels,0,1) / M00

    eta = [[0]*4 for p in range(4)]
    for p in range(4):
        for q in range(4):
            eta[p][q] = cmoment(pixels,p,q,xbar,ybar) * M00**(-1-(p+q)/2)

    I1 = eta[2][0] + eta[0][2]
    I2 = (eta[2][0] - eta[0][2])**2 + 4*eta[1][1]**2
    I8 = eta[1][1]*( (eta[3][0]+eta[1][2])**2 - (eta[0][3]+eta[2][1])**2) - (eta[2][0]-eta[0][2])*(eta[3][0]-eta[1][2])*(eta[0][3]-eta[2][1])
    I4 = (eta[3][0]+eta[1][2])**2 + (eta[2][1]+eta[0][3])**2
    I5 = (eta[3][0] - 3*eta[1][2])*(eta[3][0] + eta[1][2]) * (  (eta[3][0] + eta[1][2])**2 - 3* (eta[2][1] + eta[0][3])**2 ) + (
          3*eta[2][1] - eta[0][3])*(eta[2][1] + eta[0][3]) * (3*(eta[3][0] + eta[1][2])**2 -    (eta[2][1] + eta[0][3])**2 )
    I6 = (eta[2][0] - eta[0][2]) * ( (eta[3][0] + eta[1][2])**2 - (eta[2][1] + eta[0][3])**2 ) + (
         4*eta[1][1]*(eta[3][0]+eta[1][2])*(eta[2][1]+eta[0][3]))
    I7 = (3*eta[2][1] -   eta[0][3])*(eta[3][0] + eta[1][2]) * (  (eta[3][0] + eta[1][2])**2 - 3* (eta[2][1] + eta[0][3])**2 ) - (
            eta[3][0] - 3*eta[1][2])*(eta[2][1] + eta[0][3]) * (3*(eta[3][0] + eta[1][2])**2 -    (eta[2][1] + eta[0][3])**2 )
    # phi = math.arctan(2*eta[1][1]/(eta[2][0]-eta[0][2]))
    return (M00, I1,I2,I8,I4,I5,I6,I7,eta[3][0], eta[0][3])
    # good:   0   1  2     4        7    8

def featureExtraction(pixels):
    return moments(pixels)
    # intensity
    # symmetry
    # eigenvalues?

def turnImgToBW(pngReader):
    pixels = pngReader.asFloat()[2]
    # turn to gray by averaging
    pixelsGray = [[(i+j+k)/3 for i,j,k in zip(*(itertools.islice(row,i,None,3) for i in range(3)))] for row in pixels]

    # turn pic to b/w by cutting off
    cutoff = 0.7
    # note: BW denotes "normal", ie black digit on white background
    pixelsBW = [[(1 if p > cutoff else 0) for p in row] for row in pixelsGray]
    return pixelsBW

def truncateAndInvertDigits(pixelsWB, cuts):
    cuts = [0]+cuts+[len(pixelsWB[0])]
    # print(cuts)
    pixT = list([(1-p) for p in r] for r in zip(*pixelsWB)) # transpose and invert
    digitPixs = [list(zip(*(pixT[l:r]))) for l,r in pairwise(cuts)] # note: probably have some overlap here...
    return digitPixs

def guessDigit(pixels):
    glbmeans = [305.530, 0.26446, 0.01655942, -3.6018e-06, 0.0002300769, 3.0538e-07, 1.52023e-05, 7.46647e-07, 0.00337966348, -0.00192048951]
    glbstd   = [68.22927, 0.0415, 0.02645950, 8.39806e-05, 0.000513669, 1.3828625e-06, 7.4462673e-05, 2.085475e-06, 0.01624549, 0.0052139189  ]
    meansPerGuess = [
[ 0.546,  0.194, -0.460,  0.043, -0.130, -0.719, -0.630, -0.314, -0.238,  0.400  ], # 0
[-2.329,  1.937,  2.795,  0.014, -0.419, -0.221, -0.199, -0.358, -0.218,  0.336  ], # 1
[ 0.136,  0.501, -0.062,  0.010, -0.299, -0.229, -0.287, -0.361, -0.622, -0.585  ], # 2
[-0.047,  0.286, -0.131,  0.035, -0.141, -0.249, -0.440, -0.343, -0.390, -0.480  ], # 3
[ 0.205, -1.357, -0.613,  0.014, -0.444, -0.221, -0.204, -0.358, -0.692, -1.225  ], # 4
[ 0.346, -0.299, -0.411,  0.044, -0.409, -0.220, -0.199, -0.358, -0.107,  0.398  ], # 5
[ 0.726, -0.808, -0.556,  0.038, -0.359, -0.225, -0.229, -0.358, -0.442,  0.708  ], # 6
[-1.097,  1.085,  0.353, -0.248,  2.616,  2.125,  2.180,  2.702,  2.594, -0.277  ], # 7
[ 0.853, -1.051, -0.536,  0.053, -0.446, -0.221, -0.204, -0.358, -0.379,  0.724  ], # 8 
[ 0.786, -0.799, -0.539,  0.043, -0.372, -0.224, -0.224, -0.358,  0.049, -0.146  ]] # 9


    featuresThisDigit = featureExtraction(pixels)
    normalizedFeaturesThisDigit = [(f-m)/s for f,m,s in zip(featuresThisDigit,glbmeans,glbstd) ]
    weights = [1,1,1,0,1,0,0,1,1,0]
    discrepancyPerGuess = [ sum(w*(d-m)**2 for w,d,m in zip(weights,normalizedFeaturesThisDigit, featuresGuess)) for featuresGuess in meansPerGuess]

    bestDiscrepancy,bestGuess = discrepancyPerGuess[0],0
    for digit,disc in enumerate(discrepancyPerGuess):
        if disc < bestDiscrepancy:
            bestDiscrepancy = disc
            bestGuess = digit
    logging.debug(("  {: 7.3f} "*10).format(*normalizedFeaturesThisDigit))
    logging.debug(str(bestGuess)+("{: 7.3f} "*10).format(*discrepancyPerGuess))
    return bestGuess



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit a ProjectEuler solution for you.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("problemNumber", type=int, help="the problem number")
    parser.add_argument("solution", help="the solution")
    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',level=logging.ERROR)
    if args.verbose:
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',level=logging.DEBUG)

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',level=logging.INFO)

    submit(args.problemNumber, args.solution)

# default: info and above (i,w,e,c)
# verbose: debug and above (d,i,w,e,c)
# quiet: above warning (e,c)
