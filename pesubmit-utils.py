# coding: utf-8
import itertools as it
import functools as ft

import collections
import png
import glob
import math
import random


from pesubmit import * 

def levelPic(levels, M,N, cuts = None):
    mi, ma = min(levels), max(levels)+1
    lPic = [[1]*N for row in range(M)]
    for col in range(N):
        y = math.floor(M*(levels[col] - mi)/(ma-mi))
        lPic[y][col] = 0
        if cuts is not None:
            if col in cuts:
                for y in range(M):
                    lPic[y][col] = 0
    return lPic



def extractDigitsFromPath(testImagePath):
    return [int(c) for c in testImagePath[-9:-4]]


def truncateAndInvertDigits(pixelsWB, cuts):
    cuts = [0]+cuts+[len(pixelsWB[0])]
    # print(cuts)
    pixT = list([(1-p) for p in r] for r in zip(*pixelsWB)) # transpose and invert
    digitPixs = [list(zip(*(pixT[l:r]))) for l,r in pairwise(cuts)] # note: probably have some overlap here...
    return digitPixs

def turnImgToBW(pngReader):
    pixels = pngReader.asFloat()[2]
    # turn to gray by averaging
    pixelsGray = [[(i+j+k)/3 for i,j,k in zip(*(it.islice(row,i,None,3) for i in range(3)))] for row in pixels]

    # turn pic to b/w by cutting off
    cutoff = 0.7
    # note: BW denotes "normal", ie black digit on white background
    pixelsBW = [[(1 if p > cutoff else 0) for p in row] for row in pixelsGray]
    return pixelsBW


def test2():
#    testimageFolder = "/Users/frl/Documents/Meins/Coding/HackerSchool/PEAnswer/captchaExamples"
    testImagePathPattern = "/Users/frl/Documents/Meins/Coding/HackerSchool/fizz/pesubmit/captchaExamples/[0-9][0-9][0-9][0-9][0-9].png"
    digitCounter = collections.defaultdict(int)
    for testImagePath in glob.glob(testImagePathPattern):
        print(testImagePath)
        writeBW = ft.partial(writeModBWPng,testImagePath[:-4])

        # read image in
        img = png.Reader(testImagePath)
        pixelsBW = turnImgToBW(img)

        M = len(pixelsBW)
        N = len(pixelsBW[0])
        colSum = [sum(col)/M for col in zip(*pixelsBW)]
        for i in range(N):
            if colSum[i] < 0.999:
                iLeft = i
                break
        for i in range(N-1,0,-1):
            if colSum[i] < 0.999:
                iRight = i+1
                break
        colSumTrunc = colSum[iLeft:iRight]

        cuts = [iLeft + c for c in findCuts(colSumTrunc)]

        digitPics = truncateAndInvertDigits(pixelsBW,cuts)

        digits = extractDigitsFromPath(testImagePath)

        if True:
            for digit, digitPic in zip(digits, digitPics):
                digitPath = testImagePath[:-9]+"Digits/"+str(digit)
                suffix = "v{:0>3}".format(digitCounter[digit])
                digitCounter[digit] += 1
                # print(digit,digitPath+"-"+suffix)
                writeModBWPng(digitPath, suffix, digitPic)


        # and plot cuts
        for col in cuts:
            for y in range(M):
                pixelsBW[y][col] = 0
        writeBW("bw", pixelsBW)

        # compute average column darkness, from gray
        colSum = [sum(col) for col in zip(*pixelsGray)]
        colSumPic = levelPic(colSum,M,N,cuts)
        writeBW("cold",colSumPic)

        # compute average column darkness, from BW
        colSum = [sum(col) for col in zip(*pixelsBW)]
        colSumPic = levelPic(colSum,M,N,cuts)
        writeBW("colb",colSumPic)


        # # compute shifted copies (for gradient/energy computation)
        # su = pixelsGray[1:] + [pixelsGray[-1]]
        # sd = [pixelsGray[1]] + pixelsGray[:-1]
        # sl = [row[1:] + [row[-1]] for row in pixelsGray]
        # sr = [[row[1]] + row[:-1] for row in pixelsGray]
        # # energy
        # energy = [[ (p-u)**2 + (p-d)**2 + (p-l)**2 + (p-r)**2 for p,u,d,l,r in zip(rp,ru,rd,rl,rr)] for rp,ru,rd,rl,rr in zip(pixelsGray,su,sd,sl,sr)]
        # # determine cutoff point
        # flatEnergy = it.chain.from_iterable(energy)
        # cutoff = sorted(flatEnergy)[87*M*N//100]
        # # turn energy to b/w by cutting off
        # gradBW = [[(1 if p > cutoff else 0) for p in row] for row in energy]
        # writeBW("grad", gradBW)

def analyzeFeaturesList(featuresList, digit):
    # analyze featuresList
    flt = list(zip(*featuresList))
    # flt now is an array, for different features, of arrays of values of that feature per digit
    means = [sum(featureList)/len(featureList) for featureList in flt]
    std   = [math.sqrt(   sum( (f-fbar)**2 for f in featureList ) /len(featureList) )  for featureList,fbar in zip(flt,means)]
    print("Avg:",digit," Feat: {: 6.1f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f}".format(*means))
    print("Std:",digit," Feat: {: 6.1f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f} {: 3.6f}".format(*std))
    return (means,std)

def examineDigits():
    digitsPattern = "/Users/frl/Documents/Meins/Coding/HackerSchool/fizz/pesubmit/captchaExamples/digits/[0-9]-v[0-9][0-9][0-9].png"
    featuresPerDigit = [[]]*10
    oldDigit, featuresList = 0, []
    # first, we collect the list of features for each version of each digit
    for digitPath in glob.glob(digitsPattern): # NOTE: we're relying on pulling in the files alphabetically here
        digit = int(digitPath[-10])
        if digit != oldDigit:
            # we've reached a new digit, so let's store features of all versions of this digit (the featureList)
            featuresPerDigit[oldDigit] = featuresList
            oldDigit, featuresList = digit, []
        # read the next digit image
        i1 = png.Reader(digitPath)
        pixels = list(i1.asFloat()[2])
        # and extract its features
        features = featureExtraction(pixels)
        featuresList.append(features)

    # append featuresList for 9
    featuresPerDigit[9] = featuresList

    # now we can analyze these features
    # featuresPerDigit now is an array with 10 elements 0..9, with 
    # element k an array of the different versions of digit k, and for each version an array of features

    # flatten by one: [item for inner_list in outer_list for item in inner_list]
    glbFeaturesListT = list(zip(*[feature for featureList in featuresPerDigit for feature in featureList]))
    glbmeans = [sum(glb)/len(glb) for glb in glbFeaturesListT]
    glbstd   = [math.sqrt(   sum( (f-fbar)**2 for f in glb ) /len(glb) )  for glb,fbar in zip(glbFeaturesListT,glbmeans)]
    print("Avg:","glb",(" Feat: [" + "{}, "*10).format(*glbmeans),"],")
    print("Std:","glb",(" Feat: [" + "{}, "*10).format(*glbstd),"],")

    # normalize
    featuresPerDigit = [[[(f-m)/s for f,m,s in zip(featuresOneVersion,glbmeans,glbstd)] for featuresOneVersion in featuresOneDigit] for featuresOneDigit in featuresPerDigit]

    # # test - expect 0,1
    # glbFeaturesListT = list(zip(*[feature for featureList in featuresPerDigit for feature in featureList]))
    # glbmeans = [sum(glb)/len(glb) for glb in glbFeaturesListT]
    # glbstd   = [math.sqrt(   sum( (f-fbar)**2 for f in glb ) /len(glb) )  for glb,fbar in zip(glbFeaturesListT,glbmeans)]
    # print("Avg:",digit,(" Feat:" + "{: 6.3f} "*10).format(*glbmeans))
    # print("Std:",digit,(" Feat:" + "{: 6.3f} "*10).format(*glbstd))

    meansPerDigit, stdPerDigit = [],[]
    for digit, featuresOneDigit in enumerate (featuresPerDigit):
        featuresOneDigitT = list(zip(*featuresOneDigit))
        means = [sum(vpf)/len(vpf) for vpf in featuresOneDigitT]
        std   = [math.sqrt( sum( (f-fbar)**2 for f in vpf ) /len(vpf) ) for vpf,fbar in zip(featuresOneDigitT,means)]
        print("Avg:",digit,(" Feat:" + "{: 6.3f} "*10).format(*means))
        print("Std:",digit,(" Feat:" + "{: 6.3f} "*10).format(*std))
        meansPerDigit.append(means)
        stdPerDigit.append(std)
        
    featuresAllDigitsT = list(zip(*meansPerDigit))
    meansm = [sum(vpf)/len(vpf) for vpf in featuresAllDigitsT]
    stdm   = [math.sqrt( sum( (f-fbar)**2 for f in vpf ) /len(vpf) ) for vpf,fbar in zip(featuresAllDigitsT,means)]
    print("Avg  m:  Feat:" + ("{: 6.3f} "*10).format(*meansm))
    print("Std  m:  Feat:" + ("{: 6.3f} "*10).format(*stdm))

    featuresAllDigitsT = list(zip(*stdPerDigit))
    meanss = [sum(vpf)/len(vpf) for vpf in featuresAllDigitsT]
    stds   = [math.sqrt( sum( (f-fbar)**2 for f in vpf ) /len(vpf) ) for vpf,fbar in zip(featuresAllDigitsT,means)]
    print("Avg  s:  Feat:" + ("{: 6.3f} "*10).format(*meanss))
    print("Std  s:  Feat:" + ("{: 6.3f} "*10).format(*stds))

    # for high quality distinguisher, want low std dev within classes, ie low meanss, and high std dev of means, 
    # ie high stdm
    quality = [s/m for s,m in zip(stdm,meanss)]
    print("quality: Feat:" + ("{: 6.3f} "*10).format(*quality))
    print("number:  Feat:" + ("{: 6.3f} "*10).format(*range(10)))

    # result: good features: 0, 1, 2, 4, 7, 8, 9

    print("\nidx " + ("{: 6.0f} "*10).format(*range(10)))
    for row in meansPerDigit:
        print("[" + ("{: 6.3f}, "*10).format(*row), "]")

    mpdT = list(zip(*meansPerDigit))
    prod = [[0]*10 for dummy in range(10)]
    for x,r1 in enumerate(mpdT):
        print(r1)
        for y,r2 in enumerate(mpdT):
            prod[x][y] = sum(e1*e2 for e1,e2 in zip(r1,r2))

    print("\nidx " + ("{: 6.0f} "*10).format(*range(10)))
    for row in prod:
        print("prod " + ("{: 6.3f} "*10).format(*row))


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
    print(0, ("{: 7.3f} "*10).format(*normalizedFeaturesThisDigit))
    print(bestGuess, ("{: 7.3f} "*10).format(*discrepancyPerGuess))
    return bestGuess


def guessDigits():
    digitsPattern = "/Users/frl/Documents/Meins/Coding/HackerSchool/fizz/pesubmit/captchaExamples/digits/[0-9]-v[0-9][0-9][0-9].png"
    for digitPath in glob.glob(digitsPattern): # NOTE: we're relying on pulling in the files alphabetically here
        # read the next digit image
        digit = int(digitPath[-10])
        i1 = png.Reader(digitPath)
        pixels = list(i1.asFloat()[2])
        print(digit, digitPath)
        guessDigit(pixels)


def findCuts(pixelsBW, numCuts = 5):
    M = len(pixelsBW)
    N = len(pixelsBW[0])
    colSum = [sum(col)/M for col in zip(*pixelsBW)]

    for i in range(N):
        if colSum[i] < 0.999:
            iLeft = i
            print("yay")
            break
    for i in range(N-1,0,-1):
        if colSum[i] < 0.999:
            iRight = i+1
            break
    potential = colSum[iLeft:iRight]

    g = 200
    means = [i/numCuts for i in range(1,numCuts)] # [0.2,0.4,0.6,0.8]
    std = 0.3/numCuts
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




def writeModBWPng(path, suffix, pixels):
    testimageModPath = path+"-"+suffix+".png"
    with open(testimageModPath, 'wb') as f:
        w = png.Writer(len(pixels[0]), len(pixels), greyscale=True, bitdepth=1)
        w.write(f, pixels)

# Notes: 

# login successful:
 # <body>
 #  <div class="noprint" id="message">
 #   Login successful
 #  </div>


# already solved:
    # <div class="noprint" style="text-align:center;">
    #  <form action="problem=50" method="post" name="form">
    #   <table align="center" cellpadding="10" width="400">
    #    <tr>
    #     <td>
    #      <table>
    #       <tr>
    #        <td>
    #         <div style="text-align:right;">
    #          Answer:
    #         </div>
    #        </td>
    #        <td style="text-align:left;">
    #         <b>
    #          997651
    #         </b>
    #        </td>
    #       </tr>
    #       <td colspan="2">
    #        <span style="font-size:90%;color:#999;">
    #         Completed on Thu, 24 Jan 2013, 17:18
    #        </span>

# Correct Captcha, Correct result:
  # <div id="content">
  #   <div>
  #    <img alt="Correct" class="dark_border" src="images/answer_correct.png" style="vertical-align:middle;" title="Correct"/>
  #   </div>
  #   <p>
  #    Congratulations, the answer you gave to problem 128 is correct.
  #   </p>
  #   <p>
  #    You are the 2329th person to have solved this problem.
  #   </p>
  #   <p>

# Correct Captcha, Incorrect result:
    #<div id="content">
    # <div>
    #  <img alt="Wrong" class="dark_border" src="images/answer_wrong.png" style="vertical-align:middle;" title="Wrong"/>
    # </div>
    # <p>
    #  Sorry, but the answer you gave appears to be incorrect.
    # </p>
    # <p>


# Incorrect captcha:
  # <div class="noprint" id="message">
  #  The confirmation code you entered was not valid
  # </div>

# too many submissions:
 # <body>
 #  <div class="noprint" id="message">
 #   You are not permitted to submit another guess within 30 seconds of your previous submission
 #  </div>




if __name__ == "__main__":
    # test1()
    # test2()
    # examineDigits()
    # guessDigits()
    # submit(150,12345)
    pass
