# Vendored from the ALMA pipeline, pipeline/extern/findContinuum.py,
# version 2026.3.0.7 (commit 29cff76c4).
#
# Copyright (C) 2011 Associated Universities, Inc. (AUI), European Southern
# Observatory (ESO), and National Astronomical Observatory of Japan (NAOJ), in
# the framework of the Atacama Large Millimeter/submillimeter Array (ALMA)
# collaboration. All rights reserved.
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version. It is distributed WITHOUT ANY WARRANTY; see the GNU Lesser
# General Public License <https://www.gnu.org/licenses/> for details.
#
# AstroVIPER changes: the CASA tasks and tools are imported from
# casa_image_tasks (AstroVIPER implementations on xradio images), and the
# atmospheric transmission and sky temperature come from the caller instead of
# CASA's atmosphere tool. Nothing else is changed.
"""
This file implements the algorithm to determine continuum channel ranges to 
use from an ALMA image cube (dirty or clean) in CASA format.  All dependencies 
on what were originally analysisUtils functions have been pasted into this 
file for convenience.  This function is meant to be run inside CASA.  
Simple usage:
  import findContinuum as fc
  fc.findContinuum('my_dirty_cube.image')
This file can be found in a typical pipeline distribution directory, e.g.:
/lustre/naasc/sciops/comm/rindebet/pipeline/branches/trunk/pipeline/extern
As of March 7, 2019 (version 3.36), it is compatible with both python 2 and 3.

Code changes for Pipeline2026 (as of June 29, 2026)
1) Fix for PIPE-3102
2) Convert np.int64/float64 to int/float for lists that are printed to the log
3) Fix for PIPE-3103 (incl. typo found by Rui in code review)
4) Fix for PIPE-3107
5) Fix for PIPE-850 (y-axis limits for strong masers)
6) Moved the automatic check of singleContinuum upward prior to first use of 
   singleContinuum in findContinuum().  (Does not affect pipeline use case)

Code changes for Pipeline2025 (as of July 18, 2025)
0) Fix crash in recalcMomDiffSNR (not used by pipeline)
1) Prevent crash due to no pyfits module available in CASA 6.6.6.
2) Change np.string_ to np.bytes_
3) Fix for PIPE-2417 (erroneous AllCont = True when intersecting channel ranges 
   in onlyExtraMask/autoLower/autoLower2)
4) Fix for PIPE-2374
5) Fix for PIPE-2544  (put onto main in mid-April 2025)
6) Prevent incorrect warning upon import in CASA 4.7
7) Prevent undefined variable jointMaskTemp (PIPEREQ-368 / PIPE-2563)
8) Fix for PIPE-2261 (touch original momDiff image when reverting)
9) Fix for PIPE-2355 (draw AllCont label on plot)
10) Fixes for PIPE-2673 (remove DeprecationWarnings)


Code changes for Pipeline2024 (as of July 30, 2024)
0) No longer disable onlyExtraMask when returnBluePoints==True
1) No longer search for vis in cube dir if not specified, only if vis=='auto'
2) Remove two print statements at import
3) Restore PL2022 behavior of allowing autoLower if returnBluePoints==True
4) add tooManyPixels parameter to control amendMaskDecision
5) widen selections narrower than 4 to make them visible but with thinner linewidth
6) add useBaseline option (default to 'auto')
7) force use of middleChannels in extraMask/onlyExtraMask/autoLower/autoLower2
8) import copy
9) change algorithm in robustMADofContinuumRanges to avoid problems with
   taking median of the MADs of lots of narrow ranges; fit slope to medians 
   to remove impact of slope on the global MAD of all findcont channels,
   and use peakSNR>7 instead of 15.   2023-08-08
10) Allow transition names of "Continuum(ID=X)" for X>0 to also be considered 
    continuum if there is only (at most) one appearance of "ID="  2023-08-08
11) Add chans parameter to imageSNR.   2024-04-29
12) Fix for PIPE-2034 (incorrect AllCont display)  2024-04-29
13) Fix for PIPE-2158 (warningStrings is now always a list of length 2, but see 15) 2024-04-29
14) Fix for PIPE-1644 (reverted plot now shows the difSNR value)  2024-04-30
15) Fix for PIPE-1629 (pb mask is applied to joint.mask and joint.mask2) and 
    further fix PIPE-2158     2024-04-30
16) New code for PIPE-2171 (write dirtyCubeStats.txt file)   2024-05-09
17) Fix for PIPE-1579 (annulus fully inscribed)   2024-05-09
18) Fix for PIPE-1657 (speed up pbmom calculation)  2024-05-09
19) Add CASA version to end of name of dirtyCubeStats_CASAx.y.z.txt file. 2024-05-23
20) Reduce the length of cyan line for single channel case    2024-06-01
21) Fix for PIPE-2188 (try harder to avoid LowSpread)  2024-06-03
22) Fix for PIPE-1992: if dynamic range of mom0 is high (>90), raise the mom0minsnr mask threshold to 8
23) Add spectralDynamicRangeString to first line of legend
24) Final fix for PIPE-1629 (pb mask is applied to jointMask when it is pb-based)

Code changes for Pipeline2023 (as of June 12, 2023)
0) Enable amendMaskIterations=4: .autoLower2 will run if LowBW not yet reached
1) Set amendMaskIterations='auto' to invoked amendMaskIterations=4
2) Set madRatioCheck to False instead of True when middle channels used as baseline
3) Add parameter for momDiffLevelForProceeding with default of 8.0
4) Use scalingFactor = 6/7 in extraMask and .onlyExtraMask if groups > 32
5) Add minGroupsForSFCAdjustment as a control parameter, with default of 7
6) Add maxGroups as a control parameter, with a default of 15
7) When entering autoLower, use 6/7 instead of 5/7 if momDiffSNR<momDiffApproachLevel
8) Add myia option to imageChannelFrequencies (to support recalcMomDiffSNR)
9) Compute mom0 and mom8 at the same time in findContinuum, plotPickleFile, 
    recalcMomDiffSNR, and meanSpectrumFromMom0Mom8JointMask()
10) Fix plotPickleFile crash for single channel ranges
11) Support a list of intersectRanges in recalcMomDiffSNR
12) Add useUncorrectedMedian as a manual usage option
13) Add dotted cyan line marking the median of the selected continuum channels 
and display its value in the second line of the legend
14) Elevate storeExtraMask as a main control parameter
15) Add returnBluePoints, enableInitialQuadratic and removeOutlierBaselineChannels 
as main control parameters.
16) Add skipCubeNoise option
17) Add returnMomDiffSNR option (for hif_findcont to set to True)
18) import astropy.io.fits as pyfits, if possible
19) fix for PIPE-1636 (set nbin=1 if 'cont' found in transition name), incl. search for vis in cube dir
20) Add 'auto' option to enableInitialQuadratic (but keep True as the default)
21) import gaussian_filter from scipy.ndimage (instead of scipy.ndimage.filters)
22) Fix onlyExtraMaskYesOrNo to allow YesMom when skipCubeNoise is set to True by the user
23) Raise momDiffLevel by 3.5 (from 8 to 11.5) for 7m SingleContinuum to get more AllCont
24) Avoid fix for CAS-13894 in casa >= 6.5.3
25) Add amendMaskIterations as parameter to compute4LetterCodeAndUpdateLegend
26) Add spectralDynamicRangeBandWidth parameter for PIPE-1566 with default=None

Code changes for Pipeline2022 from PIPE-1221: (as of August 21, 2022)
0) fix for PIPE-1227 (look for .virtspw first)
1) Do not remove pbmom if it already exists (for speeding up manual use case)
2) import pickle and add useJointMaskPrior control parameter for mom0mom8jointMask
3) PIPE-1433: set definition for TDM for VLA data, add telescope parameter
4) add optional img parameter to plotPickleFile (for manual use case)
5) add optional window parameter and automatic smooth when nbin>1 (PIPE-848)
6) add optional intersectRanges parameter to plotPickleFile (for manual use case)
7) fix for PIPE-1518 and PIPE-1498
8) abbreviate Frequency to Freq. in the upper x-axis label to make room for nbin
9) add removal of median in calculation of initialPeakOverMad (PIPE-848)
10) Show horizontal cyan line even for ranges that are only one channel wide
11) add control parameter for peakOverMadCriterion, set to 65
12) add control parameter for minPeakOverMadForSFCAdjustment, set to 25
13) add additional clause for momDiffSNR<15 in order to adjust sFC in .autoLower
14) prevent undefined var if returnSFC=True
15) in autoLower, change scalingFactor for sFC from 5/7 to 6/7
16) in ['.extraMask','.onlyExtraMask','.autoLower'], set finalSigmaFindContinuum
17) skyThresholds adjusted to 5.0, 0.40
18) change maxMadRatioForSFCAdjustmentInLaterIterations from 1.20 to 1.18
19) in .extraMask, reset sigmaFindContinuum to sigmaFindContinuumForExtraMask=2.5
20) fix crash in representativeSpw, remove call to isSingleContinuum
21) do not revert if momDiffSNR < momDiffLevel
22) cast spw list intersection to int (since msmd produces inconsistent types)
23) fix crash in cubeFrameToTopo when vis not specified

Code changes for Pipeline2021 from PIPE-824: (as of July 20, 2021)
0) fix for PRTSPR-50321
1) change .rstrip('.image')+'.mask' to .replace('.image','.mask')
2) fix for PIPE-1181 (protection for all joint mask pixels between pb=0.20-21)
3) add nbin=1 parameter in preparation for PIPE-849
4) fix for PIPE-1211 (does not revert to first range in Result6)
5) add separate ALL_CONTINUUM fraction threshold of 0.91 for nchan<75 (PIPE-825)
6) fix for PIPE-1213 (and correction to roundFigures, unused by current PL)

Code changes for Pipeline2020: 
0) fix for PIPE-554 (bug found in cube imaging of calibrators by ARI-L project)
1) fix for PIPE-525 (divide by zero in a 130-target MOUS)
2) new feature PIPE-702 (expand mask if mom8fc image has emission outside it)
3) Add warning messages for too little bandwidth or too little spread.
4) if outdir is specified and does not exist, stop and print error message.
5) Modified the is_binary() function to work in both python 2 and 3.
6) Add LSRK frequency ranges in a new final line of the .dat file
7) Increase to 2 significant digits on sigmaFC in the png name
8) Allow manually specified narrow value to be used everywhere instead of 'auto'
9) Print Final selection message to casa log
10) fix bug in convertSelectionIntoChannelList (only relevant if avoidance used)
11) allow single value in convertSelection
12) fix bug in avoidance range when gaps were trimmed (not relevant to PL)
13) remove warning about no beam in cube header
14) Add totalChannels & medianWidthOfRanges parameters+logic to pickAutoTrimChannels()
15) Insure that channel ranges are in increasing order if a second range is added to a solo range
16) add imstatListit=False to all imstat calls
17) add minor ticks to lower x-axis (channel number) of the plot
18) added 21 new control parameters to findContinuum (default values listed) for a total of 98:
   * returnWarnings=False (PL 2020 might set this to True if Dirk has time)
   * sigmaFindContinuumMode='auto' (split from sigmaFindContinuum)
   * returnSigmaFindContinuum=False 
   * enableRejectNarrowInnerWindows=True (to be able to disable it manually)
   * avoidExtremaInNoiseCalcForJointMask=False
   * buildMom8fc=True
   * momentdir=''
   * amendMaskIterations='auto'
   * skipchan=1
   * checkIfMaskWouldHaveBeenAmended=False
   * fontsize=10
   * thresholdForSame=0.01 (to control the new 4-letter codes)
   * keepIntermediatePngs=True
   * enableOnlyExtraMask=True
   * useMomentDiff=True
   * smallBandwidthFraction=0.05
   * smallSpreadFraction=0.33
   * skipAmendMask=False (for manual usage)
   * useAnnulus=True
   * cubeSigmaThreshold=7.5
   * npixThreshold=7 (for onlyExtraMask)
19) Added 27 new functions:
   * round_half_up (to support the same rounding in python 3 as 2)
   * byteDecode (to convert bytes to string for python 3)
   * imageSNR
   * imageSNRAnnulus
   * plotChannelSelections
   * compute4LetterCodeAndUpdateLegend
   * compute4LetterCode
   * cubeNoiseLevel
   * updateChannelRangesOnPlot
   * computeNpixCubeMedian
   * computeNpixMom8Median
   * computeNpixMom8MedianBadAtm
   * replaceLineFullRangesWithNoise
   * gatherWarnings
   * tooLittleBandwidth
   * tooLittleSpread
   * computeSpread
   * amendMaskYesOrNo
   * extraMaskYesOrNo
   * onlyExtraMaskYesOrNo
   * invertChannelRanges
   * robustMADofContinuumRanges
   * findWidestContiguousListInChannelRange
   * imagePercentileNoMask (not used by PL because avoidExtremaInNoiseCalcForJointMask=False)
   * combineContDat (not used by pipeline)
   * getSpwFromPipelineImageName (not used by pipeline)
   * getFieldnameFromPipelineImageName (not used by pipeline)
-Todd Hunter
"""
from __future__ import print_function  # prevents adding old-style print statements

import os
import warnings
import decimal
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)   # for debugging large arrays
#import matplotlib.pyplot as pl  # used through Cycle 7 but avoided with python3 starting in PL2020
#import pylab as pl  # used from Pipeline 2020-2023
import matplotlib.pyplot as pl # restored for Pipeline 2024: PIPE-1865
import matplotlib.ticker
import time as timeUtilities
import pickle
import copy
difSNRFontsizeAdjust = -2
DISABLE_CUBE_NOISE = 1e9  # shortcut for not passing skipCubeNoise all the way through to extraMaskYesOrNo, this is the value that MADCubeOutside is set to if skipCubeNoise is True
ACA_MAX_BASELINE = 60
MAX_RANGES_FOR_IMMOMENTS = 240
try:
    from importlib import reload
except:
    pass  # reload is already available in python 2.x
# Check if this is CASA6  CASA 6
try:
    from astroviper.distributed_applications.imaging.find_continuum.casa_image_tasks import casaVersion
except:
    # either we are importing into python, or CASA < 6
    if os.getenv('CASAPATH') is not None:
        import casadef
        if casadef.casa_version >= '5.0.0':
            import casa as mycasa
            if 'cutool' in dir(mycasa):
                cu = mycasa.cutool()
                casaVersion = '.'.join([str(i) for i in cu.version()[:-1]]) + '-' + str(cu.version()[-1])
            else:
                casaVersion = mycasa.casa['build']['version'].split()[0]
        else:
            casaVersion = casadef.casa_version
    else:
        import sys
        print("findContinuumCycle12: You appear to be importing analysisUtils into python (not CASA). version = ", '.'.join([str(i) for i in sys.version_info[:3]]))
        print("This is not supported.")
        exit

if (casaVersion >= '5.9.9'):
    try:
        import astropy.io.fits as pyfits
    except:
        if casaVersion < '6.6.4':
            with warnings.catch_warnings():
                # ignore pyfits deprecation message, maybe casa will include astropy.io.fits soon
                # Note: you cannot set category=DeprecationWarning in the call to filterwarnings() because the actual 
                #       warning is PyFITSDeprecationWarning, which of course is not defined until you import pyfits!
                warnings.filterwarnings("ignore") #, category=DeprecationWarning)  
                import pyfits
        else:
            print("Neither astropy.io nor pyfits are present.")
else:
    try:
        import pyfits
    except:
        pass

from astroviper.distributed_applications.imaging.find_continuum.casa_image_tasks import (
    attool,
    casalog,
    iatool,
    imcollapse,
    imhead,
    immath,
    immoments,
    imregrid,
    imsmooth,
    imstat,
    imsubimage,
    makemask,
    metool,
    msmdtool,
    mstool,
    qatool,
    rgtool,
    tbtool,
)
#    print("imported casatasks and casatools individually")

if casaVersion < '5.9.9':
    try:
        synthesismaskhandler = casac.synthesismaskhandler
    except:
        print("This casa does not contain casac.synthesismaskhandler(), so the joint mask cannot be pruned.")
    from immath_cli import immath_cli as immath # only used if pbcube is not passed and no emission is found
    from imhead_cli import imhead_cli as imhead
    from imregrid_cli import imregrid_cli as imregrid
    from imsmooth_cli import imsmooth_cli as imsmooth
    from immoments_cli import immoments_cli as immoments
    from makemask_cli import makemask_cli as makemask
    from imsubimage_cli import imsubimage_cli as imsubimage
    from imcollapse_cli import imcollapse_cli as imcollapse
    from imstat_cli import imstat_cli as imstat  # used by computeMadSpectrum
else:
    from astroviper.distributed_applications.imaging.find_continuum.casa_image_tasks import synthesismaskhandler

import warnings
import subprocess
import sys
import scipy
import glob
import shutil
import random
from scipy.stats import scoreatpercentile, percentileofscore
from scipy.ndimage import gaussian_filter

casaVersionString = casaVersion
casaMajorVersion = int(casaVersion[0])

if casaMajorVersion < 5:
    from scipy.stats import nanmean as scipy_nanmean
    import casadef  # This still works in CASA 5.0, but might not someday.
else:
    # scipy.nanmean still exists, but is deprecated in favor of numpy's version
    from numpy import nanmean as scipy_nanmean

def version(showfile=True):
    """
    Returns the CVS revision number.
    """
    myversion = "$Id: findContinuumCycle13.py,v 9.6 2026/06/29 17:37:59 we Exp $" 
    if (showfile):
        print("Loaded from %s" % (__file__))
    return myversion

def casalogPost(mystring, debug=True, priority='INFO'):
    """
    Generates an INFO message prepended with the version number of findContinuum.
    """
    if (debug): print(mystring)
    token = version(False).split()
    origin = token[1].replace(',','_') + token[2]
    casalog.post(mystring.replace('\n',''), origin=origin, priority=priority)
    
SFC_FACTOR_WHEN_MOM8MASK_EXISTS = 0.8  # was 0.6 on Jan 26; 0.5 was too low on 2015.1.00131.S G09_0847+0045 spw 17
                                        # 0.8 was too high on 2018.1.00828.S
ALL_CONTINUUM_CRITERION = 0.925 # for totalChannels >= 75 (was 0.94 for 2019 Jan 17 test  )
ALL_CONTINUUM_CRITERION_TDM_FULLPOL = 0.91 # for totalChannels < 75
DYNAMIC_RANGE_LIMIT_PLOT = 800
AMENDMASK_PIXEL_RATIO_EXCEEDED = -1
AMENDMASK_PIXELS_ABOVE_THRESHOLD_EXCEEDED = -2
NBIN_THRESHOLD = 2 # if nbin >= this value, then apply smoothing
maxTrimDefault = 20
dpiDefault = 150
imstatListit = False
imstatVerbose = False
meanSpectrumMethods = ['mom0mom8jointMask', 'peakOverMad', 'peakOverRms', 
                       'meanAboveThreshold', 'meanAboveThresholdOverRms', 
                       'meanAboveThresholdOverMad', 'auto'] 

def help(match='', debug=False):
    """
    Print an alphabetized list of all the defined functions at the top level in
    this python file.
    match: limit the list to those functions containing this string (case insensitive)
    -- Todd Hunter
    """
    myfile = __file__
    if (myfile[-1] == 'c'):
        if (debug): print("au loaded from .pyc file, looking at .py file instead")
        myfile = myfile[:-1]
    aufile = open(myfile,'r')
    lines = aufile.readlines()
    if (debug): print("Read %d lines from %s" % (len(lines), __file__))
    aufile.close()
    commands = []
    for line in lines:
        if (line.find('def ') == 0):
            commandline = line.split('def ')[1]
            tokens = commandline.split('(')
            if (len(tokens) > 1):
                command = tokens[0]
            else:
                command = commandline.strip('\n\r').strip('\r\n')
            if (match == '' or command.lower().find(match.lower()) >= 0):
                commands.append(command)
    commands.sort()
    for command in commands:
        print(command)

def round_half_up(x):
    return float(decimal.Decimal(float(x)).to_integral_value(rounding=decimal.ROUND_HALF_UP))

def is_binary(filename):
    """
    This function is called by runFindContinuum.
    Return true if the given filename appears to be binary.
    Works in python 2.7 and 3
    """
    f = os.popen('file %s'%filename, 'r')
    result = f.read().find('text')
    f.close() # fix for PL2026
    if result == -1:
        return True
    else:
        return False

# Old method: fails in python 3
#    File is considered to be binary if it contains a NULL byte.
#    This approach returns True for .fits files, but
#    incorrectly reports UTF-16 as binary.
#    with open(filename, 'rb') as f:
#        for block in f:
#            if '\0' in block:
#                return True
#    return False

def getMemorySize():
    """
    This function is called by findContinuum and runFindContinuum.
    """
    try:
        return(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except ValueError:
        # SC_PHYS_PAGES can be missing on OS X
        return int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip())

MOM8MINSNR_DEFAULT = 4
MOM0MINSNR_DEFAULT = 5

def removeIfNecessary(imgname):
    """
    If an image exists, remote its directory tree.
    """
    if os.path.exists(imgname):
        shutil.rmtree(imgname)
        if os.path.exists(imgname): 
            print("WARNING:  shutil.rmtree failed for %s" % imgname)

def findContinuum(img='', pbcube=None, psfcube=None, minbeamfrac=0.3, spw='', 
                  transition='', 
                  baselineModeA='min', baselineModeB='min', sigmaCube=3, 
                  nBaselineChannels=0.19, sigmaFindContinuum='auto', 
                  sigmaFindContinuumMode='auto',
                  verbose=False, png='', pngBasename=False, nanBufferChannels=1,
                  source='', useAbsoluteValue=True, trimChannels='auto', 
                  percentile=20, continuumThreshold=None, narrow='auto', 
                  separator=';', overwrite=True, titleText='', # 25
                  maxTrim=maxTrimDefault, maxTrimFraction=1.0,
                  meanSpectrumFile='', centralArcsec='auto', 
                  alternateDirectory='.',
                  plotAtmosphere='transmission', airmass=1.5, pwv=1.0,
                  channelFractionForSlopeRemoval=0.75, mask='', 
                  invert=False, meanSpectrumMethod='mom0mom8jointMask', 
                  peakFilterFWHM=10,  # 38
                  skyTempThreshold=5.0, # was 1.5 in C4R2, reduced to 1.2 after slope removal added, increased to 5 in PL2022
                  skyTransmissionThreshold=0.40, # was 0.08 prior to PL2022
                  maxGroupsForSkyThreshold=5,
                  minBandwidthFractionForSkyThreshold=0.2, regressionTest=False,
                  quadraticFit=True, triangleFraction=0.83, maxMemory=-1, # 46
                  tdmSkyTempThreshold=0.65, negativeThresholdFactor=1.15,
                  vis='', singleContinuum=None, applyMaskToMask=False, 
                  plotBaselinePoints=True, dropBaselineChannels=2.0,
                  madRatioUpperLimit=1.5, madRatioLowerLimit=1.15,
                  projectCode='', useIAGetProfile=True, 
                  useThresholdWithMask=False, 
                  dpi=dpiDefault, normalizeByMAD='auto', returnSnrs=False, # 61
                  overwriteMoments=False,minPeakOverMadForSFCAdjustment=25,
                  minPixelsInJointMask=3, userJointMask='', signalRatioTier1=0.967, 
                  snrThreshold=23, mom0minsnr=MOM0MINSNR_DEFAULT, mom8minsnr=MOM8MINSNR_DEFAULT, overwriteMasks=True, 
                  rmStatContQuadratic=False, quadraticNsigma=1.8, bidirectionalMaskPhase2=False,
                  makeMovie=False, returnAllContinuumBoolean=True, outdir='', allowBluePruning=True,
                  avoidance='', returnSigmaFindContinuum=False, 
                  enableRejectNarrowInnerWindows=True, avoidExtremaInNoiseCalcForJointMask=False,  # 81
                  buildMom8fc=True, momentdir='',
                  amendMaskIterations='auto', skipchan=1,  # 85
                  checkIfMaskWouldHaveBeenAmended=False, fontsize=10, thresholdForSame=0.01, # 88
                  keepIntermediatePngs=True, returnWarnings=True, enableOnlyExtraMask=True,
                  useMomentDiff=True, smallBandwidthFraction=0.05, smallSpreadFraction=0.33,
                  skipAmendMask=False, useAnnulus=True, cubeSigmaThreshold=7.5, 
                  npixThreshold=7, nbin=1, useJointMaskPrior=False, telescope='ALMA', 
                  window='flat', subimage=False, maxMadRatioForSFCAdjustment=1.18,  # 104
                  maxMadRatioForSFCAdjustmentInLaterIterations=1.18, 
                  sigmaFindContinuumForExtraMask=2.5, momDiffLevel=8.0, 
                  peakOverMadCriterion=65, momDiffLevelForProceeding=8.0, # 109
                  minGroupsForSFCAdjustment=7, maxGroups=15, momDiffApproachLevel=15.0,
                  mom08simultaneous=True, useUncorrectedMedian=False, storeExtraMask=False,
                  returnBluePoints=False, removeOutlierBaselineChannels='',
                  enableInitialQuadratic=True, skipCubeNoise=False, 
                  returnMomDiffSNR=False, spectralDynamicRangeBandWidth=None,
                  tooManyPixels=850, useBaseline='auto'): # 123
    """
    This function calls functions to:
    1) compute a representative 'mean' spectrum of a dirty cube
    2) find the continuum channels 
    3) plot the results and writes a text file with channel ranges and frequency ranges
    It calls runFindContinuum up to 2 times.  
    If meanSpectrumMethod != 'mom0mom8jointMask', then it runs it a second 
    time with a reduced field size if it finds only one range of continuum 
    channels. 

    Please note that the channel ranges do not necessarily map back to 
    the visibility spectra in the measurement set(s).  In order to produce 
    channel ranges that are guaranteed to map back correctly, then you must 
    supply a list of all desired measurement sets to the vis parameter.  In 
    this case, the final line in the .dat file created will contain the 
    channel list in the correct frequency frame and order.

    Returns:
    --------
    * A channel selection string suitable for the spw parameter of clean.
    * The name of the final png produced.
    * The aggregate bandwidth in continuum in GHz.
    If returnAllContinuumBoolean = True, then it also returns a Boolean desribing
      if it thinks the whole spectrum is continuum (True) or not (False)
      Cycle 7 pipeline usage should set returnAllContinuumBoolean = True
    If returnWarnings = True, then it also returns a list of warning strings,
       which may be empty [], for tooLittleBandwidth and/or tooLittleSpread (PL2020+)
       and the name of the joint mask image (PL2022+)
    If returnSnrs = True, then it also returns two more lists: mom0snrs and mom8snrs,
          and the max baseline (floating point meters) inferred from the psf image
    if returnSigmaFindContinuum = True (new in PL2020), then it also returns 17 more things:
       1) the final value used for sigmaFindContinuum (PL2022: at the end of the .original stage)
       2) the Boolean intersectionOfSelections 
       3) number of central ranges that were (or would have been) dropped, summed
          over all runs of findContinuumChannels by the final run of runFindContinuum.
       4) the result of amendMaskYesOrNo (if it was run;  False otherwise)
       5) the result of extraMaskYesOrNo (if it was run;  False otherwise)
       6) the result of extraMaskYesOrNo test 2 (if it was run;  False otherwise)
       7) the result of autoLower
       8) the name of the finalMomDiff image (if amendMask=True) else the final mom8fc image
       9) the logic path code (for amendMask) (pathCode)
      10) the 4-letter code (for amendMask) (momDiffCode)
      11) the SNR in the final momDiff image (momDiffSNR)
      12) name of meanSpectrumFile produced
      13) name of the "_findContinuum.dat" file produced
      14) positiveThreshold (upper black dotted line)
      15) negativeThreshold (lower black dotted line)
      16) median (solid black line)
      17) median of selected continuum channels (dotted cyan line)
    If returnMomDiffSNR = True (new in PL2023) and returnSigmaFindContinuum & returnSNRs=False
      then also return the momDiffSNR value.
    PL2023 use case (returnWarnings=True, returnAllContinuumBoolean=True returnMomDiffSNR=False) 
      will return: 
      selection, png, aggregateBandwidth, allContinuum, warningStrings, jointMask

    Required Inputs:
    ----------------
    img: the image cube to operate upon (as opposed to specifying a
         pre-existing meanSpectrumFile from a previous run)
     --or--
    meanSpectrumFile: an alternative pre-existing ASCII text file to use
         for the mean spectrum, instead of operating on the img to create one.  
    Note: if you specify meanSpectrumFile and vis, you must also specify img, but
          only its metadata will be used.
      Parameters relevant only to meanSpectrumFile:
      * invert: if True, then invert the sign of intensity and add the minimum

    Optional inputs (see also General Parameters below):
    ---------------------------------------------------
    spw: the spw name or number (integer or string integer) to put in the x-axis label;
         also, it will be used to select the spw for which to generate topo channels for the 
         *.dat file if vis is also specified; if vis is specified and spw is 
         not, then it will search the name of the image for an spw number 
         and use it for generating the topo channel list.
    transition: the name of the spectral transition (for the plot title)
    avoidance: a CASA channel selection string to avoid selecting (for manual use);
           invoked at the end of each execution of runFindContinuum()
    meanSpectrumMethod: 'auto', 'mom0mom8jointMask', 'peakOverMad', 'peakOverRms', 
       'meanAboveThreshold', 'meanAboveThresholdOverRms', 'meanAboveThresholdOverMad', 
        where 'peak' refers to the peak in a spatially-smoothed version of cube
        'auto' invokes 'meanAboveThreshold' unless sky temperature range 
           (max-min) from the atmospheric model is more than skyTempThreshold 
           in which case it invokes 'peakOverMad' instead
        ALMA Cycle 5 used 'auto', while ALMA Cycle 6 will use 'mom0mom8jointMask'

        Parameters relevant only to 'mom0mom8jointMask' (i.e., ALMA Cycle 6 onward)
        ---------------------------------------------------------------------
        * nbin: if greater than NBIN_THRESHOLD (4), then smooth spectrum by this
                many channels
        * spectralDynamicRangeBandWidth: frequency string with units; 
               compute nbin based on this value (overrides nbin parameter), 
               example: '5.1MHz'
        * window: the smoothing window to use with nbin (one of: 'gauss',
                 'flat','hanning','hamming','bartlett','blackman')
        * minPeakOverMadForSFCAdjustment: scalar value
        * pbcube: the primary beam response for img, but only used by 
            meanSpectrumMethod='mom0mom8jointMask'. If not specified, 
            it will be searched for by changing file suffix from .residual to .pb
        * psfcube: the psf response for img, but only used by 
            meanSpectrumMethod='mom0mom8jointMask'. If not specified, 
            it will be searched for by changing file suffix from .residual to .psf,
            if still not found, then pruneMask will default to 6 pixels
        * minbeamfrac: only used if psfcube is specified and meanSpectrumMethod='mom0mom8jointMask'
             floating point value or 'auto' for 0.3 for 12m and 0.1 for 7m (maxBaseline<60m)
             Set to zero to prevent pruning of the joint mask entirely; otherwise pruneMask
             will use np.max([4,minbeamfrac*pixelsPerBeam])
        * minPixelsInJointMask: if fewer than these pixels are found (after pruning), then
            use all pixels above pb=0.3 (or equivalent higher value on mitigated images).
            This option was essentially obseleted by the addition of pruneMask with the
            knowledge of beamsize from psfcube.  To avoid having any effect, it should be set 
            to < minbeamfrac*minPixelsPerBeam, which is typically 0.5*7=3.5; so use 3 or less.
        * userJointMask: if specified, use this joint mask instead of computing one
                         (this option is meant for test purposes only, typically *.mask2_bi)
        * useJointMaskPrior: if True, and if the <img>_mom0mom8jointMaskPrior.pkl file exists,
                then it will skip the creation of joint.mask and joint.mask2 on the first 
                iteration of amendMask to save some time. 
        * normalizeByMAD: can be True, False or 'auto' (default); 
            if 'auto', then if atmospheric transmission varies enough across the spw, 
              set to True, otherwise set to False.  
            If True, then request normalization of the resulting mean spectrum by the 
              xmadm spectrum using an outer annulus and outside the joint mask.  The 
              nominal annulus is the 0.2-0.3 pb response, but the range will 
              automatically be scaled upward with minimum value in the pb cube, 
              in order to preserve the fractional number of pixels contained in 
              a single field's 0.2-0.3 annulus.  
        * mom0minsnr: sets the threshold for the mom0 image (i.e. median + mom0minsnr*scaledMAD)
        * mom8minsnr: sets the threshold for the mom8 image (i.e. median + mom8minsnr*scaledMAD)
        * snrThreshold: if SNR is higher than this, then run a phase 2 mask calculation
        * bidirectionalMaskPhase2: True=extend mask to negative values beyond threshold; 
            False=Cycle6+ behavior (True was tried but gives worse results in some cases)
        * enableInitialQuadratic: if True, assess the need for quadratic removal, and use 
             the old-style method to remove it; if 'auto', then set to True only
             if the lowercase version of the source name includes venus, mars,
                 jupiter, saturn, uranus, or neptune
        * rmStatContQuadratic: if True, then do not do the initialQuadratic removal (if 
            enabled and deemed necessary), but instead always perform the new-style 
            quadratic removal
        * makeMovie: if True, then make a png for each quadratic fit as channels are 
             removed one-by-one (in the rmStatContQuadratic option)
        * quadraticNsigma: the stopping threshold to use when ignoring outliers prior to 
            fitting the quadratic (in the rmStatContQuadratic option). To more quickly test 
            different values, be sure to set the meanSpectrumFile option and overwrite=False)
        * allowBluePruning: use continuum range pruning heuristic added in Cycle 6 (default=True)
        * amendMaskIterations (new in PL2020): if 'auto', then choose 0 for TDM and 3 for FDM;
            if 0, then do not attempt any amendMask (i.e. mimic the Cycle 7 pipeline release)
            if >=1, then when finished the initial findContinuum trial, create a mom8fc image and
            mom0fc image scaled by nchan and chanwidth in kms, and subtract them to create momDiff 
            image. If the momDiff image has sufficient signal remaining in it, then expand the
            joint mask, and repeat the findContinuum process with this expanded mask as the
            userJointMask;  if >=2, and there is still sufficient signal in the new momDiff image from
            the second run of findContinuum, then check if further analysis of the extra portion of the 
            mask is necessary. If so, first check if intersecting the ranges of the first two rounds
            eliminates the excess, otherwise create a new spectrum from it and pass it into a third
            round of findContinuum with this as the meanSpectrum file.  If >=3, and there is stil
            sufficient signal in the new momDiff, image then set sigmaFindContinuumMode='autolower'
            and sigmaFindContinuum=most recent value, and run another round of findContinuum.
            If >= 4, and and there is sufficient signal in the new momDiff, run another round
            of autolower.
        * skipAmendMask: if True, then skip the test for amend mask, and move on to onlyExtraMask
        * skipCubeNoise: if True, then do not run cubeNoiseLevel, instead set MADCubeOutside to
                 DISABLE_CUBE_NOISE and the cubeMedian to zero
        * tooManyPixels: value of pixels in mask beyond which amendMask will not 
                proceed, and causes result code 'P#'
        * storeExtraMask: if True, the write the extraMask as a mask image (for debugging)
        * momDiffLevel: if amendMaskIterations>0, sets the level above which pixels are counted.
            For SingleContinuum 7m spws, the value is automatically incremented by 3.5. It is not
            recommended to change this parameter, as it affects many heuristics. Change the next one.
        * momDiffLevelForProceeding: if amendMaskIterations>0, sets the level above pixels must be
            in the mom8-mom0 image difference to be counted when determining whether to amend the mask.
        * useAnnulus: if amendMaskIterations>0, sets whether median/MAD in imageSNR be assessed in 
               annulus but outside the jointmask (True) or simply outside the joint mask (False)
        * subimage: if True, then add 0.2 to both radii of the annulus
        * enableOnlyExtraMask: if False, then do not do the extraMask or onlyExtraMask iterations
        * useMomentDiff: if False, use the mom8fc image to assess AmendMask decision instead of momDiff
        * checkIfMaskWouldHaveBeenAmended (new in PL2020): if True, then if amendMaskIterations=0, 
              then it will still check to see if the mask would have been amended and return the 
              result (assuming that returnSigmaFindContinuum==True) -- for test purposes only
        * maxMadRatioForSFCAdjustment: one of the controls for lowering sigmaFindContinuum in
              the .original amendMask iteration 
        * maxMadRatioForSFCAdjustmentInLaterIterations: one of the controls for lowering 
              sigmaFindContinuum in amendMask iterations after the .original iteration
        
        Parameters relevant only to 'peakOverMad' or 'mom0mom8jointMask':
        -----------------------------------------------------------
        * maxGroups: maximum number of continuum regions below which the potential reduction 
              factor of sigmaFindContinuum will be scaled by (1 - groups/maxGroups)**(bandwidth/1875e6)
              via the following equation:
          sFC_factor = np.max([scalingFactor/7.0, (1-groups/float(maxGroups)) ** (spwBandwidth/1.875e9)])

        Parameters relevant only to meanSpectrumMethod='auto' and 'mom0mom8jointMask':
        ------------------------------------------------------------------------------
        * skyTempThreshold: rms value in Kelvin (for FDM spectra<1000MHz), above which 
            the meanSpectrumMethod is changed in 'auto' mode to peakOverMad; and above
            which normalizeByMAD is set True in meanSpectrumMethod='mom0mom8jointMask'
            when normalizeByMAD='auto' mode
        * tdmSkyTempThreshold: rms value in Kelvin (for TDM and 1875MHz FDM spectra) 
        * skyTransmissionThreshold: threshold on (max-min)/mean, above which the
            meanSpectrumMethod is changed in 'auto' mode to peakOverMad;  and above
            which normalizeByMAD is set True in meanSpectrumMethod='mom0mom8jointMask'
            when normalizeByMAD='auto' mode
        * overwriteMoments: if True, then overwrite any existing moment0 or moment8 image 
        * overwriteMasks: if True, then overwrite any existing moment0 or moment8 mask image 
        * rmStatContQuadratic: if True, then fit and remove quadratic after removing outliers
        * outdir: directory to write the .mom0 and .mom8 images, .png, .dat and meanSpectrum
              (default is same directory as the cube)
        * avoidExtremaInNoiseCalcForJointMask (new in PL2020): experimental Boolean to avoid 
             pixels <5%ile and >95%ile in the chauvenet imstat of mom0 and mom8 images

        Parameters relevant only to 'auto':
        -----------------------------------
        * triangleFraction: remove a triangle shape if the MAD of the dual-linfit residual 
           is less than this fraction*originalMAD; set this parameter to zero to turn it off.

        Parameters relevant only to 'peakOverMad' or 'meanAboveThreshold*':
        -------------------------------------------------------------------
        * maxMemory: behave as if the machine has this many GB of memory
        * nanBufferChannels: when removing or replacing NaNs, do this many extra channels
                       beyond their extent
        * channelFractionForSlopeRemoval: if this many channels are initially selected, 
               then fit and remove a linear slope and re-identify continuum channels
               Set to 1 to turn off.
        * quadraticFit: if True, fit quadratic polynomial to the noise regions when 
            deemed appropriate; Otherwise, fit only a linear slope
        * centralArcsec: radius of central box within which to compute the mean spectrum
            default='auto' means start with whole field, then reduce to 1/10 if only
            one window is found (unless mask is specified); or 'fwhm' for the central square
            with the same radius as the PB FWHM; or a floating point value in arcseconds
        * mask: a mask image preferably equal in shape to the parent image that is used to determine
            the region to compute the noise (outside the mask, i.e. mask=0) and 
            (in the 'meanAboveThresholdMethod') the mean spectrum (inside the mask, i.e. mask=1).  
            The spatial union of all masked pixels in all channels is used as the mask for each channel.
            Option 'auto' will look for the <filename>.mask that matches the <filename>.image
            and if it finds it, it will use it; otherwise it will use no mask.
            If the mask does not match in shape but is multi-channel, it will be regridded to match
            and written out as <filename>.mask.regrid.
            If it matches spatially but is single-channel, this channel will be used for all.
            To convert a .crtf file to a mask, use:
            makemask(mode='copy', inpimage=img, inpmask='chariklo.crtf', output=img+'.mask')
        * applyMaskToMask: if True, apply the mask inside the user mask image to set its masked pixels to 0

        Parameters relevant only to 'peakOverRms' and 'peakOverMad':
        ------------------------------------------------------------
        * peakFilterFWHM: value (in pixels) to presmooth each plane with a Gaussian kernel of
             this width.  Set to 1 to not do any smoothing.
             Default = 10 which is typically about 2 synthesized beams.

        Parameters relevant only to 'meanAboveThreshold':
        -------------------------------------------------
        * useAbsoluteValue: passed to meanSpectrum, then avgOverCube -- take absolute value of
             the cube before producing mean spectrum
        * useThresholdWithMask: if False, then just take mean rather than meanAboveThreshold
             when a mask has been specified
        * useIAGetProfile: if True, then for baselineMode='min', use ia.getprofile instead of 
             ia.getregion and subsequent arithmetic (faster, less memory exhaustive)
        * continuumThreshold: if specified, begin by using only pixels above this intensity level
        * sigmaCube: multiply this value by the MAD to get the threshold above 
               which a voxel is included in the mean spectrum.
        * baselineModeA: 'min' or 'edge', method to define the baseline in meanSpectrum()
            Parameters relevant to 'min':
            -----------------------------
            * percentile: control parameter for 'min'

            Parameters relevant to 'edge':
            * nBaselineChannels: if integer, then the number of channels to use in:
              a) computing the mean spectrum with the 'meanAboveThreshold' methods.
              b) computing the MAD of the lowest/highest channels in findContinuumChannels
                 if float, then the fraction of channels to use (i.e. the percentile)
                 default = 0.19, which is 24 channels (i.e. 12 on each side) of a TDM window

        Parameters relevant only when amendMaskIterations > 0 (new for PL2020)
        ----------------------------------------------------------------------
        * keepIntermediatePngs: if True, keep changing the png name to avoid overwriting them; only
             the final one is returned
        * thresholdForSame: fractional value used to determine if the residual in the mom8fc image
             improved ('L'), worsened ('H') or stayed the same ('S')
        * cubeSigmaThreshold: minimum cube SNR for onlyExtraMask to trigger
        * npixThreshold: minimum momDiff pixels for onlyExtraMask to trigger

    Parameters for function findContinuumChannels (which is called by all meanSpectrum heuristics):
    -----------------------------------------------------------------------------------------------
    baselineModeB: 'min' (default) or 'edge', method to define the baseline 
        Parameters only relevant if baselineModeB='min':
        * dropBaselineChannels: percentage of extreme values to drop
        * removeOutlierBaselineChannels: after dropBaselineChannels, remove >4*scMAD 
              outliers;  options: 'high', 'low', 'both', or '' == none/False
    useBaseline: 'low', 'middle', 'high' or 'auto'; auto tries to automatically pick 'low' for
       spectra that are emission-line dominated, 'high' if absorption line dominated, or 'both' if
       both types of lines are present
    nBaselineChannels: if integer, then the number of channels to use in computing
         standard deviation/MAD of the baseline channels (i.e. the blue points in the plot)
         if float, then it is the fraction of channels to use (i.e. the percentile)
    sigmaFindContinuum: starting value for sigmaFindContinuum; passed to findContinuumChannels, 
        'auto' starts with value:
        TDM: singleContinuum: 9, meanAboveThreshold: 4.5, peakOverMAD: 6.5,
             mom0mom8jointMask: 7.2
        FDM: meanAboveThreshold: 3.5, peakOverMAD: 6.0, 
             mom0mom8jointMask: nchan<750: 4.2 (strong lines) or 4.5 (weak lines) 
                                nchan>=759: 2.5 (strong lines) or 3.2 (weak lines)
        and adjusts it depending on results
    sigmaFindContinuumMode (new in PL2020): 'auto', 'autolower', 'fixed'
    trimChannels: after doing best job of finding continuum, remove this many 
         channels from each edge of each block of channels found (for margin of safety)
         If it is a float between 0..1, then trim this fraction of channels in each 
         group (rounding up). If it is 'auto' (default), use 0.1 but not more than 
         maxTrim channels, and not more than maxTrimFraction
    narrow: the minimum number of channels that a group of channels must have to survive
            if 0<narrow<1, then it is interpreted as the fraction of all
                           channels within identified blocks
            if 'auto', then use int(ceil(log10(nchan)))
    maxTrim: in trimChannels='auto', this is the max channels to trim per group for 
        TDM spws; it is automatically scaled upward for FDM spws.
    maxTrimFraction: in trimChannels='auto', the max fraction of channels to trim per group
    singleContinuum: if True, treat the cube as having come from a Single_Continuum setup;
        For testing purpose, if None, then check the contents of vis (if specified).
    negativeThresholdFactor: scale the nominal negative threshold by this factor (to 
        adjust the sensitivity to absorption features: smaller values=more sensitive)
    signalRatioTier1: threshold for signalRatio, above which we desensitize the level to
        consider line emission in order to avoid small differences in noise levels from 
        affecting the result (e.g. that occur between serial and parallel tclean cubes)
        signalRatio=1 means: no lines seen, while closer to zero: more lines seen
    enableRejectNarrowInnerWindows (new in PL2020): if True, then remove any inner groups 
        of channels that are narrower than both edge windows (when there are 3-15 groups)

    General parameters:
    -------------------
    verbose: if True, then print additional information during processing
    useUncorrectedMedian: if True, then do not adjust the median (solid black line) upward
       from the median of the baseline points (dashed blue line)
    returnBluePoints: if True, then simply set the continuum ranges to the baseline points 
         (i.e. the blue points) inside each run of findContinuumChannels().  If using
         mom0mom8jointMask with amendMask option, then also stop after amendMask (and
         automatically disable the onlyExtraMask step, even if enableOnlyExtraMask=True)
    momentdir (new in PL2020): alternative directory to look for (and write) the 
         moment 0&8 images
    png: the name of the png to produce ('' yields default name)
    pngBasename: if True, then remove the directory from img name before generating png name
    source: the name of the source, to be shown in the title of the spectrum.
            if None, then use imhead('object')
    overwrite: if True, or ASCII file does not exist, then recalculate the mean spectrum
                      writing it to <img>.meanSpectrum
               if False, then read the mean spectrum from the existing ASCII file
    separator: the character to use to separate groups of channels in the string returned
    titleText: default is img name and transition and the control parameter values
    plotAtmosphere: '', 'tsky', or 'transmission'
    airmass: for plot of atmospheric transmission
    skipchan (newly-exposed in PL2020): used in making the spectral plot, the number of highest 
         and lowest intensity channels to avoid when constructing the y-axis range
    pwv: in mm (for plot of atmospheric transmission)
    vis: comma-delimited list or python list of measurement sets to use to convert channel
         ranges to topocentric frequency ranges (for use in uvcontfit or uvcontsub)
         or 'auto' which will look for any/all vis in the cube directory
    plotBaselinePoints: if True, then plot the baseline-defining points as black dots
    fontsize (newly-exposed in PL2020): size to use for channel ranges, axis labels and plot legend
    dpi: dots per inch to use in writing the png (106 produces 861x649 pixels)
            150 produces 1218x918
    projectCode: a string to prepend to the title of the plot (useful for regression testing
        where you can put the page number from the full PDF when running only a subset)
    buildMom8fc (new in PL2020): if True, then when finished, generate "mom8fc" and "mom0fc"
              images using selected channels
    smallBandwidthFraction: threshold for returning a warning for too little bandwidth found
    smallSpreadFraction: threshold for returning a warning for too little bandwidth spread found
    telescope: name of telescope to use (if no image is passed from which to 
        read it from the header)
    window: the smoothing window to apply to the mean spectrum (when nbin>4)
    mom08simultaneous: if True, set moments=[0,8] in one call to immoments instead of two
    """
    executionTimeSeconds = timeUtilities.time()
    if singleContinuum is None and vis == '':
        singleContinuum = False
    if meanSpectrumMethod not in meanSpectrumMethods:
        print("Unrecognized option for meanSpectrumMethod: %s" % meanSpectrumMethod)
        print("Available options: %s " % meanSpectrumMethods)
        return
    if useBaseline not in ['low','high','middle','auto']:
        print("useBaseline must be one of: low, high, middle, auto")
        return
    if img == '' and meanSpectrumFile != '' and vis != '':
        print("If you specify meanSpectrumFile and vis, then you must also specify img (needed only to retrieve the osbserving date and direction).")
        return
    if userJointMask != '' and amendMaskIterations != 0:
        print("You cannot set userJointMask at the same time as amendMaskIterations != 0")
        return
    img = img.rstrip('/')
    if type(centralArcsec) == str:
        if centralArcsec.isdigit():
            centralArcsecValue = centralArcsec
            centralArcsec = float(centralArcsec)
        else:
            centralArcsecValue = "'"+centralArcsec+"'"
    else:
        centralArcsecValue = str(centralArcsec)
    if len(outdir) > 0:
        if not os.path.exists(outdir):
            print("Requested outdir does not exist.")
            return
    if meanSpectrumMethod == 'mom0mom8jointMask':
        validWindows = ['gauss','flat','hanning','hamming','bartlett','blackman']
        if window not in validWindows:
            print("Invalid window: %s, must be one of: " % (window), validWindows)
            return
        if mask != '':
            print("The mask parameter is not relevant for meanSpectrumMethod='mom0mom8jointMask'.  Use the userJointMask parameter instead.")
            return
        if isinstance(trimChannels, (str, np.bytes_)):
            trimChannelsString = "'" + trimChannels + "'"
        else:
            trimChannelsString = str(trimChannels)
        if isinstance(spectralDynamicRangeBandWidth, (str, np.bytes_)):
            spectralDynamicRangeString = "'" + spectralDynamicRangeBandWidth + "'"
        else:
            spectralDynamicRangeString = str(spectralDynamicRangeBandWidth)
        if isinstance(narrow, (str, np.bytes_)):
            narrowString = "'" + narrow + "'"
        else:
            narrowString = str(narrow)
        if isinstance(amendMaskIterations, (str, np.bytes_)):
            amendMaskIterationsString = "'" + amendMaskIterations + "'"
        else:
            amendMaskIterationsString = str(amendMaskIterations)
        casalogPost("\n BEGINNING: %s findContinuum.findContinuum('%s', overwriteMoments=%s, sigmaFindContinuum='%s', sigmaFindContinuumMode='%s', meanSpectrumMethod='%s', meanSpectrumFile='%s', singleContinuum=%s, outdir='%s', userJointMask='%s', useJointMaskPrior=%s, momentdir='%s', amendMaskIterations=%s, nbin=%d, mom0minsnr=%f, mom8minsnr=%f, momDiffLevel=%f, momDiffLevelForProceeding=%f, returnBluePoints=%s, useUncorrectedMedian=%s, removeOutlierBaselineChannels='%s', skipCubeNoise=%s, trimChannels=%s, narrow=%s, spectralDynamicRangeBandWidth=%s, vis='%s')" % (version().split()[2], img, overwriteMoments, str(sigmaFindContinuum), sigmaFindContinuumMode, meanSpectrumMethod, meanSpectrumFile, singleContinuum, outdir, userJointMask, useJointMaskPrior, momentdir, amendMaskIterationsString, nbin, mom0minsnr, mom8minsnr, momDiffLevel, momDiffLevelForProceeding, returnBluePoints, useUncorrectedMedian, removeOutlierBaselineChannels, skipCubeNoise, trimChannelsString, narrowString, spectralDynamicRangeString, vis))
    else:
        casalogPost("\n BEGINNING: %s findContinuum.findContinuum('%s', centralArcsec=%s, mask='%s', overwrite=%s, sigmaFindContinuum='%s', sigmaFindContinuumMode='%s', meanSpectrumMethod='%s', peakFilterFWHM=%.0f, meanSpectrumFile='%s', triangleFraction=%.2f, singleContinuum=%s, useIAGetProfile=%s, outdir='%s', mask='%s', nbin=%d, vis='%s')" % (version().split()[2], img, centralArcsecValue, mask, overwrite, str(sigmaFindContinuum), sigmaFindContinuumMode, meanSpectrumMethod, peakFilterFWHM, meanSpectrumFile, triangleFraction, singleContinuum, useIAGetProfile, outdir, mask, nbin, vis))
    img = img.rstrip('/')
    imageInfo = [] # information returned from getImageInfo
    xyChanRange0 = None # set to None in case it does not get defined (probably not needed)
    if (len(vis) > 0):
        # convert vis to a list (if it is a string)
        # vis is a non-blank list or non-blank string
        if isinstance(vis, (str,np.bytes_)):
            vis = vis.split(',')
        # vis is now assured to be a non-blank list
        for v in vis:
            if not os.path.exists(v):
                print("Could not find measurement set: ", v)
                return
        
    if (img != ''):
        if meanSpectrumFile != '' and overwrite:
            casalogPost("Setting overwrite to False because both img and meanSpectrumFile were specified!")
            overwrite = False
        if (not os.path.exists(img)):
            casalogPost("Could not find image = %s" % (img))
            return
        imageInfo = getImageInfo(img)
        if (imageInfo is None): 
            return 
        bmaj, bmin, bpa, cdelt1, cdelt2, naxis1, naxis2, freq, imgShape, crval1, crval2, maxBaselineIgnore, telescope = imageInfo
        chanInfo = numberOfChannelsInCube(img, returnChannelWidth=True, returnFreqs=True) 
        nchan,firstFreq,lastFreq,channelWidth = chanInfo
        if (nchan < 2):
            casalogPost("You must have more than one channel in the image.")
            return
        channelWidth = abs(channelWidth)
    else:
        # we need to define chanInfo for the tooLittleBandwidth function to work later
        avgspectrum, avgSpectrumNansReplaced, threshold, edgesUsed, nchan, nanmin, firstFreq, lastFreq, centralArcsec, mask, percentagePixelsNotMasked = readPreviousMeanSpectrum(meanSpectrumFile)
        channelWidth = (lastFreq-firstFreq)/(nchan-1)
        chanInfo = [nchan,firstFreq,lastFreq,channelWidth] 
        print("Set chanInfo to ", chanInfo)

    meanFreqHz = np.mean([firstFreq,lastFreq])
    # Look for the .pb image if it was not specified
    if pbcube == '' or pbcube is None:
        pbcube = locatePBCube(img)
        if pbcube is not None:
            if os.path.exists(pbcube):
                casalogPost("Found PB cube: %s" % (pbcube))
        elif amendMaskIterations != 0:
            casalogPost("No PB cube found, cannot use amendMask option.")
            return
    # Look for the .psf image if it was not specified
    if psfcube == '' or psfcube is None:
        if img.find('.residual') >= 0: 
            if os.path.islink(img):
                psfcube = os.readlink(img).replace('.residual','.psf')
            else:
                psfcube = img.replace('.residual','.psf')
            if not os.path.exists(psfcube):
                psfcube = None
            else:
                casalogPost("Found PSF cube: %s" % (psfcube))
    if psfcube == '' or psfcube is None:
        maxBaseline = 0 # i.e. it is not known if no image is passed
    else:
        maxBaseline = getImageInfo(psfcube)[11]
        casalogPost('Inferred max baseline = %f m' % (maxBaseline))
        # copy to the dirty cube for later use in the plot legend in runFindContinuum
        imageInfo[11] = maxBaseline

    # channelWidth, nchan, and maxBaseline are now defined, so we can define amendMaskIterations
    bandwidth = np.abs(nchan*channelWidth) # of the cube
    minRegions = 13 # 2021.1.00265.S spw18
    originalNbin = 1*nbin
    if spectralDynamicRangeBandWidth not in [None,'None']:
        casalogPost('Checking if nbin (%d) is too wide compared to spectralDynamicRangeBandWidth of %s (=linewidth/3)' % (nbin,spectralDynamicRangeBandWidth))
        spectralDynamicRangeBandWidthChannels = parseFrequencyArgumentToHz(spectralDynamicRangeBandWidth) / channelWidth
        casalogPost('spectralDynamicRangeBandWidth = %.2f channels' % (spectralDynamicRangeBandWidthChannels))
        nbin = np.min([nchan // minRegions, int(round(spectralDynamicRangeBandWidthChannels))])
        if nbin < originalNbin:
            casalogPost('Reducing nbin from %d to %d' % (originalNbin, nbin))
    else:
      casalogPost('Checking if nbin (%d) is too wide compared to repBW' % (nbin))
      if len(vis) == 1:
          if vis[0] == 'auto':
              # look for any .ms in the img directory (sometimes needed for transition name and repSpw BW)
              if verbose:
                  print("Searching for vis in cube directory")
              vis = glob.glob(os.path.join(os.path.dirname(img),'*ms'))
              if verbose:
                  print("Found list of vis: ", vis)
      if (nchan / nbin) < minRegions:
#        if len(vis) == 0:
#            # look for any .ms in the img directory
#            print("Searching for vis in cube directory")
#            vis = glob.glob(os.path.join(os.path.dirname(img),'*ms'))
#            print("Found list of vis: ", vis)
        if len(vis) > 0:
            print("vis = %s." % (vis))
            if os.path.exists(vis[0]):
                result = representativeSpwBandwidth(vis[0])
                if result is None:
                    repSpw, repBW, repNchan, minBW, maxBW = surmiseRepresentativeSpw(vis[0])
                else:
                    repBW, repSpw, repNchan, minBW, maxBW = result
                bandwidthForSensitivity = (repBW/repNchan)*nbin
                casalogPost('bandwidthForSensitivity = %.6f GHz' % (bandwidthForSensitivity*1e-9))
                myspw = getSpwFromPipelineImageName(img)
                if myspw is not None:
                    if (minBW < 215e6 and bandwidth > 650e6):
                        # If I am wide (937/1875) and any other window is 
                        # narrow (58/117), then limit nbin
                        if True:
                            nbin = 1
                        else:
                            if maxBaseline <= ACA_MAX_BASELINE:
                                nbin = 3
                            else:
                                nbin = 2
                        casalogPost('Because my spw is wide and another spw is narrow, limiting nbin to %d' % (nbin))
                    elif (bandwidthForSensitivity > bandwidth):
                        # If the bwForSensitivity is 
                        # wider than me, then limit nbin
                        if True:
                            nbin = 1
                        else:
                            if maxBaseline <= ACA_MAX_BASELINE:
                                nbin = 3
                            else:
                                nbin = 2
                        casalogPost('Because my spw bandwidth (%f) is narrower than the bandwidthForSensitivity (%f), limiting nbin to %d' % (bandwidth,bandwidthForSensitivity,nbin))
                    else:
                        if repBW <= 0:
                            casalogPost('rep spw not found from ms.')
                        else:
                            casalogPost('My bandwidth is not much wider than the narrowest spw, nor is it narrower than the bandwidthForSensitivity.')
                        nbin = nchan // minRegions
                        casalogPost('Setting nbin to %d//%d = %d' % (nchan,minRegions,nbin))
                else:
                    casalogPost('spw ID not parsed from img name.')
                    nbin = nchan // minRegions
            else:
                casalogPost('Could not find vis=%s.' % (vis[0]))
                nbin = nchan // minRegions
        else:
            casalogPost('No vis specified. Skipping search for repSpw.')
            nbin = nchan // minRegions
        if nbin > 3:
            casalogPost('For nchan=%d, limiting nbin from %d to %d' % (nchan, originalNbin, nbin))
      else:  #  (nchan / nbin) >= minRegions:
        casalogPost('nchan/nbin = %d/%d = %f >= %d=minRegions' % (nchan,nbin,nchan/nbin,minRegions))
    
    if amendMaskIterations == 'auto':
        if tdmSpectrum(channelWidth, nchan, telescope):
            if maxBaseline > 400:
                amendMaskIterations = 0
                casalogPost('TDM spectrum detected with maxBaseline>400m detected: setting amendMaskIterations=%d' % (amendMaskIterations))
            else:
                amendMaskIterations = 4 # 3
                casalogPost('TDM spectrum detected with maxBaseline<=400m: setting amendMaskIterations=%d' % (amendMaskIterations))
        else:
            amendMaskIterations = 4 # 3
            casalogPost('FDM spectrum detected: setting amendMaskIterations=%d' % (amendMaskIterations))
        if False: # returnBluePoints:  (restriction is no longer desired) 
            if enableOnlyExtraMask:
                casalogPost('returnBluePoints=True: disabling onlyExtraMask')
                enableOnlyExtraMask = False
            if amendMaskIterations > 1:
                casalogPost('returnBluePoints=True: reducing amendMaskIterations from %d to 1' % (amendMaskIterations))
                amendMaskIterations = 1

    if (mask == 'auto'):
        mask = img.replace('.image','.mask')
        if (not os.path.exists(mask)):
            mask = ''
        else:
            maskInfo = getImageInfo(mask)
            maskShape = maskInfo[8]
            if (maskShape == imgShape).all():
                centralArcsec = -1
            else:
                casalogPost("Shape of mask (%s) does not match the image (%s)." % (maskShape,imgShape))
                casalogPost("If you want to automatically regrid the mask, then set its name explicitly.")
                return
    else:
        if mask == False: 
            mask = ''
        if (len(mask) > 0):
            if (not os.path.exists(mask)):
                casalogPost("Could not find image mask = %s" % (mask))
                return
            maskInfo = getImageInfo(mask)
            maskShape = maskInfo[8]
            if (maskShape == imgShape).all():
                casalogPost("Shape of mask matches the image.")
            else:
                casalogPost("Shape of mask (%s) does not match the image (%s)." % (maskShape,imgShape))
                # check if the spatial dimension matches.  If so, then simply add channels
                if (maskShape[0] == imgShape[0] and
                    maskShape[1] == imgShape[1]):
                    myia = iatool()
                    myia.open(img) 
                    axis = findSpectralAxis(myia) # assume img & mask have same spectral axis number
                    myia.close()
                    if (imgShape[axis] != 1):
                        if (os.path.exists(mask+'.regrid') and not overwrite):
                            casalogPost("Regridded mask already exists, so I will use it.")
                        else:
                            casalogPost("Regridding the spectral axis of the mask with replicate.")
                            imregrid(mask, output=mask+'.regrid', template=img,
                                     axes=[axis], replicate=True, 
                                     interpolation='nearest', 
                                     overwrite=overwrite)
                    else:
                        casalogPost("This single plane mask will be used for all channels.")
                else:
                    casalogPost("Regridding the mask spatially and spectrally.")
                    imregrid(mask, output=mask+'.regrid', template=img, asvelocity=False, interpolation='nearest')
                mask = mask+'.regrid'
                maskInfo = getImageInfo(mask)
                maskShape = maskInfo[8]

    bytes = getMemorySize()
    if meanSpectrumMethod == 'mom0mom8jointMask':
        minGroupsForSFCAdjustment = minGroupsForSFCAdjustment # was 8 on april 3, 2018
    else:
        minGroupsForSFCAdjustment = 10
        try:
            hostname = os.getenv('HOST')
        except:
            hostname = "?"
        casalogPost("Total memory on %s = %.3f GB" % (hostname,bytes/(1024.**3)))
        if (maxMemory > 0 and maxMemory < bytes/(1024.**3)):
            bytes = maxMemory*(1024.**3)
            casalogPost("but behaving as if it has only %.3f GB" % (bytes/(1024.**3)))
    meanSpectrumMethodRequested = meanSpectrumMethod
    meanSpectrumMethodMessage = ''
    if img != '':
        npixels = float(nchan)*naxis1*naxis2
    else:
        npixels = 1
    triangularPatternSeen = False
    badAtmosphere = None
    momDiffSNR = -1 # initialize it because it is passed to runFindContinuum for use in .autoLower
    if (source is None or source == ''):
        source = imhead(img, mode='get', hdkey='object')
        print("Read sourcename from cube = %s" % (source))
        if len(vis) > 0:
            print('vis=%s'%(vis))
    if enableInitialQuadratic == 'auto':
        sourceLower = source.lower()
        bigPlanets = ['venus','mars','moon','jupiter','saturn','uranus','neptune']
        enableInitialQuadratic = False
        for bigPlanet in bigPlanets:
            if sourceLower.find(bigPlanet) >= 0:
                enableInitialQuadratic = True
                casalogPost('Found large planet: changing enableInitialQuadratic from "auto" to True.')
    if spw == '':
        # Try to find the name of the spw from the image name
        if os.path.basename(img).find('spw') > 0:
            spw = os.path.basename(img).split('spw')[1].split('.')[0]
    if nbin > 1 and tdmSpectrum(channelWidth, nchan, telescope) and len(vis) > 0:
        # PIPE-1636
        transitionName = transitions(vis[0], spw, source, verbose=verbose)
        if verbose or True:
            casalogPost('%d transition names for source=%s in %s: = %s' % (len(transitionName),source,os.path.basename(vis[0]), str(transitionName)))
        if len(transitionName) > 0:
            transitionName = ','.join(transitionName).lower()
            n_ids = transitionName.count('id=')
            if transitionName.find('cont') >= 0 and \
               n_ids <= 1:   # PL2024
                casalogPost("Resetting nbin=1 because 'cont' found in transition names and only (up to) one ID= is present: %s" % (transitionName))
                #  PL2023 are the lines below
#               transitionName.find('id=0') >= 0 and \
#               transitionName.find('id=1') < 0 and \
#               transitionName.find('id=2') < 0 and \
#               transitionName.find('id=3') < 0 and \
#               transitionName.find('id=4') < 0 and \
#               transitionName.find('id=5') < 0 and \
#               transitionName.find('id=6') < 0 and \
#               transitionName.find('id=7') < 0 and \
#               transitionName.find('id=8') < 0 and \
#               transitionName.find('id=9') < 0:
#                casalogPost("Resetting nbin=1 because 'cont' found in transition names with no non-zero IDs: ", transitionName)
                nbin = 1

    if (meanSpectrumMethod.find('auto') >= 0):
        # Cycle 4+5 Heuristic
        meanSpectrumMethod = 'meanAboveThreshold'
        if (img != ''):
            a,b,c,d,e = atmosphereVariation(img, imageInfo, chanInfo, airmass, pwv, source=source, vis=vis, spw=spw)
            if (b > skyTransmissionThreshold or e > skyTempThreshold):
                meanSpectrumMethod = 'peakOverMad'
                badAtmosphere = True
                meanSpectrumMethodMessage = "Set meanSpectrumMethod='%s' since atmos. variation %.2f>%.2f or %.3f>%.1fK." % (meanSpectrumMethod,b,skyTransmissionThreshold,e,skyTempThreshold)
            elif (e > tdmSkyTempThreshold and abs(channelWidth*nchan) > 1e9): # tdmSpectrum(channelWidth,nchan)):
                meanSpectrumMethod = 'peakOverMad'
                meanSpectrumMethodMessage = "Set meanSpectrumMethod='%s' since atmos. variation %.2f>%.2f or %.3f>%.2fK." % (meanSpectrumMethod,b,skyTransmissionThreshold,e,tdmSkyTempThreshold)
                badAtmosphere = True
            else:
                badAtmosphere = False
                # Maybe comment this out once thresholds are stable
                if abs(channelWidth*nchan > 1e9): # tdmSpectrum(channelWidth,nchan):
                    myThreshold = tdmSkyTempThreshold
                else:
                    myThreshold = skyTempThreshold
                triangularPatternSeen, value = checkForTriangularWavePattern(img,triangleFraction)
                if (triangularPatternSeen):
                    meanSpectrumMethod = 'peakOverMad'
                    meanSpectrumMethodMessage = "Set meanSpectrumMethod='%s' since triangular pattern was seen (%.2f<%.2f)." % (meanSpectrumMethod, value, triangleFraction)
                else:
                    if value == False:
                        triangleMsg = 'slopeTest=F'
                    else:
                        triangleMsg = '%.2f>%.2f' % (value,triangleFraction)
                    meanSpectrumMethodMessage = "Did not change meanSpectrumMethod since atmos. variation %.2f<%.2f & %.3f<%.1fK (%s)." % (b,skyTransmissionThreshold,e,myThreshold,triangleMsg)
#                    meanSpectrumMethodMessage = "Did not change meanSpectrumMethod from %s since atmos. variation %.2f<%.2f & %.3f<%.1fK (%s)." % (meanSpectrumMethod,b,skyTransmissionThreshold,e,myThreshold,triangleMsg)

            casalogPost(meanSpectrumMethodMessage)

    if meanSpectrumMethod == 'mom0mom8jointMask':
        # Cycle 6 Heuristic
        centralArcsecField = None
        centralArcsecLimitedField = None
        if normalizeByMAD == 'auto' and img != '': # second phrase added on March 22, 2019
            a,b,c,d,e = atmosphereVariation(img, imageInfo, chanInfo, airmass=airmass, pwv=pwv, source=source, vis=vis, spw=spw)
            if (b > skyTransmissionThreshold or e > skyTempThreshold):
                normalizeByMAD = True
                meanSpectrumMethodMessage = "will potentially normalize by MAD since atmospheric variation %.2f>%.2f or %.3f>%.1fK." % (b,skyTransmissionThreshold,e,skyTempThreshold)
                badAtmosphere = True
#            elif (e > tdmSkyTempThreshold and abs(channelWidth*nchan) > 1e9): # tdmSpectrum(channelWidth,nchan)): # commented out whole elif when PIPE-848 implemented
#                normalizeByMAD = True
#                meanSpectrumMethodMessage = "will potentially normalize by MAD since atmospheric variation %.2f>%.2f or %.3f>%.2fK." % (b,skyTransmissionThreshold,e,tdmSkyTempThreshold)
#                badAtmosphere = True
            else:
                badAtmosphere = False
                normalizeByMAD = False
                meanSpectrumMethodMessage = "normalizeByMAD='auto' but atmos. variation is too small to use it (%.2f<=%.2f and %.3f<=%.2fK)." % (b,skyTransmissionThreshold,e,skyTempThreshold)
            casalogPost(meanSpectrumMethodMessage)
    else:
        # Cycle 4+5 Heuristic
        maxpixels = bytes/67 # float(1024)*1024*960
        centralArcsecField = centralArcsec                    
        centralArcsecLimitedField = -1
        if (centralArcsec == 'auto' and img != ''):
            pixelsNotAProblem = useIAGetProfile and meanSpectrumMethod in ['meanAboveThreshold','peakOverMad'] and mask != ''
            if (npixels > maxpixels and (not pixelsNotAProblem)):
                casalogPost("Excessive number of pixels (%.0f > %.0f) %dx%dx%d" % (npixels,maxpixels,naxis1,naxis2,nchan))
                totalWidthArcsec = abs(cdelt2*naxis2)
                if (mask == ''):
                    centralArcsecField = totalWidthArcsec*maxpixels/npixels
                else:
                    casalogPost("Finding size of the central square that fully contains the mask.")
                    centralArcsecField = widthOfMaskArcsec(mask, maskInfo)
                    casalogPost("Width = %.3f arcsec" % (centralArcsecField))
                newWidth = int(naxis2*centralArcsecField/totalWidthArcsec)
                centralArcsecLimitedField = float(centralArcsecField)  # Remember in case we need to reinvoke later
                maxpixpos = getMaxpixpos(img)
                if (maxpixpos[0] > naxis2/2 - newWidth/2 and 
                    maxpixpos[0] < naxis2/2 + newWidth/2 and
                    maxpixpos[1] > naxis2/2 - newWidth/2 and 
                    maxpixpos[1] < naxis2/2 + newWidth/2):
                    npixels = float(nchan)*newWidth*newWidth
                    casalogPost('Data max is within the smaller field')
                    casalogPost("Reducing image width examined from %.2f to %.2f arcsec to avoid memory problems." % (totalWidthArcsec,centralArcsecField))
                else:
                    centralArcsecField = centralArcsec
                    if (meanSpectrumMethod == 'peakOverMad'):
                        casalogPost('Data max is NOT within the smaller field. Keeping meanSpectrumMethod of peakOverMad over full field.')
                    else:
                        meanSpectrumMethod = 'peakOverMad'
                        meanSpectrumMethodMessage = "Data max NOT within the smaller field. Switch to meanSpectrumMethod='peakOverMad'"
                        casalogPost(meanSpectrumMethodMessage)
            else:
                casalogPost("Using whole field since npixels=%d < maxpixels=%d" % (npixels, maxpixels))
                centralArcsecField = -1  # use the whole field
        elif (centralArcsec == 'fwhm' and img != ''):
            centralArcsecField = primaryBeamArcsec(frequency=np.mean([firstFreq,lastFreq]),
                                                   showEquation=False)
            npixels = float(nchan)*(centralArcsecField**2/abs(cdelt2)/abs(cdelt1))
        else:  
            centralArcsecField = centralArcsec
            if img != '':
                npixels = float(nchan)*(centralArcsec**2/abs(cdelt2)/abs(cdelt1))
    if amendMaskIterations != 0:
        # 0=normal run, 1=after amending mask (if necessary), 2=extraMask or onlyExtraMask (if necessary) 
        # 3=lower sigmaFindContinuum (if necessary)
        selection = ''
        aggregateBandwidth = 0
        amendMask = True
    else:
        amendMask = False
    intersectionOfSelections = False
    better = ''
    amendMaskDecision = False  # result from amendMaskYesOrNo (need to initialize it here in case it never gets run)
    extraMaskDecision = False  # result from extraMaskYesOrNo (need to initialize it here in case it never gets run)
    extraMaskDecision2 = False  # result from second extraMaskYesOrNo (need to initialize it here in case it never gets run)
    autoLowerDecision = False  # result from third extraMaskYesOrNo (need to initialize it here in case it never gets run)
    autoLower2Decision = False  # result from fourth extraMaskYesOrNo (need to initialize it here in case it never gets run)
    finalMom8fc = ''
    finalMom0fc = ''
    if checkIfMaskWouldHaveBeenAmended:
        buildMom8fc = True # this option needs to be True for this to work

    ##############################################################################################
    # Each iteration is associated with a name, which corresponds to the heuristic that shall be
    # followed.  The name for the next iteration is set before the prior iteration concludes.
    # This name is appended to all output files produced during this iteration to avoid confusion.
    ##############################################################################################
    if amendMaskIterations == 0:
        amendMaskIterationNames = ['']  
    else:
        # Start with just the original loop, additional names will be added later
        amendMaskIterationNames = ['.original'] + ['']*amendMaskIterations
        # other possible names are:  '.amendedMask', '.extraMask', '.onlyExtraMask', '.autoLower', '.autoLower2'
    mom8fc = {}  # dictionary of image names, keyed by amendMaskIterationName
    mom0fc = {}  # dictionary of image names, keyed by amendMaskIterationName
    momDiff = {} # dictionary of image names, keyed by amendMaskIterationName
    idxNonZeroMAD = None
    
    # Define the default name (used if amendMaskIterations==0), so that we can also use it to create 
    # the longer names generated when amendMaskIterations > 0
    if mom08simultaneous:
        momRoot = {} # dictionary of image names, keyed by amendMaskIterationName

    if outdir == '':
        # img is guaranteed to not end in a '/' since img.rstrip('/') is called above
        if mom08simultaneous:
            momRoot[''] = img + '.momfc'
            pbmom = momRoot[''].replace('.residual.momfc','.pbmom')
        else:
            mom8fc[''] = img + '.mom8fc'
            mom0fc[''] = img + '.mom0fc'
            # Make sure that we write the pbmom to the local directory that contains mom8fc and mom0fc
            pbmom = mom0fc[''].replace('.residual.mom0fc','.pbmom')
    else:
        pbmom = os.path.join(outdir,os.path.basename(pbcube)) + 'mom'
        if mom08simultaneous:
            momRoot[''] = os.path.join(outdir,os.path.basename(img)) + '.momfc'
        else:
            mom8fc[''] = os.path.join(outdir,os.path.basename(img)) + '.mom8fc'
            mom0fc[''] = os.path.join(outdir,os.path.basename(img)) + '.mom0fc'

    if vis != '' and singleContinuum is None:  # this is now passed as a parameter from hif_findcont
        # automatic check of singleContinuum if vis parameter is populated but singleContinuum is not
        singleContinuum = isSingleContinuum(vis, spw)
    if (amendMask or checkIfMaskWouldHaveBeenAmended):
#       This is now a control parameter in PL2022, for the manual use case.
#        momDiffLevel = 8.0 # 8.0 is what we want in general (for 16293 maser: 8 gives 6 pix, while 7.5 gives 13 pix)
        #  momDiffLevel gets modified further down by -0.5 if mom8fc is more diagnostic (as it is for that maser)
        if maxBaseline <= ACA_MAX_BASELINE and singleContinuum:
            momDiffLevel += 3.5
            casalogPost('Raising momDiffLevel by 3.5 to %.1f because this is 7m singleContinuum' % (momDiffLevel))
        momDiffLevelBadAtm = 11.5  # for amendMask and extraMask
        momDiffLevelBadAtmOnlyExtraMask = 11.0
        mom8level = 8.5
        mom8levelBadAtm = 12
        cubeLevel = 7.5
        cubeLevel2 = 7.25  # needs to be somewhat smaller than cubeLevel
        momDiffCode = None
        mom8code = None # mom8code
        momDiffPeak0 = None # necessary in case neither AmendMask or ExtraMask is invoked but LowBW+LowSpread triggers a new range
    for amendMaskIteration in range(amendMaskIterations+1):
        if amendMaskIteration <= 1: # prevent undefined var if returnSFC=True
            sigmaFindContinuumAtEndOfOriginalIteration = sigmaFindContinuum
        # On subsequent iterations, the amendMaskIterationName (and hence heuristics to follow)
        # will change based on decisions made from each new .mom8fc image results.
        amendMaskIterationName = amendMaskIterationNames[amendMaskIteration]
        if amendMask:
            if amendMaskIteration > 0:
                casalogPost('At start of amendMaskIteration %d: aggregateBandwidth = %.6f GHz' % (amendMaskIteration, aggregateBandwidth))
            casalogPost("=======================================================================")
            casalogPost("---------- amendMaskIteration %d (%s) -------------------- sFC=%s" % (amendMaskIteration, amendMaskIterationName,str(sigmaFindContinuum)))
            casalogPost("=======================================================================")
            previousPng = png
            previousSelection = selection
            previousAggregateBandwidth = aggregateBandwidth
        iteration = 0  # of runFindContinuum
        fullLegend = False
        if meanSpectrumMethod != 'mom0mom8jointMask':
            # There can be multiple iterations, so highlight the first one in the log.
            casalogPost('---------- runFindContinuum iteration 0')
        if False:
            # dump all parameter values to casa log (for debugging purposes)
            for v,variable in enumerate([img, pbcube, psfcube, minbeamfrac, spw, transition,
                         baselineModeA,baselineModeB, sigmaCube, nBaselineChannels, sigmaFindContinuum,
                         sigmaFindContinuumMode, verbose, png, pngBasename, nanBufferChannels, 
                         source, useAbsoluteValue, trimChannels, percentile, continuumThreshold, narrow, 
                         separator, overwrite, titleText, maxTrim, maxTrimFraction,
                         meanSpectrumFile, centralArcsecField, 
                         channelWidth, alternateDirectory, imageInfo, chanInfo, 
                         plotAtmosphere, airmass, pwv, channelFractionForSlopeRemoval, mask, 
                         invert, meanSpectrumMethod, peakFilterFWHM, 
                         fullLegend, iteration, meanSpectrumMethodMessage,
                         minGroupsForSFCAdjustment, regressionTest, quadraticFit,
                         npixels*1e-6, triangularPatternSeen, maxMemory, negativeThresholdFactor,
                         bytes, singleContinuum, applyMaskToMask, plotBaselinePoints,
                         dropBaselineChannels, madRatioUpperLimit, madRatioLowerLimit, projectCode,
                         useIAGetProfile,useThresholdWithMask, dpi, normalizeByMAD, overwriteMoments,
                         minPeakOverMadForSFCAdjustment, minPixelsInJointMask, userJointMask,
                         signalRatioTier1, snrThreshold, mom0minsnr, mom8minsnr, overwriteMasks,
                         rmStatContQuadratic, quadraticNsigma, 
                         bidirectionalMaskPhase2, makeMovie, outdir, allowBluePruning,
                         avoidance, enableRejectNarrowInnerWindows, 
                         avoidExtremaInNoiseCalcForJointMask, amendMask, momentdir], useUncorrectedMedian=useUncorrectedMedian):
                casalogPost('%d) %s.'%(v,str(variable)))
        if amendMaskIterationName == '.extraMask':
            sigmaFindContinuum = sigmaFindContinuumForExtraMask
        casalogPost("amendMaskIteration%d (%s) runFindContinuum: sFC=%s" % (amendMaskIteration, amendMaskIterationName, str(sigmaFindContinuum)))

        result = runFindContinuum(img, pbcube, psfcube, minbeamfrac, spw, transition, 
                                  baselineModeA,baselineModeB,
                                  sigmaCube, nBaselineChannels, sigmaFindContinuum, sigmaFindContinuumMode,
                                  verbose, png, pngBasename, nanBufferChannels, 
                                  source, useAbsoluteValue, trimChannels, 
                                  percentile, continuumThreshold, narrow, 
                                  separator, overwrite, titleText, 
                                  maxTrim, maxTrimFraction,
                                  meanSpectrumFile, centralArcsecField, 
                                  channelWidth,
                                  alternateDirectory, imageInfo, chanInfo, 
                                  plotAtmosphere, airmass, pwv, 
                                  channelFractionForSlopeRemoval, mask, 
                                  invert, meanSpectrumMethod, peakFilterFWHM, 
                                  fullLegend, iteration, meanSpectrumMethodMessage,
                                  minGroupsForSFCAdjustment=minGroupsForSFCAdjustment,
                                  regressionTest=regressionTest, quadraticFit=quadraticFit,
                                  megapixels=npixels*1e-6, triangularPatternSeen=triangularPatternSeen,
                                  maxMemory=maxMemory, negativeThresholdFactor=negativeThresholdFactor,
                                  byteLimit=bytes, singleContinuum=singleContinuum,
                                  applyMaskToMask=applyMaskToMask, plotBaselinePoints=plotBaselinePoints,
                                  dropBaselineChannels=dropBaselineChannels,
                                  madRatioUpperLimit=madRatioUpperLimit, 
                                  madRatioLowerLimit=madRatioLowerLimit, projectCode=projectCode,
                                  useIAGetProfile=useIAGetProfile,useThresholdWithMask=useThresholdWithMask,
                                  dpi=dpi, normalizeByMAD=normalizeByMAD,
                                  overwriteMoments=overwriteMoments,
                                  minPeakOverMadForSFCAdjustment=minPeakOverMadForSFCAdjustment,
                                  maxMadRatioForSFCAdjustment=maxMadRatioForSFCAdjustment,
                                  maxMadRatioForSFCAdjustmentInLaterIterations=maxMadRatioForSFCAdjustmentInLaterIterations,
                                  minPixelsInJointMask=minPixelsInJointMask, userJointMask=userJointMask,
                                  signalRatioTier1=signalRatioTier1, snrThreshold=snrThreshold, 
                                  mom0minsnr=mom0minsnr, mom8minsnr=mom8minsnr, overwriteMasks=overwriteMasks,
                                  rmStatContQuadratic=rmStatContQuadratic, quadraticNsigma=quadraticNsigma, 
                                  bidirectionalMaskPhase2=bidirectionalMaskPhase2, 
                                  makeMovie=makeMovie, 
                                  outdir=outdir, allowBluePruning=allowBluePruning,
                                  avoidance=avoidance, 
                                  enableRejectNarrowInnerWindows=enableRejectNarrowInnerWindows, 
                                  avoidExtremaInNoiseCalcForJointMask=avoidExtremaInNoiseCalcForJointMask,
                                  amendMask=amendMask, momentdir=momentdir, skipchan=skipchan, 
                                  amendMaskIterationName=amendMaskIterationName, fontsize=fontsize, 
                                  vis=vis, useJointMaskPrior=useJointMaskPrior, nbin=nbin, window=window, 
                                  subimage=subimage, momDiffSNR=momDiffSNR, peakOverMadCriterion=peakOverMadCriterion,
                                  maxGroups=maxGroups, momDiffApproachLevel=momDiffApproachLevel,
                                  mom08simultaneous=mom08simultaneous, 
                                  useUncorrectedMedian=useUncorrectedMedian,
                                  returnBluePoints=returnBluePoints,
                                  removeOutlierBaselineChannels=removeOutlierBaselineChannels, 
                                  enableInitialQuadratic=enableInitialQuadratic,
                                  useBaseline=useBaseline, spectralDynamicRangeString=spectralDynamicRangeString, idxNonZeroMAD=idxNonZeroMAD)
        if result is None:
            return
        if amendMaskIterationName in ['.extraMask','.onlyExtraMask','.autoLower','.autoLower2']:
            # We only want to keep the new selection, not the other info
            selection = result[0]
            selectionPreBluePruning = result[9]   # fix for PIPE-2034
            finalSigmaFindContinuum = result[10]  # new in PL2022
        else:
            selection, mypng, slope, channelWidth, nchan, useLowBaseline, mom0snrs, mom8snrs, useMiddleChannels, selectionPreBluePruning, finalSigmaFindContinuum, jointMask, avgspectrumAboveThreshold, medianTrue, labelDescs, ax1, ax2, positiveThreshold, areaString, rangesDropped, effectiveSigma, baselineMAD, upperXlabel, allBaselineChannelsXY, nbin, initialPeakOverMad, negativeThreshold, xyChanRange0, idxNonZeroMAD = result
        casalogPost('Returned from runFindContinuum with finalSigmaFindContinuum = %.2f' % (finalSigmaFindContinuum))
        if png == '':
            png = mypng
        mytest = False
        if meanSpectrumMethod != 'mom0mom8jointMask':
            # Cycle 4+5 Heuristic
            if (centralArcsec == 'auto' and img != '' and len(selection.split(separator)) < 2):
                # Only one range was found, so look closer into the center for a line
                casalogPost("Only one range of channels was found")
                myselection = selection.split(separator)[0]
                if (myselection.find('~') > 0):
                    a,b = [int(i) for i in myselection.split('~')]
                    print("Comparing range of %d~%d to %d/2" % (a,b,nchan))
                    mytest = b-a+1 > nchan/2
                else:
                    mytest = True
                if (mytest):
                    reductionFactor = 10.0
                    if (naxis1 > 1*6*reductionFactor): # reduced field must be at least 1 beam across (assuming 6 pix per beam)
                        # reduce the field size to one tenth of the previous
                        bmaj, bmin, bpa, cdelt1, cdelt2, naxis1, naxis2, freq, imgShape, crval1, crval2, maxBaselineIgnore, telescope = imageInfo 
                        imageWidthArcsec = 0.5*(np.abs(naxis2*cdelt2) + np.abs(naxis1*cdelt1))
                        npixels /= reductionFactor**2
                        centralArcsecField = imageWidthArcsec/reductionFactor
                        casalogPost("Reducing the field size to 1/10 of previous (%f arcsec to %f arcsec) (%d to %d pixels)" % (imageWidthArcsec,centralArcsecField,naxis1,naxis1/10))
                        # could change the 128 to tdmSpectrum(channelWidth,nchan), but this heuristic may also help
                        # for excerpts of larger cubes with narrower channel widths.
                        if (nBaselineChannels < 1 and nchan <= 128):  
                            nBaselineChannels = float(np.min([0.5, nBaselineChannels*1.5]))
                        overwrite = True
                        casalogPost("Re-running findContinuum over central %.1f arcsec with nBaselineChannels=%g" % (centralArcsecField,nBaselineChannels))
                        iteration += 1
                        result = runFindContinuum(img, pbcube, psfcube, minbeamfrac, spw, transition, baselineModeA, baselineModeB,
                                                  sigmaCube, nBaselineChannels, sigmaFindContinuum, sigmaFindContinuumMode,
                                                  verbose, png, pngBasename, nanBufferChannels, 
                                                  source, useAbsoluteValue, trimChannels, 
                                                  percentile, continuumThreshold, narrow, 
                                                  separator, overwrite, titleText, 
                                                  maxTrim, maxTrimFraction,
                                                  meanSpectrumFile, centralArcsecField, channelWidth,
                                                  alternateDirectory, imageInfo, chanInfo, 
                                                  plotAtmosphere, airmass, pwv, 
                                                  channelFractionForSlopeRemoval, mask, 
                                                  invert, meanSpectrumMethod, peakFilterFWHM, 
                                                  fullLegend,iteration,meanSpectrumMethodMessage,
                                                  minGroupsForSFCAdjustment=minGroupsForSFCAdjustment,
                                                  regressionTest=regressionTest, quadraticFit=quadraticFit,
                                                  megapixels=npixels*1e-6, triangularPatternSeen=triangularPatternSeen,
                                                  maxMemory=maxMemory, negativeThresholdFactor=negativeThresholdFactor,
                                                  byteLimit=bytes, singleContinuum=singleContinuum,
                                                  applyMaskToMask=applyMaskToMask, plotBaselinePoints=plotBaselinePoints,
                                                  dropBaselineChannels=dropBaselineChannels,
                                                  madRatioUpperLimit=madRatioUpperLimit, 
                                                  madRatioLowerLimit=madRatioLowerLimit, projectCode=projectCode,
                                                  useIAGetProfile=useIAGetProfile,useThresholdWithMask=useThresholdWithMask,
                                                  dpi=dpi, overwriteMoments=overwriteMoments,
                                                  minPeakOverMadForSFCAdjustment=minPeakOverMadForSFCAdjustment,
                                                  maxMadRatioForSFCAdjustment=maxMadRatioForSFCAdjustment,
                                                  maxMadRatioForSFCAdjustmentInLaterIterations=maxMadRatioForSFCAdjustmentInLaterIterations,
                                                  minPixelsInJointMask=minPixelsInJointMask, userJointMask=userJointMask,
                                                  signalRatioTier1=signalRatioTier1, snrThreshold=snrThreshold, 
                                                  mom0minsnr=mom0minsnr, mom8minsnr=mom8minsnr, 
                                                  overwriteMasks=overwriteMasks, rmStatContQuadratic=rmStatContQuadratic, 
                                                  quadraticNsigma=quadraticNsigma, bidirectionalMaskPhase2=bidirectionalMaskPhase2, 
                                                  makeMovie=makeMovie, outdir=outdir, allowBluePruning=allowBluePruning, avoidance=avoidance, 
                                                  enableRejectNarrowInnerWindows=enableRejectNarrowInnerWindows, 
                                                  avoidExtremaInNoiseCalcForJointMask=avoidExtremaInNoiseCalcForJointMask, 
                                                  amendMask=amendMask, momentdir=momentdir, skipchan=skipchan,
                                                  amendMaskIterationName=amendMaskIterationName, fontsize=fontsize, 
                                                  vis=vis, useJointMaskPrior=useJointMaskPrior, nbin=nbin, 
                                                  window=window, subimage=subimage, momDiffSNR=momDiffSNR, 
                                                  peakOverMadCriterion=peakOverMadCriterion,
                                                  maxGroups=maxGroups, momDiffApproachLevel=momDiffApproachLevel,
                                                  mom08simultaneous=mom08simultaneous, useUncorrectedMedian=useUncorrectedMedian,
                                                  returnBluePoints=returnBluePoints,
                                                  removeOutlierBaselineChannels=removeOutlierBaselineChannels,
                                                  enableInitialQuadratic=enableInitialQuadratic, useBaseline=useBaseline, spectralDynamicRangeString=spectralDynamicRangeString, idxNonZeroMAD=idxNonZeroMAD)

                        if result is None:
                            return
                        selection, mypng, slope, channelWidth, nchan, useLowBaseline, mom0snrs, mom8snrs, useMiddleChannels, selectionPreBluePruning, finalSigmaFindContinuum, jointMask, avgspectrumAboveThreshold, medianTrue, labelDescs, ax1, ax2, positiveThreshold, areaString, rangesDropped, effectiveSigma, baselineMAD, upperXlabel, allBaselineChannelsXY, nbin, initialPeakOverMad, negativeThreshold, xyChanRange0, idxNonZeroMAD  = result
                        if png == '':
                            png = mypng
                            print("************ b) png passed into initial call was blank")
                    else:
                        casalogPost("*** Not reducing field size since it would be less than 1 beam across")
            else:
                casalogPost("*** Not reducing field size since more than 1 range found")
        aggregateBandwidth = computeBandwidth(selection, channelWidth, 0)
        if (meanSpectrumMethodRequested == 'auto'):
            # Cycle 4+5 Heuristic
            # Here we check to see if we need to switch the method of computing the mean spectrum
            # based on any undesirable characteristics of the results, and if so, re-run it.
            groups = len(selection.split(separator))
            if (aggregateBandwidth <= 0.00001 or 
                  # the following line was an attempt to solve CAS-9269 case reported by Ilsang Yoon for SCOPS-2571
      #            (mytest and meanSpectrumMethod!='peakOverMad' and groups < 2) or
                  (not useLowBaseline and meanSpectrumMethod=='peakOverMad' and not tdmSpectrum(channelWidth,nchan,telescope)) or
                  (meanSpectrumMethod=='peakOverMad' and meanSpectrumMethodMessage != ''
                   and groups > maxGroupsForSkyThreshold
                   # the following line maintains peakOverMad for dataset in CAS-8908 (and also OMC1_NW spw39 in CAS-8720)
                   and aggregateBandwidth < minBandwidthFractionForSkyThreshold*nchan*channelWidth*1e-9
                   and not tdmSpectrum(channelWidth,nchan,telescope))):
                # If less than 10 kHz is found (or two other situations), then try the other approach.
                if (meanSpectrumMethod == 'peakOverMad'):
                    meanSpectrumMethod = 'meanAboveThreshold'
                    if (aggregateBandwidth <= 0.00001):
                        meanSpectrumMethodMessage = "Reverted to meanSpectrumMethod='%s' because no continuum found." % (meanSpectrumMethod)
                        casalogPost("Re-running findContinuum with the other meanSpectrumMethod: %s (because aggregateBW=%eGHz is less than 10kHz)" % (meanSpectrumMethod,aggregateBandwidth))
                    elif (not useLowBaseline and meanSpectrumMethod=='peakOverMad' and not tdmSpectrum(channelWidth,nchan,telescope)):
                        meanSpectrumMethodMessage = "Reverted to meanSpectrumMethod='%s' because useLowBaseline=F." % (meanSpectrumMethod)
                        casalogPost("Re-running findContinuum with the other meanSpectrumMethod: %s (because useLowBaseline=False)" % (meanSpectrumMethod))
                    elif (aggregateBandwidth < minBandwidthFractionForSkyThreshold*nchan*channelWidth*1e-9 and groups>maxGroupsForSkyThreshold):
                        meanSpectrumMethodMessage = "Reverted to meanSpectrumMethod='%s' because groups=%d>%d and not TDM." % (meanSpectrumMethod,groups,maxGroupsForSkyThreshold)
                        casalogPost("Re-running findContinuum with the other meanSpectrumMethod: %s because it is an FDM spectrum with many groups (%d>%d) and aggregate bandwidth (%f GHz) < %.2f of total bandwidth." % (meanSpectrumMethod,groups,maxGroupsForSkyThreshold,aggregateBandwidth,minBandwidthFractionForSkyThreshold))
                    else:
                        if groups < 2:
                            if sigmaFindContinuum == 'auto':
                                # Fix for CAS-9639: strong line near band edge and odd noise characteristic
                                sigmaFindContinuum = 6.5
                                casalogPost('Setting sigmaFindContinuum = %.1f since groups < 2' % sigmaFindContinuum)
                                casalogPost("Re-running findContinuum with meanSpectrumMethod: %s because groups=%d<2." % (meanSpectrumMethod,groups))
                            else: # a value was specified for sigmaFindContinuum
                                if sigmaFindContinuumMode == 'auto':
                                    sigmaFindContinuum += 3.0
                                    meanSpectrumMethodMessage = "Increasing sigmaFindContinuum by 3 to %.1f because groups=%d<2 and not TDM." % (sigmaFindContinuum,groups)
                                    casalogPost("Re-running findContinuum with meanSpectrumMethod: %s because groups=%d<2." % (meanSpectrumMethod,groups))
                                else:
                                    meanSpectrumMethodMessage = ''
                        elif groups > maxGroupsForSkyThreshold:
                            # hot core FDM case
                            meanSpectrumMethodMessage = "Reverted to meanSpectrumMethod='%s' because groups=%d>%d and not TDM." % (meanSpectrumMethod,groups,maxGroupsForSkyThreshold)
                            casalogPost("Re-running findContinuum with meanSpectrumMethod: %s." % (meanSpectrumMethod))
                        else:
                            meanSpectrumMethodMessage = "Reverted to meanSpectrumMethod='%s' because groups=%d and not TDM." % (meanSpectrumMethod,groups)
                            casalogPost("Re-running findContinuum with meanSpectrumMethod: %s." % (meanSpectrumMethod))
                    if (centralArcsecLimitedField > 0):
                        # Re-establish the previous limit
                        centralArcsecField = centralArcsecLimitedField
                        casalogPost("Re-establishing the limit on field width determined earlier: %f arcsec" %(centralArcsecField)) 
                else:
                    meanSpectrumMethod = 'peakOverMad'
                    if (mytest and groups < 2):
                        centralArcsecField = -1
                        casalogPost("Re-running findContinuum with meanSpectrumMethod: %s (because still only 1 group found after zoom)" % (meanSpectrumMethod))
                    else:
                        casalogPost("Re-running findContinuum with the other meanSpectrumMethod: %s (because aggregateBW=%eGHz is less than 10kHz)" % (meanSpectrumMethod,aggregateBandwidth))
                    
                iteration += 1
                if os.path.exists(png):
                    os.remove(png)
                result = runFindContinuum(img, pbcube, psfcube, minbeamfrac, spw, transition, baselineModeA, baselineModeB,
                                          sigmaCube, nBaselineChannels, sigmaFindContinuum, sigmaFindContinuumMode,
                                          verbose, png, pngBasename, nanBufferChannels, 
                                          source, useAbsoluteValue, trimChannels, 
                                          percentile, continuumThreshold, narrow, 
                                          separator, overwrite, titleText, 
                                          maxTrim, maxTrimFraction,
                                          meanSpectrumFile, centralArcsecField, channelWidth,
                                          alternateDirectory, imageInfo, chanInfo, 
                                          plotAtmosphere, airmass, pwv, 
                                          channelFractionForSlopeRemoval, mask, 
                                          invert, meanSpectrumMethod, peakFilterFWHM, 
                                          fullLegend, iteration, 
                                          meanSpectrumMethodMessage,
                                          minGroupsForSFCAdjustment=minGroupsForSFCAdjustment,
                                          regressionTest=regressionTest, quadraticFit=quadraticFit,
                                          megapixels=npixels*1e-6, triangularPatternSeen=triangularPatternSeen,
                                          maxMemory=maxMemory, negativeThresholdFactor=negativeThresholdFactor,
                                          byteLimit=bytes, singleContinuum=singleContinuum, 
                                          applyMaskToMask=applyMaskToMask, plotBaselinePoints=plotBaselinePoints,
                                          dropBaselineChannels=dropBaselineChannels,
                                          madRatioUpperLimit=madRatioUpperLimit, 
                                          madRatioLowerLimit=madRatioLowerLimit, projectCode=projectCode,
                                          useIAGetProfile=useIAGetProfile,useThresholdWithMask=useThresholdWithMask,
                                          dpi=dpi, overwriteMoments=overwriteMoments,
                                          minPeakOverMadForSFCAdjustment=minPeakOverMadForSFCAdjustment, 
                                          maxMadRatioForSFCAdjustment=maxMadRatioForSFCAdjustment,
                                          maxMadRatioForSFCAdjustmentInLaterIterations=maxMadRatioForSFCAdjustmentInLaterIterations,
                                          minPixelsInJointMask=minPixelsInJointMask, userJointMask=userJointMask, 
                                          signalRatioTier1=signalRatioTier1, snrThreshold=snrThreshold, 
                                          mom0minsnr=mom0minsnr, mom8minsnr=mom8minsnr, 
                                          overwriteMasks=overwriteMasks, rmStatContQuadratic=rmStatContQuadratic, 
                                          quadraticNsigma=quadraticNsigma, bidirectionalMaskPhase2=bidirectionalMaskPhase2, 
                                          makeMovie=makeMovie, outdir=outdir, allowBluePruning=allowBluePruning, 
                                          avoidance=avoidance, enableRejectNarrowInnerWindows=enableRejectNarrowInnerWindows, 
                                          avoidExtremaInNoiseCalcForJointMask=avoidExtremaInNoiseCalcForJointMask, 
                                          amendMask=amendMask, momentdir=momentdir, skipchan=skipchan,
                                          amendMaskIterationName=amendMaskIterationName, fontsize=fontsize, 
                                          vis=vis, useJointMaskPrior=useJointMaskPrior, nbin=nbin, window=window, 
                                          subimage=subimage, momDiffSNR=momDiffSNR, peakOverMadCriterion=peakOverMadCriterion,
                                          maxGroups=maxGroups, momDiffApproachLevel=momDiffApproachLevel,
                                          mom08simultaneous=mom08simultaneous, useUncorrectedMedian=useUncorrectedMedian,
                                          returnBluePoints=returnBluePoints,
                                          removeOutlierBaselineChannels=removeOutlierBaselineChannels,
                                          enableInitialQuadratic=enableInitialQuadratic, useBaseline=useBaseline, spectralDynamicRangeString=spectralDynamicRangeString, idxNonZeroMAD=idxNonZeroMAD)
                selection, mypng, slope, channelWidth, nchan, useLowBaseline, mom0snrs, mom8snrs, useMiddleChannels, selectionPreBluePruning, finalSigmaFindContinuum, jointMask, avgspectrumAboveThreshold, medianTrue, labelDescs, ax1, ax2, positiveThreshold, areaString, rangesDropped, effectiveSigma, baselineMAD, upperXlabel, allBaselineChannelsXY, nbin, initialPeakOverMad, negativeThreshold, xyChanRange0, idxNonZeroMAD  = result   # used to say selectionPreBlueTruncation
                if png == '' or amendMask:
                    if png == '':
                        print("************ c) png passed into initial call was blank")
                    png = mypng  # update to the latest png name
            else:
                casalogPost("Not re-running findContinuum with the other method because the results are deemed acceptable.")
#        endif (meanSpectrumMethodRequested == 'auto'):
        if amendMaskIteration == 0:
            originalWarnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction) # only used to prevent reversion
            originalSelection = selection  # only used in the final reversion based on 4-letter code
            originalPng = png  # only used in the final reversion based on 4-letter code
        # At this point, we are done the (potentially multiple) run(s) of runFindContinuum for this amendMaskIteration
        if (amendMask or buildMom8fc) and img != '':
            if amendMaskIteration == 0:
                if pbcube is not None:
                    # create a mean pb image on the first iteration, as it is needed for image statistics
                    #removeIfNecessary(pbmom)  # in case there was a prior run of findContinuum
                    if not os.path.exists(pbmom):
                        if nchan < 16:  # PIPE-1657
                            pbmomChans = ''
                        else:
                            pbmomChans = '%d,%d,%d,%d,%d' % (nchan//8, nchan//4, nchan//2, 3*nchan//4, 7*nchan//8)
                        casalogPost("Running immoments('%s', moments=[-1], outfile='%s', chans='%s')" % (pbcube,pbmom,pbmomChans))
                        immoments(pbcube, moments=[-1], outfile=pbmom, chans=pbmomChans)
                # This mask test should remain as the original joint mask throughout the 
                # process, not change to the amended mask, which is what happens to jointMask 
                # in the return call from runFindContinuum(userJointMask=amendedMask).
                if jointMask is None:
                    jointMaskTest = None
                else:
                    jointMaskTest = '"' + jointMask + '"==0'
                if useAnnulus and pbcube is not None:
                    lowerAnnulusLevel, higherAnnulusLevel = findOuterAnnulusForPBCube(pbcube, False, False, subimage, imageInfo=imageInfo)
                    if jointMaskTest is None:  # this happens when meanSpectrumFile is specified
                        jointMaskTestAnnulus = '"%s">%f && "%s"<%f' % (pbmom,lowerAnnulusLevel,pbmom,higherAnnulusLevel)
                    else:  # this is the pipeline use case
                        jointMaskTestAnnulus = jointMaskTest + ' && "%s">%f && "%s"<%f' % (pbmom,lowerAnnulusLevel,pbmom,higherAnnulusLevel)
                else:
                    jointMaskTestAnnulus = ''

            # start of moved block for Cycle 11 PIPE-1644: we need to run this here so that we can label reverted spectra with difSNR
            ##################################################
            # build a new mom0fc and mom8fc on every iteration
            ##################################################
            if mom08simultaneous:
                momRoot[amendMaskIterationName] = momRoot[''] + amendMaskIterationName  # here is where .original is appended on the first round
                mom8fc[amendMaskIterationName] = momRoot[amendMaskIterationName] + '.maximum'
                mom0fc[amendMaskIterationName] = momRoot[amendMaskIterationName] + '.integrated'
            else:
                mom8fc[amendMaskIterationName] = mom8fc[''] + amendMaskIterationName   # here is where .original is appended on the first round
                mom0fc[amendMaskIterationName] = mom0fc[''] + amendMaskIterationName   # here is where .original is appended on the first round
            mymom8fc = mom8fc[amendMaskIterationName]  # just a short-hand notation for easier use
            mymom0fc = mom0fc[amendMaskIterationName]  # just a short-hand notation for easier use
            removeIfNecessary(mymom8fc)  # in case there was a prior run of findContinuum
            removeIfNecessary(mymom0fc)  # in case there was a prior run of findContinuum
            if mom08simultaneous:
                immoments(img, moments=[0,8], chans=selection, outfile=momRoot[amendMaskIterationName])
            else:
                if not os.path.exists(mymom8fc):  # save time during development phase
                    print("Running immoments('%s', moments=[8], chans='%s', outfile='%s')" % (img,selection,mymom8fc))
                    immoments(img, moments=[8], chans=selection, outfile=mymom8fc)
                if not os.path.exists(mymom0fc):  # save time during development phase
                    print("Running immoments('%s', moments=[0], chans='%s', outfile='%s')" % (img,selection,mymom0fc))
                    immoments(img, moments=[0], chans=selection, outfile=mymom0fc)

            # To-Do PL2024: if dynamic range of mom0 is high, raise the mom0minsnr mask threshold
            
            #################################
            # create scaled version of mom0fc
            #################################
            mom0fcScaled = mymom0fc+'.scaled'
            removeIfNecessary(mom0fcScaled)  # in case there was a prior run of findContinuum
            chanWidthKms = 299792.458*channelWidth/meanFreqHz 
            fcChannels = len(convertSelectionIntoChannelList(selection))
            factor = 1.0/chanWidthKms/fcChannels
            immath([mymom0fc], mode='evalexpr', expr='IM0*%f'%(factor), outfile=mom0fcScaled)
            ##########################################
            # Create a new momDiff on every iteration 
            ##########################################
            momDiff[amendMaskIterationName] = mymom8fc + '.minus.mom0fcScaled' 
            mymomDiff = momDiff[amendMaskIterationName]
            removeIfNecessary(mymomDiff)  # in case there was a prior run of findContinuum
            if pbcube is None:
                mymask = ''
            else:
                mymask = '"%s">0.23' % (pbmom)
            print("immath(['%s','%s'], mode='evalexpr', expr='IM0-IM1', mask='%s', outfile='%s')" % (mymom8fc,mom0fcScaled,mymask,mymomDiff))
            immath([mymom8fc, mom0fcScaled], mode='evalexpr', expr='IM0-IM1', mask=mymask, 
                   outfile=mymomDiff)
            finalMom8fc = mymom8fc
            finalMom0fc = mymom0fc
            finalMomDiff = mymomDiff
            print("Wrote %s" % (mymom8fc))
            useTrimmedMADIfNeeded = False # experimental method that was abandonded
            ##############################################
            # Recalculate stats for mom8fc every iteration
            ##############################################
            casalogPost("%d) Running imageSNR('%s',mask='%s',applyMaskToAll=False)" % (amendMaskIteration, mymom8fc, jointMaskTest))

            # This is the SNR with peak measured over the whole image; but median and MAD from outside mask (and in annulus if useAnnulus=True)
            mom8fcSNR, mom8fcPeak, mom8fcMedian, mom8fcMAD = imageSNR(mymom8fc, mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus, 
                                                                      useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
            casalogPost("%d) Running imageSNR('%s',mask='%s',applyMaskToAll=True)" % (amendMaskIteration, mymom8fc,jointMaskTestAnnulus))
            # This is the SNR outside the mask
            if useAnnulus:
                mom8fcSNRMom8, mom8fcPeakOutside = imageSNRAnnulus(mymom8fc, jointMaskTest, jointMaskTestAnnulus)
            else:
                mom8fcSNRMom8, mom8fcPeakOutside, ignore0, ignore1 = imageSNR(mymom8fc, mask=jointMaskTest, returnAllStats=True, 
                                                                              applyMaskToAll=True)
            mom8fcSum = imstat(mymom8fc, listit=imstatListit)['sum'][0]
            casalogPost("%d) mom8fcMedian = %f, mom8fcScaledMAD = %f,  mom8fcSNRinside = %f, mom8fcSNROutside=%f, mom8fcSum = %f, mom8fcPeak=%f, mom8fcPeakOutside=%f" % (amendMaskIteration, mom8fcMedian, mom8fcMAD, mom8fcSNR, mom8fcSNRMom8, mom8fcSum, mom8fcPeak, mom8fcPeakOutside))

            ###############################################
            # Recalculate stats for momDiff every iteration
            ###############################################
            momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD = imageSNR(mymomDiff, mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus, 
                                                                          useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
            # end of moved block for Cycle 11 PIPE-1644
            
            if amendMaskIteration == 0:  # copied this 'if' statement from above for PIPE-1644
                # save the first png again, with the tag "reverted" added, in case we need it later
                labelDescs[-1].remove()
#                revertedAreaString = areaString + ', reverted'
                revertedAreaString = areaString + ', reverted, difSNR=%.1f' % (momDiffSNR)
                warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                for warning in warnings:
                    if warning.find('amount') > 0:
                        revertedAreaString += ', LowBW'
                    if warning.find('spread') > 0:
                        revertedAreaString += ', LowSpread'
                casalogPost("Drawing new areaString: %s" % (revertedAreaString))
                labelDesc = ax1.text(0.5,0.99-3*0.03, revertedAreaString, transform=ax1.transAxes, ha='center', size=fontsize-1)
                labelDescs.append(labelDesc)
                revertedPng = png.replace('.png','.reverted.png')
                pl.savefig(revertedPng, dpi=dpi)

                # put back the original label
                labelDescs[-1].remove()
                labelDesc = ax1.text(0.5,0.99-3*0.03, areaString, transform=ax1.transAxes, ha='center', size=fontsize-1)
                labelDescs.append(labelDesc)
                pl.draw()
                # Write out the original _findContinuum.dat file, for testing future improvements of heuristics
                if (meanSpectrumFile == ''): 
                    meanSpectrumFile = buildMeanSpectrumFilename(img, meanSpectrumMethod, peakFilterFWHM, amendMaskIterationName, nbin)
                    if outdir != '':
                        meanSpectrumFile = os.path.join(outdir, os.path.basename(meanSpectrumFile))
                # only the base name of the meanSpectrumFile is used by writeContDat, not the contents
                contDat = writeContDat(meanSpectrumFile, selection, png, aggregateBandwidth,
                                      firstFreq, lastFreq, channelWidth, img, imageInfo, 
                                      vis, spw=spw, source=source)
            # endif amendMaskIteration == 0

            
            if amendMaskIterationName in ['.extraMask','.onlyExtraMask','.autoLower','.autoLower2']:
                prefix = amendMaskIterationName + ': '
            else:
                prefix = ''
            casalogPost('%smomDiff: %s' % (prefix,mymomDiff))
            casalogPost('%smomDiffSNR: %f  peakAnywhere: %f  median: %f  scaledMAD: %f' % (prefix,momDiffSNR,momDiffPeak, momDiffMedian, momDiffMAD))
            if not amendMask: # put momDiffSNR label even on the amendMask=False results
                labelDescs[-1].remove()
                areaString += ', difSNR=%.1f' % (momDiffSNR)
                labelDesc = ax1.text(0.5,0.99-3*0.03, areaString, transform=ax1.transAxes, ha='center', size=fontsize+difSNRFontsizeAdjust)
                labelDescs.append(labelDesc)
                pl.savefig(png, dpi=dpi)
            # This is the SNR outside the mask
            if useAnnulus:
                momDiffSNROutside, momDiffPeakOutside = imageSNRAnnulus(mymomDiff, jointMaskTest, jointMaskTestAnnulus)
            else:
                momDiffSNROutside, momDiffPeakOutside, ignore0, ignore1 = imageSNR(mymomDiff, mask=jointMaskTest, returnAllStats=True, 
                                                                                   applyMaskToAll=True)
            momDiffSum = imstat(mymomDiff, listit=imstatListit)['sum'][0]
#        endif (amendMask or buildMom8fc) and img != '':

        if meanSpectrumMethod == 'mom0mom8jointMask' and (amendMask or checkIfMaskWouldHaveBeenAmended):
            if amendMaskIterationName == '.original':
                if badAtmosphere is None:
                    # in case this was not established above (i.e. someone ran it with normalizeByMAD=False)
                    a,b,c,d,e = atmosphereVariation(img, imageInfo, chanInfo, airmass=airmass, pwv=pwv, source=source, vis=vis, spw=spw)
                    badAtmosphere = False
                    if (b > skyTransmissionThreshold or e > skyTempThreshold):
                        badAtmosphere = True
                        meanSpectrumMethodMessage = "setting badAtmosphere=True since atmospheric variation %.2f>%.2f or %.3f>%.1fK." % (b,skyTransmissionThreshold,e,skyTempThreshold)
#                    elif (e > tdmSkyTempThreshold and abs(channelWidth*nchan) > 1e9): # commented out whole elif when PIPE-848 implemented
#                        badAtmosphere = True
#                        meanSpectrumMethodMessage = "setting badAtmosphere=True since atmospheric variation %.2f>%.2f or %.3f>%.2fK." % (b,skyTransmissionThreshold,e,tdmSkyTempThreshold)
                    else:
                        badAtmosphere = False
                        meanSpectrumMethodMessage = "atmospheric variation is considered too small to set badAtmosphere=True"
                    casalogPost(meanSpectrumMethodMessage)
                else:
                    casalogPost('badAtmosphere is already %s' % (badAtmosphere))
                # Moment8 image: Set levels and count number of pixels above each level (and outside the joint mask)
                NpixMom8Median = computeNpixMom8Median(mom8fcMedian, mom8level, mom8fcMAD, mymom8fc, jointMaskTest)
                NpixMom8MedianBadAtm = computeNpixMom8MedianBadAtm(mom8fcMedian, mom8levelBadAtm, mom8fcMAD, mymom8fc, jointMaskTest)
                if badAtmosphere:
                    casalogPost("NpixMom8MedianBadAtm = %d" % (NpixMom8MedianBadAtm))
                else:
                    casalogPost("NpixMom8Median = %d" % (NpixMom8Median))
                if skipCubeNoise:
                    MADCubeOutside = DISABLE_CUBE_NOISE
                    cubeMedian = 0
                else:
                    casalogPost("Running fc.cubeNoiseLevel('%s', pbcube='%s', mask='%s', chans='%s', subimage=%s)" % (img, pbmom, jointMaskTest, selection, subimage))
                    MADCubeOutside, cubeMedian = cubeNoiseLevel(img, pbcube=pbmom, mask=jointMaskTest, chans=selection, subimage=subimage, imageInfo=imageInfo) # no need for jointMaskTestAnnulus since pbmom is passed already
                mom8fcSNRCube = (mom8fcPeakOutside-mom8fcMedian)/MADCubeOutside
                npix = imstat(img, listit=imstatListit)['npts']
                npix = pickFirstElementIfNecessary(npix)   # fix for PIPE-2673
                TenEventSigma = oneEvent(npix, 10)
                casalogPost("%d pixels yields 10-event sigma = %f,  MAD of cube outside joint mask = %f" % (npix, TenEventSigma, MADCubeOutside))
                NpixCubeMedian = computeNpixCubeMedian(mom8fcMedian, cubeLevel, MADCubeOutside, mymom8fc, jointMaskTest)
                NpixCubeMedian2 = computeNpixCubeMedian(mom8fcMedian, cubeLevel2, MADCubeOutside, mymom8fc, jointMaskTest)

                # MomentDiff image: Set levels and count number of pixels above each level (and outside the joint mask)
                # This is the SNR with peak measured over the whole image.
                momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD = imageSNR(mymomDiff, mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus, 
                                                                              useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
                casalogPost('momDiff: %s' % (momDiff))
                casalogPost('momDiffSNR: %f  peakAnywhere: %f  median: %f  scaledMAD: %f' % (momDiffSNR,momDiffPeak, momDiffMedian, momDiffMAD))
                # This is the SNR outside the mask
                if useAnnulus:
                    # Need 2 calls to get peak anywhere outside the joint mask, but median and MAD from the annulus
                    momDiffSNROutside, momDiffPeakOutside = imageSNRAnnulus(mymomDiff, jointMaskTest, jointMaskTestAnnulus)
                else:
                    momDiffSNROutside, momDiffPeakOutside, ignore0, ignore1 = imageSNR(mymomDiff, mask=jointMaskTest, returnAllStats=True, 
                                                                                       applyMaskToAll=True)

                momDiffSNRCube = (momDiffPeakOutside-momDiffMedian)/MADCubeOutside
                casalogPost('momDiffSNRCube: %f  peakOutside: %f  median: %f  cubeScaledMAD: %f' % (momDiffSNRCube, momDiffPeakOutside, momDiffMedian, MADCubeOutside))
                momDiffSum = imstat(mymomDiff, listit=imstatListit)['sum'][0]
                if mom8fcSNRMom8 > 0:
                    momDiffRatio = momDiffSNR/mom8fcSNRMom8
                    casalogPost('Test for reducing momDiffLevel: momDiffSNR/mom8fcSNRMom8 = %f < 0.3?' % (momDiffRatio))
                    if momDiffRatio < 0.3:
                        momDiffLevel -= 0.5
                        casalogPost('Reducing momDiffLevel by 0.5 to %f' % (momDiffLevel))

                NpixMomDiff = computeNpixMom8Median(momDiffMedian, momDiffLevel, momDiffMAD, mymomDiff, jointMaskTest)
                NpixMomDiff2 = computeNpixMom8Median(momDiffMedian, momDiffLevel+0.5, momDiffMAD, mymomDiff, jointMaskTest)
                NpixMomDiffBadAtm = computeNpixMom8MedianBadAtm(momDiffMedian, momDiffLevelBadAtm, momDiffMAD, mymomDiff, jointMaskTest)
                NpixMomDiff2BadAtm = computeNpixMom8MedianBadAtm(momDiffMedian, momDiffLevelBadAtm+0.5, momDiffMAD, mymomDiff, jointMaskTest)
                if badAtmosphere:
                    casalogPost("NpixMomDiffBadAtm = %d" % (NpixMomDiffBadAtm))
                    casalogPost("NpixMomDiff2BadAtm = %d" % (NpixMomDiff2BadAtm))
                else:
                    casalogPost("NpixMomDiff = %d" % (NpixMomDiff))
                    casalogPost("NpixMomDiff2 = %d" % (NpixMomDiff2))
                NpixDiffCubeMedian = computeNpixCubeMedian(momDiffMedian, cubeLevel, MADCubeOutside, mymomDiff, jointMaskTest) # used by amendMaskYesOrNo
                NpixDiffCubeMedian2 = computeNpixCubeMedian(momDiffMedian, cubeLevel2, MADCubeOutside, mymomDiff, jointMaskTest) # used by amendMaskYesOrNo
                if skipAmendMask:
                    casalogPost('Skipping amendMask due to skipAmendMask request')
                    amendMaskDecision = 'No'
                    sigmaUsedToAmendMask = 0
                else:
                    npixels = imstat(mymom8fc, listit=imstatListit)['npts'][0]
                    mymask = '%s && "%s"<0' % (jointMaskTest,mymom8fc)
                    print("Calling imstat('%s', mask='%s')" % (mymom8fc,mymask))
                    mystats = imstat(mymom8fc, mask=mymask, listit=imstatListit)['npts']
                    if len(mystats) == 0:
                        negativePixels = 0
                    else: 
                        negativePixels = mystats[0]
                    fractionNegativePixels = float(negativePixels)/npixels
                    casalogPost('Fraction of negative pixels: %f' % (fractionNegativePixels))
                    if useMomentDiff:
                        # MomDiff Decision
                        casalogPost('------------Assessing moment difference image first--------------')
                        amendMaskDecision, AmendMaskLevel, sigmaUsedToAmendMask = amendMaskYesOrNo(badAtmosphere, momDiffMedian, momDiffSNROutside, momDiffSNRCube, 
                                                                                                   NpixMomDiff, NpixMomDiffBadAtm, NpixDiffCubeMedian, 
                                                                                                   TenEventSigma, momDiffMAD, MADCubeOutside, momDiffLevelForProceeding, 
                                                                                                   momDiffLevelBadAtm, cubeLevel, NpixDiffCubeMedian2,
                                                                                                   fractionNegativePixels, NpixMomDiff2, NpixMomDiff2BadAtm, skipCubeNoise, tooManyPixels)
                    else:
                        # Moment8 decision
                        casalogPost('----------------Assessing moment8 image first------------------')
                        amendMaskDecision, AmendMaskLevel, sigmaUsedToAmendMask = amendMaskYesOrNo(badAtmosphere, mom8fcMedian, mom8fcSNRMom8, mom8fcSNRCube, 
                                                                                                   NpixMom8Median, NpixMom8MedianBadAtm, NpixCubeMedian, 
                                                                                                   TenEventSigma, mom8fcMAD, MADCubeOutside, mom8level, 
                                                                                                   mom8levelBadAtm, cubeLevel, NpixCubeMedian2,
                                                                                                   fractionNegativePixels, skipCubeNoise=skipCubeNoise, tooManyPixels=tooManyPixels)
                if amendMaskDecision == 'No' or not amendMask: 
                    # Normally, the decision is all we need, but we also check the directive amendMask, because it 
                    # could be False if the user set the parameter checkIfMaskWouldHaveBeenAmended=True.
                    casalogPost("Not amending mask")
                    if amendMask and enableOnlyExtraMask and sigmaUsedToAmendMask >= -1: # sigmaUsedToAmendMask=-1 means PixelRatio, -2 means PixelsAboveThreshold are too high, and we want to exit on that
                        # Assess whether onlyExtraMask should be run
                        result = imstat(img, chans=selection, listit=imstatListit)
                        cubePeak = result['max'][0]  # in whole cube (not merely outside the mask), so we need a fresh call
                        casalogPost('channel selection: %s' % (selection))
                        casalogPost("cube peak anywhere = %f, medianOutside = %f, scaledMADoutside = %f,  (peak-medianOutside)/scaledMADoutside=%f" % (cubePeak,cubeMedian,MADCubeOutside,(cubePeak-cubeMedian)/MADCubeOutside))
                        NpixMomDiff = computeNpixMom8Median(momDiffMedian, momDiffLevel, momDiffMAD, mymomDiff, '')
                        NpixMomDiffBadAtm = computeNpixMom8MedianBadAtm(momDiffMedian, momDiffLevelBadAtmOnlyExtraMask, momDiffMAD, mymomDiff, '')
                        NpixCubeDiffMedian = computeNpixCubeMedian(momDiffMedian, cubeLevel, MADCubeOutside, mymomDiff, '') # unused now
                        # This is the one of the remaining places where the choice of useMomentDiff is assumed to be True.
                        extraMaskDecision, ExtraMaskLevel, extraMaskSigmaUsed = onlyExtraMaskYesOrNo(badAtmosphere, momDiffMedian, momDiffSNR, 
                                                                                                     momDiffSNRCube, NpixMomDiff, NpixMomDiffBadAtm,
                                                                                                     NpixCubeDiffMedian, TenEventSigma, momDiffMAD, 
                                                                                                     MADCubeOutside, cubePeak, cubeMedian, 
                                                                                                     momDiffLevelForProceeding, momDiffLevelBadAtmOnlyExtraMask, cubeLevel,
                                                                                                     cubeSigmaThreshold, npixThreshold, skipCubeNoise)
                    warning = tooLittleBandwidth(selection, chanInfo, smallBandwidthFraction)
                    if warning is not None:
                        casalogPost('Current fractional bandwidth is too small to allow Extra Mask.')
                        extraMaskDecision = False
                    else:
                        casalogPost('Current fractional bandwidth is large enough to allow Extra Mask.')
                    if extraMaskDecision in [False,'No']:
                        casalogPost("Only Extra Mask will not be tried")
                        if amendMask:
                            #########################################################
                            # Result0: No Amend Mask, no Extra Mask ---------------------
                            #########################################################
                            # write an 'S' after the number of pixels in the legend (remove the zeros since they are not
                            # yet calculated and will not be used anyway)
                            warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                            channelRanges = len(selection.split(separator))
                            if not useMomentDiff:
                                labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, 
                                                              mom8fcPeak, mom8fcPeakOutside, 
                                                              mom8fcPeakOutside, mom8fcSum, mom8fcSum, # there is no difference between mom8fcSum,mom8fcSum0
                                                              mom8fcMAD, mom8fcMAD, thresholdForSame, 
                                                              areaString, labelDescs, fontsize, ax1,
                                                              amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision,
                                                              intersectionOfSelections, useMomentDiff, 
                                                              warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR, 
                                                                                                     autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                            else:
                                labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                      momDiffPeak, momDiffPeakOutside, 
                                                      momDiffPeakOutside, momDiffSum, momDiffSum, 
                                                      momDiffMAD, momDiffMAD, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                      warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                        autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                            pl.savefig(png, dpi=dpi)
                        break # do not run a second round
                    else:
                        # The following step is needed, e.g. for Nessie_F1 spw 18 and 24
                        casalogPost("Extra Mask will be tried")
                        # remember the previous values in case the extra mask is indeed used
                        mom8fcPeak0 = mom8fcPeak
                        mom8fcPeakOutside0 = mom8fcPeakOutside
                        mom8fcMAD0 = mom8fcMAD
                        mom8fcSum0 = mom8fcSum
                        momDiffPeak0 = momDiffPeak
                        momDiffPeakOutside0 = momDiffPeakOutside
                        momDiffMAD0 = momDiffMAD
                        momDiffSum0 = momDiffSum
                        print("extraMaskSigmaUsed = %f, ExtraMaskLevel = %f" % (extraMaskSigmaUsed,ExtraMaskLevel))
                        print("--------------------------------------------------------------------------")
                        print("momDiff = ", mymomDiff)
                        print("pbmom = ", pbmom)
                        print("--------------------------------------------------------------------------")
                        channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(img, '"%s">%f' % (mymomDiff,ExtraMaskLevel), '"%s">0.23'%(pbmom), imageInfo, 'mean', normalizeByMAD, outdir=outdir, jointMaskForNormalize=userJointMask, subimage=subimage)
                        lineFullSelection = invertChannelRanges(selection, nchan)  # selection = line-free ranges
                        myChannelList = convertSelectionIntoChannelList(selection) # myChannelList = list of line-free channels
                        casalogPost('Continuum channels so far: %s' % (selection))
                        casalogPost('Line-full channels so far: %s' % (lineFullSelection))
                        scaledMADFCSubset, medianFCSubset = robustMADofContinuumRanges(myChannelList, intensity)
                        edgesUsed = 0
                        meanSpectrumFile = mymom8fc + '.meanSpectrumFile_fromOnlyExtraMask_beforeNoiseReplacement'
                        writeMeanSpectrum(meanSpectrumFile, frequency, intensity, intensity, ExtraMaskLevel, 
                                          nchan, edgesUsed, centralArcsec='mom0mom8jointMask')
                        intensity = replaceLineFullRangesWithNoise(intensity, lineFullSelection, medianFCSubset, scaledMADFCSubset)
                        meanSpectrumFile = mymom8fc + '.meanSpectrumFile_fromOnlyExtraMask'
                        writeMeanSpectrum(meanSpectrumFile, frequency, intensity, 
                                          intensity, ExtraMaskLevel, nchan, 
                                          edgesUsed, centralArcsec='mom0mom8jointMask')
                        # update selection0 to hold the amendMask selection
                        selection0 = selection 
                        if keepIntermediatePngs:
                            png = '' # reset name so that new name will be chosen and prior plot not overwritten
                        amendMaskIterationNames[amendMaskIteration+1] = '.onlyExtraMask'
                        signalRatioTier1 = 1.0
                        if False:
                            sigmaFindContinuumMode = 'autolower'
                            sigmaFindContinuum = finalSigmaFindContinuum
                            casalogPost("Using sigmaFindContinuumMode='autolower' with sigmaFindContinuum=%f" % (sigmaFindContinuum))
                        overwrite = False  # do not overwrite the spectrum we just wrote when runFindContinuum runs again!
                        continue  # from  .original iteration to .onlyExtraMask iteration
                # endif amendMaskDecision=='No'

                # Remember the previous values if we are proceeding with amending the mask.
                # These *0's will be used to compute the mom8fc 4-letter code generation
                selection0 = selection 
                mom8fcPeak0 = mom8fcPeak
                mom8fcPeakOutside0 = mom8fcPeakOutside
                mom8fcMAD0 = mom8fcMAD
                mom8fcSum0 = mom8fcSum
                # These *0's will be used to compute the MomDiff 4-letter code generation
                momDiffPeak0 = momDiffPeak
                momDiffPeakOutside0 = momDiffPeakOutside
                momDiffMAD0 = momDiffMAD
                momDiffSum0 = momDiffSum
                if outdir == '':
                    userJointMask =  img + '.amendedJointMask' + amendMaskIterationName
                    if storeExtraMask:
                        addedMask = img + '.addedMask' + amendMaskIterationName
                else:
                    userJointMask = os.path.join(outdir, os.path.basename(img) + '.amendedJointMask' + amendMaskIterationName)
                    if storeExtraMask:
                        addedMask = os.path.join(outdir, os.path.basename(img) + '.addedMask' + amendMaskIterationName)
                # here is the problem where the jointMask was being removed because it had same name as userJointMask
                if useJointMaskPrior:
                    # need to change the name because jointMask was set equal to userJointMask earlier
                    userJointMask += '.prior'
                removeIfNecessary(userJointMask)
                casalogPost("Amending the mask with new fluxThreshold = %f" % (AmendMaskLevel))
                expression = 'iif((IM0==1)||(IM1>%f), 1, 0)' % (AmendMaskLevel)
                print("Running immath(imagename=['%s','%s'], outfile='%s', mode='evalexpr', expr='%s'" % (jointMask,mymom8fc,userJointMask,expression))
                immath(imagename=[jointMask,mymom8fc], outfile=userJointMask, mode='evalexpr', 
                       expr=expression)
                if storeExtraMask:
                    # Make spectrum of the newly-added portion of the amendedJointMask
                    expression = 'iif((IM0==0)&&(IM1==1), 1, 0)'
                    removeIfNecessary(addedMask)
                    print("Running immath(imagename=['%s','%s'], mode='evalexpr', outfile='%s', expr='%s')" % (jointMask,userJointMask,addedMask,expression))
                    immath(imagename=[jointMask,userJointMask], mode='evalexpr', outfile=addedMask, expr=expression)
                    plotStatisticalSpectrumFromMask(img, addedMask, pbcube, statistic='mean', normalizeByMAD=normalizeByMAD, png=addedMask+'.meanspec.png')
                
                maskedPixelsBefore = countPixelsAboveZero(jointMask, pbmom)
                maskedPixelsAfter = countPixelsAboveZero(userJointMask, pbmom) 
                newPixels = maskedPixelsAfter - maskedPixelsBefore
                print("---------------------------------------------------------------------")
                casalogPost("amending mask %s into %s, adding %d-%d=%d pixels" % (jointMask,userJointMask,maskedPixelsAfter,maskedPixelsBefore,newPixels))
                if maskedPixelsBefore > 0:
                    # protect against divide by zero
                    if (float(maskedPixelsAfter)/maskedPixelsBefore) < 2:
                        sigmaFindContinuumMode = 'autolower'
                        sigmaFindContinuum = finalSigmaFindContinuum
                        casalogPost("Setting sigmaFindContinuumMode = %s" % (sigmaFindContinuumMode))
                        casalogPost("Setting sigmaFindContinuum = finalSigmaFindContinuum = %f" % (finalSigmaFindContinuum))
                    else:
                        casalogPost("Not changing sigmaFindContinuumMode for the next run.")
                if keepIntermediatePngs:
                    png = '' # reset name so that new name will be chosen and prior plot not overwritten
                # Set the name/heuristic of the next iteration
                amendMaskIterationNames[amendMaskIteration+1] = '.amendedMask'
                continue  # from  .original iteration to .amendedMask iteration
            # endif amendMaskIterationName == '.original' 
            if amendMaskIterationName == '.amendedMask':  
                print("---------------------------------------------------------------------")
                # mom8fcSNRMom8 (outside) is already calculated fresh (above) on the latest .mom8fc on every iteration
                mom8fcSNRCube = (mom8fcPeakOutside-mom8fcMedian)/MADCubeOutside
                NpixMom8Median = computeNpixMom8Median(mom8fcMedian, np.max([5.5,TenEventSigma]), mom8fcMAD, mymom8fc, jointMaskTest)
                NpixMom8MedianBadAtm = computeNpixMom8MedianBadAtm(mom8fcMedian, 8, mom8fcMAD, mymom8fc, jointMaskTest)
                # use mom8fc image
                NpixCubeMedian = computeNpixMom8Median(mom8fcMedian, np.max([5.0,TenEventSigma]), mom8fcMAD, mymom8fc, jointMaskTest) # unused now
                if useMomentDiff:
                    # These values are no longer used in extraMaskYesOrNo, so could probably not bother calculating them now.
                    NpixMomDiff = computeNpixMom8Median(momDiffMedian, momDiffLevel, momDiffMAD, mymomDiff, '')
                    NpixMomDiffBadAtm = computeNpixMom8MedianBadAtm(momDiffMedian, momDiffLevelBadAtm, momDiffMAD, mymomDiff, '')
                    NpixCubeDiffMedian = computeNpixCubeMedian(momDiffMedian, cubeLevel, MADCubeOutside, mymomDiff, '')  # unused now
                    extraMaskDecision, ExtraMaskLevel, extraMaskSigmaUsed = extraMaskYesOrNo(badAtmosphere, momDiffMedian, momDiffSNR, 
                                                                                             momDiffSNRCube, NpixMomDiff, NpixMomDiffBadAtm, 
                                                                                             NpixCubeDiffMedian, TenEventSigma, momDiffMAD, 
                                                                                             MADCubeOutside, momDiffLevelForProceeding, momDiffLevelBadAtm,
                                                                                             cubeLevel)
                else:
                    extraMaskDecision, ExtraMaskLevel, extraMaskSigmaUsed = extraMaskYesOrNo(badAtmosphere, mom8fcMedian, mom8fcSNRMom8, 
                                                                                             mom8fcSNRCube, NpixMom8Median, NpixMom8MedianBadAtm, 
                                                                                             NpixCubeMedian, TenEventSigma, mom8fcMAD, 
                                                                                             MADCubeOutside, mom8level, mom8levelBadAtm,
                                                                                             cubeLevel)
                casalogPost('momDiff: %s' % (mymomDiff))
                casalogPost('momDiffSNR: %f  peakAnywhere: %f  median: %f  scaledMAD: %f' % (momDiffSNR,momDiffPeak, momDiffMedian, momDiffMAD))
                casalogPost('momDiffSNRCube: %f  scaledMAD: %f' % (momDiffSNRCube, MADCubeOutside))
                warning = tooLittleBandwidth(selection, chanInfo, smallBandwidthFraction)
                if warning is not None:
                    casalogPost('Current fractional bandwidth is too small to allow Extra Mask.')
                    extraMaskDecision = False
                else:
                    casalogPost('Current fractional bandwidth is enough to allow Extra Mask.')
                if extraMaskDecision == 'No': # crystal relies on this when using returnBluePoints # or returnBluePoints: # latter will prevent ADIX  
                    # If No, we are done
                    ########################################
                    # Result1: Amended Mask but no Extra Mask
                    ########################################
                    if warning is None:
                        casalogPost("Extra Mask not deemed necessary, so we are done.")
                    # derive 4-letter codes and update the final line of plot legend
                    warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                    channelRanges = len(selection.split(separator))
                    if not useMomentDiff:
                        labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, mom8fcPeak0, mom8fcPeakOutside, 
                                                      mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                                      mom8fcMAD, mom8fcMAD0, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                                                             warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                             autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                    else:
                        # Needs momDiff 4-letter code
                        labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                      momDiffPeak0, momDiffPeakOutside, 
                                                      momDiffPeakOutside0, momDiffSum, momDiffSum0, 
                                                      momDiffMAD, momDiffMAD0, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                                                                warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                    pl.savefig(png, dpi=dpi)
                    break
                else:
                    casalogPost("Extra Mask is needed")
                #################################################################
                # There is still excess emission.
                # Can intersect help?  build mom8fc.amendedMask - mom8fc.original
                #################################################################
                if useMomentDiff:
                    momDiff['.diff'] = momDiff['.amendedMask'] + '.diff'
                    removeIfNecessary(momDiff['.diff'])
                    immath([momDiff['.original'], momDiff['.amendedMask']], expr='IM1-IM0', outfile=momDiff['.diff'])
                    if useAnnulus:
                        casalogPost("Running fc.imageSNRAnnulus('%s', '%s', '%s', applyMaskToAll=False)" % (momDiff['.diff'], jointMaskTest, jointMaskTestAnnulus))
#                        diffStats = imageSNRAnnulus(momDiff['.diff'], jointMaskTest, jointMaskTestAnnulus) # this is what we used on July 30
                        diffStats = imageSNRAnnulus(momDiff['.diff'], jointMaskTest, jointMaskTestAnnulus, applyMaskToAll=False) 
                    else:
                        diffStats = imageSNR(momDiff['.diff'], returnAllStats=True, mask=jointMaskTest) 
                else:
                    mom8fc['.diff'] = mom8fc['.amendedMask'] + '.diff'
                    removeIfNecessary(mom8fc['.diff'])
                    immath([mom8fc['.original'], mom8fc['.amendedMask']], expr='IM1-IM0', outfile=mom8fc['.diff'])
                    if useAnnulus:
                        casalogPost("Running fc.imageSNRAnnulus('%s', '%s', '%s', applyMaskToAll=False)" % (mom8fc['.diff'], jointMaskTest, jointMaskTestAnnulus))
                        diffStats = imageSNRAnnulus(mom8fc['.diff'], jointMaskTest, jointMaskTestAnnulus, applyMaskToAll=False) 
                    else:
                        diffStats = imageSNR(mom8fc['.diff'], returnAllStats=True, mask=jointMaskTest)
                if badAtmosphere:
                    testDiffLev = mom8levelBadAtm * np.sqrt(2)
                else:
                    testDiffLev = mom8level * np.sqrt(2)
                if diffStats[0] > testDiffLev:
                    casalogPost('**************************************************************')
                    casalogPost('There is significant emission in difference image (%f > %f):  Needs intersection of ranges.' % (diffStats[0], testDiffLev))
                    casalogPost('**************************************************************')
                    channelList0 = convertSelectionIntoChannelList(selection0)
                    channelList1 = convertSelectionIntoChannelList(selection)
                    channelList = np.intersect1d(channelList0,channelList1)
                    casalogPost("Taking intersection of %d channels with %d channels, to get %d channels" % (len(channelList0), len(channelList1), len(channelList)))
                    casalogPost("first channel list = %s" % (selection0))
                    casalogPost("second channel list = %s" % (selection))
                    if len(channelList) == 0:
                        casalogPost("No intersecting ranges found: Reverting to first channel selection and stopping. (Result2)")
                        channelList = channelList0  # reverting
                        selection = convertChannelListIntoSelection(channelList)
                        ##########################################################################################
                        # Result2: Amended Mask and Extra Mask was tried but reverted due to no remaining channels
                        ##########################################################################################
                        # derive 4-letter codes and update the final line of the plot legend
                        warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                        channelRanges = len(selection.split(separator))
                        if not useMomentDiff:
                            labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, mom8fcPeak0, mom8fcPeakOutside, 
                                                          mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                                          mom8fcMAD, mom8fcMAD0, thresholdForSame, 
                                                          areaString, labelDescs, fontsize, ax1,
                                                          amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision,
                                                          intersectionOfSelections, useMomentDiff, 
                                                          warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                 autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                        else:
                            # Needs momDiff 4-letter code
                            labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                      momDiffPeak0, momDiffPeakOutside, 
                                                      momDiffPeakOutside0, momDiffSum, momDiffSum0, 
                                                      momDiffMAD, momDiffMAD0, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                      warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                    autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                        pl.savefig(png, dpi=dpi)
                        break
                    intersectionOfSelections = True
                    ###############################################################
                    # create mom8fcIntersect from original and amendedMask regions
                    ###############################################################
                    selection = convertChannelListIntoSelection(channelList)
                    casalogPost("(b) Due to use of intersected channels, setting selectionPreBluePruning to ", selection)
                    selectionPreBluePruning = selection  # PIPE-2417
                    casalogPost("Building new mom8fc with chans='%s'" % (selection))
                    if mom08simultaneous:
                        momRoot['.intersect'] = momRoot[''] + '.intersectChannelRanges'
                        mom8fc['.intersect'] = momRoot['.intersect'] + '.maximum'
                        mom0fc['.intersect'] = momRoot['.intersect'] + '.integrated'
                    else:
                        mom8fc['.intersect'] = mom8fc[''] + '.intersectChannelRanges'
                        mom0fc['.intersect'] = mom0fc[''] + '.intersectChannelRanges'
                    mom8fcIntersect = mom8fc['.intersect']  # just a shorthand
                    mom0fcIntersect = mom0fc['.intersect']
                    removeIfNecessary(mom8fcIntersect)
                    removeIfNecessary(mom0fcIntersect)  # in case there was a prior run of findContinuum
                    if mom08simultaneous:
                        immoments(img, moments=[0,8], chans=selection, outfile=momRoot['.intersect'])
                    else:
                        immoments(img, moments=[8], chans=selection, outfile=mom8fcIntersect)
                        print("Running immoments('%s', moments=[0], chans='%s', outfile='%s')" % (img,selection,mom0fcIntersect))
                        immoments(img, moments=[0], chans=selection, outfile=mom0fcIntersect)
                    finalMom8fc = mom8fcIntersect
                    finalMom0fc = mom0fcIntersect

                    # Compute statistics of mom8fcIntersect image
                    # Assess mom8fc.intersect using original mask
                    mom8fcSNR, mom8fcPeak, mom8fcMedian, mom8fcMAD = imageSNR(mom8fcIntersect, mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus,
                                                                              useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
                    if useAnnulus:
                        # This function uses 2 calls of imageSNR in order to get peak anywhere outside the joint mask, but median and MAD from the annulus (in order to calculate SNR).
                        mom8fcSNROutside, mom8fcPeakOutside = imageSNRAnnulus(mom8fcIntersect, jointMaskTest=jointMaskTest, jointMaskTestWithAnnulus=jointMaskTestAnnulus, applyMaskToAll=True)
                    else:
                        mom8fcSNROutside, mom8fcPeakOutside, ignore0, ignore1 = imageSNR(mom8fcIntersect, mask=jointMaskTest, maskWithAnnulus='', 
                                                                                     useAnnulus=False, returnAllStats=True, applyMaskToAll=True)
                    mom8fcSNRCube = (mom8fcPeakOutside-mom8fcMedian)/MADCubeOutside
                    mom8fcSum = imstat(mom8fcIntersect, listit=imstatListit)['sum'][0]
                    casalogPost("new  mom8fcMedian = %f, mom8fcScaledMAD = %f,  mom8fcSNR = %f, mom8fcSNROutside = %f, mom8fcSum = %f" % (mom8fcMedian, mom8fcMAD, mom8fcSNR, mom8fcSNROutside, mom8fcSum))
                    NpixMom8Median = computeNpixMom8Median(mom8fcMedian, np.max([5.5,TenEventSigma]), mom8fcMAD, mom8fcIntersect, jointMaskTest)
                    NpixMom8MedianBadAtm = computeNpixMom8MedianBadAtm(mom8fcMedian, 8, mom8fcMAD, mom8fcIntersect, jointMaskTest)
                    NpixCubeMedian = computeNpixMom8Median(mom8fcMedian, np.max([5.0,TenEventSigma]), mom8fcMAD, mom8fcIntersect, jointMaskTest)

                    # create scaled version of mom0fcIntersect
                    mom0fcIntersectScaled = mom0fcIntersect + '.scaled'
                    removeIfNecessary(mom0fcIntersectScaled)  # in case there was a prior run of findContinuum
                    chanWidthKms = 299792.458*channelWidth/meanFreqHz 
                    fcChannels = len(convertSelectionIntoChannelList(selection))
                    factor = 1.0/chanWidthKms/fcChannels
                    immath([mom0fcIntersect], mode='evalexpr', expr='IM0*%f'%(factor), outfile=mom0fcIntersectScaled)

                    # Need to build a new momDiff image
                    momDiff['.intersect'] = mom8fcIntersect + '.minus.mom0fcScaled'
                    momDiffIntersect = momDiff['.intersect']
                    removeIfNecessary(momDiffIntersect)  # in case there was a prior run of findContinuum
                    mymask = '"%s">0.3' % (pbmom)
                    print("immath(['%s','%s'], mode='evalexpr', expr='IM0-IM1', mask='%s', outfile='%s')" % (mom8fcIntersect,mom0fcIntersectScaled,mymask,momDiffIntersect))
                    immath([mom8fcIntersect, mom0fcIntersectScaled], mode='evalexpr', expr='IM0-IM1', mask='"%s">0.23'%(pbmom), outfile=momDiffIntersect)
                    mymomDiff = momDiffIntersect
                    finalMomDiff = momDiffIntersect

                    ###########################################################################################
                    # Recalculate momDiff stats for purposes of extraMaskYesOrNo, and the second 4-letter code
                    ###########################################################################################
                    #  momDiffSNR, momDiffPeak = peak anywhere in image
                    momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD = imageSNR(mymomDiff, mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus, 
                                                                                  useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
                    casalogPost('momDiff: %s' % (mymomDiff))
                    casalogPost('momDiffSNR: %f  peakAnywhere: %f  median: %f  scaledMAD: %f' % (momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD))
                    # This is the SNR outside the mask
                    if useAnnulus:
                        # Need 2 calls to get peak anywhere outside the joint mask, but median and MAD from the annulus
                        momDiffSNROutside, momDiffPeakOutside = imageSNRAnnulus(mymomDiff, jointMaskTest, jointMaskTestAnnulus)
                    else:
                        momDiffSNROutside, momDiffPeakOutside, ignore0, ignore1 = imageSNR(mymomDiff, mask=jointMaskTest, returnAllStats=True, 
                                                                                           applyMaskToAll=True)
                    momDiffSNRCube = (momDiffPeakOutside-momDiffMedian)/MADCubeOutside
                    casalogPost('momDiffSNRCube: %f  scaledMAD: %f' % (momDiffSNRCube, MADCubeOutside))
                    momDiffSum = imstat(mymomDiff, listit=imstatListit)['sum'][0]
                    if useMomentDiff:
                        # These values are no longer used in extraMaskYesOrNo, so could probably not bother calculating them now.
                        NpixMomDiff = computeNpixMom8Median(momDiffMedian, momDiffLevel, momDiffMAD, mymomDiff, '')
                        NpixMomDiffBadAtm = computeNpixMom8MedianBadAtm(momDiffMedian, momDiffLevelBadAtm, momDiffMAD, mymomDiff, '')
                        NpixCubeDiffMedian = computeNpixCubeMedian(momDiffMedian, cubeLevel, MADCubeOutside, mymomDiff, '') # unused now
                        extraMaskDecision2, ExtraMaskLevel, extraMaskSigmaUsed = extraMaskYesOrNo(badAtmosphere, momDiffMedian, momDiffSNR, 
                                                                                                  momDiffSNRCube, NpixMomDiff, NpixMomDiffBadAtm, 
                                                                                                  NpixCubeDiffMedian, TenEventSigma, momDiffMAD, 
                                                                                                  MADCubeOutside, momDiffLevelForProceeding, momDiffLevelBadAtm,
                                                                                                  cubeLevel)
                    else:
                        # NpixMom8Median used here was defined above as max([5.5,TenEventSigma])
                        extraMaskDecision2, ExtraMaskLevel, extraMaskSigmaUsed = extraMaskYesOrNo(badAtmosphere, mom8fcMedian, mom8fcSNRMom8, 
                                                                                                  mom8fcSNRCube, NpixMom8Median, NpixMom8MedianBadAtm, 
                                                                                                  NpixCubeMedian, TenEventSigma, mom8fcMAD, 
                                                                                                  MADCubeOutside, mom8level, mom8levelBadAtm, cubeLevel)

                    if extraMaskDecision2 == 'No': 
                        casalogPost('Extra Mask is no longer deemed necessary. Stopping (Result3).')
                        # If No, stop: because Taking intersection was enough, so do not proceed with extraMask.
                        ###################################################
                        # Result3: Amended Mask and Intersection was enough
                        ###################################################
                        # Derive 4-letter codes and update the plot (channel ranges and legend)
                        aggregateBandwidth, xyChanRange0 = updateChannelRangesOnPlot(labelDescs, selection, ax1, separator, 
                                                                       avgspectrumAboveThreshold, skipchan, 
                                                                       medianTrue, positiveThreshold, 
                                                                       upperXlabel, ax2, channelWidth, fontsize, nbin=nbin, initialPeakOverMad=initialPeakOverMad)
                        warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                        channelRanges = len(selection.split(separator))
                        if not useMomentDiff:
                            labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, 
                                                          mom8fcPeak0, mom8fcPeakOutside, 
                                                          mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                                          mom8fcMAD, mom8fcMAD0, thresholdForSame, 
                                                          areaString, labelDescs, fontsize, ax1,
                                                          amendMaskDecision, extraMaskDecision, 
                                                          extraMaskDecision2, autoLowerDecision,
                                                          intersectionOfSelections, useMomentDiff, 
                                                          warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                 autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                        else:
                            # Needs momDiff 4-letter code
                            labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                      momDiffPeak0, momDiffPeakOutside, 
                                                      momDiffPeakOutside0, momDiffSum, momDiffSum0, 
                                                      momDiffMAD, momDiffMAD0, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                      warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                    autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                        # give the modified figure a new name so that it does not overwrite the pre-extraMask png
                        png = png.replace('.amendedMask.meanSpectrum','.amendedMaskIntersect.meanSpectrum')
                        pl.savefig(png, dpi=dpi)
                        break
                    if len(amendMaskIterationNames) <= amendMaskIteration+1: # len will be 2 if amendMaskIterations was 1
                        extraMaskDecision = 'No'
                        #############################################################
                        # Result4: amendMaskIterations was set to only 1 (not 2 or 3)
                        #############################################################
                        aggregateBandwidth, xyChanRange0 = updateChannelRangesOnPlot(labelDescs, selection, ax1, separator, 
                                                                       avgspectrumAboveThreshold, skipchan, 
                                                                       medianTrue, positiveThreshold, 
                                                                       upperXlabel, ax2, channelWidth, fontsize, nbin=nbin, initialPeakOverMad=initialPeakOverMad)
                        warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                        channelRanges = len(selection.split(separator))
                        if not useMomentDiff:
                            labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, mom8fcPeak0, mom8fcPeakOutside, 
                                                          mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                                          mom8fcMAD, mom8fcMAD0, thresholdForSame, 
                                                          areaString, labelDescs, fontsize, ax1,
                                                          amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision,
                                                          intersectionOfSelections, useMomentDiff, 
                                                          warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                 autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                        else:
                            labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                      momDiffPeak0, momDiffPeakOutside, 
                                                      momDiffPeakOutside0, momDiffSum, momDiffSum0, 
                                                      momDiffMAD, momDiffMAD0, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                      warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                    autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                        casalogPost('Stopping because amendMaskIterations was set less than 2 (Result4)')
                        pl.savefig(png, dpi=dpi)
                        break
                    # If Yes, do the extraMask procedure:
                    print("extraMaskSigmaUsed = %f, ExtraMaskLevel = %f" % (extraMaskSigmaUsed,ExtraMaskLevel))
                    if useMomentDiff:
                        channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(img, '"%s">%f' % (momDiffIntersect,ExtraMaskLevel), pbcube, imageInfo, 'mean', normalizeByMAD, outdir=outdir, jointMaskForNormalize=userJointMask, subimage=subimage)
                    else:
                        # use mom8fc.intersect to set the mask level from fc.imageSNR calcs  (using original mask)
                        channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(img, '"%s">%f' % (mom8fcIntersect,ExtraMaskLevel), pbcube, imageInfo, 'mean', normalizeByMAD, outdir=outdir, jointMaskForNormalize=userJointMask, subimage=subimage)
                    lineFullSelection = invertChannelRanges(selection, nchan)  # converts a string to a string
                    myChannelList = convertSelectionIntoChannelList(selection)
                    casalogPost('Continuum channels so far: %s' % (selection))
                    casalogPost('Line-full channels so far: %s' % (lineFullSelection))
                    scaledMADFCSubset, medianFCSubset = robustMADofContinuumRanges(myChannelList, intensity)
                    casalogPost('Intersect was done: Median and scaledMAD of %d continuum channels: %f, %f' % (len(myChannelList),medianFCSubset, scaledMADFCSubset))
                    edgesUsed = 0
                    meanSpectrumFile = mymom8fc + '.meanSpectrumFile_fromExtraMask_beforeNoiseReplacement'
                    writeMeanSpectrum(meanSpectrumFile, frequency, intensity, intensity, ExtraMaskLevel, 
                                      nchan, edgesUsed, centralArcsec='mom0mom8jointMask')
                    intensity = replaceLineFullRangesWithNoise(intensity, lineFullSelection, medianFCSubset, scaledMADFCSubset)
                    meanSpectrumFile = mymom8fc + '.meanSpectrumFile_fromExtraMask'
                    writeMeanSpectrum(meanSpectrumFile, frequency, intensity, intensity, ExtraMaskLevel, 
                                      nchan, edgesUsed, centralArcsec='mom0mom8jointMask')
                    # update selection0 to hold the amendMask/intersect selection
                    selection0 = selection 
                    if keepIntermediatePngs:
                        png = '' # reset name so that new name will be chosen and prior plot not overwritten
                    if False:
                        sigmaFindContinuumMode = 'autolower'
                        sigmaFindContinuum = finalSigmaFindContinuum
                        casalogPost("Using sigmaFindContinuumMode='autolower' with sigmaFindContinuum=%f" % (sigmaFindContinuum))
                    amendMaskIterationNames[amendMaskIteration+1] = '.extraMask'
                    signalRatioTier1 = 1.0
                    overwrite = False  # do not overwrite the spectrum we just wrote when runFindContinuum runs again!
                    continue  # from  .amendedMask iteration to .extraMask iteration
                else:
                    casalogPost('**************************************************************')
                    casalogPost('There is no significant emission in difference image (%f < %f).' % (diffStats[0], testDiffLev))
                    casalogPost('**************************************************************')
                    if len(amendMaskIterationNames) <= amendMaskIteration+1:
                        ########################################################
                        # Result5: amendMaskIterations was set to only 1 (not 2)
                        ########################################################
                        extraMaskDecision = 'No'
                        aggregateBandwidth, xyChanRange0 = updateChannelRangesOnPlot(labelDescs, selection, ax1, separator, 
                                                                       avgspectrumAboveThreshold, skipchan, 
                                                                       medianTrue, positiveThreshold, 
                                                                       upperXlabel, ax2, channelWidth, fontsize, nbin=nbin, initialPeakOverMad=initialPeakOverMad)
                        warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                        channelRanges = len(selection.split(separator))
                        if not useMomentDiff:
                            labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, mom8fcPeak0, mom8fcPeakOutside, 
                                                          mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                                          mom8fcMAD, mom8fcMAD0, thresholdForSame, 
                                                          areaString, labelDescs, fontsize, ax1,
                                                          amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision,
                                                          intersectionOfSelections, useMomentDiff, 
                                                          warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                 autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                        else:
                            labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                      momDiffPeak0, momDiffPeakOutside, 
                                                      momDiffPeakOutside0, momDiffSum, momDiffSum0, 
                                                      momDiffMAD, momDiffMAD0, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                      warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                    autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                        casalogPost('Stopping because amendMaskIterations was set less than 2 (Result5)')
                        pl.savefig(png, dpi=dpi)
                        break
                    # do the extraMaskProcedure:
                    # jointMaskTest will be the original mask, unless I take action here
                    casalogPost("sigmaUsed to set level = %f, computing spectrum from pixels > %f" % (extraMaskSigmaUsed,ExtraMaskLevel))
                    if useMomentDiff:
                        channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(img, '"%s">%f' % (mymomDiff,ExtraMaskLevel), pbcube, imageInfo, 'mean', normalizeByMAD, outdir=outdir, jointMaskForNormalize=userJointMask, subimage=subimage)
                    else:
                        channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(img, '"%s">%f' % (mymom8fc,ExtraMaskLevel), pbcube, imageInfo, 'mean', normalizeByMAD, outdir=outdir, jointMaskForNormalize=userJointMask, subimage=subimage)
                    lineFullSelection = invertChannelRanges(selection, nchan) # converts a string to a string
                    myChannelList = convertSelectionIntoChannelList(selection)
                    casalogPost('Continuum channels so far: %s' % (selection))
                    casalogPost('Line-full channels so far: %s' % (lineFullSelection))
                    scaledMADFCSubset, medianFCSubset = robustMADofContinuumRanges(myChannelList, intensity)
                    edgesUsed = 0
                    casalogPost('Intersect was not done: Median and scaledMAD of %d continuum channels: %f, %f' % (len(myChannelList),medianFCSubset, scaledMADFCSubset))
                    meanSpectrumFile = mymom8fc + '.meanSpectrumFile_fromExtraMask_beforeNoiseReplacement'
                    writeMeanSpectrum(meanSpectrumFile, frequency, intensity, intensity, ExtraMaskLevel, 
                                      nchan, edgesUsed, centralArcsec='mom0mom8jointMask')
                    intensity = replaceLineFullRangesWithNoise(intensity, lineFullSelection, medianFCSubset, scaledMADFCSubset)
                    meanSpectrumFile = mymom8fc + '.meanSpectrumFile_fromExtraMask'
                    writeMeanSpectrum(meanSpectrumFile, frequency, intensity, 
                                      intensity, ExtraMaskLevel, nchan, 
                                      edgesUsed, centralArcsec='mom0mom8jointMask')
                    # update selection0 to hold the amendMask/intersect selection
                    selection0 = selection 
                    if keepIntermediatePngs:
                        png = '' # reset name so that new name will be chosen and prior plot not overwritten
                    amendMaskIterationNames[amendMaskIteration+1] = '.extraMask'
                    signalRatioTier1 = 1.0
                    if False:
                        sigmaFindContinuumMode = 'autolower'
                        sigmaFindContinuum = finalSigmaFindContinuum
                        casalogPost("Using sigmaFindContinuumMode='autolower' with sigmaFindContinuum=%f" % (sigmaFindContinuum))
                    overwrite = False  # do not overwrite the spectrum we just wrote when runFindContinuum runs again!
                    continue  # from .amendedMask iteration to .extraMask iteration
                # end if diffStats[0] > testDiffLev  else:
            # end if amendMaskIterationName == '.amendedMask'
            if amendMaskIterationName in ['.extraMask','.onlyExtraMask','.autoLower', '.autoLower2']:
                # Intersect extraMask spectral (cyan) ranges with amendMask spectral (cyan) ranges
                # selection0 will hold the amendMask selection if .extraMask was run
                channelList0 = convertSelectionIntoChannelList(selection0)
                channelList1 = convertSelectionIntoChannelList(selection)
                channelList = np.intersect1d(channelList0,channelList1)
                casalogPost("Taking intersection of %d channels with %d channels, to get %d channels (aggregateBW=%.4f GHz)" % (len(channelList0), len(channelList1), len(channelList), len(channelList)*channelWidth*1e-9))
                casalogPost("first channel list = %s" % (selection0))
                casalogPost("second channel list = %s" % (selection))
                if len(channelList) == 0:
                    ###########################################################################################
                    # Result6: ExtraMask or onlyExtraMask was used but no channels were left after intersection
                    ###########################################################################################
                    casalogPost("No intersection: Reverting to first channel selection (Result6)")
                    channelList = channelList0  # reverting
                    selection = convertChannelListIntoSelection(channelList) # this line was added to fix PIPE-1211
                    # derive 4-letter codes and update the final line of the plot legend
                    warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                    channelRanges = len(selection.split(separator))
                    if not useMomentDiff:
                        labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, 
                                                          mom8fcPeak0, mom8fcPeakOutside, 
                                                          mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                                          mom8fcMAD, mom8fcMAD0, thresholdForSame, 
                                                          areaString, labelDescs, fontsize, ax1,
                                                          amendMaskDecision, extraMaskDecision, 
                                                          extraMaskDecision2, autoLowerDecision,
                                                          intersectionOfSelections, useMomentDiff, 
                                                          warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                          autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                    else:
                        # Needs momDiff 4-letter code, but only if it was .extraMask 
                        labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                      momDiffPeak0, momDiffPeakOutside, 
                                                      momDiffPeakOutside0, momDiffSum, momDiffSum0, 
                                                      momDiffMAD, momDiffMAD0, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                      warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR,
                                                                                                autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                    pl.savefig(png, dpi=dpi)
                    break
                if amendMaskIterationName in ['.onlyExtraMask']:
                    # save the .original.meanSpectrum.*.png before modifying the channel ranges and legend below
                    # save the .amendedMask.meanSpectrum.*.png before modifying the channel ranges and legend below
                    casalogPost("%s: Saving %s" % (amendMaskIterationName, png))
                    pl.savefig(png, dpi=dpi)
                selection = convertChannelListIntoSelection(channelList)
                if list(channelList) == list(channelList0):
                    if amendMaskIterationName == '.extraMask':
                        extraMaskDecision = 'NoImprovement'
                    elif amendMaskIterationName == '.autoLower':
                        autoLowerDecision = 'NoImprovement'
                    elif amendMaskIterationName == '.autoLower2':
                        autoLower2Decision = 'NoImprovement'
                    casalogPost('No change in channel list')
                elif len(channelList) == 1 and nchan > 500:
                    # disallow first and final channel if either is the adjacent channel
                    casalogPost('intersected channel list: %s' % (selection))
                    onlyChannel = channelList[0]
                    channelList = range(np.max([1,onlyChannel-1]),np.min([onlyChannel+1,nchan-2])+1)
                    casalogPost('Forcing single channel to include the adjacent channels because nchan>500')
                    selection = convertChannelListIntoSelection(channelList)
                                   
                casalogPost("(a) Due to use of intersected channels, setting selectionPreBluePruning to ", selection)
                selectionPreBluePruning = selection  # PIPE-2417
#                else:  # process new channel selection (even if it is the same, because we want to produce the autoLowerIntersect momDiff)
                casalogPost('intersected channel list: %s' % (selection))
                ##########################################
                # Build a (potentially) final mom8fc image
                ##########################################
                myIntersectName = amendMaskIterationName+'Intersect'
                if mom08simultaneous:
                    momRoot[myIntersectName] = momRoot[amendMaskIterationName] + 'Intersect'
                    mom8fc[myIntersectName] = momRoot[myIntersectName] + '.maximum'
                    mom0fc[myIntersectName] = momRoot[myIntersectName] + '.integrated'
                else:
                    mom8fc[myIntersectName] = mom8fc[amendMaskIterationName] + 'Intersect'
                    mom0fc[myIntersectName] = mom0fc[amendMaskIterationName] + 'Intersect'
                removeIfNecessary(mom8fc[myIntersectName])
                removeIfNecessary(mom0fc[myIntersectName])
                if mom08simultaneous:
                    immoments(img, moments=[0,8], chans=selection, outfile=momRoot[myIntersectName])
                else:
                    casalogPost("Running immoments('%s', moments=[8], chans='%s', outfile='%s')" % (img,selection, mom8fc[myIntersectName]))
                    immoments(img, moments=[8], chans=selection, outfile=mom8fc[myIntersectName])
                    casalogPost("Running immoments('%s', moments=[0], chans='%s', outfile='%s')" % (img, selection, mom0fc[myIntersectName]))
                    immoments(img, moments=[0], chans=selection, outfile=mom0fc[myIntersectName])
                finalMom0fc = mom0fc[myIntersectName]
                finalMom8fc = mom8fc[myIntersectName]

                # compute final 4-letter code and update the final line of the plot legend
                mom8fcSNR, mom8fcPeak, mom8fcMedian, mom8fcMAD = imageSNR(mom8fc[myIntersectName], mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus, 
                                                                          useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
                if useAnnulus:
                    # Need 2 calls to get peak anywhere outside the joint mask, but median and MAD from the annulus
                    mom8fcSNROutside, mom8fcPeakOutside = imageSNRAnnulus(mom8fc[myIntersectName], jointMaskTest, jointMaskTestAnnulus)
                else:
                    mom8fcSNROutside, mom8fcPeakOutside, ignore0, ignore1 = imageSNR(mom8fc[myIntersectName], mask=jointMaskTest, 
                                                                                     returnAllStats=True, applyMaskToAll=True)
                # need to recalculate mom8fcSNRCube because the numerator has changed
                mom8fcSNRCube = (mom8fcPeakOutside-mom8fcMedian)/MADCubeOutside
                mom8fcSum = imstat(mom8fc[myIntersectName], listit=imstatListit)['sum'][0]

                # create scaled version of mom0fcIntersect
                mom0fcIntersectScaled = mom0fc[myIntersectName] + '.scaled'
                removeIfNecessary(mom0fcIntersectScaled)  # in case there was a prior run of findContinuum
                chanWidthKms = 299792.458*channelWidth/meanFreqHz 
                fcChannels = len(convertSelectionIntoChannelList(selection))
                factor = 1.0/chanWidthKms/fcChannels
                immath(mom0fc[myIntersectName], mode='evalexpr', expr='IM0*%f'%(factor), outfile=mom0fcIntersectScaled)

                # Need to build a new momDiff image since we have a new mom8fc[myIntersectName]
                momDiff[myIntersectName] = mom8fc[myIntersectName] + '.minus.mom0fcScaled'
                removeIfNecessary(momDiff[myIntersectName])  # in case there was a prior run of findContinuum
                mymask = '"%s">0.3' % (pbmom)
                print("immath(['%s','%s'], mode='evalexpr', expr='IM0-IM1', mask='%s', outfile='%s')" % (mom8fc[myIntersectName],mom0fcIntersectScaled,mymask,momDiff[myIntersectName]))
                immath([mom8fc[myIntersectName], mom0fcIntersectScaled], mode='evalexpr', expr='IM0-IM1', mask='"%s">0.23'%(pbmom), outfile=momDiff[myIntersectName])
                finalMomDiff = momDiff[myIntersectName]

                # Recalculate for purposes of second 4-letter code
                momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD = imageSNR(momDiff[myIntersectName], mask=jointMaskTest, 
                                                                              maskWithAnnulus=jointMaskTestAnnulus, useAnnulus=useAnnulus,
                                                                              returnAllStats=True, applyMaskToAll=False)
                casalogPost('momDiff: %s' % (momDiff[myIntersectName]))
                casalogPost('momDiffSNR: %f  peakAnywhere: %f  median: %f  scaledMAD: %f' % (momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD))
                # This is the SNR outside the mask
                if useAnnulus:
                    # Need 2 calls to get peak anywhere outside the joint mask, but median and MAD from the annulus
                    momDiffSNROutside, momDiffPeakOutside = imageSNRAnnulus(momDiff[myIntersectName], jointMaskTest, jointMaskTestAnnulus)
                else:
                    momDiffSNROutside, momDiffPeakOutside, ignore0, ignore1 = imageSNR(momDiff[myIntersectName], mask=jointMaskTest, 
                                                                                       returnAllStats=True, applyMaskToAll=True)
                momDiffSum = imstat(momDiff[myIntersectName], listit=imstatListit)['sum'][0]
                momDiffSNRCube = (momDiffPeakOutside-momDiffMedian)/MADCubeOutside
                casalogPost('momDiffSNRCube: %f  scaledMAD: %f' % (momDiffSNRCube, MADCubeOutside))
                if (amendMaskIterationName == '.extraMask' and amendMaskIterations > 2) or (amendMaskIterationName == '.onlyExtraMask' and amendMaskIterations > 1):   # PL2022
#                if not returnBluePoints and ((amendMaskIterationName == '.extraMask' and amendMaskIterations > 2) or (amendMaskIterationName == '.onlyExtraMask' and amendMaskIterations > 1)):  # PL2023 (only needed if amendMaskIterations != 'auto')
                    warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                    if len(warnings[0]) + len(warnings[1]) == 0:
#                        These values are no longer used in extraMaskYesOrNo(), so no need to compute them
#                        NpixMomDiff = computeNpixMom8Median(momDiffMedian, momDiffLevel, momDiffMAD, momDiff[myIntersectName], '')
#                        NpixMomDiffBadAtm = computeNpixMom8MedianBadAtm(momDiffMedian, momDiffLevelBadAtm, momDiffMAD, momDiff[myIntersectName], '')
                        # NpixCubeDiffMedian is unused now
                        autoLowerDecision, autoLowerLevel, autoLowerSigmaUsed = extraMaskYesOrNo(badAtmosphere, 
                                                                                                 momDiffMedian, momDiffSNR, 
                                                                                                 momDiffSNRCube, NpixMomDiff, NpixMomDiffBadAtm, 
                                                                                                 NpixCubeDiffMedian, TenEventSigma, momDiffMAD, 
                                                                                                 MADCubeOutside, momDiffLevelForProceeding, momDiffLevelBadAtm, cubeLevel)
                        casalogPost('autoLowerDecision = %s, autoLowerLevel = %f' % (autoLowerDecision, autoLowerLevel))
                        if autoLowerDecision in ['YesMom','YesCube']:
                            if keepIntermediatePngs:
                                png = '' # reset name so that new name will be chosen and prior plot not overwritten
                            amendMaskIterationNames[amendMaskIteration+1] = '.autoLower'
                            signalRatioTier1 = 1.0
                            sigmaFindContinuumMode = 'autolower'
                            if momDiffSNR < momDiffApproachLevel:
                                casalogPost('Scaling sigmaFindContinuum by 6/7 because momDiffSNR=%f < %f' % (momDiffSNR,momDiffApproachLevel))
                                sigmaFindContinuum = finalSigmaFindContinuum*6/7.  # added *6/7 in PL2023
                            else:
                                casalogPost('Scaling sigmaFindContinuum by 5/7 because momDiffSNR=%f >= %f' % (momDiffSNR,momDiffApproachLevel))
                                sigmaFindContinuum = finalSigmaFindContinuum*5/7.  # added *5/7 in PL2022
                            selection0 = selection # be sure we intersect against this latest channel selection
                            casalogPost("Using sigmaFindContinuumMode='autolower' with sigmaFindContinuum=%f" % (sigmaFindContinuum))
                            continue # from .extraMask/.onlyExtraMask to .autoLower
                    else:
                        nWarnings =  len(warnings[0]) + len(warnings[1])
                        casalogPost("Not trying .autoLower because we have %d bandwidth warning(s)" % (nWarnings))             

                else:
                    if amendMaskIterationName != '.autoLower':
                        casalogPost("Not trying .autoLower2 because amendMaskIteration=%d/%d (amendMaskIterationName=%s)" % (amendMaskIteration+1,amendMaskIterations,amendMaskIterationName))
                        autoLower2Decision = 'YesMom' # need to keep this setting so 'Y' is displayed on the png
                    else:  # .autoLower2
                        autoLower2Decision, autoLower2Level, autoLower2SigmaUsed = extraMaskYesOrNo(badAtmosphere, 
                                                                                                    momDiffMedian, momDiffSNR, 
                                                                                                    momDiffSNRCube, NpixMomDiff, NpixMomDiffBadAtm, 
                                                                                                    NpixCubeDiffMedian, TenEventSigma, momDiffMAD, 
                                                                                                    MADCubeOutside, momDiffLevelForProceeding, momDiffLevelBadAtm, cubeLevel)
                        casalogPost('autoLower2Decision = %s, autoLower2Level = %f' % (autoLower2Decision, autoLower2Level))
                        if autoLower2Decision in ['YesMom','YesCube'] and amendMaskIterations>3:
                            if keepIntermediatePngs:
                                png = '' # reset name so that new name will be chosen and prior plot not overwritten
                            amendMaskIterationNames[amendMaskIteration+1] = '.autoLower2'
                            signalRatioTier1 = 1.0
                            sigmaFindContinuumMode = 'autolower'
                            sigmaFindContinuum = finalSigmaFindContinuum*6/7. 
                            selection0 = selection # be sure we intersect against this latest channel selection
                            casalogPost("Using sigmaFindContinuumMode='autolower' with sigmaFindContinuum=%f" % (sigmaFindContinuum))
                            continue # from .autoLower to .autoLower2
                # was: endif list(channelList) == list(channelList0)  else:
                warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                channelRanges = len(selection.split(separator))
                # update the plot channel ranges 
                casalogPost('A) calling updateChannelRangesOnPlot prior to Result7, autoLower2Decision=%s' % str(autoLower2Decision))
                aggregateBandwidth, xyChanRange0 = updateChannelRangesOnPlot(labelDescs, selection, ax1, separator, 
                                                               avgspectrumAboveThreshold, skipchan, 
                                                               medianTrue, positiveThreshold, 
                                                               upperXlabel, ax2, channelWidth, fontsize, 
                                                               nbin=nbin, initialPeakOverMad=initialPeakOverMad)
                ########################################################################
                # Result7: ExtraMask or onlyExtraMask, and possibly autoLower were used 
                ########################################################################
                if not useMomentDiff:
                    labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, 
                                                      mom8fcPeak0, mom8fcPeakOutside, 
                                                      mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                                      mom8fcMAD, mom8fcMAD0, thresholdForSame, 
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                      warnings, channelRanges, 
                                                      sigmaUsedToAmendMask, momDiffSNR, 
                                                      autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                else:
                    # Needs momDiff 4-letter code
                    labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                  momDiffPeak0, momDiffPeakOutside, 
                                                  momDiffPeakOutside0, momDiffSum, momDiffSum0, 
                                                  momDiffMAD, momDiffMAD0, thresholdForSame, 
                                                  areaString, labelDescs, fontsize, ax1,
                                                  amendMaskDecision, extraMaskDecision, 
                                                  extraMaskDecision2, autoLowerDecision,
                                                  intersectionOfSelections, useMomentDiff,
                                                  warnings, channelRanges, sigmaUsedToAmendMask, 
                                                  momDiffSNR, 
                                                  autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                # give the modified figure a new name so that it does not overwrite the pre-extraMask png
                if amendMaskIterationName == '.onlyExtraMask': 
                    print("onlyExtraMask) Changing name of png: ", png)
                    png = png.replace('.original.meanSpectrum','.onlyExtraMaskIntersect.meanSpectrum')
                    casalogPost("Saving %s" % png)
                    pl.savefig(png, dpi=dpi)
                    # do not allow it to try a 3rd iteration, since we are done
                    break 
                elif amendMaskIterationName == '.extraMask':  
                    print("extraMask) Changing name of png: ", png)
                    png = png.replace('.extraMask.meanSpectrum','.extraMaskIntersect.meanSpectrum')
                    png = png.replace('.amendedMask.meanSpectrum','.extraMaskIntersect.meanSpectrum')
                    casalogPost("Saving %s" % png)
                    pl.savefig(png, dpi=dpi)
                    break # not needed if amendMaskIterations=2, but needed if it is 3
                elif amendMaskIterationName == '.autoLower':  
                    print("autoLower) Changing name of png: ", png)
                    png = png.replace('.amendedMask.meanSpectrum','.autoLowerIntersect.meanSpectrum')
                    png = png.replace('.original.meanSpectrum','.autoLowerIntersect.meanSpectrum')
                    casalogPost("%s) Saving %s" % (amendMaskIterationName,png))
                    pl.savefig(png, dpi=dpi)
                    if amendMaskIterations == 3:
                        break # needed if amendMaskIterations == 3 and we did not amendMask (because autoLower would be 2nd iteration)
                    if amendMaskIterations == 4:
                        break # also needed if amendMaskIterations == 4 and autoLower2Decision is No (which is how we can get here)
                    haveLowBW = False
                    for warning in warnings:
                        if warning.find('amount') > 0:
                            casalogPost("Not allowing .autoLower2 because we are in the LowBW state already")
                            haveLowBW = True
                    if haveLowBW:
                        break
                    amendMaskIterationNames[amendMaskIteration+1] = '.autoLower2'
                elif amendMaskIterationName == '.autoLower2':  # new
                    print("autoLower2) Changing name of png: ", png)
                    png = png.replace('.amendedMask.meanSpectrum','.autoLowerIntersect2.meanSpectrum')
                    png = png.replace('.original.meanSpectrum','.autoLowerIntersect2.meanSpectrum')
                    casalogPost("%s) Saving %s" % (amendMaskIterationName,png))
                    pl.savefig(png, dpi=dpi)
                    if amendMaskIterations == 4:
                        break # needed if amendMaskIterations == 4 and we did not amendMask (because autoLower2 would be 3rd iteration)
            # endif amendMaskIterationName in ['.extraMask','.onlyExtraMask','.autoLower']
        # endif meanSpectrumMethod=='mom0mom8jointMask' and (amendMask or checkIfMaskWouldHaveBeenAmended):
    # end 'for' loop over amendMaskIterations
    if amendMaskIterations < 1:
        sigmaFindContinuumAtEndOfOriginalIteration = finalSigmaFindContinuum

    if selection == '' and amendMask:
        # I think this should never happen due to protection above, but I leave this code here, just in case.
        casalogPost("Blank selection found: Reverting to prior iteration with selection: %s" % (selection))
        selection = previousSelection
        png = previousPng
        aggregateBandwidth = previousAggregateBandwidth

    ##########################################################################################
    # At this point, all of the images and pngs have been saved with the correct names.
    # Now compare the 4-letter 'code' to see if we need to revert the channel range results,
    # the png returned, and the final image returned to the .original versions of these files.
    ##########################################################################################
    if amendMask:
        casalogPost("Check for need to revert.")
        reversionCodes = ['HHHH','HHHS']
        revert = False
        if momDiffCode is None:
            code = mom8code
            if mom8code in reversionCodes:
                revert = True
                casalogPost('This mom8fc code triggers reversion to original result.')
        else: 
            code = momDiffCode
            if momDiffCode in reversionCodes:
                if (momDiffSNR > momDiffLevel):
                    revert = True
                    casalogPost('This momDiff code with momDiffSNR>%.1f triggers reversion to original result.'%(momDiffLevel))
        if not revert:
            casalogPost("No need for reversion.")
            warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
#            if len(warnings[0]) > 0 and len(warnings[1]) > 0: # PL2023
            if len(warnings[1]) > 0:  # LowSpread    # PL2024
                casalogPost("Will try to fix the LowSpread condition")    # PL2024
                # do our addition of new channel range(s)
                continuumChannels = convertSelectionIntoChannelList(selection)
                if continuumChannels[0] > (nchan-continuumChannels[-1]):
                    # channels are in increasing order;  don't look beyond the half-way point of the spectrum
                    endchan = int(np.min([continuumChannels[-1] - nchan*smallSpreadFraction - 1, nchan//2]))
                    channelRange = [0,endchan]
                else:  # look on the right side of the spectrum
                    startchan = int(continuumChannels[0] + nchan*smallSpreadFraction + 1)
                    channelRange = [startchan,int(nchan)] # prevent np.int64 from printing to log
                mylist = findWidestContiguousListInChannelRange(allBaselineChannelsXY[0], channelRange, continuumChannels)
                if mylist is None:
                    triedComparableIntensity = True
                    casalogPost('  %s ****** No blue points found in the wider side of the spectrum: %s' % (projectCode,str(channelRange)))
                    # now need to look for non-blue points within the min/max range of blue points
                    mylist = findChannelsInChannelRangeWithComparableIntensity(allBaselineChannelsXY[0], channelRange, continuumChannels, avgspectrumAboveThreshold)  # PL2024
                    if len(mylist) == 0:
                        mylist = None
                else:
                    triedComparableIntensity = False
                    
#                    mylist = [mylist] # convert single range to list of 1 range

                # Need to check if most of the points are beyond 1 sigma MAD of all points (for PIPE-2188)
                # and if so, then reject it since it might be the shoulders of a real line.
                if mylist is not None:
                    MADofBaselineChannels = MAD(avgspectrumAboveThreshold[allBaselineChannelsXY[0]])
                    medianNegativeSigma = (negativeThreshold - np.median(avgspectrumAboveThreshold[np.array(mylist).flatten()])) / MADofBaselineChannels
                    medianPositiveSigma = (np.median(avgspectrumAboveThreshold[mylist]) - positiveThreshold) / MADofBaselineChannels
                    casalogPost('medianNegativeSigma = %f, medianPositiveSigma = %f' % (medianNegativeSigma, medianPositiveSigma))
                    if medianNegativeSigma > 1:
                        casalogPost('Rejecting new range (%s) because the median value is %.1f (> 1) scaled MAD beyond one of the dotted lines (negativeThreshold).' % (mylist, medianNegativeSigma))
                        mylist = None
                    if medianPositiveSigma > 1:
                        casalogPost('Rejecting new range (%s) because the median value is %.1f (> 1) scaled MAD beyond one of the dotted lines (positiveThreshold).' % (mylist, medianPositiveSigma))
                        mylist = None
                    if mylist is None and not triedComparableIntensity:  # PIPE-3103
                        mylist = findChannelsInChannelRangeWithComparableIntensity(allBaselineChannelsXY[0], channelRange, continuumChannels, avgspectrumAboveThreshold, medianNegativeSigma, negativeThreshold, positiveThreshold)
                        if len(mylist) == 0:
                            casalogPost('No channels returned with comparable intensity (final attempt to add channels).')
                            mylist = None
                        else:
                            mylist = splitListIntoContiguousLists(mylist)
                            # take only the widest list
                            widestList = mylist[np.argmax([len(i) for i in mylist])]
                            mylist = widestList
                if mylist is None:
                    casalogPost('  %s ****** No points with intensity in the range of the blue points were found in the wider side of the spectrum' % (projectCode))
                else:
                    # we added 1 (or more) new selections, so need to update everything
#                    print("mylist before split: ", str(mylist))
                    mylist = splitListIntoContiguousLists(mylist)
#                    print("mylist after split: ", str(mylist))
                    groups = len(mylist)
                    for group in range(groups):
#                        print("group %d: %s" % (group,mylist[group]))
                        secondSelectionAdded = '%d~%d' % (mylist[group][0],mylist[group][-1])
                        if selection == '':
                            selection = secondSelectionAdded
                        else:
                            if mylist[group][0] > continuumChannels[-1]:
                                # add to end of string (to maintain increasing order)
                                selection += separator + secondSelectionAdded
                            else:
                                # add to beginning of string (to maintain increasing order)
                                selection = secondSelectionAdded + separator + selection
                        casalogPost('  %s ****** Added another selection in wider side of the spectrum: %s' % (projectCode,secondSelectionAdded))
                    # end 'for' loop over groups
                    if momDiffPeak0 is None:
                        # No AmendMask nor ExtraMask was done, so we need to store the original values.
                        # These *0's will be used to compute the MomDiff 4-letter code generation
                        momDiffPeak0 = momDiffPeak
                        momDiffPeakOutside0 = momDiffPeakOutside
                        momDiffMAD0 = momDiffMAD
                        momDiffSum0 = momDiffSum
                        mom8fcPeak0 = mom8fcPeak
                        mom8fcPeakOutside0 = mom8fcPeakOutside
                        mom8fcMAD0 = mom8fcMAD
                        mom8fcSum0 = mom8fcSum
                        removeRanges = True
                    else:
                        removeRanges = False
                    
                    # update the plot again
                    casalogPost('B) calling updateChannelRangesOnPlot')
                    aggregateBandwidth, xyChanRange0 = updateChannelRangesOnPlot(labelDescs, selection, ax1, separator, 
                                                                   avgspectrumAboveThreshold, skipchan, 
                                                                   medianTrue, positiveThreshold, 
                                                                   upperXlabel, ax2, channelWidth, fontsize,
                                                                   remove=removeRanges, nbin=nbin, initialPeakOverMad=initialPeakOverMad)
                    warnings = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
                    # recompute a new mom8fc, mom0fc, momDiff, mom8fc.minus.mom0fcScaled
                    ##################################
                    # Build a final mom8fc image
                    ##################################
                    myIntersectName = amendMaskIterationName+'Intersect'
                    if mom08simultaneous:
                        momRoot[myIntersectName] = momRoot[amendMaskIterationName] + 'Intersect'
                        mom8fc[myIntersectName] = momRoot[myIntersectName] + '.maximum'
                        mom0fc[myIntersectName] = momRoot[myIntersectName] + '.integrated'
                    else:
                        mom8fc[myIntersectName] = mom8fc[amendMaskIterationName] + 'Intersect'
                        mom0fc[myIntersectName] = mom0fc[amendMaskIterationName] + 'Intersect'
                    removeIfNecessary(mom8fc[myIntersectName])
                    removeIfNecessary(mom0fc[myIntersectName])
                    if mom08simultaneous:
                        immoments(img, moments=[0,8], chans=selection, outfile=momRoot[myIntersectName])
                    else:
                        casalogPost("Running immoments('%s', moments=[8], chans='%s', outfile='%s')" % (img,selection, mom8fc[myIntersectName]))
                        immoments(img, moments=[8], chans=selection, outfile=mom8fc[myIntersectName])
                        casalogPost("Running immoments('%s', moments=[0], chans='%s', outfile='%s')" % (img, selection, mom0fc[myIntersectName]))
                        immoments(img, moments=[0], chans=selection, outfile=mom0fc[myIntersectName])
                    finalMom0fc = mom0fc[myIntersectName]
                    finalMom8fc = mom8fc[myIntersectName]
                    mom8fcSNR, mom8fcPeak, mom8fcMedian, mom8fcMAD = imageSNR(mom8fc[myIntersectName], mask=jointMaskTest, 
                                                                              maskWithAnnulus=jointMaskTestAnnulus, useAnnulus=useAnnulus,
                                                                              returnAllStats=True, applyMaskToAll=False)
                    if useAnnulus:
                        # Need 2 calls to get peak anywhere outside the joint mask, but median and MAD from the annulus
                        mom8fcSNROutside, mom8fcPeakOutside = imageSNRAnnulus(mom8fc[myIntersectName], jointMaskTest, jointMaskTestAnnulus)
                    else:
                        mom8fcSNROutside, mom8fcPeakOutside, ignore0, ignore1 = imageSNR(mom8fc[myIntersectName], mask=jointMaskTest, 
                                                                                         returnAllStats=True, applyMaskToAll=True)
                    # need to recalculate mom8fcSNRCube because the numerator has changed
                    mom8fcSNRCube = (mom8fcPeakOutside-mom8fcMedian)/MADCubeOutside
                    mom8fcSum = imstat(mom8fc[myIntersectName], listit=imstatListit)['sum'][0]

                    # create scaled version of mom0fcIntersect
                    mom0fcIntersectScaled = mom0fc[myIntersectName] + '.scaled'
                    removeIfNecessary(mom0fcIntersectScaled)  # in case there was a prior run of findContinuum
                    chanWidthKms = 299792.458*channelWidth/meanFreqHz 
                    fcChannels = len(convertSelectionIntoChannelList(selection))
                    factor = 1.0/chanWidthKms/fcChannels
                    immath(mom0fc[myIntersectName], mode='evalexpr', expr='IM0*%f'%(factor), outfile=mom0fcIntersectScaled)

                    # Need to build a new momDiff image since we have a new mom8fc[myIntersectName]
                    momDiff[myIntersectName] = mom8fc[myIntersectName] + '.minus.mom0fcScaled'
                    removeIfNecessary(momDiff[myIntersectName])  # in case there was a prior run of findContinuum
                    mymask = '"%s">0.3' % (pbmom)
                    print("immath(['%s','%s'], mode='evalexpr', expr='IM0-IM1', mask='%s', outfile='%s')" % (mom8fc[myIntersectName],mom0fcIntersectScaled,mymask,momDiff[myIntersectName]))
                    immath([mom8fc[myIntersectName], mom0fcIntersectScaled], mode='evalexpr', expr='IM0-IM1', mask='"%s">0.23'%(pbmom), outfile=momDiff[myIntersectName])
                    finalMomDiff = momDiff[myIntersectName]

                    # Recalculate for purposes of second 4-letter code
                    momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD = imageSNR(momDiff[myIntersectName], mask=jointMaskTest,
                                                                                  maskWithAnnulus=jointMaskTestAnnulus, useAnnulus=useAnnulus,
                                                                                  returnAllStats=True, applyMaskToAll=False)
                    casalogPost('momDiff: %s' % (momDiff[myIntersectName]))
                    casalogPost('momDiffSNR: %f  peakAnywhere: %f  median: %f  scaledMAD: %f' % (momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD))
                    # This is the SNR outside the mask
                    if useAnnulus:
                        # Need 2 calls to get peak anywhere outside the joint mask, but median and MAD from the annulus
                        momDiffSNROutside, momDiffPeakOutside = imageSNRAnnulus(momDiff[myIntersectName], jointMaskTest, jointMaskTestAnnulus)
                    else:
                        momDiffSNROutside, momDiffPeakOutside, ignore0, ignore1 = imageSNR(momDiff[myIntersectName], mask=jointMaskTest, 
                                                                                           returnAllStats=True, applyMaskToAll=True)
                    momDiffSum = imstat(momDiff[myIntersectName], listit=imstatListit)['sum'][0]
                    momDiffSNRCube = (momDiffPeakOutside-momDiffMedian)/MADCubeOutside
                    casalogPost('momDiffSNRCube: %f  scaledMAD: %f' % (momDiffSNRCube, MADCubeOutside))
                    if not useMomentDiff:
                        labelDescs, areaString, mom8code = compute4LetterCodeAndUpdateLegend(mom8fcPeak, 
                                                          mom8fcPeak0, mom8fcPeakOutside, 
                                                          mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                                          mom8fcMAD, mom8fcMAD0, thresholdForSame*5,  # use looser definition of Same
                                                          areaString, labelDescs, fontsize, ax1,
                                                          amendMaskDecision, extraMaskDecision, 
                                                          extraMaskDecision2, autoLowerDecision,
                                                          intersectionOfSelections, useMomentDiff, 
                                                          warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR, 
                                                          replace=True, addedSelection=True, autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                    else:
                        # Needs momDiff 4-letter code
                        labelDescs, areaString, momDiffCode = compute4LetterCodeAndUpdateLegend(momDiffPeak, 
                                                      momDiffPeak0, momDiffPeakOutside, 
                                                      momDiffPeakOutside0, momDiffSum, momDiffSum0, 
                                                      momDiffMAD, momDiffMAD0, thresholdForSame*5,  # use looser definition of Same
                                                      areaString, labelDescs, fontsize, ax1,
                                                      amendMaskDecision, extraMaskDecision, 
                                                      extraMaskDecision2, autoLowerDecision,
                                                      intersectionOfSelections, useMomentDiff, 
                                                      warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR, replace=True, 
                                                      addedSelection=True, autoLower2Decision=autoLower2Decision, amendMaskIterations=amendMaskIterations)
                    if png == originalPng:
                        png = originalPng.replace('.png','_addedRange.png')
                    pl.savefig(png, dpi=dpi)
                    # but we *might* have made it worse, so check again
                    if momDiffCode is None:
                        code = mom8code
                        if mom8code in reversionCodes:
                            revert = True
                            casalogPost('This mom8fc code triggers reversion to original result.')
                    else: 
                        code = momDiffCode
                        if momDiffCode in reversionCodes:
                            if (momDiffSNR > momDiffLevel):
                                revert = True
                                casalogPost('This momDiff code with momDiffSNR>%.1f triggers reversion to original result.'%(momDiffLevel))
                if len(warnings[0]) > 0 and len(warnings[1]) > 0:
                    revert = True
                    casalogPost('This combination of bandwidth warnings triggers reversion to original result, which had only %d warning(s).' % (len(originalWarnings)))
        if revert:
            finalMomDiff = momDiff['.original']
            selection = originalSelection
            casalogPost('code=%s: Reverting to original selection: %s' % (code,selection))
            aggregateBandwidth = computeBandwidth(selection, channelWidth, 0)
            if originalPng != png:
                png = revertedPng
                # update the modification time to current time (like Unix 'touch') so that it appears to be most recent
                os.utime(png,None)
                os.utime(finalMomDiff,None) # fix for PIPE-2261
        else:
            os.remove(revertedPng)
#    endif amendMask:

    ##########################################################################################
    # Determine the name of the finalImageToReturn, in case it was requested
    ##########################################################################################
    pathCode = ''
    if amendMask:
        if useMomentDiff:
            finalImageToReturn = finalMomDiff
        else:
            finalImageToReturn = finalMom8fc
        casalogPost('finalImageToReturn = %s' % (finalImageToReturn))
        momDiffSNR = imageSNR(finalImageToReturn, returnAllStats=False, applyMaskToAll=False, 
                              mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus, useAnnulus=useAnnulus)
        casalogPost('********************************************')
        casalogPost('***** SNR in final momDiff image: %.1f *****' % (momDiffSNR))
        casalogPost('********************************************')
        if areaString.find(',') > 0:
            pathCode = areaString.split(',')[1].strip()
        casalogPost('***** pathCode: %s   momDiffCode: %s *****' % (pathCode,momDiffCode))
    else:
        momDiffCode = ''
        momDiffSNR = 0
        finalImageToReturn = '' # could return the image if I find the right name

    ##########################################################################
    # Finish up by generating warnings (if necessary), checking for AllCont, 
    # and writing summary of results to the _findContinuum.dat file
    ##########################################################################
    nranges = len(selection.split(';'))
    nkeep = MAX_RANGES_FOR_IMMOMENTS
    if nranges > nkeep and casaVersion < '6.5.3':
        # avoid CAS-13894
        selection = cutSomeNarrowRanges(selection, nkeep)
        channelList = convertSelectionIntoChannelList(selection)
        aggregateBandwidth = computeBandwidth(selection, channelWidth, 0)
        casalogPost('Cut %d narrowest ranges to avoid CAS-13894 in casa-%s.  New aggregate bandwidth = %f' % (casaVersion,aggregateBandwidth))
    if (meanSpectrumFile == ''): 
        meanSpectrumFile = buildMeanSpectrumFilename(img, meanSpectrumMethod, peakFilterFWHM, amendMaskIterationName, nbin)
        if outdir != '':
            meanSpectrumFile = os.path.join(outdir, os.path.basename(meanSpectrumFile))
    # only the base name of the meanSpectrumFile is used by writeContDat, not the contents
    contDat = writeContDat(meanSpectrumFile, selection, png, aggregateBandwidth,
                           firstFreq, lastFreq, channelWidth, img, imageInfo, vis, 
                           spw=spw, source=source)
    executionTimeSeconds = timeUtilities.time() - executionTimeSeconds
    casalogPost("Final selection: %s" % (selection))
    channelList = convertSelectionIntoChannelList(selection)
    pdiff = 100*(np.median(avgspectrumAboveThreshold[channelList])-medianTrue) / medianTrue
    dottedCyanLevel = np.median(avgspectrumAboveThreshold[channelList])
    casalogPost("Median of cyan ranges spw%s: %f  medianTrue(black): %f  percentDiff=%f" % (str(spw), dottedCyanLevel, medianTrue, pdiff))
    #####################################################################################
    # Post Final warnings to the log (as they might have changed due to reversion above)
    #####################################################################################
    warningStringList = gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction)
    warningStrings = []
    for w in warningStringList:  # PIPE-2158 for Cycle 11
        if len(w) == 0:
            warningStrings.append(w)
        else:
            warningStrings.append(w.split()[-1])
#    warningStrings = [w.split()[-1] for w in warningStrings]  # picks out LowBW and/or LowSpread at end of string (see tooLittleSpread() and tooLittleBandwidth())
    casalogPost("Final warnings: %s" % (str(warningStrings)))
    if selection.find(';') > 0 or aggregateBandwidth*1e9 < 0.5*np.abs(channelWidth*nchan):
        # There is more than 1 group after blue pruning, page 258, so there might be a line in that gap
        # or less than half was selected for continuum, so this is dubious enough that we should clean it
        allContinuum = False
    else:
        allContinuum = allContinuumSelected(selectionPreBluePruning, nchan, spw=spw)
        if allContinuum and xyChanRange0 is not None:
            # fix for PIPE-2355: add a label to the .png
            xcenter = np.mean(xyChanRange0, axis=0)[0]
            ycenter = np.mean(xyChanRange0, axis=0)[1]
            yspan = np.abs(xyChanRange0[0][1] - xyChanRange0[1][1])
            y0 = xyChanRange0[0][1] - yspan*1.2
#            casalogPost('xyChanRange=%s, xcenter=%f, yspan=%f, y0=%f'%(xyChanRange0,xcenter,yspan,y0))
            ax1.text(xcenter, y0, 'AllCont', size=14, ha='center')
            allcontpng = png.replace('.png','_allcont.png')
            pl.savefig(allcontpng, dpi=dpi)
            casalogPost('Wrote %s' % (allcontpng))
            os.remove(png)
            png = allcontpng
            
    casalogPost("final png: %s" % (png))
    casalogPost("final jointMask: %s" % jointMask)
    casalogPost("Finished findContinuum.py. Execution time: %.1f seconds" % (executionTimeSeconds))
    ########################################################################
    # There are 16 possible return lists, based on control parameters.
    # The pipeline use case (which has changed with release) is noted below.
    ########################################################################
    if returnSnrs:
        if returnAllContinuumBoolean:
            if returnSigmaFindContinuum:
                if returnWarnings:
                    return(selection, png, aggregateBandwidth, allContinuum, warningStrings, jointMask, mom0snrs, mom8snrs, maxBaseline, sigmaFindContinuumAtEndOfOriginalIteration, intersectionOfSelections, rangesDropped, amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision, finalImageToReturn, pathCode, momDiffCode, momDiffSNR)
                else:
                    return(selection, png, aggregateBandwidth, allContinuum, mom0snrs, mom8snrs, maxBaseline, sigmaFindContinuumAtEndOfOriginalIteration, intersectionOfSelections, rangesDropped, amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision, finalImageToReturn, pathCode, momDiffCode, momDiffSNR)
            else:
                if returnWarnings:
                    return(selection, png, aggregateBandwidth, allContinuum, warningStrings, jointMask, mom0snrs, mom8snrs, maxBaseline)  
                else:
                    return(selection, png, aggregateBandwidth, allContinuum, mom0snrs, mom8snrs, maxBaseline)
        else:
            if returnSigmaFindContinuum:
                if returnWarnings:
                    return(selection, png, aggregateBandwidth, warningStrings, jointMask, mom0snrs, mom8snrs, maxBaseline, sigmaFindContinuumAtEndOfOriginalIteration, intersectionOfSelections, rangesDropped, amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision, finalImageToReturn, pathCode, momDiffCode, momDiffSNR)   
                else:
                    return(selection, png, aggregateBandwidth, mom0snrs, mom8snrs, maxBaseline, sigmaFindContinuumAtEndOfOriginalIteration, intersectionOfSelections, rangesDropped, amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision, finalImageToReturn, pathCode, momDiffCode, momDiffSNR)
            else:
                if returnWarnings:
                    return(selection, png, aggregateBandwidth, allContinuum, warningStrings, jointMask, mom0snrs, mom8snrs, maxBaseline)
                else:
                    return(selection, png, aggregateBandwidth, allContinuum, mom0snrs, mom8snrs, maxBaseline)
    else: # not returnSnrs
        if returnAllContinuumBoolean:  # default=True
            if returnSigmaFindContinuum:
                if returnWarnings:  # default=True
                    return(selection, png, aggregateBandwidth, allContinuum, warningStrings, jointMask, sigmaFindContinuumAtEndOfOriginalIteration, intersectionOfSelections, rangesDropped, amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision, finalImageToReturn, pathCode, momDiffCode, momDiffSNR, meanSpectrumFile, contDat, positiveThreshold, negativeThreshold, medianTrue, dottedCyanLevel) 
                else:
                    return(selection, png, aggregateBandwidth, allContinuum, sigmaFindContinuumAtEndOfOriginalIteration, intersectionOfSelections, rangesDropped, amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision, finalImageToReturn, pathCode, momDiffCode, momDiffSNR, meanSpectrumFile, contDat, positiveThreshold, negativeThreshold, medianTrue, dottedCyanLevel)
            else:
                if returnWarnings:
                    if returnMomDiffSNR:
                        return(selection, png, aggregateBandwidth, allContinuum, warningStrings, jointMask, momDiffSNR)
                    else:
                        # Pipeline 2020+ use case
                        return(selection, png, aggregateBandwidth, allContinuum, warningStrings, jointMask) #  added mask name here for PL2022+
                else:
                    if returnMomDiffSNR:
                        return(selection, png, aggregateBandwidth, allContinuum, momDiffSNR)
                    else:
                        # Pipeline Cycle 7 use case
                        return(selection, png, aggregateBandwidth, allContinuum)
        else:
            if returnSigmaFindContinuum:
                if returnWarnings:
                    return(selection, png, aggregateBandwidth, warningStrings, jointMask, sigmaFindContinuumAtEndOfOriginalIteration, intersectionOfSelections, rangesDropped, amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision, finalImageToReturn, pathCode, momDiffCode, momDiffSNR)    
                else:
                    return(selection, png, aggregateBandwidth, sigmaFindContinuumAtEndOfOriginalIteration, intersectionOfSelections, rangesDropped, amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision, finalImageToReturn, pathCode, momDiffCode, momDiffSNR)
            else:
                if returnWarnings:
                    if returnMomDiffSNR:
                        return(selection, png, aggregateBandwidth, returnWarningStrings, jointMask, momDiffSNR)
                    else:
                        return(selection, png, aggregateBandwidth, returnWarningStrings, jointMask)
                else:
                    if returnMomDiffSNR:
                        # Pipeline Cycle 6 (and prior) use case
                        return(selection, png, aggregateBandwidth)
                    else:
                        return(selection, png, aggregateBandwidth, momDiffSNR)
# end of findContinuum   

def cutSomeNarrowRanges(selection, nkeep=MAX_RANGES_FOR_IMMOMENTS, 
                        delimiter=';'):
    """
    Remove the N narrowest ranges from a channel selection string to avoid 
    CAS-13894 crash of immoments.
    selection: e.g. '14~18;50~60'
    nkeep: number of ranges to keep (failures of imstat have been seen at 244 
        and 245)
    """
    lengths = countChannelsInRanges(selection)
    ncut = len(lengths)-nkeep
    if ncut < 0:
        return selection
    elif nkeep == 0:
        return ''
    casalogPost("Cutting the %d narrowest ranges to avoid CAS-13894" % (ncut))
    indexShortestFirst = np.argsort(lengths)
    j = []
    for i in range(len(lengths)):
        if i not in indexShortestFirst[:ncut]:
            j.append(i)
    selectionRanges = np.array(selection.split(';'))
    selectionRanges = selectionRanges[np.array(j)]
    selection = delimiter.join(selectionRanges)
    return selection

def gauss_kern(size):
    """ 
    Returns a normalized 2D gauss kernel array for convolutions.  Used by smooth.
    """
    size = int(size)
    x= np.mgrid[-size:size+1]
    g = np.exp(-(x**2/float(size)))
    return g / g.sum()

def smooth(x, window_len=10, window='hanning', verbose=False, newMethod=True):
    """
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    Date: 2017-07-13

    Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t = linspace(-2,2,0.1)
    x = sin(t)+random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        if verbose:
            print("Returning unsmoothed input data")
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'gauss']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'gauss'")

    if newMethod:  # superior
        s = np.r_[x[window_len:0:-1], x, x[-2:-window_len:-1]] 
    else:  # this method introduces rollups or rolldowns at the edges
        s = np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    if verbose:
        print("kernel length: %d, window_len: %d, data_len: %d" % (len(s), window_len, len(x)))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    elif window == 'gauss':
        w = gauss_kern(window_len)
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]

def robustMADofContinuumRanges(myChannelList, myIntensity):
    """
    Used in Pipeline2020 going forward.
    myChannelList: subset of channels that are currently the continuum selection
    intensity: list of intensities in *all* channels of the spectrum
    """
    intensity = copy.deepcopy(myIntensity) # prevent changing the input!
    totalChannels = len(intensity)
    intensityInFCSubset = intensity[myChannelList]
    medianFCSubset = np.median(intensityInFCSubset)
    casalogPost('Median of the %d continuum channels: %f' % (len(myChannelList), medianFCSubset))
    if len(myChannelList) >= 10:
#        Original simpler method:        
#        scaledMADFCSubset = MAD(intensityInFCSubset)
#        casalogPost('using MAD of the %d continuum channels: %f' % (len(myChannelList),scaledMADFCSubset))
        # This method removes the effect of a slope in the baseline on the MAD
        myChannelLists = splitListIntoContiguousLists(myChannelList)
        MADs = []
        Medians = []
        for mylist in myChannelLists:
            Medians.append(np.median(intensity[mylist]))
            MADs.append(MAD(intensity[mylist]))
        scaledMADFCSubset = np.median(MADs) / np.sqrt(2)
        snr = (np.max(intensityInFCSubset)-medianFCSubset) / scaledMADFCSubset
        casalogPost('(median of the MADs)/sqrt(2) of the %d ranges: %f' % (len(myChannelLists),scaledMADFCSubset))
        casalogPost('old SNR = (peak-median)/medianMAD = %f' % (snr))

        # But taking the MAD of very narrow ranges (width 2-4 channels) will severely underestimate the true MAD
        # So we should go back to  using all channels at once.  In order to still be able to remove the effect of
        # a slope in the baseline, we could do a linear fit to all of their median values if len(myChannelLists) > 1.
        if len(myChannelLists) > 1:
            slope, intercept = linfit(np.arange(len(Medians)), np.array(Medians), np.array(Medians)*0.01)
            casalogPost('slope=%f, intercept=%f' % (slope,intercept))
            for i,mylist in enumerate(myChannelLists):
                bias = intercept + slope*i
#                casalogPost('subtracting bias = %f' % (bias))
                intensity[mylist] -= bias
            intensityInFCSubset = intensity[myChannelList]
        scaledMADFCSubset = MAD(intensityInFCSubset)   # dividing by 3 makes it close to the prior method for G35.03 spw27
        casalogPost('MAD of all FC channels together = %f,  1/3 of that = %f' % (MAD(intensityInFCSubset), scaledMADFCSubset/3))
        intensityInFCSubset += medianFCSubset   # restore the global median of all FC channels
        snr = (np.max(intensityInFCSubset)-medianFCSubset) / scaledMADFCSubset
        casalogPost('new SNR = (peak-median) / MAD = %f' % (snr))
        idx = np.argsort(intensityInFCSubset)
        npts = len(myChannelList)
        nBaselineChannels = int(np.ceil(0.19 * npts))
        # If there are strong lines in the "continuum" regions, then the MAD can be not representative
        # So we examine the 19% lowest channels instead
        if nBaselineChannels > 35 and snr > 4:  # put the cutoff between TDM and FDM240 (35/.19=184chan)
            mad = MAD(intensityInFCSubset[idx][:nBaselineChannels])
#            percentile = 100.0*nBaselineChannels / totalChannels
            correctionFactor = 1.0 # sigmaCorrectionFactor('min', npts, percentile)
            newMAD = mad * correctionFactor
            casalogPost('Because there is significant signal in the continuum-free ranges, computing the MAD inferred from the lowest %d channels: %f (using correctionFactor=%f)' % (nBaselineChannels, newMAD, correctionFactor))
            if newMAD < scaledMADFCSubset:
                newMedianFCSubset = np.median(intensityInFCSubset[idx][:nBaselineChannels])
                casalogPost('Because the MAD is lower, computing the MAD and median (%f) of the baseline channels.' % (newMedianFCSubset))
                scaledMADFCSubset = np.mean([scaledMADFCSubset, newMAD])
                medianFCSubset = np.mean([medianFCSubset, newMedianFCSubset])
                casalogPost('Using the mean of the two methods: median=%f MAD=%f' % (medianFCSubset, scaledMADFCSubset))
#            medianFCSubset = medianCorrected('min', percentile, np.median(intensityInFCSubset[idx][:nBaselineChannels]), 
#                                             scaledMADFCSubset, 1, True)
#            casalogPost('Inferred median = %f' % (medianFCSubset))
    else:  # there are too few channels to compute a reliable MAD
        casalogPost('using MAD of all channels because there are too few channels in line-free regions (%d)' % (len(myChannelList)))
        scaledMADFCSubset = MAD(intensity)
    return scaledMADFCSubset, medianFCSubset

def gatherWarnings(selection, chanInfo, smallBandwidthFraction, smallSpreadFraction):
    """
    Used in Pipeline2020 going forward.
    Return list of warning strings (often empty).
    """
    warningStrings = []
    warning = tooLittleBandwidth(selection, chanInfo, smallBandwidthFraction)
    if warning is not None:
        warningStrings.append(warning)
    else:
        warningStrings.append('')  # PIPE-2158
    warning = tooLittleSpread(selection, chanInfo, smallSpreadFraction)
    if warning is not None:
        warningStrings.append(warning)
    else:
        warningStrings.append('')  # PIPE-2158
    return warningStrings

def compute4LetterCodeAndUpdateLegend(mom8fcPeak, mom8fcPeak0, mom8fcPeakOutside, 
                                      mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                      mom8fcMAD, mom8fcMAD0, thresholdForSame, areaString, 
                                      labelDescs, fontsize, ax1, amendMaskDecision, 
                                      extraMaskDecision, extraMaskDecision2, autoLowerDecision,
                                      intersectionOfSelections, 
                                      useMomentDiff, warnings, channelRanges, sigmaUsedToAmendMask, momDiffSNR=0,
                                      skipDecisionCode=False, replace=False, addedSelection=False,
                                      autoLower2Decision=False, amendMaskIterations=4):
    """
    Used in Pipeline2020 going forward.
    Add a variable length code and a 4-letter code to the end of the last line of the upper legend.
    useMomentDiff: specified in the call to findContinuum()
    Returns:  labelDescs, areaString, mycode
    """
    casalogPost('compute4LetterCodeAndUpdateLegend: replace=%s' % replace)
    if replace:
        areaString = areaString.split(',')[0]
    if not skipDecisionCode:
        decisionCode = ''
        print("amend=%s, extra=%s, extra2=%s, autoLower=%s" % (amendMaskDecision, extraMaskDecision, extraMaskDecision2, autoLowerDecision))
        if amendMaskDecision in ['No',False]:
            if extraMaskDecision in ['No',False]:
                if sigmaUsedToAmendMask == AMENDMASK_PIXEL_RATIO_EXCEEDED:
                    casalogPost("Adding ', PR' to the plot legend")
                    decisionCode += ', PR'
                elif sigmaUsedToAmendMask == AMENDMASK_PIXELS_ABOVE_THRESHOLD_EXCEEDED:
                    casalogPost("Adding ', P#' to the plot legend")
                    decisionCode += ', P#'
                else:
                    casalogPost("Adding ', S' to the plot legend")
                    decisionCode += ', S'
            elif extraMaskDecision == 'NoImprovement':
                casalogPost("Adding ', Se' to the plot legend")
                decisionCode += ", Se"
            elif extraMaskDecision == 'YesMom':
                casalogPost("Adding ', ED' to the plot legend")
                decisionCode += ', ED'
            elif extraMaskDecision == 'YesCube':
                casalogPost("Adding ', EC' to the plot legend")
                decisionCode += ', EC'
        else:
            decisionCode += ', A'
            if amendMaskDecision == 'YesMom':
                if useMomentDiff:
                    decisionCode += 'D'
                else:
                    decisionCode += '8'
            elif amendMaskDecision == 'YesCube':
                decisionCode += 'C'
            if extraMaskDecision in ['YesMom','YesCube'] or extraMaskDecision2 in ['YesMom','YesCube']:
#                if extraMaskDecision in ['YesMom','YesCube']:
#                    decisionCode += 'E'
                if intersectionOfSelections:
                    decisionCode += 'I'
                    if extraMaskDecision2 in ['YesMom','YesCube']:
                        decisionCode += 'E'
                else:
                    decisionCode += 'E'
            elif intersectionOfSelections:
                # this will only happen if findContinuum was run with amendMaskIterations=1 instead of 2
                decisionCode += 'I'
        if autoLowerDecision in ['YesMom','YesCube']:
            decisionCode += 'X'
        elif autoLowerDecision == 'NoImprovement':
            decisionCode += 'x'
        elif autoLowerDecision not in ['No',False]:
            decisionCode += '?'
        if amendMaskIterations > 3:
            print("autoLower2Decision = ", autoLower2Decision)
            if autoLower2Decision in ['YesMom','YesCube']:
                decisionCode += 'Y'
            elif autoLower2Decision == 'NoImprovement':
                decisionCode += 'y'
            elif autoLower2Decision not in ['No',False]:
                decisionCode += '?'
        if addedSelection:
            decisionCode += '+' # added a selection of channels at the end due to LowBW/LowSpread
        areaString += decisionCode
        casalogPost('Decision code: %s' % (decisionCode.split(',')[1]))
    mycode = compute4LetterCode(mom8fcPeak, mom8fcPeak0, mom8fcPeakOutside, 
                                mom8fcPeakOutside0, mom8fcSum, mom8fcSum0, 
                                mom8fcMAD, mom8fcMAD0, thresholdForSame)
    areaString += ', ' + mycode + ', difSNR=%.1f' % (momDiffSNR)
    if not skipDecisionCode:
        for warning in warnings:
            if warning.find('amount') > 0:
                areaString += ', LowBW'
            if warning.find('spread') > 0:
                areaString += ', LowSpread'
    myaxis = pl.gca()
    for f in myaxis.texts:
        casalogPost("Label: %s" % (f.get_text()))
    labelDescs[-1].remove()
    token = areaString.split(';')
    areaString = ';'.join([token[0], ' %d ranges'%(channelRanges), token[2]])
    casalogPost("Drawing new areaString: %s" % (areaString))
    labelDesc = ax1.text(0.5, 0.99-3*0.03, areaString, transform=ax1.transAxes, ha='center', size=fontsize+difSNRFontsizeAdjust)
    labelDescs.append(labelDesc)
#    casalogPost("calling pl.draw")
    pl.draw()
#    casalogPost("done pl.draw")
    return labelDescs, areaString, mycode

def updateChannelRangesOnPlot(labelDescs, selection, ax1, separator, avgspectrumAboveThreshold, 
                              skipchan, medianTrue, positiveThreshold, upperXlabel, ax2, 
                              channelWidth, fontsize=10, remove=True, nbin=1, initialPeakOverMad=-1):
    """
    Used in Pipeline2020 going forward.
    channelWidth: in Hz
    Returns: new aggregate bandwidth in GHz
    """
    if remove:
        # exclude final item, which is the 3rd line of the legend
#        print("labelDescs: ", labelDescs)
        for ld,labelDesc in enumerate(labelDescs[:-1]): 
#            print("type(labelDesc) = ", type(labelDesc))
            if type(labelDesc) == list:  # plot artists (i.e., the cyan lines)
                if len(labelDesc) > 0:
                    labelDesc.pop(0).remove()
            else: # text artists (i.e., the chan0~chan1 labels above the cyan lines)
                try:
                    labelDesc.remove()
                except:
                    # prevent a crash; not sure why this happens sometimes on the first call
                    pass
    print("Calling plotChannelSelections: ", selection)
    labelDescs, xyChanRange0 = plotChannelSelections(ax1, selection, separator, 
                          avgspectrumAboveThreshold, skipchan, medianTrue, 
                          positiveThreshold)
    loc = upperXlabel.find('contBW')
    aggregateBandwidth = computeBandwidth(selection, channelWidth, 0)
    if loc > 0:
        upperXlabel = upperXlabel[:loc] + 'contBW: %.2f MHz' % (aggregateBandwidth * 1000)
        if nbin >= NBIN_THRESHOLD: # PIPE-848
            upperXlabel += ', iPoM=%.1f, nbin=%d' % (initialPeakOverMad, nbin)
        casalogPost("Setting new upper x-axis label: %s"  % (upperXlabel))
        ax2.set_xlabel(upperXlabel, size=fontsize) # this artist cannot be removed in CASA 5.6, so just overwrite it
    return aggregateBandwidth, xyChanRange0

def computeNpixCubeMedian(mom8fcMedian, cubeLevel, MADCubeOutside, mom8fc, jointMaskTest):
    """
    Used in Pipeline2020 going forward.
    jointMaskTest: can be a mask, a mask statement or compound mask statement
    """
    threshold = mom8fcMedian + cubeLevel*MADCubeOutside
    if jointMaskTest == '' or jointMaskTest is None:
        npixels = imstat(mom8fc, listit=imstatListit, mask='"%s" > %f'%(mom8fc, threshold))['npts']
    else:
        npixels = imstat(mom8fc, listit=imstatListit, mask='"%s" > %f && %s'%(mom8fc, threshold, jointMaskTest))['npts']
    if len(npixels) == 0:
        NpixCubeMedian = 0
    else:
        NpixCubeMedian = npixels[0]
    return NpixCubeMedian

def computeNpixMom8MedianBadAtm(mom8fcMedian, mom8levelBadAtm, mom8fcMAD, mom8fc, jointMaskTest):
    """
    Used in Pipeline2020 going forward.
    jointMaskTest: can be a mask, a mask statement or compound mask statement
    """
    threshold = mom8fcMedian + mom8levelBadAtm*mom8fcMAD
    if jointMaskTest == '' or jointMaskTest is None:
        npixels = imstat(mom8fc, listit=imstatListit, mask='"%s" > %f'%(mom8fc, threshold))['npts']
    else:
        npixels = imstat(mom8fc, listit=imstatListit, mask='"%s" > %f && %s'%(mom8fc, threshold, jointMaskTest))['npts']
    if len(npixels) == 0:
        NpixMom8MedianBadAtm = 0
    else:
        NpixMom8MedianBadAtm = npixels[0]
    return NpixMom8MedianBadAtm

def computeNpixMom8Median(mom8fcMedian, mom8level, mom8fcMAD, mom8fc, jointMaskTest, verbose=True):
    """
    Used in Pipeline2020 going forward.
    mom8fcMedian: the median of the moment8 image being used 
    mom8level: a sigma value to use
    mom8fcMAD: a scaled MAD
    jointMaskTest: can be a mask, a mask statement or compound mask statement
    """
    threshold = mom8fcMedian + mom8level*mom8fcMAD
    if jointMaskTest == '' or jointMaskTest is None:
        results = imstat(mom8fc, listit=imstatListit, mask='"%s" > %f'%(mom8fc, threshold))
        npixels = results['npts']
    else:
        mymask = '"%s" > %f && %s' % (mom8fc, threshold, jointMaskTest)
        casalogPost("computeNpixMom8Median(): Running imstat('%s', mask='%s')" % (mom8fc,mymask))
        results = imstat(mom8fc, mask=mymask, listit=imstatListit)
        npixels = results['npts']
    if len(npixels) == 0:
        NpixMom8Median = 0
        peak = 0
    else:
        peak = results['max'][0]
        NpixMom8Median = npixels[0]
    if verbose:
        casalogPost("computeNpixMom8Median(): momMedian=%f, momlevel=%f, momMAD=%f, peak=%f, jointMaskTest=%s" % (mom8fcMedian, mom8level, mom8fcMAD, peak, jointMaskTest))
        casalogPost("                         threshold=%f, NpixMom=%d" % (threshold,NpixMom8Median))
    return NpixMom8Median

def replaceLineFullRangesWithNoise(intensity, selection, median, scaledMAD):
    """
    Used in Pipeline2020 going forward.
    intensity: an array of intensity values of *all* channels in the spectrum
    selection: a channel selection string in CASA format, whose intensities should be replaced
    median: of the channels *not* in the selection string (for the noise generation)
    scaledMAD: of the channels *not* in the selection string (for the noise generation)
    Returns: a new intensity array, of the same dimension as the input array
    """
    ranges = selection.split(';')
    random.seed(2001)  # A Space Odyssey (guarantees that .onlyExtraMask will get same result everytime)
    for myrange in ranges:
        c0,c1 = [int(i) for i in myrange.split('~')]
#        print("len(intensity)=%d, c0,c1 = %d,%d" % (len(intensity), c0,c1))
        peakSNR = (np.max(intensity[c0:c1+1]) - median) / scaledMAD
        if peakSNR > 7:  # was 15 in PL2023 (when the SNR values were anomalously high)
            # Note because the continuum is still in this spectrum, it can have a slope which
            # will make the MAD higher than normal, so we may need to account for this he scaledMAD 
            # somewhat.
            intensity[c0:c1+1] = median + pickRandomErrors(c1-c0+1) * scaledMAD
            casalogPost("Replacing channel range %d:%d (peakSNR=%f) with median=%f +- randomNumber*scaledMAD=%f" % (c0,c1,peakSNR,median,scaledMAD))
        else:
            casalogPost("Not replacing channel range %d:%d (peakSNR=%f)" % (c0,c1,peakSNR))
    return intensity

def compute4LetterCode(mom8fcPeak, mom8fcPeak0, mom8fcPeakOutside, mom8fcPeakOutside0, mom8fcSum, 
                       mom8fcSum0, mom8fcMAD, mom8fcMAD0, threshold=0.01, returnRatios=False, momDiffSNR=0):
    """
    Used in Pipeline2020 going forward.
    threshold: to use for Peak, PeakOutside, and Sum;  double this value is used for MAD (i.e. 2%)
    """
    better = ''
    changeInPeakInside = (mom8fcPeak-mom8fcPeak0)/mom8fcPeak0
    ratios = [] # these will be negative values when the quantity improved
    ratios.append(changeInPeakInside)
    if changeInPeakInside > threshold:
        casalogPost('peak inside = H because (%f-%f)/%f=%f is > %f' % (mom8fcPeak,mom8fcPeak0,mom8fcPeak0,changeInPeakInside,threshold))
        better += 'H'
    elif changeInPeakInside < -threshold:
        casalogPost('peak inside = L because (%f-%f)/%f=%f is < -%f' % (mom8fcPeak,mom8fcPeak0,mom8fcPeak0,changeInPeakInside,threshold))
        better += 'L'
    else:
        casalogPost('peak inside = S because (%f-%f)/%f=%f is between -%f and +%f' % (mom8fcPeak,mom8fcPeak0,mom8fcPeak0,changeInPeakInside,threshold,threshold))
        better += 'S'
    changeInPeakOutside = (mom8fcPeakOutside-mom8fcPeakOutside0)/mom8fcPeakOutside0
    ratios.append(changeInPeakOutside)
    if changeInPeakOutside > threshold:
        casalogPost('peak outside = H because (%f-%f)/%f=%f is > %f' % (mom8fcPeakOutside,mom8fcPeakOutside0,mom8fcPeakOutside0,changeInPeakOutside,threshold))
        better += 'H'
    elif changeInPeakOutside < -threshold:
        casalogPost('peak outside = L because (%f-%f)/%f=%f is < -%f' % (mom8fcPeakOutside,mom8fcPeakOutside0,mom8fcPeakOutside0,changeInPeakOutside,threshold))
        better += 'L'
    else:
        casalogPost('peak outside = S because (%f-%f)/%f=%f is between -%f and +%f' % (mom8fcPeakOutside,mom8fcPeakOutside0,mom8fcPeakOutside0,changeInPeakOutside,threshold,threshold))
        better += 'S'
    changeInSum = (mom8fcSum-mom8fcSum0)/mom8fcSum0
    ratios.append(changeInSum)
    if changeInSum > threshold:
        casalogPost('sum = H because (%f-%f)/%f=%f is > %f ' % (mom8fcSum,mom8fcSum0,mom8fcSum0,changeInSum,threshold))
        better += 'H'
    elif changeInSum < -threshold:
        casalogPost('sum = L because (%f-%f)/%f=%f is < -%f ' % (mom8fcSum,mom8fcSum0,mom8fcSum0,changeInSum,threshold))
        better += 'L'
    else:
        casalogPost('sum = S because (%f-%f)/%f=%f is between -%f and +%f ' % (mom8fcSum,mom8fcSum0,mom8fcSum0,changeInSum,threshold,threshold))
        better += 'S'
    changeInMad = (mom8fcMAD-mom8fcMAD0)/mom8fcMAD0
    ratios.append(changeInMad)
    madThreshold = 2*threshold
    if changeInMad > madThreshold:
        casalogPost('MAD = H because (%f-%f)/%f=%f is > %f ' % (mom8fcMAD,mom8fcMAD0,mom8fcMAD0,changeInMad,madThreshold))
        better += 'H'
    elif changeInMad < -madThreshold:
        casalogPost('MAD = L because (%f-%f)/%f=%f is < -%f ' % (mom8fcMAD,mom8fcMAD0,mom8fcMAD0,changeInMad,madThreshold))
        better += 'L'
    else:
        casalogPost('MAD = S because (%f-%f)/%f=%f is between -%f and +%f' % (mom8fcMAD,mom8fcMAD0,mom8fcMAD0,changeInMad,madThreshold,madThreshold))
        better += 'S'
    casalogPost("4-letter code: %s" % (better))
    if returnRatios:
        return better, ratios
    else:
        return better

def cubeNoiseLevel(cube, pbcube='', mask='', percentile=50, chans='', subimage=False, imageInfo=None):
    """
    Used in Pipeline2020 going forward.
    Computes median of the per-channel scaled MAD within the standard noise annulus
    and outside the specified mask
    chans: channel selection string (passed to imstat)
    mask: either mask image to use (e.g. the string passed to imstat will be '"mask" == 0')
          or a proper mask string containing >, < or ==
    percentile: compute this percentile value of the per-channel scaled MAD
          (examples: 50 = median,  90 = close to the highest value)
          if it is a list, then a list of values is returned
    Returns: median of MAD, median
          or [list of MADs], median
    """
    if not os.path.exists(cube):
        print("Could not find the cube")
        return
    if pbcube == '':
        pbcube = cube.replace('.image','.pb').replace('.residual','.pb')
        if not os.path.exists(pbcube):
            casalogPost("Could not find the corresponding pbcube: %s" % (pbcube))
            casalogPost("Will use the whole image (or the specified mask) to compute the MAD.")
    if mask == '':
        mask = cube.replace('.image','.mask').replace('.residual','.mask')
        if not os.path.exists(mask):
            mask = cube + '.joint.mask2'
            if not os.path.exists(mask):
                mask = cube + '.joint.mask'
                if not os.path.exists(mask):
                    print("Could not find any corresponding clean mask or findContinuum jointmask")
                    mask = ''
    if os.path.exists(pbcube):
        lowerAnnulusLevel, higherAnnulusLevel = findOuterAnnulusForPBCube(pbcube, False, False, subimage, imageInfo)
        if mask != '' and mask is not None:
            if mask.find("==") < 0 and mask.find(">") < 0 and mask.find("<") < 0:
                mask = '"'+mask+'"==0'
            print("    Using mask: ", mask)
            mymask = '"%s">%f && "%s"<%f && %s' % (pbcube, lowerAnnulusLevel, pbcube, higherAnnulusLevel, mask)
        else:
            mymask = '"%s">%f && "%s"<%f' % (pbcube, lowerAnnulusLevel, pbcube, higherAnnulusLevel)
    else:
        mymask = ''
    print("    mymask expression = ", mymask)
    results = imstat(cube, axes=[0,1,2], chans=chans, mask=mymask, stretch=True, listit=imstatListit)
    cubeMedian = results['median'][0]
    result = results['medabsdevmed']
    if type(percentile) != list and type(percentile) != np.ndarray:
        percentile = [percentile]
    cubeScaledMAD = []
    for p in percentile:
        cubeScaledMAD.append(scoreatpercentile(result,p) / 0.6745)
    if len(cubeScaledMAD) > 1:
        return cubeScaledMAD, cubeMedian
    else:
        return cubeScaledMAD[0], cubeMedian

def tooLittleBandwidth(selection, chanInfo, fraction=0.05):
    """
    Used in Pipeline2020 going forward.
    Returns: a string if there is a warning, or None if not
    """
    nchan, firstFreq, lastFreq, channelWidth = chanInfo
    aggBandwidthHz = computeBandwidth(selection, channelWidth, 1) * 1e9
    myfraction = aggBandwidthHz/float(channelWidth*nchan)
    if myfraction < fraction: # compute4LetterCodeAndUpdateLegend() keys on the presence of the word 'amount'
        warning = 'WARNING: Small amount of fractional bandwidth found (%.3f < %.3f) = LowBW' % (myfraction,fraction)
        casalogPost(warning, priority='WARN')
        return warning
    else:
        casalogPost('Current fractional bandwidth is %.3f' % (myfraction))
    return None

def computeSpread(selection, channelWidth):
    """
    Used in Pipeline2020 going forward.
    """
    ranges = selection.split(';')
    for i,r in enumerate(ranges):
        result = r.split('~')
        if (len(result) == 2):
            a,b = result
            if i == 0:
                startchan = int(a)
            # keep extending the endchan where there is more than 1 group
            endchan = int(b)
    return (endchan-startchan+1)*channelWidth

def tooLittleSpread(selection, chanInfo, fraction=0.33):
    """
    Used in Pipeline2020 going forward.
    Returns: a string if there is a warning, or None if not
    """
    nchan, firstFreq, lastFreq, channelWidth = chanInfo
    spread = computeSpread(selection, channelWidth)
    myfraction = spread/float(nchan*channelWidth)
    if myfraction < fraction:  # compute4LetterCodeAndUpdateLegend() keys on the presence of the word 'spread'
        warning = 'WARNING: Fractional spread of frequency coverage is small (%.3f < %.3f) = LowSpread' % (myfraction,fraction)
        casalogPost(warning, priority='WARN')
        return warning
    return None

def amendMaskYesOrNo(badAtmosphere, median, momSNR, momSNRCube, Npix, NpixBadAtm, NpixCube, 
                     TenEventSigma, MADMomOutside, MADCubeOutside, momLevel, 
                     momLevelBadAtm, cubeLevel, NpixCube2, fractionNegativePixels,
                     Npix2=None, Npix2BadAtm=None, skipCubeNoise=False,
                     tooManyPixels=850, verbose=True):
    """
    Used in Pipeline2020 going forward.
    median: median of mom8fc img if fc(useMomentDiff=False), or momDiff (mom8-mom0scaled) if useMomentDiff=True
    momSNR: SNR of mom8fc img if fc(useMomentDiff=False), or momDiff (mom8-mom0scaled) if useMomentDiff=True
    etc.
    badAtmosphere: either a boolean or a string ('goodAtm' or 'badAtm')
    momLevel: for the non-badAtmosphere case, this is passed in from momDiffLevelForProceeding 
    """
    AMENDMASK_PIXEL_RATIO_EXCEEDED = -1
    AMENDMASK_PIXELS_ABOVE_THRESHOLD_EXCEEDED = -2
    decision = 'No'
    AmendMaskLevel = 0.0
    sigmaUsed = AMENDMASK_PIXEL_RATIO_EXCEEDED
    if Npix2 is not None:
        if Npix2 > 0:
            pixelRatio = float(Npix)/Npix2
            if pixelRatio > 2.2:  # IRAS16293 maser gives 2.1667, NGC6334I spw16 gives 2.333
                casalogPost('Pixel ratio is too high to trigger AmendMask: %f > 2.2' % (pixelRatio))
                return decision, AmendMaskLevel, sigmaUsed
    if badAtmosphere in ['badAtm',True]:
        badAtmosphere = True
        if fractionNegativePixels > 0.1 and NpixBadAtm > tooManyPixels:
            sigmaUsed = AMENDMASK_PIXELS_ABOVE_THRESHOLD_EXCEEDED
            casalogPost('Fraction of negative pixels (%f) > 0.1 and too many pixels (%d) above threshold to trigger AmendMask' % (fractionNegativePixels,NpixBadAtm))
            return decision, AmendMaskLevel, sigmaUsed
    elif badAtmosphere in ['goodAtm',False]:
        badAtmosphere = False
        if fractionNegativePixels > 0.1 and Npix > tooManyPixels:
            sigmaUsed = AMENDMASK_PIXELS_ABOVE_THRESHOLD_EXCEEDED
            casalogPost('Fraction of negative pixels (%f) > 0.1 and too many pixels (%d) above threshold to trigger AmendMask' % (fractionNegativePixels,Npix))
            return decision, AmendMaskLevel, sigmaUsed
        else:
            casalogPost('fractionNegativePixels = %f, Npix = %d, allows it to trigger AmendMask' % (fractionNegativePixels,Npix))
    else:
        casalogPost('badAtmosphere has unrecognized value = %s' % (badAtmosphere))
    sigmaUsed = 0 # this is the value that will be returned if threshold are not met
    if badAtmosphere:
        casalogPost("Using badAtmosphere thresholds: momLevelBadAtm=%f" % (momLevelBadAtm))
        if momSNR >= momLevelBadAtm and NpixBadAtm >= 9:
            decision = 'YesMom'
            sigmaUsed = 8
            AmendMaskLevel = median + sigmaUsed*MADMomOutside
            casalogPost('Found YesMom (BadAtm) with sigma=%g yielding AmendMaskLevel = %f' % (sigmaUsed,AmendMaskLevel))
    else:
        if verbose:
            print("-------------------------------------")
            print("amendMaskYesOrNo(): median=%f, MADMomOutside=%f, momSNR=%f, Npix=%.f" % (median, MADMomOutside, momSNR, Npix))
            print("amendMaskYesOrNo(): momSNRCube=%f, MADCubeOutside=%f, NpixCube=%.1f" % (momSNRCube, MADCubeOutside, NpixCube))
            print("-------------------------------------")
        if momSNR >= momLevel and Npix >= 9:
            decision = 'YesMom'
            AmendMaskLevel = median + np.max([5.5,TenEventSigma]) * MADMomOutside
            sigmaUsed = np.max([5.5,TenEventSigma])
            casalogPost('Found YesMom with sigma=%g (because momSNR=%f >= momLevel=%f) yielding AmendMaskLevel = %f' % (sigmaUsed,momSNR,momLevel,AmendMaskLevel))
        elif momSNRCube >= cubeLevel and NpixCube >= 9:
            if skipCubeNoise:
                casalogPost('Skipping the check for AmendMask needed on the basis of MADCubeOutside (due to skipCubeNoise=True)')
            else:
                # Need to guard against divide by zero below, so we only assess the
                # ratio if we know that NpixCube > 0, rather than making it
                # a 3-term 'elif' clause above.
                # The float() below is to insure real arithmetic in CASA < 6.
                if float(NpixCube2)/NpixCube < 2.0:
                    decision = 'YesCube'
                    AmendMaskLevel = median + np.max([5.0,TenEventSigma]) * MADCubeOutside
                    sigmaUsed = np.max([5.0,TenEventSigma])
                    casalogPost('YesCube: sigmaUsed = %f' % (sigmaUsed))
    return decision, AmendMaskLevel, sigmaUsed

def extraMaskYesOrNo(badAtmosphere, median, momSNR, momSNRCube, NpixMom, NpixMomBadAtm, NpixCubeAnywhere, 
                     TenEventSigma, MADMomOutside, MADCubeOutside, momDiffLevel, momDiffLevelBadAtm, 
                     cubeLevel, verbose=True):
    """
    Used in Pipeline2020 going forward.  Also used for autoLower decision logic.
    badAtmosphere: either a boolean or a string ('goodAtm' or 'badAtm')
    momSNR: SNR outside the mask, i.e. will be momDiffSNR if useMomentDiff=True (which PL uses)
    NpixMom: unused now
    NpixMomBadAtm: unused now
    NpixCubeAnywhere: unused now
    momDiffLevel: for the non-badAtmosphere case, this is passed in from momDiffLevelForProceeding 
    MADCubeOutside: not currently used in this function!
    """
    decision = 'No'
    ExtraMaskLevel = 0.0
    sigmaUsed = 0
    if badAtmosphere == 'badAtm':
        badAtmosphere = True
    elif badAtmosphere == 'goodAtm':
        badAtmosphere = False
    if badAtmosphere:
        sigmaUsed = 8
#        if momSNR >= sigmaUsed and NpixMomBadAtm >= 9:
        if momSNR >= momDiffLevelBadAtm: # and NpixMomBadAtm >= 9:
            decision = 'YesMom'
            ExtraMaskLevel = median + sigmaUsed*MADMomOutside
            casalogPost('badAtm: Found YesMom (BadAtm) with sigma=%g yielding ExtraMaskLevel = %f' % (sigmaUsed,ExtraMaskLevel))
        else:
            casalogPost('badAtm: decision=No because momSNR=%f < %f' % (momSNR,momDiffLevelBadAtm))
    else:
        if verbose:
            print("-------------------------------------")
            print("extraMaskYesOrNo(): median=%f, MADMomOutside=%f, momDiffSNR=%f, NpixMom=%.1f (unused here)" % (median, MADMomOutside, momSNR, NpixMom))
            print("extraMaskYesOrNo(): momSNRCube=%f, NpixCubeAnywhere=%.1f (unused here)" % (momSNRCube, NpixCubeAnywhere))
            print("-------------------------------------")
        sigmaUsed = np.max([5.5,TenEventSigma])
        cubeSigmaUsed = np.max([5.0,TenEventSigma])
#        if momSNR >= sigmaUsed and NpixMom >= 9:
        if momSNR >= momDiffLevel: # and NpixMom >= 9:
            decision = 'YesMom'
            ExtraMaskLevel = median + sigmaUsed * MADMomOutside
            casalogPost('YesMom: sigmaUsed = %f because momDiffSNR=%f >= momDiffLevel=%f' % (sigmaUsed,momSNR,momDiffLevel))
#        elif momSNRCube >= cubeSigmaUsed and NpixCubeAnywhere >= 9:
#        elif momSNRCube >= cubeLevel:  # and NpixCubeAnywhere >= 9:
#            decision = 'YesCube'
#            ExtraMaskLevel = median + cubeSigmaUsed * MADCubeOutside
#            sigmaUsed = cubeSigmaUsed
#            casalogPost('YesCube: sigmaUsed = %f' % (sigmaUsed))
        else:
#            casalogPost('%s.  Neither YesMom nor YesCube was triggered.' % (decision))
            casalogPost('%s.  YesMom was not triggered because momDiffSNR=%f < momDiffLevel=%f.' % (decision,momSNR,momDiffLevel))
    return decision, ExtraMaskLevel, sigmaUsed

def onlyExtraMaskYesOrNo(badAtmosphere, median, momDiffSNR, momDiffSNRCube, 
                         NpixMomDiff, NpixMomDiffBadAtm, 
                         NpixCubeAnywhere, TenEventSigma, momDiffMAD, 
                         MADCubeOutside, cubePeak, cubeMedian,
                         momDiffLevel, momDiffLevelBadAtm, cubeLevel, 
                         sigmaThreshold=7.5, npixThreshold=7, skipCubeNoise=False, verbose=True):
    """
    Used in Pipeline2020 going forward.
    badAtmosphere: either a boolean or a string ('goodAtm' or 'badAtm')
    momDiffSNR: SNR where peak is measured over whole image (instead of outside mask), 
                   (mom scaledMAD is in denominator)
    momDiffSNRCube: SNR where peak is measured over whole image (instead of outside mask), 
                   (cube scaledMAD is in denominator)
    NpixCubeAnywhere: unused now
    momDiffLevel: for the non-badAtmosphere case, this is passed in from momDiffLevelForProceeding 
    """
    if False:
        print("***** arguments:", badAtmosphere, median, momDiffSNR, momDiffSNRCube, 
                         NpixMomDiff, NpixMomDiffBadAtm, NpixCubeAnywhere, 
                         TenEventSigma, momDiffMAD, MADCubeOutside, cubePeak,
                         cubeMedian)
    decision = 'No'
    onlyExtraMaskLevel = 0.0
    sigmaUsed = 0
    casalogPost('*******************************************')
    casalogPost('npixThreshold = %d' % (npixThreshold))
    casalogPost('*******************************************')
    if badAtmosphere == 'badAtm':
        badAtmosphere = True
    elif badAtmosphere == 'goodAtm':
        badAtmosphere = False
    if badAtmosphere:
        sigmaUsed = 8 # to set the mask level
        if momDiffSNR >= momDiffLevelBadAtm and NpixMomDiffBadAtm >= npixThreshold and ((cubePeak-cubeMedian)>sigmaThreshold*MADCubeOutside or MADCubeOutside==DISABLE_CUBE_NOISE):
            decision = 'YesMom'
            onlyExtraMaskLevel = median + sigmaUsed*momDiffMAD
            casalogPost('Found YesMom (BadAtm) with sigma=%g yielding onlyExtraMaskLevel = %f' % (sigmaUsed,onlyExtraMaskLevel))
        else:
            casalogPost('No onlyExtraMask: momSNR=%f,NpixMomBadAtm=%.1f '%(momDiffSNR,NpixMomDiffBadAtm))
    else:
        if verbose:
            print("-------------------------------------")
            print("onlyExtraMaskYesOrNo(): median=%f, momDiffMAD=%f, momSNRMom=%f, NpixMom=%.1f" % (median, momDiffMAD, momDiffSNR, NpixMomDiff))
            print("onlyExtraMaskYesOrNo(): momSNRCube=%f, (cubePeak=%f - cubeMedian=%f) = %f, %g*MADCubeOutside=%f" % (momDiffSNRCube, cubePeak, cubeMedian, cubePeak-cubeMedian, sigmaThreshold, sigmaThreshold*MADCubeOutside))
            print("-------------------------------------")
        sigmaUsed = np.max([5.5,TenEventSigma])  # to set the mask level (when YesMom is triggered)
        cubeSigmaUsed = np.max([5.0,TenEventSigma])  # to set the mask level (when YesCube is triggered)
        if momDiffSNR >= momDiffLevel and NpixMomDiff >= npixThreshold and (skipCubeNoise or (cubePeak-cubeMedian)>sigmaThreshold*MADCubeOutside):
            decision = 'YesMom'
            onlyExtraMaskLevel = median + sigmaUsed * momDiffMAD
            casalogPost('YesMom: sigmaUsed = %f because momDiffSNR=%f >= momDiffLevel=%f' % (sigmaUsed,momDiffSNR,momDiffLevel))
#        elif momDiffSNRCube >= cubeLevel and NpixCubeAnywhere >= 9  and (cubePeak-cubeMedian)>sigmaThreshold*MADCubeOutside:
#            decision = 'YesCube'
#            onlyExtraMaskLevel = median + cubeSigmaUsed * MADCubeOutside
#            sigmaUsed = cubeSigmaUsed
#            casalogPost('YesCube: sigmaUsed = %f' % (sigmaUsed))
        else:
#            casalogPost('%s.  Neither YesMom nor YesCube was triggered. ' % (decision))
            casalogPost('%s.  YesMom was not triggered. ' % (decision))
            if momDiffSNR < momDiffLevel:
                casalogPost('  because momDiffSNR=%f < momDiffLevel=%f' % (momDiffSNR, momDiffLevel))
            if NpixMomDiff < npixThreshold:
                casalogPost('  because NpixMomDiff=%d < npixThreshold=%d' % (NpixMomDiff, npixThreshold))
            if (cubePeak-cubeMedian) <= sigmaThreshold*MADCubeOutside:
                casalogPost('  because (cubePeak-cubeMedian)=%f <= (sigmaThreshold*MADCubeOutside)=%f' % (cubePeak-cubeMedian, sigmaThreshold*MADCubeOutside))
    return decision, onlyExtraMaskLevel, sigmaUsed

def allContinuumSelected(selection, nchan, fraction=ALL_CONTINUUM_CRITERION, 
                         fraction2=ALL_CONTINUUM_CRITERION_TDM_FULLPOL, spw=''):
    """
    Returns True if only one range is selected and that range covers sufficient
    bandwidth (default fraction = 92.5%)
    nchan: number of channels in the spectrum
    fraction: for nchan >= 75 (default = 0.925 which is 92.5%)
    fraction2: for nchan < 75 (default = 0.91)
    """
    ranges = selection.split(';')
    if nchan < 75:
        threshold = fraction2
    else:
        threshold = fraction
    if len(ranges) == 1:
        if ranges[0].find('~') > 0:
            c0,c1 = [int(i) for i in ranges[0].split('~')]
        elif ranges[0] == '':
            c0 = 1
            c1 = 0
        else: # in case a single channel is selected
            c0,c1 = [int(ranges[0]),int(ranges[0])]
#       Require this single range to be centered:
#        if c0 <= fraction*nchan and c1 >= (1-fraction)*(nchan-1):
#       Range is not required to be centered:
        myfraction = (c1-c0+1)*1.0 / nchan
        if (myfraction >= threshold):
            casalogPost("continuum fraction (spw%s): %d-%d: %f >= %f" % (str(spw),c0,c1,myfraction, threshold))
            return True
        else:
            casalogPost("continuum fraction (spw%s): %d-%d: %f < %f" % (str(spw),c0,c1,myfraction, threshold))
    return False

def byteDecode(buffer):
      if casaVersion >= '5.9.9':
          return buffer.decode('utf-8')
      else:
          return buffer

def maskArgumentMismatch(mask, meanSpectrumFile, useThresholdWithMask):
    """
    This function is called by checkForMismatch.
    Determines if the requested mask does not match what was used to generate 
    the specified meanSpectrumFile.
    Returns: True or False
    """
#    casalogPost('Entered maskArgumentMismatch()')
    grepResult = byteDecode(grep(meanSpectrumFile,'mask')[0])
    if (mask == '' or mask == False):
        if (grepResult != ''): 
            casalogPost('Mismatch: mask was used previously, but we did not request one this time')
            # mask was used previously, but we did not request one this time
            return True
        else:
            # mask was not used previously and we are not requesting one this time
            casalogPost('No mismatch: mask was not used previously and we are not requesting one this time')
            False
    elif (grepResult == ''):
        # requested mask was not used previously
        casalogPost('Mismatch: requested mask was not used previously')
        return True
    else:
        # requested mask was used previously.  Except if the new mask is a subset 
        # of the old name, so check for that possibility.
        tokens = grep(meanSpectrumFile,mask)[0].split()
        newtoken = []
        for token in tokens:
            newtoken.append(byteDecode(token))
        tokens = newtoken
        if mask not in tokens:
            casalogPost('Mismatch: mask name not in meanSpectrumFile')
            return True
        else:
            # check if threshold was 0.00 before and we are using one now
            f = open(meanSpectrumFile,'r')
            lines = f.readlines()
            f.close()
            token = lines[1].split()
            threshold = float(token[0])
            if (useThresholdWithMask and threshold == 0.0) or (not useThresholdWithMask and threshold != 0.0):
                casalogPost('Mismatch in threshold: useThresholdWithMask=%s, threshold=%f' % (useThresholdWithMask, threshold))
                return True
            casalogPost('No mismatch in threshold')
            return False

def centralArcsecArgumentMismatch(centralArcsec, meanSpectrumFile, iteration):
    """
    This function is called by checkForMismatch.
    Determines if the requested centralArcsec value does not match what was used to 
    generate the specified meanSpectrumFile.
    Returns: True or False
    """
    if (centralArcsec == 'auto' or centralArcsec == -1):
        if (grep(meanSpectrumFile,'centralArcsec=auto')[0] == '' and 
            grep(meanSpectrumFile,'centralArcsec=-1')[0] == ''):
            casalogPost("request for auto but 'centralArcsec=auto' not found and 'centralArcsec=-1' not found")
            return True
        else:
            result = grep(meanSpectrumFile,'centralArcsec=auto')[0]
            if (iteration == 0 and result != ''):
                # This will force re-run of any result that went to a smaller size.
                token = result.split('=auto')
                if (len(token) > 1):
                    if (token[1].strip().replace('.','').isdigit()):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
    elif (grep(meanSpectrumFile,'centralArcsec=auto %s'%(str(centralArcsec)))[0] == '' and
          grep(meanSpectrumFile,'centralArcsec=mom0mom8jointMask True')[0] == '' and
          grep(meanSpectrumFile,'centralArcsec=mom0mom8jointMask False')[0] == '' and
          grep(meanSpectrumFile,'centralArcsec=%s'%(str(centralArcsec)))[0] == ''):
        token = grep(meanSpectrumFile,'centralArcsec=')[0].split('centralArcsec=')
        # the Boolean after centralArcsec=mom0mom8jointMask is: initialQuadraticRemoved
        if (len(token) < 2):
            # This should never happen, but if it does, prevent a crash by returning now.
            casalogPost("Did not find string 'centralArcsec=' with a value in the meanSpectrum file.")
            return True
        value = token[1].replace('auto ','').split()[0]
        print("              value = ", value)
        if value.find('mom0mom8jointMask') == 0:
            return False
        try:
            previousRequest = float(value)
            centralArcsecThresholdPercent = 2
            if (100*abs(previousRequest-centralArcsec)/centralArcsec < centralArcsecThresholdPercent):
                casalogPost("request for specific value (%s) and previous value (%s) is within %d percent" % (str(centralArcsec),str(previousRequest),centralArcsecThresholdPercent))
                return False
            else:
                casalogPost("request for specific value but 'centralArcsec=auto %s' not within %d percent" % (str(centralArcsec),centralArcsecThresholdPercent))
                return True
        except:
            # If there is any trouble reading the previous value, then return now.
            casalogPost("Failed to parse floating point value: %s" % str(value), debug=True)
            return True
    else:
        return False

def plotMeanSpectrum(meanSpectrumFile, plotfile='', selection=''):
    """
    Not used by pipeline.
    Reads a *.findcont.residual.meanSpectrum.<method> file and plots it as
    intensity vs. channel.
    plotfile: set to a string to generate a png
    selection: channel selection string to draw in cyan horizontal lines,
       and its median intensity as a dotted cyan line
    """
    if not os.path.exists(meanSpectrumFile):
        print("Could not find file: ", meanSpectrumFile)
        return
    f = open(meanSpectrumFile,'r')
    lines = f.readlines()
    f.close()
    channel = []
    intensity = []
    for line in lines[3:]:
        a,b,c,d = line.split()
        channel.append(int(a))
        intensity.append(float(c))
    intensity = np.array(intensity)
    pl.clf()
    pl.plot(channel,intensity,'k-')
    pl.xlabel('Channel')
    pl.ylabel('Intensity')
    if selection != '':
        channelList = convertSelectionIntoChannelList(selection)
        myChannelLists = splitListIntoContiguousLists(channelList)
        cyanLevel = np.median(intensity[channelList])
        level = (np.max(intensity)-cyanLevel)/3. + cyanLevel
        for myChannelList in myChannelLists:
#            print("myChannelList = ", myChannelList)
            if len(myChannelList) < 2: # add a channel to length of line
                # channel range has only one channel, so cyan line doesn't appear
                myChannelList[0] = myChannelList[0]-0.5
                myChannelList.append(myChannelList[-1]+1)
            elif len(myChannelList) < 3:  # add a channel to length of line
                myChannelList[0] = myChannelList[0]-0.5
                myChannelList[1] = myChannelList[1]+0.5
            elif myChannelList[0] == myChannelList[1]:
                # this should never happen
                myChannelList[0] = myChannelList[0]-0.5
                myChannelList[1] = myChannelList[1]+0.5
            pl.plot(myChannelList,[level]*len(myChannelList),'c-',lw=2)
        pl.plot(pl.xlim(), 2*[cyanLevel], 'c:')
    pl.draw()
    if plotfile != '':
        if plotfile == True:
            plotfile = meanSpectrumFile + '.png'
        pl.savefig(plotfile)
        print("Wrote ", plotfile)

def writeContDat(meanSpectrumFile, selection, png, aggregateBandwidth, 
                 firstFreq, lastFreq, channelWidth, img, imageInfo, vis='', 
                 numchan=None, chanInfo=None, lsrkwidth=None, spw='', source=''):
    """
    This function is called by findContinuum.
    Writes out an ASCII file called <meanSpectrumFile>_findContinuum.dat
    that contains the selected channels, the png name and the aggregate 
    bandwidth in GHz. 
    Returns: name of file written
    meanSpectrumFile: only used to generate the name of the .dat file, unless
       firstFreq <= 0, in which case the readPreviousMeanSpectrum is called on it
       (It splits off name before '.meanSpectrum' and appends _findContinuum.dat)
    vis: if specified, then also write a line with the topocentric velocity ranges
    spw: integer or string ID number; if specified (along with vis), then also write 
         a final line with the topocentric channel ranges for this spw
    """
    if (meanSpectrumFile.find('.meanSpectrumFile') > 0):
#        contDat = meanSpectrumFile.split('.meanSpectrum')[0] + '_findContinuum.dat'
        # remove meanSpectrum from the name to avoid confusion that this is not a mean spectrum file
        contDat = meanSpectrumFile.replace('.meanSpectrumFile','') + '_findContinuum.dat'
    else:
        contDat = meanSpectrumFile + '_findContinuum.dat'
    contDatDir = os.path.dirname(contDat)
    if firstFreq <= 0:
        result = readPreviousMeanSpectrum(meanSpectrumFile)
        nchan = result[4]
        firstFreq = result[6]
        lastFreq = result[7]
    if (firstFreq > lastFreq and channelWidth > 0):
        # restore negative channel widths if appropriate
        channelWidth *= -1
#    print("firstFreq=%f, channelWidth=%f" % (firstFreq,channelWidth))
    lsrkfreqs = 1e-9*np.arange(firstFreq, lastFreq+channelWidth*0.5, channelWidth)
    if (len(contDatDir) < 1):
        contDatDir = '.'
    if (not os.access(contDatDir, os.W_OK) and contDatDir != '.'):
        # Tf there is no write permission, then use the directory of the png
        # since this has been established as writeable in runFindContinuum.
        contDat = os.path.join(os.path.dirname(png), os.path.basename(contDat))
#    try:
    if True:
        f = open(contDat, 'w')
        casalogPost('writeContDat(): aggregateBandwidth = %s' % (aggregateBandwidth))
        f.write('%s %s %g\n' % (selection, png, aggregateBandwidth))
        # Now write the LSRK frequency ranges on a new line
        freqRange = ''
        for i,s in enumerate(selection.split(';')):
            c0,c1 = [int(j) for j in s.split('~')]
            minFreq = np.min([lsrkfreqs[c0],lsrkfreqs[c1]])-0.5*abs(channelWidth*1e-9)
            maxFreq = np.max([lsrkfreqs[c0],lsrkfreqs[c1]])+0.5*abs(channelWidth*1e-9)
            freqRange += '%.9fGHz~%.9fGHz LSRK ' % (minFreq,maxFreq)
        f.write(freqRange+'\n')
        if (len(vis) > 0):
            # vis is a non-blank list or non-blank string
            if (type(vis) == str):
                vis = vis.split(',')
            # vis is now assured to be a non-blank list
            topoFreqRanges = [] # this will be a list of lists
            frame = getFreqType(img).upper()
            for v in vis:
                casalogPost("Converting continuum ranges from %s to TOPO for vis = %s" % (frame,v))
                vselection = ''
                for i,s in enumerate(selection.split(';')):
                    c0,c1 = [int(j) for j in s.split('~')]
                    minFreq = np.min([lsrkfreqs[c0],lsrkfreqs[c1]])-0.5*abs(channelWidth*1e-9)
                    maxFreq = np.max([lsrkfreqs[c0],lsrkfreqs[c1]])+0.5*abs(channelWidth*1e-9)
                    freqRange = '%.5fGHz~%.5fGHz' % (minFreq,maxFreq)
#                    casalogPost("%2d) %s channelRange in cube = %s" % (i, frame, s))
#                    casalogPost("    %s freqRange in cube = %s" % (frame,str(freqRange)))
#                    casalogPost("    img=%s  spw=%s" % (img, str(spw)))
                    if img == '':
                        result, fromFrame = cubeFrameToTopo(img, imageInfo, freqRange, vis=v, nchan=nchan, f0=firstFreq, f1=lastFreq, chanwidth=channelWidth, source=source, spw=spw)
                    else:
                        result, fromFrame = cubeFrameToTopo(img, imageInfo, freqRange, vis=v, source=source, spw=spw)
                    result *= 1e-9 # convert from Hz to GHz
                    # pipeline calls uvcontfit with GHz label only on upper freq
                    freqRange = '%.5f~%.5fGHz' % (np.min(result),np.max(result))
                    casalogPost("    TOPO freqRange for this vis = %s" % str(freqRange))
                    if (i > 0): vselection += ';'
                    vselection += freqRange
                f.write('%s %s\n' % (v,vselection))
                topoFreqRanges.append(vselection)
            if spw != '':
                for i,v in enumerate(vis):
                    topoChanRanges = topoFreqRangeListToChannel(freqlist=topoFreqRanges[i], spw=spw, vis=v)
                    f.write('%s %s\n' % (v,topoChanRanges))
        f.close()
        casalogPost("Wrote %s" % (contDat))
    return contDat
#    except:
#        casalogPost("Failed to write %s" % (contDat))

def getFieldnameFromPipelineImageName(img, verbose=False):
    """
    Extracts the field name from the pipeline image file name.
    -Todd Hunter
    """
    basename = os.path.basename(img)
    fieldname = ''
    styles = ['sci','chk','bp','ph','flux','pol'] # I am anticpating that polcal will be called pol.
    for style in styles:
        if basename.find('_'+style) > 0:
            fieldname = basename.split('_'+style)[0]
    if fieldname == '': 
        print("No image found of these styles: ", styles)
        return fieldname
    # this will now be: a_Xb.s35_0.NGC300
    fieldname = fieldname.split('_0.')[1]
    fieldname = fieldname.strip('_')
    return fieldname

def getSpwFromPipelineImageName(img, verbose=False):
    """
    Extracts the spw ID from the pipeline image file name.
    -Todd Hunter
    """
    sourceName = os.path.basename(img.rstrip('/'))
    if (sourceName.find('.mfs.') < 0):
        if (sourceName.find('.cube.') < 0):
            return 'sourcename'
        else:
            sourceName = sourceName.split('.cube.')[0]
    else:
        sourceName = sourceName.split('.mfs.')[0]
    if sourceName.find('.virtspw') > 0:  # PIPE-1105
        sourceName = sourceName.split('.virtspw')[1].split('.')[0]
    else:
        sourceName = sourceName.split('.spw')[1].split('.')[0]
    if sourceName.isdigit():
        return int(sourceName)
    else:
        return None

def combineContDat(contdatlist, outputfile='cont.dat', fieldname='', spw=''):
    """
    Takes a list of *.dat files created by findContinuum for a single field/spw
    and pulls the line that contains LSRK frequency ranges from each one, and 
    writes it to a new cont.dat file suitable for the pipeline that contains an
    entry for each spw of that field.  
    contdatlist: a list or comma-delimited string of the .dat files, or of
          the cube names (to which '_findContinuum.dat' will be appended)
    fieldname: use this value if specified, otherwise parse from filename
    spw: integer or string integer, use this value if specified, otherwise parse
         from the filename
    Note: It assumes that none of the files originate from an ALLcont spw, so it will
    never write the "ALL" line that the pipeline triggers on.  But this feature could be
    added by calling the following function if there is only one range of channels read:
       allContinuumSelected(selection, nchan, fraction=ALL_CONTINUUM_CRITERION)
    We would only need to know nchan.
    """
    if type(contdatlist) == str:
        contdatlist = contdatlist.split(',')
    for c,contdat in enumerate(contdatlist):
        if not os.path.exists(contdat) or os.path.isdir(contdat):
            if contdat.find('.dat') < 0:
                contdatlist[c] = contdat + '_findContinuum.dat'
    for c,contdat in enumerate(contdatlist):
        if not os.path.exists(contdat):
            print("Could not find file: ", contdat)
            return
    o = open(outputfile,'w')
    previousFieldname = ''
    for contdat in contdatlist:
        cube = contdat.split('_findContinuum.dat')[0]
        if fieldname == '':
            fieldname = getFieldnameFromPipelineImageName(cube)
        if spw == '':
            spw = getSpwFromPipelineImageName(cube)
        f = open(contdat,'r')
        lines = f.readlines()
        f.close()
        ranges = lines[1].split('LSRK')
        if fieldname != previousFieldname:
            o.write('Field: %s\n\n' % (fieldname))
        o.write('SpectralWindow: %s\n' % (str(spw)))
        for myrange in ranges:
            myrange = myrange.strip()
            if len(myrange) > 2: # could be a line with only a few blanks
                if myrange.count('GHz') > 1:
                    # remove GHz from the first frequency (if it is present on both)
                    myrange = myrange.replace('GHz', '', 1)
                o.write('%s LSRK\n' % (myrange))
        o.write('\n')
        o.flush()
        previousFieldname = fieldname
        spw = ''
    o.close()

def drawYlabel(img, typeOfMeanSpectrum, meanSpectrumMethod, meanSpectrumThreshold,
               peakFilterFWHM, fontsize, mask, useThresholdWithMask, normalized=False):
    """
    This function is called by runFindContinuum.
    Draws a descriptive y-axis label based on the origin and type of mean spectrum used.
    Returns: None
    """
    if (img == ''):
        label = 'Mean spectrum passed in as ASCII file'
    else:
        if (meanSpectrumMethod.find('meanAboveThreshold') >= 0):
            if (meanSpectrumMethod.find('OverMad') > 0):
                label = '(Average spectrum > threshold=(%g))/MAD' % (roundFigures(meanSpectrumThreshold,3))
            elif (meanSpectrumMethod.find('OverRms') > 0):
                label = '(Average spectrum > threshold=(%g))/RMS' % (roundFigures(meanSpectrumThreshold,3))
            elif (useThresholdWithMask or mask==''):
                label = 'Average spectrum > threshold=(%g)' % (roundFigures(meanSpectrumThreshold,3))
            else:
                label = 'Average spectrum within mask'
        elif (meanSpectrumMethod.find('peakOverMad')>=0):
            if peakFilterFWHM > 1:
                label = 'Per-channel (Peak / MAD) of image smoothed by FWHM=%d pixels' % (peakFilterFWHM)
            else:
                label = 'Per-channel (Peak / MAD)'
        elif (meanSpectrumMethod.find('peakOverRms')>=0):
            if peakFilterFWHM > 1:
                label = 'Per-channel (Peak / RMS) of image smoothed by FWHM=%d pixels' % (peakFilterFWHM)
            else:
                label = 'Per-channel (Peak / RMS)'
        elif meanSpectrumMethod == 'mom0mom8jointMask':
            label = 'Mean profile from mom0+8 joint mask'
            if normalized:
                label += ' (normalized by MAD)'
            else:
                label += ' (not normalized by MAD)'
        else:
            label = 'Unknown method'
    pl.ylabel(typeOfMeanSpectrum+' '+label, size=fontsize)

def computeBandwidth(selection, channelWidth, loc=-1):
    """
    This function is called by runFindContinuum and findContinuum.
    selection: a string of format:  '5~6;9~20'
    channelWidth: in Hz
    Returns: bandwidth in GHz
    """
    ranges = selection.split(';')
    channels = 0
    for r in ranges:
        result = r.split('~')
        if (len(result) == 2):
            a,b = result
            channels += int(b)-int(a)+1
    aggregateBW = channels * abs(channelWidth) * 1e-9
    return(aggregateBW)

def buildMeanSpectrumFilename(img, meanSpectrumMethod, peakFilterFWHM, 
                              amendMaskIterationName='', nbin=1):
    """
    This function is called by findContinuum and runFindContinuum.
    Creates the name of the meanSpectrumFile to search for and/or create.
    Returns: a string
    """
    if (meanSpectrumMethod.find('peak')>=0):
        myname = img + '.meanSpectrum.'+meanSpectrumMethod+'_%d'%peakFilterFWHM+amendMaskIterationName
    else:
        if (meanSpectrumMethod == 'meanAboveThreshold'):
            myname = img + '.meanSpectrum'+amendMaskIterationName
        else:
            myname = img + '.meanSpectrum.'+meanSpectrumMethod+amendMaskIterationName
    if meanSpectrumMethod == 'mom0mom8jointMask' and nbin >= NBIN_THRESHOLD:
        myname += '.nbin%d' % (nbin)
    print("Built meanSpectrumFile name = ", myname)
    return(myname)

def tdmSpectrum(channelWidth, nchan, telescope='ALMA'):
    """
    This function is called by runFindContinuum and findContinuum.
    Use 15e6 instead of 15.625e6 because LSRK channel width can be slightly narrower than TOPO.
    Works for single, dual, or full-polarization.
    channelWidth: in Hz
    Returns: True or False
    """
    if telescope.find('VLA') >= 0:
        # PIPE-1433 requests chanwidth >= 1MHz and BW>=64 MHz
        if (channelWidth >= 0.97e6) and (channelWidth*nchan >= 62e6):
            casalogPost("Treating VLA spw as TDM")
            return True
        else:
            return False
    else:
        if ((channelWidth >= 15e6/2. and nchan>240) or # 1 pol TDM, must avoid 240-chan FDM
             (channelWidth >= 15e6)):
    #        (channelWidth >= 15e6 and nchan<96) or     # 2 pol TDM (128 chan)
    #        (channelWidth >= 30e6 and nchan<96)):      # 4 pol TDM (64 chan)
            return True
        else:
            return False

def atmosphereVariation(img, imageInfo, chanInfo, airmass=1.5, pwv=-1, removeSlope=True, vis='', source='', spw=''):
    """
    This function is called by findContinuum.
    Computes the absolute and percentage variation in atmospheric transmission 
    and sky temperature across an image cube.
    Returns 5 values: max(Trans)-min(Trans), and as percentage of mean,
                      Max(Tsky)-min(Tsky), and as percentage of mean, 
                      standard devation of Tsky
    """
    freqs, values = CalcAtmTransmissionForImage(img, imageInfo, chanInfo, airmass=airmass, pwv=pwv, value='transmission', vis=vis, source=source, spw=spw)
    if removeSlope:
        slope, intercept = linfit(freqs, values, values*0.001)
        casalogPost("Computed atmospheric variation and determined slope: %f per GHz (%.0f,%.2f)" % (slope,freqs[0],values[0]))
        values = values - (freqs*slope + intercept) + np.mean(values)
    maxMinusMin = np.max(values)-np.min(values)
    percentage = maxMinusMin/np.mean(values)
    freqs, values = CalcAtmTransmissionForImage(img, imageInfo, chanInfo, airmass=airmass, pwv=pwv, value='tsky', vis=vis, source=source, spw=spw)
    if removeSlope:
        slope, intercept = linfit(freqs, values, values*0.001)
        values = values - (freqs*slope + intercept) + np.mean(values)
    TmaxMinusMin = np.max(values)-np.min(values)
    Tpercentage = TmaxMinusMin*100/np.mean(values)
    stdValues = np.std(values)
    return(maxMinusMin, percentage, TmaxMinusMin, Tpercentage, stdValues)

def versionStringToArray(versionString):
    """
    This function is called by casaVersionCompare.
    Converts '5.3.0-22' to np.array([5,3,0,22], dtype=np.int32)
    """
    tokens = versionString.split('-')
    t = tokens[0].split('.')
    version = [np.int32(i) for i in t]
    if len(tokens) > 1:
        version += [np.int32(tokens[1])]
    return np.array(version)
    
def casaVersionCompare(comparitor, versionString):
    """
    This function is called by meanSpectrum.
    Uses cu.compare_version in CASA5, uses string comparison in CASA4 and 6.
    (but is not used by meanSpectrumMethod='mom0mom8jointMask' which is 
     the Pipeline default)
    """
    if casaMajorVersion < 5:
        if comparitor == '>=':
            comparison = casadef.casa_version >= versionString
        elif comparitor == '>':
            comparison = casadef.casa_version > versionString
        elif comparitor == '<':
            comparison = casadef.casa_version < versionString
        elif comparitor == '<=':
            comparison = casadef.casa_version <= versionString
        else:
            print("Unknown comparitor: ", comparitor)
            return False
    elif casaMajorVersion == 5:
        version = versionStringToArray(versionString)
        comparison = cu.compare_version(comparitor, version)
    else: # casa 6
        if comparitor == '>=':
            comparison = casaVersion >= versionString
        elif comparitor == '>':
            comparison = casaVersion > versionString
        elif comparitor == '<':
            comparison = casaVersion < versionString
        elif comparitor == '<=':
            comparison = casaVersion <= versionString
        else:
            print("Unknown comparitor: ", comparitor)
            return False
    return comparison

def getFreqType(img):
    """
    This function is called by runFindContinuum and cubeFrameToTopo.
    """
    myia = iatool()
    myia.open(img)
    mycs = myia.coordsys()
    mytype = mycs.referencecode('spectral')[0]
    mycs.done()
    myia.close()
    return mytype

def getEquinox(img, myia=None):
    """
    This function is called by cubeFrameToTopo.
    """
    if myia is None:
        myia = iatool()
        myia.open(img)
        needToClose = True
    else:
        needToClose = False
    mycs = myia.coordsys()
    equinox = mycs.referencecode('direction')[0]
    mycs.done()
    if needToClose: myia.close()
    return equinox

def getTelescope(img, myia=None):
    """
    This function is called by CalcAtmTransmissionForImage and cubeFrameToTopo.
    """
    if myia is None:
        myia = iatool()
        myia.open(img)
        needToClose = True
    else:
        needToClose = False
    mycs = myia.coordsys()
    telescope = mycs.telescope()
#    telescope = myia.miscinfo()['TELESCOP']  # not in images produced in 4.2.2
    mycs.done()
    if needToClose: myia.close()
    return telescope

def getDateObs(img, myia=None):
    """
    This function is called by cubeFrameToTopo.
    Returns string of format: '2014/05/22/08:47:05' suitable for lsrkToTopo
    """
    if myia is None:
        myia = iatool()
        myia.open(img)
        needToClose = True
    else:
        needToClose = False
    mycs = myia.coordsys()
    mjd = mycs.epoch()['m0']['value']
    mycs.done()
    if needToClose: myia.close()
    mydate = mjdToUT(mjd).rstrip(' UT').replace('/','-').replace(' ','/')
    return mydate

def removeInitialQuadraticIfNeeded(avgSpectrum, initialQuadraticImprovementThreshold=1.6):
    """
    This function is called by runFindContinuum when meanSpectrumMethod='mom0mom8jointMask'.
    Fits a quadratic to the specified spectrum, and removes it if the
    MAD will improve by more than a factor of threshold
    """
    index = list(range(len(avgSpectrum)))
    priorMad = MAD(avgSpectrum)
    casalogPost("preMad: %f" % (priorMad), debug=True)
    fitResult = polyfit(index, avgSpectrum, priorMad)
    order2, slope, intercept, xoffset = fitResult
    myx = np.arange(len(avgSpectrum)) - xoffset
    trialSpectrum = avgSpectrum + nanmean(avgSpectrum)-(myx**2*order2 + myx*slope + intercept)
    postMad =  MAD(trialSpectrum)
#    print("postMad=%e, trialSpectrum has %d non-zero values: " % (postMad,len(np.where(trialSpectrum != 0.0))), trialSpectrum)
    if postMad == 0.0:
        initialQuadraticRemoved = False
        improvementRatio = 1.0
        return avgSpectrum, initialQuadraticRemoved, improvementRatio
    casalogPost("preMad: %f, postMad: %f, factorReduction: %f" % (priorMad,postMad,priorMad/postMad), debug=True)
    improvementRatio = priorMad/postMad
    if improvementRatio > initialQuadraticImprovementThreshold:
        initialQuadraticRemoved = True
        avgSpectrum = trialSpectrum
        casalogPost("Initial quadratic removed because improvement ratio: %f > %f" % (improvementRatio,initialQuadraticImprovementThreshold), debug=True)
    else:
        casalogPost("Initial quadratic not removed because improvement ratio: %f <= %f" % (improvementRatio,initialQuadraticImprovementThreshold), debug=True)
        initialQuadraticRemoved = False
    return avgSpectrum, initialQuadraticRemoved, improvementRatio

def checkForMismatch(meanSpectrumFile, img, mask, useThresholdWithMask, 
                     fitsTable, iteration, centralArcsec):
    """
    This function is called by runFindContinuum.
    Returns True if there is a mismatch between the pre-existing 
    meanSpectrumFile and the requested parameters.
    """
    overwrite = False
    if (os.path.exists(meanSpectrumFile) and img != ''):
        if (maskArgumentMismatch(mask, meanSpectrumFile, useThresholdWithMask) and not fitsTable):
            casalogPost("Regenerating the meanSpectrum since there is a mismatch in the mask or useThresholdWithMask parameters. (fitsTable=%s)" % (fitsTable))
            overwrite = True
        else:
            casalogPost("No mismatch in the mask argument vs. the meanSpectrum file.")
        if (centralArcsecArgumentMismatch(centralArcsec, meanSpectrumFile, iteration) and not fitsTable):
            casalogPost("Regenerating the meanSpectrum since there is a mismatch in the centralArcsec argument (%s)." % (str(centralArcsec)))
            overwrite = True
        else:
            casalogPost("No mismatch in the centralArcsec argument vs. the meanSpectrum file. Setting overwrite=%s" % (overwrite))
    elif (img != ''):
        casalogPost("Did not find mean spectrum file = %s" % (meanSpectrumFile))
    return overwrite

def pick_sFC_TDM(meanSpectrumMethod, singleContinuum):
    """
    This function is called by runFindContinuum.
    Chooses the value of sigmaFindContinuum for TDM datasets based on
    the meanSpectrumMethod and whether the user requested single continuum in
    the Observing Tool.
    """
    if singleContinuum:
        sFC_TDM = 9.0 
    elif (meanSpectrumMethod.find('meanAboveThreshold') >= 0):
        sFC_TDM = 4.5
    elif (meanSpectrumMethod.find('peakOverMAD') >= 0):
        sFC_TDM = 6.5
    elif (meanSpectrumMethod == 'mom0mom8jointMask'):
        sFC_TDM = 7.2 # 2018-03-23 was 4.5, 2018-03-26 was 5.5, 2018-03-27 was 6.5, 2018-03-31 was 7.0
    else:
        sFC_TDM = 4.5
        casalogPost("Unknown meanSpectrumMethod: %s" % (meanSpectrumMethod))
    return sFC_TDM

def setYLimitsAvoidingEdgeChannels(avgspectrumAboveThreshold, mad, chan=1):
    """
    Called by runFindContinuum to set the y-axis limits..
    1) Avoid spikes in edge channels from skewing the plot limits, by 
       setting plot limits on the basis of channels chan..-chan
    2) Enforce a maximum dynamic range (peak-median)/mad so that weak features
       can be seen, allowing strongest ones to overflow the top.
    """
    y0,y1 = pl.ylim()
    y0naturalBuffer = np.min(avgspectrumAboveThreshold) - y0
    y1naturalBuffer = y1  - np.max(avgspectrumAboveThreshold)
    mymin = np.min(avgspectrumAboveThreshold[chan:-chan]) - y0naturalBuffer
    mymax = np.max(avgspectrumAboveThreshold[chan:-chan]) + y1naturalBuffer
    mymedian = np.median(avgspectrumAboveThreshold)
    highDynamicRange = False
    dynamicRange = (mymax-mymedian)/mad
    if dynamicRange > DYNAMIC_RANGE_LIMIT_PLOT:
        mymax = DYNAMIC_RANGE_LIMIT_PLOT*mad + mymedian
        casalogPost('Spectral dynamic range exceeds %d, limiting maximum y-axis value to %f' % (DYNAMIC_RANGE_LIMIT_PLOT,mymax))
        # The most common case is strong maser emission that can cause the baseline
        # to appear in the upper ~quarter of the plot leaving a lot of white space
        # below it.  So raise the minimum to 10% of the range from true min to mymax.
        mymin = np.min(avgspectrumAboveThreshold[chan:-chan])
        mymin = mymin - 0.1*(mymax-mymin)
        #np.mean([-mymax,np.median(avgspectrumAboveThreshold),np.min(avgspectrumAboveThreshold)])
        casalogPost('Also limiting the minimum y-axis value to %f' % (mymin))
        highDynamicRange = True
    pl.ylim([mymin, mymax]) 
    return highDynamicRange

def ExpandYLimitsForLegend():
    """
    Called by runFindContinuum.
    Make room for legend text at bottom and top of existing plot.
    """
    ylim = pl.ylim()
    yrange = ylim[1]-ylim[0]
    ylim = [ylim[0]-yrange*0.05, ylim[1]+yrange*0.2]
    pl.ylim(ylim)
    return pl.ylim()

def pickRandomErrors(nvalues=1):
    """
    Picks a series of random values from a Gaussian distribution with mean 0 and standard 
    deviation = 1 and returns it as an array.
    -Todd Hunter
    """
    p = []
    for i in range(nvalues):
        p.append(pickRandomError())
    return(np.array(p))

def pickRandomError(seed=None):
    """
    Picks a random value from a Gaussian distribution with mean 0 and standard deviation = 1.
    seed: if specified, then first reseed with random.seed(seed)
    -Todd Hunter
    """
    w = 1.0
    if seed is not None:
        random.seed(seed)
    while ( w >= 1.0 ):
      x1 = 2.0 * random.random() - 1.0
      x2 = 2.0 * random.random() - 1.0
      w = x1 * x1 + x2 * x2

    w = np.sqrt( (-2.0 * np.log( w ) ) / w )
    y1 = x1 * w
    y2 = x2 * w
    return(y1)

def createNoisyQuadratic(n=1000, nsigma=3):
    """
    Function meant simply to test removeStatContQuadratic.
    n: number of points in synthetic noise spectrum
    nsigma: passed to removeStatContQuadratic
    """
    x = np.arange(n)
    y = pickRandomErrors(n)
    x0 = (x - n/2)
    y += 0.001*x0**2 
    pl.clf()
    pl.plot(x,y,'k-')
    y1, idx, yfit = removeStatContQuadratic(y, nsigma, returnExtra=True)
    pl.plot(x,yfit,'r-')
    pl.plot(x,y1,'b-')
    pl.draw()
    
def removeStatContQuadratic(y, nsigma=2.0, returnExtra=False, minPercentage=50, makeMovie=False, img='', keepContinuum=True):
    """
    Iteratively removes outlier points down to nsigma, then fits a quadratic to the remaining 
    points, removes this quadratic from the full initial spectrum, and returns it, along
    with the list of channels used for the fit and the fit evaluated at all channels.
    minPercentage: stop removing outliers if number of remaining goes below this threshold
    makeMovie: if True, then make a png for each quadratic fit as channels are removed one-by-one
    img: only used to name the movie frame pngs
    """
    y = np.array(y)
    x = np.arange(len(y))
    idx = np.array(x)  # start with all points
    yfit = np.median(y)
    if makeMovie:
        if img != '': 
            img = img + '_'
            os.system('rm -f ' + img + '*')
        ylim = None
    diff = y[idx]-yfit
    diff = diff - np.median(diff)
    while np.max(np.abs(diff)) > nsigma*np.std(diff):
        peak = np.argmax(np.abs(diff))
#        print("Removing channel %d" % (peak))
        idx = np.delete(idx, [peak])
        if len(idx) < 3 or len(idx) < minPercentage*len(y)*0.01:
            break
        # fit only the remaining points             
        fitResult = polyfit(idx, y[idx], MAD(y[idx]))
        order2, slope, intercept, xoffset = fitResult
        yfit = (x[idx]-xoffset)**2*order2 + (x[idx]-xoffset)*slope + intercept   
        diff0 = y[idx] - yfit
        diff = diff0 - np.median(diff0)
        diffWithoutFit = y[idx] - np.median(y[idx])
        if np.max(yfit) - np.min(yfit) > 0:
            ratio = (np.max(y[idx]) - np.min(y[idx])) / (np.max(yfit) - np.min(yfit))
        else:
            ratio = 10
        peakOverMAD = (np.max(y[idx]) - np.min(y[idx])) / MAD(y[idx])
        ratio = peakOverMAD
        if ratio > 5: # 3*(np.max(yfit) - np.min(yfit)) < np.max(y[idx]) - np.min(y[idx]):
            # strong lines are still present, so revert to using the median
            fitUsed = False
            diff = diffWithoutFit
        else:
            fitUsed = True
        if makeMovie:
            pl.clf()
            pl.subplot(211)
            pl.plot(x, y, 'k-', x[idx], yfit, 'r-', x[idx], y[idx], 'b-')
            pl.plot([x[0],x[-1]], [np.median(y), np.median(y)], 'k--')
            pl.title('%d points (ratio=%.2f)' % (len(idx), ratio))
            if ylim is None:
                ylim = pl.ylim()
            else:
                pl.ylim(ylim)
            pl.subplot(212)
            pl.plot(x[idx], diff, 'k-')
            pl.plot([x[0],x[-1]], [np.median(diff), np.median(diff)], 'k--')
            if fitUsed:
                pl.title('quadratic and median removed')
            else:
                pl.title('median removed')
            png = img + 'fit%04d.png' % (len(y)-len(idx))
            pl.savefig(png)
    if len(idx) < 3:
        casalogPost('Too few points (%d) to fit a quadratic' % (len(idx)))
        idx = x
        yfit = np.zeroes(len(x))
    else:
        # remove the fit from all points
        yfit = (x-xoffset)**2*order2 + (x-xoffset)*slope + intercept   
        y = y - yfit
        if keepContinuum:
            y += np.mean(yfit)
    if returnExtra:
        return y, idx, yfit
    else:
        return y

def runFindContinuum(img='', pbcube=None, psfcube=None, minbeamfrac=0.3, 
                     spw='', transition='', 
                     baselineModeA='min', baselineModeB='min',
                     sigmaCube=3, nBaselineChannels=0.19, 
                     sigmaFindContinuum='auto', sigmaFindContinuumMode='auto',
                     verbose=False, png='', pngBasename=False, 
                     nanBufferChannels=2, 
                     source='', useAbsoluteValue=True, trimChannels='auto', 
                     percentile=20, continuumThreshold=None, narrow='auto', 
                     separator=';', overwrite=False, titleText='', 
                     maxTrim=maxTrimDefault, maxTrimFraction=1.0,
                     meanSpectrumFile='', centralArcsec=-1, channelWidth=0,
                     alternateDirectory='.', imageInfo=[], chanInfo=[], 
                     plotAtmosphere='transmission', airmass=1.5, pwv=1.0,
                     channelFractionForSlopeRemoval=0.75, mask='', 
                     invert=False, meanSpectrumMethod='peakOverMad', 
                     peakFilterFWHM=15, fullLegend=False, iteration=0,
                     meanSpectrumMethodMessage='', minSlopeToRemove=1e-8,
                     minGroupsForSFCAdjustment=10, 
                     regressionTest=False, quadraticFit=False, megapixels=0,
                     triangularPatternSeen=False, maxMemory=-1, 
                     negativeThresholdFactor=1.15, byteLimit=-1, 
                     singleContinuum=False, applyMaskToMask=False, 
                     plotBaselinePoints=False, dropBaselineChannels=2.0,
                     madRatioUpperLimit=1.5, madRatioLowerLimit=1.15, 
                     projectCode='', useIAGetProfile=True, 
                     useThresholdWithMask=False, dpi=dpiDefault, 
                     normalizeByMAD=False, overwriteMoments=False,
                     initialQuadraticImprovementThreshold=1.6,
                     minPeakOverMadForSFCAdjustment=25, 
                     maxMadRatioForSFCAdjustment=1.20,
                     maxMadRatioForSFCAdjustmentInLaterIterations=1.20,
                     minPixelsInJointMask=3, userJointMask='',
                     signalRatioTier1=0.965, signalRatioTier2=0.94, 
                     snrThreshold=23, mom0minsnr=MOM0MINSNR_DEFAULT, 
                     mom8minsnr=MOM8MINSNR_DEFAULT, 
                     overwriteMasks=True, rmStatContQuadratic=True, 
                     quadraticNsigma=2, bidirectionalMaskPhase2=False, 
                     plotQuadraticPoints=True, makeMovie=True, outdir='', 
                     allowBluePruning=True, avoidance='',
                     enableRejectNarrowInnerWindows=True, 
                     avoidExtremaInNoiseCalcForJointMask=False, 
                     amendMask=False, momentdir='', skipchan=1, 
                     amendMaskIterationName='', fontsize=10, vis='', 
                     useJointMaskPrior=False, nbin=1, window='flat',
                     subimage=False, momDiffSNR=-1, peakOverMadCriterion=85,
                     maxGroups=15, # was 25 in PL2021 # was 40 earlier
                     momDiffApproachLevel=15.0, mom08simultaneous=True,
                     useUncorrectedMedian=False, 
                     returnBluePoints=False,
                     removeOutlierBaselineChannels=False,
                     enableInitialQuadratic=False, useBaseline='auto',
                     spectralDynamicRangeString='', idxNonZeroMAD=None):
    """
    This function is called by findContinuum.  It calls functions that:
    1) compute the mean spectrum of a dirty cube
    2) find the continuum channels
    3) plot the results
    Inputs:
    See description of parameters in findContinuum.
    Additional inputs:
    channelWidth: in Hz
    iteration: order number of this call, established in calling function (findContinuum)
    Returns: 23 items
    *1 A channel selection string suitable for the spw parameter of clean.
    *2 The name of the png produced.
    *3 The slope of the linear fit (or None if no fit attempted).
    *4 The channel width in Hz (only necessary for the fitsTable option).
    *5 nchan in the cube (only necessary to return this for the fitsTable option, will be computed otherwise).
    *6 Boolean: True if it used low values as the baseline (as opposed to high values)
    *7 SNRs in the moment0 image (raw, outside mask, outside phase2 mask)
    *8 SNRs in the moment8 image (raw, outside mask, outside phase2 mask)
    *9 Boolean: True if the middle-valued channels were used as the baseline (as opposed to low or high)
    *10 list: selectionPreBluePruning
    *11 float: finalSigmaFindContinuum
    *12 string: name of jointMask
    *13 float array: avgspectrumAboveThreshold
    *14 float: the true median of the spectrum
    *15 list: of plot artist descriptors (things to potentially remove and replot)
    *16 axis instance: ax1 (lower x-axis)
    *17 axis instance: ax2 (upper x-axis)
    *18 float: threshold (the positive threshold for line detection)
    *19 string: final line of upper text legend
    *20 int: number of narrow central groups that were dropped (or would have been dropped
           if enableRejectNarrowWindows had been True, but wasn't), summed over all runs
           of findContinuumChannels() 
    *21 float: effectiveSigma (product of finalSigmaFindContinuum*correctionFactor)
    *22 float: baselineMAD
    *23 string: upper x-axis label
    *24 list of int: allBaselineChannelsXY (channel numbers of baseline)
    *25 int: nbin
    *26 float: initialPeakOverMad
    *27 float or None: negativeThreshold
    Inputs:
    img: the image cube to operate upon
    spw: the spw name or number to put in the x-axis label
    transition: the name of the spectral transition (for the plot title)
    baselineModeA: 'min' or 'edge', method to define the baseline in meanSpectrum()
    sigmaCube: multiply this value by the MAD to get the threshold above which a pixel
               is included in the mean spectrum
    nBaselineChannels: if integer, then the number of channels to use in:
      a) computing the mean spectrum with the 'meanAboveThreshold' methods.
      b) computing the MAD of the lowest/highest channels in findContinuumChannels
          if float, then the fraction of channels to use (i.e. the percentile)
          default = 0.19, which is 24 channels (i.e. 12 on each side) of a TDM window
    sigmaFindContinuum: passed to findContinuumChannels, 'auto' starts with 3.5
    sigmaFindContinuumMode: 'auto', 'autolower', or 'fixed'
    verbose: if True, then print additional information during processing
    png: the name of the png to produce ('' yields default name)
    pngBasename: if True, then remove the directory from img name before generating png name
    nanBufferChannels: when removing or replacing NaNs, do this many extra channels
                       beyond their extent
    source: the name of the source, to be shown in the title of the spectrum.
            if None, then use the filename, up to the first underscore.
    findContinuum: if True, then find the continuum channels before plotting
    overwrite: if True, or ASCII file does not exist, then recalculate the mean spectrum
                      writing it to <img>.meanSpectrum
               if False, then read the mean spectrum from the existing ASCII file
    trimChannels: after doing best job of finding continuum, remove this many 
         channels from each edge of each block of channels found (for margin of safety)
         If it is a float between 0..1, then trim this fraction of channels in each 
         group (rounding up). If it is 'auto', use 0.1 but not more than maxTrim channels
         and not more than maxTrimFraction
    percentile: control parameter used with baselineMode='min'
    continuumThreshold: if specified, only use pixels above this intensity level
    narrow: the minimum number of channels that a group of channels must have to survive
            if 0<narrow<1, then it is interpreted as the fraction of all
                           channels within identified blocks
            if 'auto', then use int(ceil(log10(nchan)))
    titleText: default is img name and transition and the control parameter values
    meanSpectrumFile: an alternative ASCII text file to use for the mean spectrum.
                      Must also set img=''.
    meanSpectrumMethod: 'peakOverMad', 'peakOverRms', 'meanAboveThreshold', 
       'meanAboveThresholdOverRms', 'meanAboveThresholdOverMad', 
        where 'peak' refers to the peak in a spatially-smoothed version of cube
    peakFilterFWHM: value used by 'peakOverRms' and 'peakOverMad' to presmooth 
        each plane with a Gaussian kernel of this width (in pixels)
        Set to 1 to not do any smoothing.
    centralArcsec: radius of central box within which to compute the mean spectrum
        default='auto' means start with whole field, then reduce to 1/10 if only
        one window is found (unless mask is specified); or 'fwhm' for the central square
        with the same radius as the PB FWHM; or a floating point value in arcseconds
    mask: a mask image equal in shape to the parent image that is used to determine the
        region to compute the noise (outside the mask) and the mean spectrum (inside the mask)
        option 'auto' will look for the <filename>.mask that matches the <filename>.image
        and if it finds it, it will use it; otherwise it will use no mask
    plotAtmosphere: '', 'tsky', or 'transmission'
    airmass: for plot of atmospheric transmission
    pwv: in mm (for plot of atmospheric transmission)
    channelFractionForSlopeRemoval: if at least this many channels are initially selected, or 
       if there are only 1-2 ranges found and the widest range has > nchan*0.3 channels, then 
       fit and remove a linear slope and re-identify continuum channels.  Set to 1 to turn off.
    quadraticFit: if True, fit quadratic polynomial to the noise regions when appropriate; 
         otherwise fit only a linear slope
    megapixels: simply used to label the plot
    triangularPatternSeen: if True, then slightly boost sigmaFindContinuum if it is in 'auto' mode
    maxMemory: only used to name the png if it is set
    applyMaskToMask: if True, apply the mask inside the user mask image to set its masked pixels to 0
    plotBaselinePoints: if True, then plot the baseline-defining points as black dots
    dropBaselineChannels: percentage of extreme values to drop in baseline mode 'min'
    useIAGetProfile: if True, then for meanAboveThreshold and baselineMode='min', then
        use ia.getprofile instead of ia.getregion and subsequent arithmetic (faster)
    initialQuadraticImprovementThreshold: if removal of a quadratic from the raw meanSpectrum reduces
        the MAD by this factor or more, then proceed with removing this quadratic (new Cycle 6 heuristic)
    maxMadRatioForSFCAdjustment: madRatio must be below this value (among other things) in order
           for automatic lowering of sigmaFindContinuum value to occur in .original iteration
    maxMadRatioForSFCAdjustmentInLaterIterations: madRatio must be below this value (among other things) in order
           for automatic lowering of sigmaFindContinuum value to occur in later iterations
    avoidance: a CASA channel selection string to avoid selecting for continuum (for 
         manual use); invoked just before making the plot and returning

    Parameters only relevant to mom0mom8jointMask:
    mom0minsnr: sets the threshold for the mom0 image (i.e. median + mom0minsnr*scaledMAD)
    mom8minsnr: sets the threshold for the mom8 image (i.e. median + mom8minsnr*scaledMAD)
    snrThreshold: if SNR is higher than this, and phase2==True, then run a phase 2 mask calculation
    minPixelsInJointMask: if fewer than these pixels are found, then use all pixels above pb=0.3
          (when meanSpectrumMethod = 'mom0mom8jointMask')
    overwriteMoments: if True, then overwrite any existing moment0 or moment8 image 
    overwriteMasks: if True, then overwrite any existing moment0 or moment8 mask image 
    enableInitialQuadratic: if True, assess the need for quadratic removal, and use the 
        old-style method to remove it
    rmStatContQuadratic: if True, then do not do the initialQuadratic removal (if 
            enabled and deemed necessary), but instead always perform the new-style 
            quadratic removal
    quadraticNsigma: the stopping threshold to use when ignoring outliers prior to fitting
            fitting the quadratic (in the rmStatContQuadratic option). To more quickly test 
            different values, be sure to set the meanSpectrumFile option and overwrite=False)
    bidirectionalMaskPhase2: True=extend mask to negative values beyond threshold; False=Cycle6 behavior
    makeMovie: if True, then make a png for each quadratic fit as channels are 
             removed one-by-one (in the rmStatContQuadratic option)
    outdir: where to write the .mom0 and .mom8 images, .dat, meanSpectrum, and the .png file
    maxGroups: maximum number of continuum regions below which the potential reduction 
          factor of sigmaFindContinuum will be scaled by (1 - groups/maxGroups)**(bandwidth/1875e6)
          via the following equation:
      sFC_factor = np.max([scalingFactor/7.0, (1-groups/float(maxGroups))**(spwBandwidth/1.875e9)])

    Parameters passed to findContinuumChannels (and only used in there):
    useUncorrectedMedian: if True, then do not adjust the median (solid black line) upward
       from the median of the baseline points (dashed blue line)
    returnBluePoints: if True, then simply set the continuum ranges to the baseline points 
         (i.e. the blue points) instead of what the other heuristics might have found
    baselineModeB: 'min' or 'edge', method to define the baseline in findContinuumChannels()
    separator: the character to use to separate groups of channels in the string returned
    maxTrim: in trimChannels='auto', this is the max channels to trim per group for TDM spws; it is automatically scaled upward for FDM spws.
    maxTrimFraction: in trimChannels='auto', the max fraction of channels to trim per group
    negativeThresholdFactor: scale the nominal negative threshold by this factor (to adjust 
        sensitivity to absorption features: smaller values=more sensitive)
    signalRatioTier1: threshold for signalRatio, above which we desensitize the level to
        consider line emission in order to avoid small differences in noise levels from 
        affecting the result (e.g. that occur between serial and parallel tclean cubes)
        signalRatio=1 means: no lines seen, while closer to zero: more lines seen
    signalRatioTier2: second threshold for signalRatio, used for FDM spws (nchan>192) and
        for cases of peakOverMad < 5.  Should be < signalRatioTier1.
    dropBaselineChannels: percentage of extreme values to drop in baseline mode 'min'
    madRatioUpperLimit, madRatioLowerLimit: determines when dropBaselineChannels is used
    mom08simultaneous: passed to meanSpectrumFromMom0Mom8JointMask
    useBaseline: 'low', 'middle', 'high' or 'auto'; auto tries to automatically pick 'low' for
       spectra that are emission-line dominated, 'high' if absorption line dominated, or 'both' if
       both types of lines are present
    """
    if amendMaskIterationName not in ['', '.original']:
        maxMadRatioForSFCAdjustment = maxMadRatioForSFCAdjustmentInLaterIterations  # added May 4, 2022
    normalized = False  # This will be set True only by return value from meanSpectrumFromMom0Mom8JointMask()
    startTime = timeUtilities.time()
    slope=None 
    replaceNans = True # This used to be a command-line option, but no longer.
    img = img.rstrip('/')
    fitsTable = False
    typeOfMeanSpectrum = 'Existing' # until proven otherwise later
    narrowValueModified = None
    RangesDropped = 0
    mom0snrs = None
    mom8snrs = None
    if img == '':
        maxBaseline = 0  # added March 22, 2019
        if minbeamfrac == 'auto':
            minbeamfrac = 0.3
            casalogPost("minbeamfrac = 'auto': using %.2f since no image was passed" % (minbeamfrac))
    else:
        maxBaseline = imageInfo[11]
        if minbeamfrac == 'auto':
            if maxBaseline <= ACA_MAX_BASELINE:
                minbeamfrac = 0.1
                casalogPost("minbeamfrac = 'auto': using %.2f since maxBaseline <= %dm" % (minbeamfrac,ACA_MAX_BASELINE))
            else:
                minbeamfrac = 0.3
                casalogPost("minbeamfrac = 'auto': using %.2f since maxBaseline > %dm" % (minbeamfrac,ACA_MAX_BASELINE))
    if (meanSpectrumFile != '' and os.path.exists(meanSpectrumFile)):
        casalogPost("Using existing meanSpectrumFile = %s" % (meanSpectrumFile))
        if (is_binary(meanSpectrumFile)):
            fitsTable = True
    if ((type(nBaselineChannels) == float or type(nBaselineChannels) == np.float64) and not fitsTable):
        # chanInfo will be == [] if an ASCII meanSpectrumFile is specified
        if len(chanInfo) >= 4:
            nchan, firstFreq, lastFreq, channelWidth = chanInfo
            channelWidth = abs(channelWidth)
            nBaselineChannels = int(round_half_up(nBaselineChannels*nchan))
            casalogPost("Found %d channels in the cube" % (nchan))
        
    if (nBaselineChannels < 2 and not fitsTable and len(chanInfo) >= 4):
        casalogPost("You must have at least 2 edge channels (nBaselineChannels = %d)" % (nBaselineChannels))
        return
    if (meanSpectrumFile == ''):
        meanSpectrumFile = buildMeanSpectrumFilename(img, meanSpectrumMethod, peakFilterFWHM, amendMaskIterationName, nbin)
    elif (not os.path.exists(meanSpectrumFile)):
        if (len(os.path.dirname(img)) > 0):
            meanSpectrumFile = os.path.join(os.path.dirname(img),os.path.basename(meanSpectrumFile))
    if not overwrite and amendMaskIterationName not in ['.extraMask','.onlyExtraMask']: 
        # do not allow overwrite to be changed from False to True if we are in the extraMaskStage
        overwrite = checkForMismatch(meanSpectrumFile, img, mask, 
                                     useThresholdWithMask, fitsTable, 
                                     iteration, centralArcsec)
    initialQuadraticRemoved = False
    initialPeakOverMad = -1
    pbBasedMask = False
    mom0peakOverMad = 0 # in case it does not get set below
    mom8peakOverMad = 0 # in case it does not get set below
    numberPixelsInMom8Mask = 0  # added March 22, 2019  
    jointMask = None # will only become populated if meanSpectrumMethod == 'mom0mom8jointMask'
    if (((overwrite or not os.path.exists(meanSpectrumFile)) and img != '') and not fitsTable):
        typeOfMeanSpectrum = 'Computed'
        if meanSpectrumMethod == 'mom0mom8jointMask':
            # There should be no Nans that were replaced, but keep this name 
            # for the spectrum for consistency with the older methods.
            saveTime = False # somehow, the returned values do not match exactly, so leave this False for now.
            if (overwrite):
                if useJointMaskPrior and iteration == 0 and saveTime:
                    casalogPost("Not regenerating the mean spectrum file because pickle file exists.")
                else:
                    casalogPost("Regenerating the mean spectrum file with method='%s' (momentdir='%s', numberPixelsInMom8Mask=%s)." % (meanSpectrumMethod,momentdir,numberPixelsInMom8Mask))
            else:
                casalogPost("Generating the mean spectrum file with method='%s'." % (meanSpectrumMethod))
            if outdir == '':
                priorValuesFile = '%s_mom0mom8jointMaskPrior.pkl'%(img)
            else:
                priorValuesFile = '%s/%s_mom0mom8jointMaskPrior.pkl'%(outdir,os.path.basename(img))
            if iteration == 0 and useJointMaskPrior and saveTime:
                # Read values from <img>_mom0mom8jointMaskPrior.pkl
                if not os.path.exists(priorValuesFile):
                    print("*****  %s does not exist, setting useJointMaskPrior to False ****" % (priorValuesFile))
                    useJointMaskPrior = False
                else:
                    print("***** priorValuesFile: %s ****" % (priorValuesFile))
            if iteration == 0 and useJointMaskPrior and saveTime:
                # This pickle file contains all the results returned by prior run of meanSpectrumFromMom0Mom8JointMask().
                open_file = open(priorValuesFile, "rb")
                result = pickle.load(open_file)
                print("loaded %d values from pickle file: %s" % (len(result), priorValuesFile))
                if len(result) == 13:
                    [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask] = result
                else:
                    [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad] = result
                open_file.close()
            else:
                results = meanSpectrumFromMom0Mom8JointMask(img, imageInfo, nchan, pbcube, psfcube, minbeamfrac, projectCode, overwriteMoments, overwriteMasks, normalizeByMAD=normalizeByMAD, minPixelsInJointMask=minPixelsInJointMask, userJointMask=userJointMask, snrThreshold=snrThreshold, mom0minsnr=mom0minsnr, mom8minsnr=mom8minsnr, rmStatContQuadratic=rmStatContQuadratic, bidirectionalMaskPhase2=bidirectionalMaskPhase2, outdir=outdir, avoidExtremaInNoiseCalcForJointMask=avoidExtremaInNoiseCalcForJointMask, momentdir=momentdir, nbin=nbin, window=window, maxBaseline=maxBaseline, subimage=subimage, mom08simultaneous=mom08simultaneous,enableInitialQuadratic=enableInitialQuadratic)
                if results is None:
                    return
                avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad, mom0peakOverMad, mom8peakOverMad = results
                # Write values to <img>_mom0mom8jointMaskPrior.pkl
                open_file = open(priorValuesFile, 'wb')
                pickle.dump(results, open_file)
                print("wrote %d values to pickle file: %s" % (len(results), priorValuesFile))
                open_file.close()
            print("returned from meanSpectrumFromMom0Mom8JointMask: numberPixelsInMom8Mask = ", numberPixelsInMom8Mask)
            nanmin = None
            edgesUsed = None
            meanSpectrumThreshold = None
        else:
            if (overwrite):
                casalogPost("Regenerating the mean spectrum file with centralArcsec=%s, mask='%s'." % (str(centralArcsec),mask))
            else:
                casalogPost("Generating the mean spectrum file with centralArcsec=%s, mask='%s'." % (str(centralArcsec),mask))
            result = meanSpectrum(img, nBaselineChannels, sigmaCube, verbose,
                                  nanBufferChannels,useAbsoluteValue,
                                  baselineModeA, percentile,
                                  continuumThreshold, meanSpectrumFile, 
                                  centralArcsec, imageInfo, chanInfo, mask,
                                  meanSpectrumMethod, peakFilterFWHM, iteration, 
                                  applyMaskToMask, useIAGetProfile, useThresholdWithMask, overwrite, nbin)
            if result is None:
                return
            avgspectrum, avgSpectrumNansRemoved, avgSpectrumNansReplaced, meanSpectrumThreshold,\
              edgesUsed, nchan, nanmin, percentagePixelsNotMasked = result
            if verbose:
                print("len(avgspectrum) = %d, len(avgSpectrumNansReplaced)=%d" % (len(avgspectrum),len(avgSpectrumNansReplaced)))
    else:
        # Here is where nchan is defined for case of FITS table or previous spectrum
        if (fitsTable):
            result = readMeanSpectrumFITSFile(meanSpectrumFile)
            if (result is None):
                casalogPost("FITS table is not valid.")
                return
            avgspectrum, avgSpectrumNansReplaced, meanSpectrumThreshold, edgesUsed, nchan, nanmin, firstFreq, lastFreq = result
            percentagePixelsNotMasked = -1
        else:
            # An ASCII file was specified as the spectrum to process
            casalogPost("Running readPreviousMeanSpectrum('%s')" % (meanSpectrumFile))
            result = readPreviousMeanSpectrum(meanSpectrumFile, verbose, invert)
            if (result is None):
                casalogPost("ASCII file is not valid, re-run with overwrite=True")
                return
            avgspectrum, avgSpectrumNansReplaced, meanSpectrumThreshold, edgesUsed, nchan, nanmin, firstFreq, lastFreq, previousCentralArcsec, previousMask, percentagePixelsNotMasked = result
            if meanSpectrumMethod == 'mom0mom8jointMask':
                numberPixelsInJointMask = edgesUsed
            # Note: previousCentralArcsec  and   previousMask  are not used yet
            regionsPruned = 0 # we don't know if pruning was done to generate this spectrum
        chanInfo = [result[4], result[6], result[7], abs(result[7]-result[6])/(result[4]-1)]
        if verbose:
            print("len(avgspectrum) = ", len(avgspectrum))
        if (len(avgspectrum) < 2):
            casalogPost("ASCII file is too short, re-run with overwrite=True")
            return
        if (firstFreq == 0 and lastFreq == 0):
            # This was an old-format ASCII file, without a frequency column
            n, firstFreq, lastFreq, channelWidth = chanInfo
            channelWidth = abs(channelWidth)
        if (fitsTable or img==''):
            if ((type(nBaselineChannels) == float or type(nBaselineChannels) == np.float64) and not fitsTable):
                print("nBaselineChannels=%d, nchan=%d" % (nBaselineChannels, nchan))
                nBaselineChannels = int(round_half_up(nBaselineChannels*nchan))
            n, firstFreq, lastFreq, channelWidth = chanInfo  # freqs are in Hz
            channelWidth = abs(channelWidth)
            print("Setting channelWidth to %g" % (channelWidth))
            if (nBaselineChannels < 2):
                casalogPost("You must have at least 2 edge channels")
                return
    # By this point, nchan is guaranteed to be defined, in case it is needed.
    if narrow == 'auto':
        narrow = pickNarrow(nchan)  # May 6, 2020 for ALMAGAL manual usage of narrow
    donetime = timeUtilities.time()
    casalogPost("%.1f sec elapsed in meanSpectrum()" % (donetime-startTime))
    casalogPost("Iteration %d (sigmaFindContinuumMode='%s', sigmaFindContinuum=%s)" % (iteration,sigmaFindContinuumMode,str(sigmaFindContinuum)))
    sFC_TDM = pick_sFC_TDM(meanSpectrumMethod, singleContinuum)
    # This definition of peakOverMad will be high if there is strong continuum
    # or if there is a strong line. However, removing the median from the peak 
    # would eliminate the sensitivity to continuum emission, which we found to
    # be a bad idea.
    peakOverMad = np.max(avgSpectrumNansReplaced) / MAD(avgSpectrumNansReplaced)
    # But for purposes of deciding when to not trigger amend mask (i.e. strong
    # continuum) then we do want to discriminate, and removing the median is 
    # good for that. Here we ignore the edge channels to calculate the peak.
    peakMinusMedianOverMad = (np.max(avgSpectrumNansReplaced[1:-1]) - np.median(avgSpectrumNansReplaced)) / MAD(avgSpectrumNansReplaced)

    if (sigmaFindContinuum in ['auto','autolower',-1]):
        # Choose the starting value for sigmaFindContinuum if automatic mode is selected.  
        # Here we could consider adding a dependency on whether numberPixelsInMom8Mask > 0
        # or not, but let's first try only adjusting this before the second run. 06 Jan 2019.
        if (tdmSpectrum(channelWidth, nchan)):
            sigmaFindContinuum = sFC_TDM
            casalogPost("Setting sigmaFindContinuum = %.1f since it is TDM" % (sFC_TDM))
        elif (meanSpectrumMethod.find('meanAboveThreshold') >= 0):
            sigmaFindContinuum = 3.5
            casalogPost("Setting sigmaFindContinuum = %.1f since we are using meanAboveThreshold" % (sigmaFindContinuum))
        elif (meanSpectrumMethod.find('peakOverMAD') >= 0):
            sigmaFindContinuum = 6.0
            casalogPost("Setting sigmaFindContinuum = %.1f since we are using peakOverMAD" % (sigmaFindContinuum))
        elif (meanSpectrumMethod == 'mom0mom8jointMask'):
            # Here is some fine tuning based on the type of spw, which we may
            # regret in the future, but it gives better practical results for ALMA.
            # Will need to revamp if storage of subregions of an spw are ever 
            # implemented in the correlator.
            if nchan < 750:
                # i.e. TDM spw or lots of channel averaging
                if peakOverMad > 6:
                    sigmaFindContinuum = 4.2 # was 3.5 on March29, 2018
                else:
                    sigmaFindContinuum = 4.5 # was 3.5 on March29, 2018
            else:
                # i.e. FDM spw with no (or little) channel averaging
                if peakOverMad > 6:
                    sigmaFindContinuum = 2.5 # was 2.6 in v4.90;  was 3.0 on Apr2, was 3.5 on Mar29
                else:
                    sigmaFindContinuum = 3.2 # was 3.0 on Apr2, was 3.5 on Mar29
            casalogPost("Setting sigmaFindContinuum = %.1f since we are using mom0mom8jointMask" % (sigmaFindContinuum))
        else:
            sigmaFindContinuum = 3.0
            print("Unknown method")
        if triangularPatternSeen and sigmaFindContinuumMode == 'auto':  
            # this cannot happen with mom0mom8jointMask because we don't check for it 
            # (because it doesn't happen in that method of constructing a mean spectrum)
            sigmaFindContinuum += 0.5
            casalogPost("Adding 0.5 to sigmaFindContinuum due to triangularPattern seen")

    fCCiteration = 0
    if (meanSpectrumMethod == 'mom0mom8jointMask' and rmStatContQuadratic):
        # We remove the quadratic from the stored mean spectrum, that way the stored spectrum still
        # contains the curved baseline, which may be useful for future reference.
        avgSpectrumNansReplaced, channelsFit, yfit = removeStatContQuadratic(avgSpectrumNansReplaced, quadraticNsigma, returnExtra=True, makeMovie=makeMovie, img=img)
        casalogPost("Removed quadratic computed from %d/%d channels" % (len(channelsFit),len(avgSpectrumNansReplaced)))
    print("Calling findContinuumChannels with nBaselineChannels = %d" % (nBaselineChannels))
    result = findContinuumChannels(avgSpectrumNansReplaced, nBaselineChannels, 
                                   sigmaFindContinuum, nanmin, baselineModeB, 
                                   trimChannels, narrow, verbose, maxTrim, 
                                   maxTrimFraction, separator, peakOverMad,
                                   negativeThresholdFactor=negativeThresholdFactor, 
                                   dropBaselineChannels=dropBaselineChannels,
                                   madRatioUpperLimit=madRatioUpperLimit, 
                                   madRatioLowerLimit=madRatioLowerLimit, 
                                   projectCode=projectCode, fCCiteration=fCCiteration,
                                   signalRatioTier1=signalRatioTier1, signalRatioTier2=signalRatioTier2,
                                   sigmaFindContinuumMode=sigmaFindContinuumMode,
                                   enableRejectNarrowInnerWindows=enableRejectNarrowInnerWindows, 
                                   useUncorrectedMedian=useUncorrectedMedian,
                                   returnBluePoints=returnBluePoints,
                                   removeOutlierBaselineChannels=removeOutlierBaselineChannels,
                                   amendMaskIterationName=amendMaskIterationName, useBaseline=useBaseline, idxNonZeroMAD=idxNonZeroMAD)
    
    continuumChannels,selection,threshold,median,groups,correctionFactor,medianTrue,mad,medianCorrectionFactor,negativeThreshold,lineStrengthFactor,singleChannelPeaksAboveSFC,allGroupsAboveSFC,spectralDiff, trimChannels, useLowBaseline, narrowValueModified, allBaselineChannelsXY, madRatio, useMiddleChannels, signalRatio, rangesDropped, idxNonZeroMAD = result
    RangesDropped += rangesDropped
    if verbose: print("aboveBelow run0")
    sumAboveMedian, sumBelowMedian, sumRatio, channelsAboveMedian, channelsBelowMedian, channelRatio = \
        aboveBelow(avgSpectrumNansReplaced,medianTrue)
    spwBandwidth = nchan*channelWidth  # in Hz
    c_mks = 2.99792458e8
    spwBandwidthKms = 0.001 * c_mks * spwBandwidth/np.mean([firstFreq,lastFreq])
    sFC_factor = 1.0
    newMaxTrim = 0
    sFC_adjusted = False
    # Due to the following code block, rejectNarrowInnerWindowsChannels() can (by reducing groups from >2 to 2)
    # can lead to a different value of maxTrim for FDM datasets.
    if (groups <= 2 and not tdmSpectrum(channelWidth, nchan) and 
        maxTrim==maxTrimDefault and trimChannels=='auto'):
        # CAS-8822
        newMaxTrim = maxTrimDefault*nchan/128
        casalogPost("Changing maxTrim from %s to %s for this FDM spw because trimChannels='%s' and groups=%d." % (str(maxTrim),str(newMaxTrim),trimChannels,groups))
        maxTrim = newMaxTrim

    # Adjust sigmaFindContinuum if necessary, and re-run findContinuumChannels()
    if (singleChannelPeaksAboveSFC==allGroupsAboveSFC and allGroupsAboveSFC>1):
        # If all the channels exceeding the threshold are single spikes and there are more
        # than one of them, then they may be noise spikes, so raise the threshold a bit
        # This 'if' block CAN sometimes be used by meanSpectrumMethod='mom0mom8jointMask'.
        if (sigmaFindContinuum < sFC_TDM):  
            if sigmaFindContinuumMode == 'autolower':
                casalogPost("Because we are in autolower, not scaling the threshold upward by a factor of %.2f to avoid apparent noise spikes (%d==%d)." % (sFC_factor, singleChannelPeaksAboveSFC,allGroupsAboveSFC))
            elif sigmaFindContinuumMode == 'auto':
                # raise the threshold a bit since all the peaks look like all noise spikes
                sFC_factor = 1.5
                sigmaFindContinuum *= sFC_factor
                casalogPost("Scaling the threshold upward by a factor of %.2f to avoid apparent noise spikes (%d==%d)." % (sFC_factor, singleChannelPeaksAboveSFC,allGroupsAboveSFC))
                fCCiteration += 1
                result = findContinuumChannels(avgSpectrumNansReplaced, 
                                               nBaselineChannels, sigmaFindContinuum, nanmin, 
                                               baselineModeB, trimChannels, narrow, verbose, 
                                               maxTrim, maxTrimFraction, separator, peakOverMad,
                                               negativeThresholdFactor=negativeThresholdFactor,
                                               dropBaselineChannels=dropBaselineChannels,
                                               madRatioUpperLimit=madRatioUpperLimit, 
                                               madRatioLowerLimit=madRatioLowerLimit, 
                                               projectCode=projectCode, fCCiteration=fCCiteration,
                                               signalRatioTier1=signalRatioTier1, signalRatioTier2=signalRatioTier2,sigmaFindContinuumMode=sigmaFindContinuumMode,                                    
                                               enableRejectNarrowInnerWindows=enableRejectNarrowInnerWindows, 
                                               useUncorrectedMedian=useUncorrectedMedian,
                                               returnBluePoints=returnBluePoints,
                                               removeOutlierBaselineChannels=removeOutlierBaselineChannels,
                                               amendMaskIterationName=amendMaskIterationName, useBaseline=useBaseline, idxNonZeroMAD=idxNonZeroMAD)
            continuumChannels,selection,threshold,median,groups,correctionFactor,medianTrue,mad,medianCorrectionFactor,negativeThreshold,lineStrengthFactor,singleChannelPeaksAboveSFC,allGroupsAboveSFC,spectralDiff,trimChannels,useLowBaseline, narrowValueModified, allBaselineChannelsXY, madRatio, useMiddleChannels, signalRatio, rangesDropped, idxNonZeroMAD  = result
            RangesDropped += rangesDropped
            if verbose: print("aboveBelow run1")
            sumAboveMedian, sumBelowMedian, sumRatio, channelsAboveMedian, channelsBelowMedian, channelRatio = \
                aboveBelow(avgSpectrumNansReplaced,medianTrue)
    elif ((groups > 3 or (groups > 1 and channelRatio < 1.0) or (channelRatio < 0.5) or (groups == 2 and channelRatio < 1.3)) and sigmaFindContinuumMode in ['auto','autolower'] and meanSpectrumMethod.find('peakOver') < 0 and meanSpectrumMethod.find('mom0mom8jointMask') < 0 and not singleContinuum):
        # If there are a lot of groups, or a lot of flux in channels above the median 
        # compared to channels below the median (i.e. the channelRatio), then lower the 
        # sigma in order to push the threshold for real lines (or line emission wings) lower.
        # However, if there is only 1 group, then there may be no real lines 
        # present, so lowering the threshold in this case can create needless 
        # extra groups, so don't allow it.
        # **** This 'elif' is NOT used by meanSpectrumMethod='mom0mom8jointMask'. *****
        print("A: groups,channelRatio=", groups, channelRatio, channelRatio < 1.0, channelRatio>0.1, tdmSpectrum(channelWidth,nchan), groups>2)
        if (channelRatio < 0.9 and channelRatio > 0.1 and (firstFreq>60e9 and not tdmSpectrum(channelWidth,nchan)) and groups>2):  # was nchan>256
            # Don't allow this much reduction in ALMA TDM mode as it chops up 
            # line-free quasar spectra too much. The channelRatio>0.1 
            # requirement prevents failures due to ALMA TFB platforming.
            sFC_factor = 0.333
        elif (groups <= 2):
            if (0.1 < channelRatio < 1.3 and groups == 2 and 
                not tdmSpectrum(channelWidth,nchan) and channelWidth>=1875e6/600.):
                if (channelWidth < 1875e6/360.):
                    if madRatioLowerLimit < madRatio < madRatioUpperLimit:
                        sFC_factor = 0.60
                    else:
                        sFC_factor = 0.50 
                    # i.e. for galaxy spectra with FDM 480 channel (online-averaging) resolution
                    # but 0.5 is too low for uid___A001_X879_X47a.s24_0.ELS26_sci.spw25.mfs.I.findcont.residual
                    # and for uid___A001_X2d8_X2c5.s24_0.2276_444_53712_sci.spw16.mfs.I.findcont.residual
                    #    the latter requires >=0.057
                    #   need to reconcile in future versions
                else:
                    sFC_factor = 0.7  # i.e. for galaxy spectra with FDM 240 channel (online-averaging) resolution
            else:
                # prevent sigmaFindContinuum going to inf if groups==1
                # prevent sigmaFindContinuum going > 1 if groups==2
                sFC_factor = 0.9
        else:
            if tdmSpectrum(channelWidth,nchan):
                sFC_factor = 1.0
            elif channelWidth>0:  # the second factor tempers the reduction as the spw bandwidth decreases
                sFC_factor = (np.log(3)/np.log(groups)) ** (spwBandwidth/1.875e9)
            else:
                sFC_factor = (np.log(3)/np.log(groups))
        casalogPost("setting factor to %f because groups=%d, channelRatio=%g, firstFreq=%g, nchan=%d, channelWidth=%e" % (sFC_factor,groups,channelRatio,firstFreq,nchan,channelWidth))
        print("---------------------")
        casalogPost("Scaling the threshold by a factor of %.2f (groups=%d, channelRatio=%f)" % (sFC_factor, groups,channelRatio))
        print("---------------------")
        sigmaFindContinuum *= sFC_factor
        fCCiteration += 1
        result = findContinuumChannels(avgSpectrumNansReplaced, 
                                       nBaselineChannels, sigmaFindContinuum, nanmin, 
                                       baselineModeB, trimChannels, narrow, verbose, maxTrim, 
                                       maxTrimFraction, separator, peakOverMad,
                                       negativeThresholdFactor=negativeThresholdFactor, 
                                       dropBaselineChannels=dropBaselineChannels,
                                       madRatioUpperLimit=madRatioUpperLimit, 
                                       madRatioLowerLimit=madRatioLowerLimit, 
                                       projectCode=projectCode, fCCiteration=fCCiteration,
                                       signalRatioTier1=signalRatioTier1, signalRatioTier2=signalRatioTier2,
                                       sigmaFindContinuumMode=sigmaFindContinuumMode,
                                       enableRejectNarrowInnerWindows=enableRejectNarrowInnerWindows, 
                                       useUncorrectedMedian=useUncorrectedMedian,
                                       returnBluePoints=returnBluePoints,
                                       removeOutlierBaselineChannels=removeOutlierBaselineChannels,
                                       amendMaskIterationName=amendMaskIterationName, useBaseline=useBaseline, idxNonZeroMAD=idxNonZeroMAD)

        continuumChannels,selection,threshold,median,groups,correctionFactor,medianTrue,mad,medianCorrectionFactor,negativeThreshold,lineStrengthFactor,singleChannelPeaksAboveSFC,allGroupsAboveSFC,spectralDiff,trimChannels,useLowBaseline, narrowValueModified, allBaselineChannelsXY, madRatio, useMiddleChannels, signalRatio, rangesDropped, idxNonZeroMAD  = result
        RangesDropped += rangesDropped
        if verbose: print("aboveBelow run2")
        sumAboveMedian, sumBelowMedian, sumRatio, channelsAboveMedian, channelsBelowMedian, channelRatio = \
            aboveBelow(avgSpectrumNansReplaced,medianTrue)
    else:
        if amendMaskIterationName == '.autoLower' and momDiffSNR < momDiffApproachLevel: # added Jun 16, 2022
            casalogPost("Not adjusting sigmaFindContinuum because we are in .autoLower and momDiffSNR=%.1f < %.1f"%(momDiffSNR,momDiffApproachLevel)) # added May 11, 2022, with momDiffSNR clause added June 11
        elif (meanSpectrumMethod.find('peakOver') < 0 and 
            meanSpectrumMethod.find('mom0mom8jointMask') < 0):
            casalogPost("Not adjusting sigmaFindContinuum, because groups=%d, channelRatio=%g, firstFreq=%g, nchan=%d" % (groups,channelRatio,firstFreq,nchan),debug=True)
        else:
            # We are using either peakOverMad or mom0mom8jointMask
            if madRatio is None: # python2 behavior for a None inequality was True, while python3 crashes
                madRatioCheck = peakOverMad>40 #   this value was set True in PL<=PL2022, note that 40 is different from default
            else:
                madRatioCheck = madRatio < maxMadRatioForSFCAdjustment or peakOverMad>peakOverMadCriterion  # added or statement on May 11, 2022
            casalogPost("numberPixelsInMom8Mask=%d, allContinuumSelected('%s',%d)=%s" % (numberPixelsInMom8Mask,selection,nchan,allContinuumSelected(selection,nchan)))
            if ((groups >= minGroupsForSFCAdjustment and not tdmSpectrum(channelWidth,nchan) 
                 or (groups >= 3 and tdmSpectrum(channelWidth,nchan)))  and 
                sigmaFindContinuumMode in ['auto','autolower'] and # added Mar 27, 2019
#               (peakOverMad>minPeakOverMadForSFCAdjustment or 
               ((peakOverMad>minPeakOverMadForSFCAdjustment and madRatioCheck) 
                or meanSpectrumMethod.find('peakOver') >= 0)): 
                # ***** This 'if' block will never be used when groups==1 *****
                #18 set by 312:2016.1.01400.S spw25 not needing it with 11.7 and
                #          431:E2E5.1.00036.S spw24 not needing it with 17.44
                #          268:2015.1.00190.S spw16 not needing it with 18.7
                #          423:E2E5.1.00034.S spw25 needing it with 20.3
                #      but 262:2015.1.01068.S spw37 does not need it with 38
                # The following heuristics for sFC achieve better results on hot cores 
                # when meanAboveThreshold cannot be used.
                if amendMaskIterationName in ['.autoLower', '.autoLower2']:
                    scalingFactor = 6 # added May 10, 2022
                elif amendMaskIterationName == '.onlyExtraMask' and groups > 32:
                    scalingFactor = 6 # added for PL2023
                elif amendMaskIterationName == '.extraMask' and groups > 32:
                    scalingFactor = 6 # added for PL2023
                else:
                    scalingFactor = 5
                if groups < maxGroups: 
                    # protect against negative number raised to a power
                    sFC_factor = np.max([scalingFactor/7.0, (1-groups/float(maxGroups)) ** (spwBandwidth/1.875e9)])
                    casalogPost("Because groups=%d < maxGroups=%d, setting sFC_factor to %f instead of %d/7." % (groups,maxGroups,sFC_factor,scalingFactor))
                else:
                    sFC_factor = scalingFactor/7.0 # was previously 2.5/sigmaFindContinuum 
                # How often do we get 5/7 vs. a higher value? (in the benchmark)
                sigmaFindContinuum *= sFC_factor
                # madRatio could be 'None' so set to string
                casalogPost("%s Adjusting sigmaFindContinuum by x%.2f to %f because groups=%d>=%d and not TDM and meanSpectrumMethod = %s and peakOverMad=%f>%g and (madRatio=%s<%f or peakOverMad>%d)" % (projectCode, sFC_factor,sigmaFindContinuum, groups, minGroupsForSFCAdjustment, meanSpectrumMethod, peakOverMad, minPeakOverMadForSFCAdjustment,str(madRatio),maxMadRatioForSFCAdjustment, peakOverMadCriterion), debug=True)
                sFC_adjusted = True # need to set this to force examination of blue points below
            elif (numberPixelsInMom8Mask >= 4) and allContinuumSelected(selection,nchan):
                normalizedMom0peak = mom0peak/spwBandwidthKms
                # before I added tdm/fdm split, it was 1.4 on Jan 26 regression;  1.5 on Feb 01 regression
                # page 260 needs <= 1.41
                if tdmSpectrum(channelWidth, nchan):
                    normalizedMom0factor = 1.5 
                else:
                    normalizedMom0factor = 1.4 # was 1.4 on Jan 26 regression;  1.5 on Feb 01 regression
                if mom8peak > normalizedMom0factor*normalizedMom0peak:
                    # New for Cycle 7:  Here we adjust sFC downward because emission was found 
                    # in the mom8 image but no line regions were found initially. - 06 Jan 2019
                    sFC_factor = SFC_FACTOR_WHEN_MOM8MASK_EXISTS
                    sigmaFindContinuum *= sFC_factor
                    if mom8peak > 3.5*normalizedMom0peak and peakOverMad > 5.5:
                        # needed for helms61 spw16 in 2018.1.00922.S
                        sFC_adjusted = True # need to set this to force examination of blue points below
#                   but do not set it between 1.5-3.5 since it removes too much continuum in line-free cases
                    casalogPost("%s Adjusting sigmaFindContinuum by %.2f to %f because groups=1 and >=%.3f of the channels were selected and mom8peak=%f > %.2f*(mom0peak=%f/bwkms=%f)=%f" % (projectCode, sFC_factor, sigmaFindContinuum, ALL_CONTINUUM_CRITERION, mom8peak, normalizedMom0factor, mom0peak, spwBandwidthKms, normalizedMom0factor*normalizedMom0peak)) 
                else:
                    casalogPost("%s Mom8 mask is present and allContinuumSelected, but mom8peak=%f not > %.2f*(mom0peak=%f/bwkms=%f)=%f" % (projectCode, mom8peak, normalizedMom0factor, mom0peak, spwBandwidthKms, normalizedMom0factor*normalizedMom0peak)) 
            elif groups == 3 and tdmSpectrum(channelWidth, nchan):
                # This is for the case of double-peaked spectrum where weaker detection in the channels between 
                # the peaks are below threshold.
                sFC_adjusted = True # need to set this to force examination of blue points below
                casalogPost('%s Not adjusting sigmaFindContinuum, but setting blue pruning because there are 3 groups and TDM' % (projectCode))
            else:
                # madRatio could be 'None' so force it to be a string
                casalogPost("%s Not adjusting sigmaFindContinuum because groups=%d < %d or peakOverMad=%f<%.0f or (madRatio=%s>=%f and peakOverMad<=%d)" % (projectCode, groups,minGroupsForSFCAdjustment,peakOverMad,minPeakOverMadForSFCAdjustment,str(madRatio),maxMadRatioForSFCAdjustment,peakOverMadCriterion), debug=True)
                
        if (newMaxTrim > 0 or 
            (allContinuumSelected(selection,nchan) and numberPixelsInMom8Mask > 0) or # line added 06 Jan 2019
            (groups>minGroupsForSFCAdjustment and not tdmSpectrum(channelWidth,nchan))): # line added Aug 22, 2016
            if (newMaxTrim > 0):
                casalogPost("But re-running findContinuumChannels with new maxTrim")
            fCCiteration += 1
            result = findContinuumChannels(avgSpectrumNansReplaced, 
                                           nBaselineChannels, sigmaFindContinuum, nanmin, 
                                           baselineModeB, trimChannels, narrow, verbose, maxTrim, 
                                           maxTrimFraction, separator, peakOverMad,
                                           negativeThresholdFactor=negativeThresholdFactor, 
                                           dropBaselineChannels=dropBaselineChannels,
                                           madRatioUpperLimit=madRatioUpperLimit, 
                                           madRatioLowerLimit=madRatioLowerLimit, 
                                           projectCode=projectCode, fCCiteration=fCCiteration,
                                           signalRatioTier1=signalRatioTier1, signalRatioTier2=signalRatioTier2,
                                           sigmaFindContinuumMode=sigmaFindContinuumMode,
                                           enableRejectNarrowInnerWindows=enableRejectNarrowInnerWindows, 
                                           useUncorrectedMedian=useUncorrectedMedian, 
                                           returnBluePoints=returnBluePoints,
                                           removeOutlierBaselineChannels=removeOutlierBaselineChannels,
                                           amendMaskIterationName=amendMaskIterationName, useBaseline=useBaseline, idxNonZeroMAD=idxNonZeroMAD)
                                           

            continuumChannels,selection,threshold,median,groups,correctionFactor,medianTrue,mad,medianCorrectionFactor,negativeThreshold,lineStrengthFactor,singleChannelPeaksAboveSFC,allGroupsAboveSFC,spectralDiff,trimChannels, useLowBaseline, narrowValueModified, allBaselineChannelsXY, madRatio, useMiddleChannels, signalRatio, rangesDropped, idxNonZeroMAD  = result
            RangesDropped += rangesDropped
            if verbose: print("aboveBelow run3")
            sumAboveMedian, sumBelowMedian, sumRatio, channelsAboveMedian, channelsBelowMedian, channelRatio = \
                aboveBelow(avgSpectrumNansReplaced,medianTrue)

    selectionPreBluePruning = selection
    secondSelectionAdded = ''  # used to indicate on the spectrum plot if the heuristic for CAS-11720 was activated to ensure a second region
    if meanSpectrumMethod == 'mom0mom8jointMask':
        casalogPost('No slope fit attempted because we are using mom0mom8jointMask.')
        if ((sFC_adjusted or ((peakOverMad>6 or len(continuumChannels) > 4*nchan/5) and not tdmSpectrum(channelWidth,nchan))) and 
             len(continuumChannels) > 0 and allowBluePruning):
            # New heuristic for Cycle 6: blue pruning (a.k.a. examining blue points)
            # Definition: remove any candidate continuum range
            # (horizontal cyan lines in the plot) that do not contain any 
            # initial baseline points (blue points in the plot), and trim 
            # those ranges that do contain baseline points to the maximal 
            # extent of the enclosed baseline points, eliminating them if 
            # they get too narrow.
            casalogPost('Activating blue pruning heuristic')
            channelSelections = selection.split(separator)
            validSelections = []
            newContinuumChannels = []
            newGroups = 0
            checkForGaps = True
            # checkForGaps means to look for large portions of a channelSelection that has no
            # baseline channels in it, and if found, then split up that selection into pieces.
            # This is meant to eliminate the remaining weak lines in a hot core spectrum that
            # are below the final threshold.
            for i,ccstring in enumerate(channelSelections): 
                validSelection = False
                cc = [int(j) for j in ccstring.split('~')]
                selectionWidth = cc[1]+1 - cc[0]
                # We set a max threshold mainly to prevent losing too much continuum if the "line" is not real
                if tdmSpectrum(channelWidth,nchan):
                    gapMaxThreshold = int(nchan/4) # was 70
                else:
                    gapMaxThreshold = int(nchan/3) # added for 478chan spw27 in 2018.1.00081.S
                if groups == 1:
                    # Enable detection of single faint narrow line, 
                    # e.g. 42 in project 54 uid___A001_X144_X7b.s21_0.IRD_C_sci.spw29
                    # 58 MHz 240 chan, 15 chan -> 3.6 km/s at 300 GHz
                    # 58 MHz 480 chan, 30 chan -> 3.6 km/s
                    # 58 MHz 960 chan, 60 chan -> 3.6 km/s
                    # 58 MHz 960 chan, 15 chan -> 3.6 km/s at 75 GHz
                    gapMinThreshold = int(30*(firstFreq/300e9)*(spwBandwidth/58e6)*(nchan/480.)) # previously was 40
                    casalogPost('initial gapMinThreshold=%d' % (gapMinThreshold))
                    if peakOverMad <= 6: # we got into here only because more than 4/5 channels were picked
                        if gapMinThreshold < 30:
                            gapMinThreshold = 30  # 30 was set to avoid gaps with no lines
                        elif gapMinThreshold > 45: #50:
                            gapMinThreshold = 45 #50  # 50 was set by E2E6.1.00030.S spw16 CO galaxy w/no continuum uid___A002_Xcff05c_X1bd.s28_0.HCG25b_sci.spw16.mfs.I.findcont.residual, but this line is 111 channels wide, so no longer appears to motivate this value
                                                 #49 was needed by 2018.1.00635.S spw29 CI1-0
                    else:
                        gapMinThreshold = np.max([16,gapMinThreshold]) # original heuristic, Note: can be a large number, even > gapMaxThreshold!
                    if (gapMinThreshold > gapMaxThreshold):  
                        # this would fix two results on CAS-5290, including 2018.1.01100.S 
                        # but has not been checked in regression
                        casalogPost('sFC was adjusted and gapMin=%d > gapMax=%d, so setting gapMin=30' % (gapMinThreshold,gapMaxThreshold))
                        gapMinThreshold = 30  # could also try 0.06*nchan (which is 28 for 478 chan)
                        if gapMinThreshold > gapMaxThreshold:
                            casalogPost('   but new gapMin=%d > gapMax=%d, so setting gapMin=%d' % (gapMinThreshold,gapMaxThreshold,gapMaxThreshold-10))
                            gapMinThreshold = gapMaxThreshold-10
                else:
                    # Must be at least half the width of the group
                    gapMinThreshold = int(np.max([selectionWidth/2, narrow]))   # splitgaps.pdf

                if checkForGaps and verbose:
                    casalogPost("%s Looking for gaps in group %d/%d: %s that are %d < gap < %d" % (projectCode,i+1,len(channelSelections),ccstring,gapMinThreshold, gapMaxThreshold), debug=True)
                theseChannels = list(range(cc[0], cc[1]+1))
                startChan = -1
                stopChan = -1
                gaps = []
                for chan in theseChannels:
                    if chan in allBaselineChannelsXY[0]:
                        validSelection = True  # valid if any of the channels appeared in the original baseline
                        if startChan == -1:
                            startChan = chan
                        else:
                            # If the blue points are contiguous, then chan will now be stopChan+1, so check
                            # if it is actually a lot more than that (meaning we just crossed a gap). But,
                            # it must be a big gap relative to the selection width in order to qualify.
                            if ((stopChan > -1) and (chan-stopChan) > gapMinThreshold and (chan-stopChan) < gapMaxThreshold and checkForGaps):
                                gaps.append([stopChan+1,chan-1])
                            else:
                                if chan-stopChan >= gapMaxThreshold:
                                    print("chan-stopChan = %d >= %d" % (chan-stopChan,gapMaxThreshold))
                            stopChan = chan
                if (len(gaps) == 1 and tdmSpectrum(channelWidth,nchan)):
                    # check if the max is in here
                    if np.max(np.abs(avgSpectrumNansReplaced)) > np.max(np.abs(avgSpectrumNansReplaced[gaps[0][0]:gaps[0][1]])):
                        casalogPost('Removing the one gap found because the peak is not in here')
                        gaps = []
                tooNarrow = False
                if validSelection:
                    if (stopChan - startChan) < narrow-1:
                        tooNarrow = True
                        validSelection = False
                if len(gaps) == 0:
                    casalogPost('%s Looking for trimmed channel ranges to drop'%(projectCode))
                    if stopChan == -1:
                        stopChan = cc[1]
                    ccstring = '%d~%d' % (startChan,stopChan)
                    if validSelection:
                        newGroups += 1
                        validSelections.append(ccstring)
                        newContinuumChannels += list(range(startChan,stopChan+1)) # theseChannels (bug fix May 14, 2020)
                    elif tooNarrow:
                        casalogPost('%s Dropping trimmed channel range %s because it is too narrow (<%d).' % (projectCode,ccstring,narrow),debug=True)
                    else:
                        ccstring = '%d~%d' % (cc[0],cc[1])
                        casalogPost('%s Dropping channel range %s because not enough baseline channels are contained and sFC was previously adjusted downwards.' % (projectCode,ccstring),debug=True)
                else:  # new heuristic on April 7, 2018
                    casalogPost('%s Splitting channel range %s into %d ranges due to wide gaps in baseline channels.' % (projectCode,ccstring,len(gaps)+1),debug=True)
                    for ngap,gap in enumerate(gaps):
                        if ngap == 0:
                            firstChan = startChan
                        else:
                            firstChan = gaps[ngap-1][1]+1
                        ccstring = '%d~%d' % (firstChan,gap[0]-1)
                        if gap[0]-firstChan >= narrow:
                            # The split piece of the range is wide enough to keep it.
                            newGroups += 1
                            validSelections.append(ccstring)
                            newContinuumChannels += list(range(firstChan, gap[0]))
                            casalogPost('  %s Defining range %s' % (projectCode,ccstring), debug=True)
                        else:
                            casalogPost('  %s Skipping range %s because it is too narrow (<%d)' % (projectCode,ccstring,narrow), debug=True)
                    # Process the final piece
                    ccstring = '%d~%d' % (gaps[len(gaps)-1][1]+1,stopChan)
                    if stopChan - gaps[len(gaps)-1][1] >= narrow:
                        newGroups += 1
                        validSelections.append(ccstring)
                        newContinuumChannels += list(range(gaps[len(gaps)-1][1]+1, stopChan+1))
                        casalogPost('  %s Defining range %s' % (projectCode,ccstring), debug=True)
                    else:
                        casalogPost('  %s Skipping range %s because it is too narrow (<%d)' % (projectCode,ccstring,narrow), debug=True)
            # end for i,ccstring in enumerate(channelSelections)
            if newGroups > 0:
                selection = separator.join(validSelections)
                groups = newGroups
                continuumChannels = newContinuumChannels
            else:
                casalogPost('%s Restoring dropped channels because no groups are left.' % (projectCode),debug=True)
        else:
            if not allowBluePruning:
                casalogPost("Not examining blue dots (at user request)")
            else:
                casalogPost("Not examining blue dots because: sFC_adjusted=%s is False or (peakOverMad=%.2f <= 6 and len(chans)=%d <= 4*nchan/5=%d) or it's TDM" % (sFC_adjusted,peakOverMad,len(continuumChannels), 4*nchan/5))
        # check if only 1 narrow group found (to activate a fix for CAS-11720 == PIPE-183)
        if groups == 1:
            if selection == '':
                print("This should not happen since PIPE-359 was fixed.")
                if not amendMask:
                    return
                channelsInSingleGroup = 0
                group0 = [-1,-1]
            else:
                group0 = [int(j) for j in selection.split(',')[0].split('~')]
                channelsInSingleGroup = np.abs(np.diff(group0))[0] + 1
            minPercentInSingleGroup = 5.0
            if float(channelsInSingleGroup)/nchan < minPercentInSingleGroup/100:
                percentage = 100.*channelsInSingleGroup / nchan
                casalogPost('  %s ****** There is only a single channel range found with only %.1f%% (< %.0f%%) of the channels' % (projectCode,percentage,minPercentInSingleGroup))
                whichHalf = np.mean(group0) < nchan/2
                mylist = findWidestContiguousListInOtherHalf(allBaselineChannelsXY[0], whichHalf, nchan)
                if mylist is not None:
                    groups += 1
                    secondSelectionAdded = '%d~%d' % (mylist[0],mylist[-1])
                    if selection == '':
                        selection = secondSelectionAdded
                    else:
                        lastChannelOfOnlyRegion = int(selection.split('~')[-1])
                        if mylist[0] > lastChannelOfOnlyRegion: # allBaselineChannelsXY[0][-1]:
                            # add to end of string (to maintain increasing order)
                            print("Adding to end of string")
                            selection += separator + secondSelectionAdded
                        else:
                            # add to beginning of string (to maintain increasing order)
                            print("Adding %s to beginning of string because %d <= %d" % (secondSelectionAdded, mylist[0],lastChannelOfOnlyRegion))
                            selection = secondSelectionAdded + separator + selection
                    casalogPost('  %s ****** Added a second selection in the other half of the spectrum: %s' % (projectCode,secondSelectionAdded))
                else:
                    casalogPost('  %s ****** No blue points found in the other half of the spectrum' % (projectCode))
    else:
        # The following Cycle 4+5 logic (on checking for the presence of 
        # candidate continuum channels in various segments of the
        # spectrum in order to decide whether to attempt to remove a first or
        # second order baseline) is not used by the mom0mom8jointMask method.
        selectedChannels = countChannels(selection)
        largestGroup = channelsInLargestGroup(selection)
        selections = len(selection.split(separator))
        slopeRemoved = False
        channelDistancesFromCenter = np.array(continuumChannels)-nchan/2
        channelDistancesFromLowerThird = np.array(continuumChannels)-nchan*0.3
        channelDistancesFromUpperThird = np.array(continuumChannels)-nchan*0.7
        if (len(np.where(channelDistancesFromCenter > 0)[0]) > 0 and 
            len(np.where(channelDistancesFromCenter < 0)[0]) > 0):
            channelsInBothHalves = True
        else:
            # casalogPost("channelDistancesFromCenter = %s" % (channelDistancesFromCenter))
            channelsInBothHalves = False  # does not get triggered by case 115, which could use it
        if ((len(np.where(channelDistancesFromLowerThird < 0)[0]) > 0) and 
            (len(np.where(channelDistancesFromUpperThird > 0)[0]) > 0)):
            channelsInBothEdgeThirds = True
        else:
            channelsInBothEdgeThirds = False
        # Too many selected windows means there might be a lot of lines, 
        # so do not attempt a baseline fit.
        maxSelections = 4 # 2    # July 27, 2017
        # If you select too few channels, you cannot get a reliable fit
        minBWFraction = 0.3
        if ((selectedChannels > channelFractionForSlopeRemoval*nchan or 
            (selectedChannels > 0.4*nchan and selections==2 and channelFractionForSlopeRemoval<1) or 
            # fix for project 00956 spw 25 is to put lower bound of 1 < selections:
            (largestGroup>nchan*minBWFraction and 1 < selections <= maxSelections and 
             channelFractionForSlopeRemoval<1))):
            previousResult = continuumChannels,selection,threshold,median,groups,correctionFactor,medianTrue,mad,medianCorrectionFactor,negativeThreshold,lineStrengthFactor,singleChannelPeaksAboveSFC,allGroupsAboveSFC        
            # remove linear slope from mean spectrum and run it again
            index = channelSelectionRangesToIndexArray(selection)
    #        if quadraticFit and (channelsInBothEdgeThirds or (channelsInBothHalves and selections==1)):  # July 27, 2017
            quadraticFitTest = quadraticFit and ((channelsInBothEdgeThirds and selections <= 2) or (channelsInBothHalves and selections==1))
            if quadraticFitTest:
                casalogPost("Fitting quadratic to %d channels (largestGroup=%d,nchan*%.1f=%.1f,selectedChannels=%d)" % (len(index), largestGroup, minBWFraction, nchan*minBWFraction, selectedChannels))
                fitResult = polyfit(index, avgSpectrumNansReplaced[index], MAD(avgSpectrumNansReplaced[index]))
                order2, slope, intercept, xoffset = fitResult
            else:
                casalogPost("Fitting slope to %d channels (largestGroup=%d,nchan*%.1f=%.1f,selectedChannels=%d)" % (len(index), largestGroup, minBWFraction, nchan*minBWFraction, selectedChannels))
                fitResult = linfit(index, avgSpectrumNansReplaced[index], MAD(avgSpectrumNansReplaced[index]))
                slope, intercept = fitResult
            if (sFC_factor >= 1.5 and maxTrim==maxTrimDefault and trimChannels=='auto'):
                #                 add these 'and' cases on July 20, 2016
                # Do not restore if we have modified maxTrim.  This prevents breaking up an FDM
                # spectrum with no line detection into multiple smaller continuum windows.
                sigmaFindContinuum /= sFC_factor
                casalogPost("Restoring sigmaFindContinuum to %f" % (sigmaFindContinuum))
                rerun = True
            else:
                rerun = False
            if quadraticFitTest:
                casalogPost("Removing quadratic = %g*(x-%f)**2 + %g*(x-%f) + %g" % (order2,xoffset,slope,xoffset,intercept))
                myx = np.arange(len(avgSpectrumNansReplaced)) - xoffset
                priorMad = MAD(avgSpectrumNansReplaced)
                trialSpectrum = avgSpectrumNansReplaced + nanmean(avgSpectrumNansReplaced)-(myx**2*order2 + myx*slope + intercept)
                postMad =  MAD(trialSpectrum)
                if postMad > priorMad:
                    casalogPost("MAD of spectrum got worse after quadratic removal: %f to %f. Switching to linear fit." % (priorMad, postMad), debug=True)
                    casalogPost("Fitting slope to %d channels (largestGroup=%d,nchan*%.1f=%.1f,selectedChannels=%d)" % (len(index), largestGroup, minBWFraction, nchan*minBWFraction, selectedChannels))
                    fitResult = linfit(index, avgSpectrumNansReplaced[index], MAD(avgSpectrumNansReplaced[index]))
                    slope, intercept = fitResult
                    if (abs(slope) > minSlopeToRemove):
                        casalogPost("Removing slope = %g" % (slope))
                        # Do not remove the offset in order to avoid putting spectrum near zero
                        avgSpectrumNansReplaced -= np.array(range(len(avgSpectrumNansReplaced)))*slope
                        slopeRemoved = True
                        rerun = True
                else:
                    avgSpectrumNansReplaced = trialSpectrum
                    casalogPost("MAD of spectrum improved after quadratic removal:  %f to %f" % (priorMad, postMad), debug=True)
                slopeRemoved = True
                rerun = True
                if lineStrengthFactor < 1.2:
                    if sigmaFindContinuumMode == 'auto':
                        if tdmSpectrum(channelWidth,nchan) or sigmaFindContinuum < 3.5:  # was 4
                            sigmaFindContinuum += 0.5
                            casalogPost("lineStrengthFactor = %f < 1.2 (increasing sigmaFindContinuum by 0.5 to %.1f)" % (lineStrengthFactor,sigmaFindContinuum)) 
                        else:
                            casalogPost("lineStrengthFactor = %f < 1.2 (but not increasing sigmaFindContinuum by 0.5 because it is TDM or already >=4)" % (lineStrengthFactor))
                    else:
                        casalogPost("lineStrengthFactor = %f < 1.2 (but not increasing sigmaFindContinuum by 0.5 because it is not 'auto')" % (lineStrengthFactor))
                else:
                    casalogPost("lineStrengthFactor = %f >= 1.2" % (lineStrengthFactor)) 
            else:
                if (abs(slope) > minSlopeToRemove):
                    casalogPost("Removing slope = %g" % (slope))
                    avgSpectrumNansReplaced -= np.array(range(len(avgSpectrumNansReplaced)))*slope
                    slopeRemoved = True
                    rerun = True
            # The following helped deliver more continuum bandwidth for HD_142527 spw3, but
            # it harmed 2013.1.00518 13co (and probably others) by including wing emission.
            # if trimChannels == 'auto':   # prevent overzealous trimming  July 20, 2016
            #     maxTrim = maxTrimDefault # prevent overzealous trimming  July 20, 2016
            discardSlopeResult = False
            if rerun:   # this is only relevant in Cycle 4 and 5; not used by mom0mom8jointMask
                fCCiteration += 1
                result = findContinuumChannels(avgSpectrumNansReplaced, 
                                               nBaselineChannels, sigmaFindContinuum, nanmin, 
                                               baselineModeB, trimChannels, narrow, verbose, 
                                               maxTrim, maxTrimFraction, separator, peakOverMad, fitResult, 
                                               negativeThresholdFactor=negativeThresholdFactor, 
                                               dropBaselineChannels=dropBaselineChannels,
                                               madRatioUpperLimit=madRatioUpperLimit, 
                                               madRatioLowerLimit=madRatioLowerLimit, 
                                               projectCode=projectCode, fCCiteration=fCCiteration,
                                               signalRatioTier1=signalRatioTier1, 
                                               signalRatioTier2=signalRatioTier2,
                                               sigmaFindContinuumMode=sigmaFindContinuumMode,
                                               enableRejectNarrowInnerWindows=enableRejectNarrowInnerWindows, 
                                               useUncorrectedMedian=useUncorrectedMedian,
                                               returnBluePoints=returnBluePoints,
                                               removeOutlierBaselineChannels=removeOutlierBaselineChannels, useBaseline=useBaseline, idxNonZeroMAD=idxNonZeroMAD)

                continuumChannels,selection,threshold,median,groups,correctionFactor,medianTrue,mad,medianCorrectionFactor,negativeThreshold,lineStrengthFactor,singleChannelPeaksAboveSFC,allGroupsAboveSFC,spectralDiff,trimChannels,useLowBaseline, narrowValueModified, allBaselineChannelsXY, madRatio, useMiddleChannels, signalRatio, rangesDropped, idxNonZeroMAD  = result
                RangesDropped += rangesDropped

                # If we had only one group and only added one or two more group after removing slope, and the
                # smallest is small compared to the original group, then discard the new solution.
                if (groups <= 3 and previousResult[4] == 1):
                    counts = countChannelsInRanges(selection)
                    if (float(min(counts))/max(counts) < 0.2):
                        casalogPost("*** Restoring result prior to linfit because %d/%d < 0.2***" % (min(counts),max(counts)))
                        discardSlopeResult = True
                        continuumChannels,selection,threshold,median,groups,correctionFactor,medianTrue,mad,medianCorrectionFactor,negativeThreshold,lineStrengthFactor,singleChannelPeaksAboveSFC,allGroupsAboveSFC = previousResult
        else:
            if channelFractionForSlopeRemoval < 1:
                casalogPost("No slope fit attempted because selected channels (%d) < %.2f * nchan(%d) or other criteria not met" % (selectedChannels,channelFractionForSlopeRemoval,nchan))
                casalogPost("  largestGroup=%d <= nchan*%.1f=%.1f or selections=%d > %d" % (largestGroup,minBWFraction,nchan*minBWFraction,selections,maxSelections))
            else:
                casalogPost("No slope fit attempted because channelFractionForSlopeRemoval >= 1")
    idx = np.where(avgSpectrumNansReplaced < threshold)
    madOfPointsBelowThreshold = MAD(avgSpectrumNansReplaced[idx])

    if avoidance != '':
        # Here we are relying on continuumChannels to match the selection string, so double-check to be sure
        if selection != convertChannelListIntoSelection(continuumChannels):
            casalogPost('THERE IS A MISMATCH BETWEEN selection AND continuumChannels!!!')
            print("selection: ", selection)
            print("continuumChannels: ", convertChannelListIntoSelection(continuumChannels))
        avoidChannels = convertSelectionIntoChannelList(avoidance)
        newContinuumChannels = sorted(list(set(continuumChannels) - set(avoidChannels)))
        print("Removing %d channels from list due to avoidance regions" % (len(continuumChannels)-len(newContinuumChannels)))
        continuumChannels = newContinuumChannels
        selection = convertChannelListIntoSelection(continuumChannels)
        groups = selection.count(';')+1
    
    #########################################
    # Plot the results (before returning)
    #########################################
    labelDescs = []
    if amendMaskIterationName in ['.extraMask','.onlyExtraMask','.autoLower','.autoLower2']:
        pl.figure(2)
    pl.clf()
    fig = pl.gcf()
    if casaVersion >= '5.9': 
        fig.set_size_inches(8, 6, forward=True)

    rows = 1
    cols = 1
    ax1 = pl.subplot(rows,cols,1)
    if replaceNans:
        avgspectrumAboveThreshold = avgSpectrumNansReplaced
    else:
        avgspectrumAboveThreshold = avgSpectrumNansRemoved
    # I have co-opted the edgesUsed field in the meanSpectrum text file to
    # hold the numberOfPixelsInMask for the mom0mom8jointMask method, so its 
    # value will be unpredictable, so we test for the method name too.
    if (edgesUsed == 2 or edgesUsed is None or 
        meanSpectrumMethod == 'mom0mom8jointMask'):
        pl.plot(list(range(len(avgspectrumAboveThreshold))), 
                avgspectrumAboveThreshold, 'r-')
        drawYlabel(img, typeOfMeanSpectrum, meanSpectrumMethod, meanSpectrumThreshold,
                   peakFilterFWHM, fontsize, mask, useThresholdWithMask, normalized)
    elif (edgesUsed == 0):
        # The upper edge is not used and can have an upward spike
        # so don't show it.
        casalogPost("Not showing final %d channels of %d" % (skipchan,\
               len(avgspectrumAboveThreshold)))
        pl.plot(list(range(len(avgspectrumAboveThreshold)-skipchan)), 
                avgspectrumAboveThreshold[:-skipchan], 'r-')
        drawYlabel(img, typeOfMeanSpectrum,meanSpectrumMethod, meanSpectrumThreshold, 
                   peakFilterFWHM, fontsize, mask, useThresholdWithMask)
    elif (edgesUsed == 1):
        # The lower edge channels are not used and the threshold mean can 
        # have an upward spike, so don't show the first channel inward from 
        # there.
        casalogPost("Not showing first %d channels of %d" % (skipchan,\
                                   len(avgspectrumAboveThreshold)))
        pl.plot(list(range(skipchan, len(avgspectrumAboveThreshold))),
                avgspectrumAboveThreshold[skipchan:], 'r-')
        drawYlabel(img, typeOfMeanSpectrum,meanSpectrumMethod, meanSpectrumThreshold, 
                   peakFilterFWHM, fontsize, mask, useThresholdWithMask)

    highDynamicRange = setYLimitsAvoidingEdgeChannels(avgspectrumAboveThreshold, mad)  # this seems to never be used (2026-06-01)
    if (baselineModeA == 'edge'):
        nEdgeChannels = nBaselineChannels/2
        if (edgesUsed == 0 or edgesUsed == 2):
            pl.plot(list(range(nEdgeChannels)), avgspectrumAboveThreshold[:nEdgeChannels], 'm-', lw=3)
        if (edgesUsed == 1 or edgesUsed == 2):
            pl.plot(list(range(nchan-nEdgeChannels, nchan)), avgspectrumAboveThreshold[-nEdgeChannels:], 'm-', lw=3)
    if plotBaselinePoints:
        pl.plot(allBaselineChannelsXY[0], allBaselineChannelsXY[1], 'b.', ms=3, mec='b')
    if plotQuadraticPoints and rmStatContQuadratic:
        pl.plot(channelsFit, avgspectrumAboveThreshold[channelsFit], 'g.', ms=3, mec='g')
    if len(selection.split(';')) > MAX_RANGES_FOR_IMMOMENTS and casaVersion < '6.5.3':
        selection = cutSomeNarrowRanges(selection) # protect against CAS-13894
    channelSelections = selection.split(separator)
    casalogPost('Drawing positive threshold at %g' % (threshold))
    pl.plot(pl.xlim(), [threshold,threshold], 'k:')
    ylims = pl.ylim()
    pl.ylim([ylims[0], np.max([ylims[1],threshold])])
    if idxNonZeroMAD is not None:
        # prevent channel labels from overwriting top legend
        ylims = pl.ylim()
        pl.ylim([ylims[0], ylims[1] + 0.1*(ylims[1]-ylims[0])])
    if (negativeThreshold is not None):
        pl.plot(pl.xlim(), [negativeThreshold,negativeThreshold], 'k:')
        casalogPost('Drawing negative threshold at %g' % (negativeThreshold))
        ylims = pl.ylim()
        pl.ylim([np.min([ylims[0],negativeThreshold]), ylims[1]])
    casalogPost('Drawing observed median as dashed line at %g' % (median))
    pl.plot(pl.xlim(), [median,median], 'b--')  # observed median (always lower than true for mode='min')
    casalogPost('Drawing inferredMedian as solid line at %g' % (medianTrue))
    pl.plot(pl.xlim(), [medianTrue,medianTrue], 'k-')
    channelList = convertSelectionIntoChannelList(selection)
    cyanLevel = np.median(avgspectrumAboveThreshold[channelList])
    casalogPost('Drawing median of selection ranges as dotted cyan line at %g' % (cyanLevel))
    pl.plot(pl.xlim(), [cyanLevel,cyanLevel], 'c:')

    if (baselineModeB == 'edge'):
        pl.plot([nEdgeChannels, nEdgeChannels], pl.ylim(), 'k:')
        pl.plot([nchan-nEdgeChannels, nchan-nEdgeChannels], pl.ylim(), 'k:')
    if (len(continuumChannels) > 0):
        labelDescs, xyChanRange0 = plotChannelSelections(ax1, selection, separator, 
                           avgspectrumAboveThreshold, skipchan, medianTrue, 
                           threshold, secondSelectionAdded)
    if (fitsTable or img==''):
        img = meanSpectrumFile
        freqType = ''
    else:
        freqType = getFreqType(img)
    if (titleText == ''):
        narrowString = pickNarrowString(narrow, len(avgSpectrumNansReplaced), narrowValueModified) 
        trimString = pickTrimString(len(avgSpectrumNansReplaced), trimChannels, len(avgSpectrumNansReplaced), maxTrim, selection)
        if len(projectCode) > 0: 
            projectCode += ', '
        titleText = projectCode + os.path.basename(img) + ' ' + transition
    ylim = ExpandYLimitsForLegend()
    xlim = [0,nchan-1]
    pl.xlim(xlim)
    titlesize = np.min([fontsize,int(np.floor(fontsize*100.0/len(titleText)))])
    if tdmSpectrum(channelWidth,nchan):
        dm = 'TDM'
    else:
        dm = 'FDM'
    if (spw != ''):
        label = '(Spw %s) %s Channels (%d)' % (str(spw), dm, nchan)
        if singleContinuum:
            label = 'SingleCont ' + label
    else:
        label = '%s Channels (%d)' % (dm,nchan)
        if singleContinuum:
            label = '(SingleContinuum) ' + label
    ax1.set_xlabel(label, size=fontsize)
    if nchan < 300:
        ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
    elif nchan < 750:
        ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(25))
    elif nchan < 1500:
        ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))
    elif nchan < 3000:
        ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100))
    else:
        ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(200))
    if casaVersion >= '5.9.9':
        titleYoffset = 1.10
    else:
        titleYoffset = 1.08
    pl.text(0.5, titleYoffset, titleText, size=titlesize, ha='center', transform=ax1.transAxes)
    pl.ylim(ylim)
#  The following line seems to have zero effect.
#    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
    ax2 = ax1.twiny()
    pl.setp(ax1.get_xticklabels(), fontsize=fontsize)
    pl.setp(ax1.get_yticklabels(), fontsize=fontsize)
    pl.setp(ax2.get_xticklabels(), fontsize=fontsize)
    ax2.set_xlim(firstFreq*1e-9,lastFreq*1e-9)
    freqRange = np.abs(lastFreq-firstFreq)
    power = int(np.log10(freqRange))-9
    ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10**power))
    numberOfTicks = 0
    visibleTicks = []
    for mytick in ax2.get_xticks():
        if mytick*1e9 >= firstFreq and mytick*1e9 <= lastFreq:
            numberOfTicks += 1
            visibleTicks.append(float(mytick))
#    numberOfTicks = len(ax2.get_xticks()) # this is not reliable
    print("Setting major locator to %f GHz, yielding %d visible major ticks: %s" % (10**power, numberOfTicks, str(visibleTicks)))
    if (numberOfTicks < 2):
        ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5*10**power))
        numberOfTicks = 0
        visibleTicks = []
        for mytick in ax2.get_xticks():
            if mytick*1e9 >= firstFreq and mytick*1e9 <= lastFreq:
                numberOfTicks += 1
                visibleTicks.append(float(mytick))
        print("Setting major locator to %f GHz to get %d visible major ticks: %s" % (0.5*10**power, numberOfTicks, str(visibleTicks)))
    ax2.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1*10**power))
    ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    aggregateBandwidth = computeBandwidth(selection, channelWidth, 1)
    if (channelWidth > 0):
        channelWidthString = ', chanwidth: %.2f kHz, BW: %g MHz, contBW: %.2f MHz' % (channelWidth*1e-3, spwBandwidth*1e-6, aggregateBandwidth*1000)
        if nbin >= NBIN_THRESHOLD:
            channelWidthString += ', iPoM=%.1f, nbin=%d' % (initialPeakOverMad, nbin)
    else:
        channelWidthString = ''
#    if (spw != ''):
#        upperXlabel = r'$\bf{%s\/Freq.\/(GHz)}$' % (str(spw), freqType) + channelWidthString
#    else:
    upperXlabel = r'$\bf{%s\/Freq.\/(GHz)}$' % freqType + channelWidthString
    ax2.set_xlabel(upperXlabel, size=fontsize) # this artist cannot be removed in CASA 5.6
    inc = 0.03
    i = 1
    if (meanSpectrumMethod == 'mom0mom8jointMask' and rmStatContQuadratic):
        bottomLegend = 'rm quad: %d/%d, ' % (len(channelsFit), len(avgSpectrumNansReplaced))
    elif initialQuadraticRemoved:
        bottomLegend = "rm quad: mad improve: %.1f>%.1f, " % (initialQuadraticImprovementRatio,initialQuadraticImprovementThreshold)
    else:
        bottomLegend = ''
    if madRatio is not None:
#        print("types: ",type(peakOverMad),type(peakMinusMedianOverMad),type(mom0peakOverMad),type(mom8peakOverMad),type(signalRatio),type(madRatio))
        bottomLegend += "peak/MAD: %.2f, %.2f, %.2f, %.2f, signalRatio: %.3f, madRatio: %.3f" % (peakOverMad,peakMinusMedianOverMad,mom0peakOverMad,mom8peakOverMad,signalRatio,madRatio)
    elif useMiddleChannels:
        bottomLegend += "peak/MAD: %.2f, %.2f, %.2f, %.2f, signalRatio: %.3f, middle chans used" % (peakOverMad,peakMinusMedianOverMad,mom0peakOverMad,mom8peakOverMad,signalRatio)
    else:
        bottomLegend += "peak/MAD: %.2f, %.2f, %.2f, %.2f, signalRatio: %.3f, madRatio not computed" % (peakOverMad,peakMinusMedianOverMad,mom0peakOverMad,mom8peakOverMad,signalRatio)
    if meanSpectrumMethod == 'mom0mom8jointMask':
        bottomLegend += ', pixMom8=%d' % (numberPixelsInMom8Mask)
    if bottomLegend != '':
        pl.text(0.5,0.01,bottomLegend, ha='center', size=fontsize-2, 
                transform=ax1.transAxes)
    effectiveSigma = sigmaFindContinuum*correctionFactor
    if meanSpectrumMethod.find('mean') >= 0:
        pl.text(0.5,0.99-i*inc,'bl=(%s,%s), sCube=%.1f, sigmaEff=%.2f*%.2f=%.2f, trim=%s' % (baselineModeA,baselineModeB,sigmaCube,sigmaFindContinuum,correctionFactor,effectiveSigma,trimString),transform=ax1.transAxes, ha='center',size=fontsize)
    else:
        pl.text(
            0.5, 0.99-i*inc, ' baselineModeB=%s, sFC=%.2f*%.2f=%.2f, trim=%s, sdrBW=%s, maxBase=%.0fm' %
            (baselineModeB, sigmaFindContinuum, correctionFactor, effectiveSigma, trimString, spectralDynamicRangeString, maxBaseline),
            transform=ax1.transAxes, ha='center', size=fontsize-2)
    i += 1
    peak = np.max(avgSpectrumNansReplaced)
    peakFeatureSigma = (peak-medianTrue)/mad
    maxPeakFeatureSigma = (peak - np.min(avgSpectrumNansReplaced))/MAD(avgSpectrumNansReplaced)
#    casalogPost("initial peakFeatureSigma = (%f-%f)/%f = %f, maxPeakFeatureSigma=%f" % (peak,medianTrue,mad,peakFeatureSigma,maxPeakFeatureSigma))
    if (signalRatio > signalRatioTier1 or (signalRatio > signalRatioTier2 and peakOverMad<5)) and peakFeatureSigma > maxPeakFeatureSigma:
        mad *= peakFeatureSigma/maxPeakFeatureSigma
        casalogPost('Limiting peakFeatureSigma from %f to %f, reset MAD to %f' % (peakFeatureSigma,maxPeakFeatureSigma,mad))
        peakFeatureSigma = maxPeakFeatureSigma
        areaString = 'limited max: %.1f*mad=%.4f; %d ranges; ' % (peakFeatureSigma, peak, len(channelSelections))
    else:
        casalogPost("maxPeakFeatureSigma=%f" % (maxPeakFeatureSigma))
        areaString = 'max: %.1f*mad=%.4f; %d ranges; ' % (peakFeatureSigma, peak, len(channelSelections))
    if (fullLegend):
        pl.text(0.5,0.99-i*inc,'rms=MAD*1.4826: of baseline chans = %f, scaled by %.1f for all chans = %f'%(mad,correctionFactor,mad*correctionFactor), 
                transform=ax1.transAxes, ha='center',size=fontsize)
        i += 1
        pl.text(0.017,0.99-i*inc,'lineStrength factor: %.2f' % (lineStrengthFactor), transform=ax1.transAxes, ha='left', size=fontsize)
        pl.text(0.983,0.99-i*inc,'MAD*1.4826: of points below upper dotted line = %f' % (madOfPointsBelowThreshold),
                transform=ax1.transAxes, ha='right', size=fontsize)
        i += 1
        pl.text(0.5,0.99-i*inc,'median: of %d baseline chans = %f, offset by %.1f*MAD for all chans = %f'%(nBaselineChannels, median,medianCorrectionFactor,medianTrue-median), 
                transform=ax1.transAxes, ha='center', size=fontsize)
        i += 1
        pl.text(0.5,0.99-i*inc,'chans>median: %d (sum=%.4f), chans<median: %d (sum=%.4f), ratio: %.2f (%.2f)'%(channelsAboveMedian,sumAboveMedian,channelsBelowMedian,sumBelowMedian,channelRatio,sumRatio),
                transform=ax1.transAxes, ha='center', size=fontsize-1)
    if (negativeThreshold is not None):
        pl.text(0.5,0.99-i*inc,'narrow=%s, mad: %.3g; levs: %.3g, %.3g (dot); median: %.3g (solid), medmin: %.3g (dash) cyan: %.3g'%(narrowString, mad, threshold,negativeThreshold,medianTrue,median,cyanLevel), transform=ax1.transAxes, ha='center', size=fontsize-3)
    else:
        pl.text(0.5,0.99-i*inc,'narrow=%s, mad: %.3g; threshold: %.3g (dot); median: %.3g (solid), medmin: %.3g (dash) cyan: %.3g'%(narrowString,mad,threshold,medianTrue,median,cyanLevel), 
                transform=ax1.transAxes, ha='center', size=fontsize-3)
    i += 1
    if meanSpectrumMethod == 'mom0mom8jointMask':
        if pbBasedMask:
            areaString += 'pixels in pb-based mask: %g' % (numberPixelsInJointMask)
            casalogPost('%s pixels in pb-based mask: %g' % (projectCode, numberPixelsInJointMask))
        else:
            if regionsPruned > 0:
                # some regions were pruned
                areaString += 'pixels in pruned mask: %g' % (numberPixelsInJointMask)
            else:
                areaString += 'pixels in joint mask: %g' % (numberPixelsInJointMask)
            casalogPost('%s pixels in joint mask: %g' % (projectCode, numberPixelsInJointMask))
    elif (centralArcsec == 'auto'):
        areaString += 'mean over area: (unknown)'
    elif (centralArcsec < 0):
        areaString += 'mean over area: whole field (%.2fMpix)' % (megapixels)
    else:
        areaString += 'mean over: central box of radius %.1f" (%.2fMpix)' % (centralArcsec,megapixels)
    labelDesc = pl.text(0.5,0.99-i*inc,areaString, transform=ax1.transAxes, ha='center', size=fontsize+difSNRFontsizeAdjust)
    labelDescs.append(labelDesc)
    if (meanSpectrumMethodMessage != ''):
        if casaVersion >= '5.9.9':
            msmm_ylabel = -0.12
        else:
            msmm_ylabel = -0.10
        pl.text(0.5,msmm_ylabel,meanSpectrumMethodMessage, 
                transform=ax1.transAxes, ha='center', size=fontsize)
        
    finalLine = ''
    if (len(mask) > 0):
        if (percentagePixelsNotMasked > 0):
            finalLine += 'mask=%s (%.2f%% pixels)' % (os.path.basename(mask), percentagePixelsNotMasked)
        else:
            finalLine += 'mask=%s' % (os.path.basename(mask))
        i += 1
        pl.text(0.5, 0.99-i*inc, finalLine, transform=ax1.transAxes, ha='center', size=fontsize-1)
        finalLine = ''
    if (slope is not None):
        if discardSlopeResult:
            discarded = ' (result discarded)'
        elif slopeRemoved:
            discarded = ' (removed)'
        else:
            discarded = ' (not removed)'
        if quadraticFitTest:
            finalLine += 'quadratic fit: %g*(x-%g)**2+%g*(x-%g)+%g %s' % (roundFigures(order2,3),roundFigures(xoffset,4),roundFigures(slope,3),roundFigures(xoffset,4),roundFigures(intercept,3),discarded)
        else:
            finalLine += 'linear slope: %g %s' % (roundFigures(slope,3),discarded)
    i += 1
#   This is the line to use to put the legend in the top group of lines.
#    pl.text(0.5, 0.99-i*inc, finalLine, transform=ax1.transAxes, ha='center', size=fontsize)
    pl.text(0.5, 0.04, finalLine, transform=ax1.transAxes, 
            ha='center', size=fontsize)

    gigabytes = getMemorySize()/(1024.**3)
    if not regressionTest:
        # Write CVS version to plot legend
        pl.text(1.06, -0.005-2*inc, ' '.join(version().split()[1:4]), size=8, 
                transform=ax1.transAxes, ha='right')
        # Write CASA version to plot legend
        casaText = "CASA "+casaVersion+' (%.0f GB'%gigabytes
    else:
        casaText = '(%.0f GB' % gigabytes
    byteLimit = byteLimit/(1024.**3)
    if maxMemory < 0 or byteLimit == gigabytes:
        casaText += ')'
    else:
        casaText += ', but limited to %.0fGB)' % (byteLimit)
    pl.text(-0.03, -0.005-2*inc, casaText, size=8, transform=ax1.transAxes, ha='left')
    if (plotAtmosphere != '' and img != meanSpectrumFile):
        if (plotAtmosphere == 'tsky'):
            value = 'tsky'
        else:
            value = 'transmission'
        freqs, atm = CalcAtmTransmissionForImage(img, imageInfo, chanInfo, airmass, pwv, value=value, vis=vis, source=source, spw=spw)
        casalogPost("freqs: min=%g, max=%g n=%d" % (np.min(freqs),np.max(freqs), len(freqs)))
        atmRange = 0.5  # how much of the y-axis should it take up
        yrange = ylim[1]-ylim[0]
        if (value == 'transmission'):
            atmRescaled = ylim[0] + 0.3*yrange + atm*atmRange*yrange
            print("atmRescaled: min=%f, max=%f" % (np.min(atmRescaled), np.max(atmRescaled)))
            pl.text(1.015, 0.3, '0%', color='m', transform=ax1.transAxes, ha='left', va='center')
            pl.text(1.015, 0.8, '100%', color='m', transform=ax1.transAxes, ha='left', va='center')
            pl.text(1.03, 0.55, 'Atmospheric Transmission', color='m', ha='center', va='center', 
                    rotation=90, transform=ax1.transAxes, size=11)
            for yt in np.arange(0.3, 0.81, 0.1):
                xtickTrans,ytickTrans = np.array([[1,1.01],[yt,yt]])
                line = matplotlib.lines.Line2D(xtickTrans,ytickTrans,color='m',transform=ax2.transAxes)
                ax1.add_line(line)
                line.set_clip_on(False)
        else:
            atmRescaled = ylim[0] + 0.3*yrange + atm*atmRange*yrange/300.
            pl.text(1.015, 0.3, '0', color='m', transform=ax1.transAxes, ha='left', va='center')
            pl.text(1.015, 0.8, '300', color='m', transform=ax1.transAxes, ha='left', va='center')
            pl.text(1.03, 0.55, 'Sky temperature (K)', color='m', ha='center', va='center', 
                    rotation=90, transform=ax1.transAxes, size=11)
            for yt in np.arange(0.3, 0.81, 0.1):
                xtickTrans,ytickTrans = np.array([[1,1.01],[yt,yt]])
                line = matplotlib.lines.Line2D(xtickTrans,ytickTrans,color='m',transform=ax2.transAxes)
                ax1.add_line(line)
                line.set_clip_on(False)
        pl.text(1.06, 0.55, '(%.1f mm PWV, 1.5 airmass)'%pwv, color='m', ha='center', va='center', 
                rotation=90, transform=ax1.transAxes, size=11)
#        pl.plot(freqs, atmRescaled, 'w.', ms=0)  # need this to finalize the ylim value
        ylim = pl.ylim()
        yrange = ylim[1]-ylim[0]
        if (value == 'transmission'):
            atmRescaled = ylim[0] + 0.3*yrange + atm*atmRange*yrange
        else:
            atmRescaled = ylim[0] + 0.3*yrange + atm*atmRange*yrange/300.
        pl.plot(freqs, atmRescaled, 'm-')
        pl.ylim(ylim)  # restore the prior value (needed for CASA 5.0)

    pl.draw()
    if pl.get_backend() == 'TkAgg' and casaVersion >= '5.9':
        fig.canvas.flush_events() # critical for TkAgg in CASA 6, or else set_fig_size never takes effect!

    if (png == ''):
        if pngBasename:
            png = os.path.basename(img) + amendMaskIterationName
        else:
            png = img + amendMaskIterationName
        if outdir != '':
            png = os.path.join(outdir,os.path.basename(png))
        transition = transition.replace('(','_').replace(')','_').replace(' ','_').replace(',','')
        if (len(transition) > 0):
            transition = '.' + transition
        narrowString = pickNarrowString(narrow, len(avgSpectrumNansReplaced), narrowValueModified) # this is used later
        trimString = pickTrimString(len(avgSpectrumNansReplaced), trimChannels, len(avgSpectrumNansReplaced), maxTrim, selection)
        if nbin >= NBIN_THRESHOLD:
            png += '.nbin%d' % (nbin)
        if userJointMask == '':
            png += '.meanSpectrum.%s.%s.%s.%.2fsigma.narrow%s.trim%s%s' % (meanSpectrumMethod, baselineModeA, baselineModeB, sigmaFindContinuum, narrowString, trimString, transition)
        elif userJointMask.find('.amendedJointMask') > 0:
            png += '.meanSpectrum.amendedJointMask.%s.%s.%.2fsigma.narrow%s.trim%s%s' % (baselineModeA, baselineModeB, sigmaFindContinuum, narrowString, trimString, transition)
        else:
            png += '.meanSpectrum.userJointMask.%s.%s.%.2fsigma.narrow%s.trim%s%s' % (baselineModeA, baselineModeB, sigmaFindContinuum, narrowString, trimString, transition)
        if maxMemory > 0:
            png += '.%.0fGB' % (maxMemory)
        if overwrite:
            png += '.overwriteTrue.png'
        else: # change the following to '.png' after my test of July 20, 2016
            png += '.png'
    pngdir = os.path.dirname(png)
    if (len(pngdir) < 1):
        pngdir = '.'
    if (not os.access(pngdir, os.W_OK) and pngdir != '.'):
        casalogPost("No permission to write to specified directory: %s. Will try alternateDirectory." % pngdir)
        if (len(alternateDirectory) < 1):
            alternateDirectory = '.'
        png = alternateDirectory + '/' + os.path.basename(png)
        pngdir = alternateDirectory
        print("png = ", png)
    if (not os.access(pngdir, os.W_OK)):
        casalogPost("No permission to write to alternateDirectory. Will not save the plot.")
    else:
        pl.savefig(png, dpi=dpi)
        casalogPost("Wrote png = %s" % (png))
    donetime = timeUtilities.time()
    casalogPost("%.1f sec elapsed in runFindContinuum" % (donetime-startTime))
    baselineMAD = mad
    if amendMaskIterationName in ['.extraMask','.onlyExtraMask','.autoLower','.autoLower2']:
        # go back to original figure now
        pl.close(2)
        pl.figure(1)

    return(selection, png, slope, channelWidth, nchan, useLowBaseline, mom0snrs, mom8snrs, useMiddleChannels, 
           selectionPreBluePruning, sigmaFindContinuum, jointMask, avgspectrumAboveThreshold, medianTrue, 
           labelDescs, ax1, ax2, threshold, areaString, RangesDropped, effectiveSigma, baselineMAD, upperXlabel,
           allBaselineChannelsXY, nbin, initialPeakOverMad, negativeThreshold, xyChanRange0, idxNonZeroMAD)
# end of runFindContinuum    
#                    [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad] = result

def invertChannelRanges(invertstring, nchan=0, startchan=0, vis='', spw='', 
                        separator=';'):
    """
    Takes a CASA channel selection string and inverts it.
    -Todd Hunter
    """
    myspw = spw
    checkForMultipleSpw = invertstring.split(',')
    mystring = ''
    if (vis != ''):
        mymsmd = msmdtool()
        mymsmd.open(vis)
    for c in range(len(checkForMultipleSpw)):
      checkspw = checkForMultipleSpw[c].split(':')
      if (len(checkspw) > 1 and spw==''):
          spw = checkspw[0]
          checkForMultipleSpw[c] = checkspw[1]
      if (vis != ''):
          nchan = mymsmd.nchan(int(spw))
#          print "Setting nchan = %d for spw %s" % (nchan,str(spw))
      goodstring = checkForMultipleSpw[c]
      goodranges = goodstring.split(separator)
      if (str(spw) != ''):
          mystring += str(spw) + ':'
      if (int(goodranges[0].split('~')[0]) > startchan):
          mystring += '%d~%d%s' % (startchan,int(goodranges[0].split('~')[0])-1, separator)
      multiWindows = False
      for g in range(len(goodranges)-1):
          r = goodranges[g]
          s = goodranges[g+1]
          mystring += '%d~%d' % (int(r.split('~')[1])+1, int(s.split('~')[0])-1)
          multiWindows = True
          if (g < len(goodranges)-2):
              mystring += separator
      if (int(goodranges[-1].split('~')[-1]) < startchan+nchan-1):
          if (multiWindows):
              mystring += separator
          mystring += '%d~%d' % (int(goodranges[-1].split('~')[-1])+1, startchan+nchan-1)
      if (c < len(checkForMultipleSpw)-1):
          mystring += ','
      spw = myspw  # reset to the initial value
    if (vis != ''):
        mymsmd.close()
    return(mystring)

def plotChannelSelections(ax1, selection, separator, avgspectrumAboveThreshold, 
                          skipchan, medianTrue, 
                          threshold, secondSelectionAdded='', lineColor='c'):
    """
    avgspectrumAboveThreshold: computed in runFindContinuum
    medianTrue, threshold: computed by findContinuumChannels
    Returns labelDescs: list of plot artists that may be removed and replaced later
    """
    labelDescs = []
    channelSelections = selection.split(separator)
#    ylevel = np.mean(avgspectrumAboveThreshold) # Cycle 0-5 heuristic
    peak = np.max(avgspectrumAboveThreshold[skipchan:-skipchan])
    if (peak-medianTrue) > 1.5*(threshold-medianTrue):
        ylevel = threshold + (threshold-medianTrue)*np.log10((peak-medianTrue)/(threshold-medianTrue))
        print("Setting cyan level (%f) to just above the positive threshold dotted line" % (ylevel))
    else:
        ylevel = threshold - 0.5*(threshold-medianTrue)
        print("Setting cyan level to half the positive threshold dotted line")
    yoffset = ylevel + 0.04*(pl.ylim()[1]-pl.ylim()[0])
    for i,ccstring in enumerate(channelSelections): 
        cc = [int(j) for j in ccstring.split('~')]
        # length of cc will always be at least 2
        mywidth = cc[1]-cc[0]  # can be 0 or larger
        if (mywidth < 2): 
#            print("Widening cyan line by %d: " % (2-mywidth))
            cc[0] = cc[0] - (2-mywidth)*0.5
            cc[1] = cc[1] + (2-mywidth)*0.5
            mylw = 1.33
        else:
            mylw = 2
        labelDesc = ax1.plot(cc, np.ones(len(cc))*ylevel, '-', color=lineColor, lw=mylw)
        labelDescs.append(labelDesc)
        mystring = ccstring
        if i==1 and secondSelectionAdded != '':
            # indicate if the heuristic for CAS-11720 was activated to ensure a second region
            mystring += ' added'
        labelDesc = ax1.text(np.mean(cc), yoffset, mystring, va='bottom', ha='center',size=8,rotation=90)
        labelDescs.append(labelDesc)
    if len(labelDescs) > 1:
        tcbox = pl.gca().transData.inverted().transform(labelDescs[1].get_window_extent())
        xyChanRange0 = tcbox # np.mean(tcbox, axis=0)
    else:
        xyChanRange0 = None
    return labelDescs, xyChanRange0

def findWidestContiguousListInOtherHalf(channels, lowerHalf, nchan):
    """
    Given a list of channels, find the longest contiguous list in the first half or second half
    example:
    CASA <43>: fc.findWidestContiguousListInOtherHalf([2,3,7,8,9,12,13,14,15], True, 20)
    Out[43]: [12, 13, 14, 15]
    channels: list of blue points channel numbers
    lowerHalf: a Boolean (True for lower half, False for upper half)
    nchan: total channels in spectrum
    """
    contiguousLists = splitListIntoContiguousLists(sorted(channels))
    mylengths = []
    widestList = None
    widestListLength = 0
    for j,contiguousList in enumerate(contiguousLists):
        mylengths.append(0)
        if (lowerHalf and contiguousList[0] >= nchan/2) or (not lowerHalf and contiguousList[0] < nchan/2):
            mylengths[j] = len(contiguousList)
        if mylengths[j] > widestListLength:
            widestListLength = mylengths[j]
            widestList = contiguousList
    return widestList

def findChannelsInChannelRangeWithComparableIntensity(channels, channelRange, continuumChannels,
                                                      intensity, withinSigma=1, negativeThreshold=None, positiveThreshold=None):
    """
    Finds channels in a portion of the spectrum (intensity) whose intensity is 
    within the range of intensities of the blue points (channels) by some sigma
    beyond the min and max.
    channels: list of blue points channel numbers
    intensity: intensity of whole spectrum (an array)
    channelRange: range of channels to look for widest set of blue points [first,final]
    continuumChannels: current selection of continuuum channels
    withinSigma: fix for PIPE-2188
    negativeThreshold: value to use instead of the min of intensity of the blue points: PIPE-3103
    positiveThreshold: value to use instead of the max of intensity of the blue points: PIPE-3103
    Returns: a list of channels, possibly non-contiguous, sorted low to high
    """
    if negativeThreshold is None:
        imin = np.min(intensity[channels])
    else:
        imin = negativeThreshold
    if positiveThreshold is None:
        imax = np.max(intensity[channels])
    else:
        imax = positiveThreshold
    newList = []
#    sigma = np.std(intensity[channels])  fc <= v7.16
    sigma = MAD(intensity[channels])   # fc v7.17
    for channel in range(int(channelRange[0]), int(channelRange[1])):
        if (imin-withinSigma*sigma) < intensity[channel] < (imax+withinSigma*sigma):
            distance = np.min([imax-intensity[channel], intensity[channel]-imin])/sigma
            casalogPost('Channel %d is within %.1f scaled MAD of nearest blue point intensity' % (channel,distance))
            newList.append(channel)
#    newLists = splitListIntoContiguousLists(sorted(newList))
    return(sorted(newList))
    
def findWidestContiguousListInChannelRange(channels, channelRange, continuumChannels):
    """
    Given a list of channels, find the longest contiguous list in the first half or second half
    example:
    CASA <43>: fc.findWidestContiguousListInChannelRange([2,3,7,8,9,12,13,14,15], [10,20], [7,8,9])
    Out[43]: [12, 13, 14, 15]
    channels: list of blue points channel numbers
    channelRange: range of channels to look for widest set of blue points
    continuumChannels: current selection of continuuum channels
    Returns: a list of contiguous channels
    """
    contiguousLists = splitListIntoContiguousLists(sorted(channels))
    currentRange = continuumChannels[-1]-continuumChannels[0] + 1
    mylengths = []
    widestList = None
    widestListLength = 0
    startchan, endchan = channelRange
    for j,contiguousList in enumerate(contiguousLists):
        mylengths.append(0)
        if (contiguousList[0] >= startchan) and (contiguousList[-1] < endchan):
            spreadFactor = np.max([contiguousList[-1]-continuumChannels[0]+1,continuumChannels[-1]-contiguousList[0]+1])/currentRange
            mylengths[j] = len(contiguousList) * spreadFactor
            mylist = [int(i) for i in contiguousList] # convert int64 to int for shorter messages
            casalogPost('Possible region: %s, %d*%f = %f' % (mylist, len(mylist), spreadFactor, mylengths[j]))
        if mylengths[j] > widestListLength:
            widestListLength = mylengths[j]
            widestList = contiguousList
    return widestList

def aboveBelow(avgSpectrumNansReplaced, threshold):
    """
    This function is called by runFindContinuum.
    Given an array of values (i.e. a spectrum) and a threshold value, this 
    function computes and returns 6 items: 
    * the number of channels above that threshold (group 1)
    * the number of channels below that threshold (group 2)
    * the ratio of these numbers (i.e., #ChanAboveThreshold / #ChanBelowThreshold))
    * sum of the intensities in group (1)    
    * sum of the intensities in group (2)
    * the ratio of these sums  (i.e., FluxAboveThreshold / FluxBelowThreshold)
      (This value is not currently used by the calling function.)
      Example with 24 channels where 3 have positive spikes, and rest are zero:
                       3      2      3        channelsAbove = 3
     +threshold=+1 ........................   sumAbove = (3-1) + (2-1) + (3-1) = 5    
           zero ---------------------------   channelsBelow = 21
     -threshold=-1 ........................   sumBelow = 21*(1-0) = 21
    """
    y = avgSpectrumNansReplaced
    aboveMedian = np.where(avgSpectrumNansReplaced > threshold)
    belowMedian = np.where(avgSpectrumNansReplaced < threshold)
    sumAboveMedian = np.sum(-threshold+avgSpectrumNansReplaced[aboveMedian])
    sumBelowMedian = np.sum(threshold-avgSpectrumNansReplaced[belowMedian])
    channelsAboveMedian = len(aboveMedian[0])
    channelsBelowMedian = len(belowMedian[0])
    channelRatio = channelsAboveMedian*1.0/channelsBelowMedian
    sumRatio = sumAboveMedian/sumBelowMedian  # this value is unused by the calling function
    return(sumAboveMedian, sumBelowMedian, sumRatio, 
           channelsAboveMedian, channelsBelowMedian, channelRatio)

def writeMeanSpectrum(meanSpectrumFile, frequency, avgspectrum, 
                      avgSpectrumNansReplaced, threshold, nchan, edgesUsed=0,
                      nanmin=0, centralArcsec='auto', mask='', iteration=0):
    """
    This function is called by meanSpectrum.
    Writes out the mean spectrum (and the key parameters used to create it), 
    so that it can quickly be restored.  This allows the manual user to quickly 
    experiment with different parameters of findContinuumChannels applied to 
    the same mean spectrum.
    Units are Hz and Jy/beam.
    meanSpectrumFile: name of file to create
    threshold: this is interpreted as mom0threshold for meanSpectrumMethod='mom0mom8jointMask'
    edgesUsed: this is interpreted as numberPixelsInMask for meanSpectrumMethod='mom0mom8jointMask'
    nanmin:    this is interpreted as mom8threshold for meanSpectrumMethod='mom0mom8jointMask'
    frequency: list or array of frequencies
    Returns: None
    """
    if len(meanSpectrumFile) == 0:
        print("WARNING: blank name requested for meanSpectrumFile!!")
        return
    f = open(meanSpectrumFile, 'w')
    if centralArcsec == 'mom0mom8jointMask':
        field1 = 'mom0threshold'
        field2 = 'numberPixelsInMask'
        field4 = 'mom8threshold'
    else:
        field1 = 'threshold'
        field2 = 'edgesUsed'
        field4 = 'nanmin'
    if (iteration == 0):
        f.write('#%s %s nchan %s centralArcsec=%s %s\n' % (field1,field2,field4,str(centralArcsec),mask))
    else:
        f.write('#threshold edgesUsed nchan nanmin centralArcsec=auto %s %s\n' % (str(centralArcsec),mask))
    #                                field1      field2           field4
    f.write('%.10f %g %g %.10f\n' % (threshold, edgesUsed, nchan, nanmin))
    f.write('#chan freq(Hz) avgSpectrum avgSpectrumNansReplaced\n')
    for i in range(len(avgspectrum)):
        f.write('%d %.1f %.10f %.10f\n' % (i, frequency[i], avgspectrum[i], avgSpectrumNansReplaced[i]))
    casalogPost('Wrote %s' % meanSpectrumFile, debug=True)
    f.close()

def findContinuumChannels(spectrum, nBaselineChannels=16, sigmaFindContinuum=3, 
                          nanmin=None, baselineMode='min', trimChannels='auto',
                          narrow='auto', verbose=False, maxTrim=maxTrimDefault, 
                          maxTrimFraction=1.0, separator=';', peakOverMad=0, fitResult=None,
                          maxGroupsForMaxTrimAdjustment=3, lowHighBaselineThreshold=1.5,
                          lineSNRThreshold=20, negativeThresholdFactor=1.15, 
                          dropBaselineChannels=2.0, madRatioUpperLimit=1.5, 
                          madRatioLowerLimit=1.15, projectCode='', fCCiteration=0,
                          signalRatioTier1=0.965, signalRatioTier2=0.94, sigmaFindContinuumMode='auto', 
                          enableRejectNarrowInnerWindows=True, useUncorrectedMedian=False,
                          returnBluePoints=False, removeOutlierBaselineChannels=False,
                          amendMaskIterationName='', useBaseline='auto', idxNonZeroMAD=None):
    """
    This function is called by runFindContinuum.
    Trys to find continuum channels in a spectrum, based on a threshold or
    some number of edge channels and their median and standard deviation.
    Inputs:
    spectrum: a one-dimensional array of intensity values
    nBaselineChannels: number of channels over which to compute standard deviation/MAD
    sigmaFindContinuum: value to multiply the standard deviation by then add 
      to median to get threshold.  Default = 3.  
    narrow: the minimum number of channels in a contiguous block to accept 
            if 0<narrow<1, then it is interpreted as the fraction of all
                           channels within identified blocks
    nanmin: the value that NaNs were replaced by in previous steps
    baselineMode: 'min' or 'edge',  'edge' will use nBaselineChannels/2 from each edge
    trimChannels: if integer, use that number of channels.  If float between
      0 and 1, then use that fraction of channels in each contiguous list
      (rounding up). If 'auto', then use 0.1 but not more than maxTrim channels.
      The 'auto' mode can also cause trimChannels to be set to 13 or 6 in certain
      circumstances (when moderately strong lines are present).
    maxTrim: if trimChannels='auto', this is the max channels to trim per group for TDM spws; it is automatically scaled upward for FDM spws.
    maxTrimFraction: in trimChannels='auto', the max fraction of channels to trim per group
    separator: the character to use to separate groups of channels in the string returned
    negativeThresholdFactor: scale the nominal negative threshold by this factor (to adjust 
        sensitivity to absorption features: smaller values=more sensitive)
    dropBaselineChannels: percentage of extreme values to drop in baseline mode 'min'
    madRatioUpperLimit, madRatioLowerLimit: if ratio of MADs (MAD of all baseline channels / 
        MAD of baseline channels with extreme channels dropped) is between these values, then
        apply dropBaselineChannels when defining the MAD of the baseline range
    fitResult: coefficients from linear or quadratic fit, only relevant when meanSpectrumMethod is
             not 'mom0mom8jointMask'
    signalRatioTier1: threshold for signalRatio, above which we desensitize the level to
        consider line emission in order to avoid small differences in noise levels from 
        affecting the result (e.g. that occur between serial and parallel tclean cubes)
        signalRatio=1 means: no lines seen, while closer to zero: more lines seen
    signalRatioTier2: second threshold for signalRatio, used for FDM spws (nchan>192) and
        for cases of peakOverMad < 5.  Should be < signalRatioTier1.
    sigmaFindContinuumMode: if 'auto', then do not desensitize with signalRatioTier1
    removeOutlierBaselineChannels: after dropBaselineChannels, remove >4*scMAD 
        outliers;  options: 'high', 'low', 'both'/True, or '' which means none/False
    returnBluePoints: if True, then simply set the continuum ranges to the baseline points 
         (i.e. the blue points) instead of what the other heuristics might have found
    useBaseline: 'low', 'middle', 'high' or 'auto'; auto tries to automatically pick 'low' for
       spectra that are emission-line dominated, 'high' if absorption line dominated, or 'both' if
       both types of lines are present

    Returns:
    1  list of channels to use (separated by the specified separator)
    2  list converted to ms channel selection syntax
    3  positive threshold used
    4  median of the baseline-defining channels
    5  number of groups found
    6  correctionFactor used
    7  inferred true median
    8  scaled MAD of the baseline-defining channels
    9  correction factor applied to get the inferred true median
    10 negative threshold used
    11 lineStrength factor
    12 singleChannelPeaksAboveSFC: how many cases where only a single channel exceeds the threshold 
    13 allGroupsAboveSFC
    14 spectralDiff (percentage of median)
    15 value of trimChannels parameter
    16 Boolean describing whether the low values were used as the baseline
    17 value of the narrow parameter
    18 tuple containing the channel numbers of the baseline channels, and their respective y-axis values
    19 value of madRatio: MAD/MAD_after_dropping_extreme_channels
         This value will be larger if there are many tall single channel peaks.
    20 number of ranges it dropped (or would have dropped if it was enabled)
    """
    if (fitResult is not None):
        myx = np.arange(len(spectrum), dtype=np.float64)
        if (len(fitResult) > 2):
            myx -= fitResult[3]
            originalSpectrum = spectrum + (fitResult[0]*myx**2 + fitResult[1]*myx + fitResult[2]) - nanmean(spectrum)
            casalog.post("min/max spectrum = %f, %f" % (np.min(spectrum), np.max(spectrum)))
            casalog.post("min/max originalSpectrum = %f, %f" % (np.min(originalSpectrum), np.max(originalSpectrum)))
        else:
            originalSpectrum = spectrum + fitResult[0]*myx
    else:
        originalSpectrum = spectrum
    if (narrow == 'auto'):
        narrow = pickNarrow(len(spectrum))
        autoNarrow = True
    else:
        autoNarrow = False
    npts = len(spectrum)
    percentile = 100.0*nBaselineChannels/npts
    correctionFactor = sigmaCorrectionFactor(baselineMode, npts, percentile)
    sigmaEffective = sigmaFindContinuum*correctionFactor
    if (fitResult is not None):
        if (len(fitResult) > 2):
            casalogPost("****** starting findContinuumChannels (%d) (polynomial=%g*(x-%.2f)**2+%g*(x-%.2f)+%g) ***********" % (fCCiteration, fitResult[0], fitResult[3], fitResult[1], fitResult[3], fitResult[2]))
        else:
            casalogPost("****** starting findContinuumChannels (%d) (slope=%g) ***********" % (fCCiteration, fitResult[0]))
    else:
        casalogPost("****** starting findContinuumChannels (%d) with nBaselineChannels=%d ***********" % (fCCiteration,nBaselineChannels))
    casalogPost("Using sigmaFindContinuum=%.2f, sigmaEffective=%.1f, percentile=%.0f for mode=%s, channels=%d/%d" % (sigmaFindContinuum, sigmaEffective, percentile, baselineMode, nBaselineChannels, len(spectrum)))
    if (baselineMode == 'edge'):
        # pick n channels on both edges
        lowerChannels = spectrum[:nBaselineChannels/2]
        upperChannels = spectrum[-nBaselineChannels/2:]
        intensityAllBaselineChannels = list(lowerChannels) + list(upperChannels)
        allBaselineXChannels = list(range(0, nBaselineChannels/2)) + \
                               list(range(len(spectrum) - nBaselineChannels/2, len(spectrum)))
        if (np.std(lowerChannels) == 0):
            mad = MAD(upperChannels)
            median = nanmedian(upperChannels)
            casalogPost("edge method: Dropping lower channels from median and std calculations")
        elif (np.std(upperChannels) == 0):
            mad = MAD(lowerChannels)
            median = nanmedian(lowerChannels)
            casalogPost("edge method: Dropping upper channels from median and std calculations")
        else:
            mad = MAD(intensityAllBaselineChannels)
            median = nanmedian(intensityAllBaselineChannels)
        useLowBaseline = True
    else:
        # Pick the n channels with the n lowest values (or highest if those 
        # have smallest MAD), but ignore edge channels inward for as long they 
        # are identical to themselves (i.e. avoid the
        # effect of TDM edge flagging.)
        myspectrum = spectrum
        allBaselineXChannels = np.array(range(len(myspectrum)))
        # the following could also be len(spectrum), since their lengths are identical
        if (len(originalSpectrum) > 10): # and len(originalSpectrum) <= 128):
            # Was opened up to all data (not just TDM) in Cycle 6
            # picks up some continuum in self-absorbed area in Cha-MMs1_CS in 200-channel spw
            if False:
                # ALMA Cycle 4+5
                idx = np.where((originalSpectrum != originalSpectrum[0]) * (originalSpectrum != originalSpectrum[-1]))
            else:
                # ALMA Cycle 6 onward
                edgeValuedChannels = np.where((originalSpectrum == originalSpectrum[0]) | (originalSpectrum == originalSpectrum[-1]))[0]
                edgeValuedChannelsLists = splitListIntoContiguousLists(edgeValuedChannels)
#                casalogPost("edgeValuedChannelsLists: %s" % (edgeValuedChannelsLists))
                idx = np.array(range(edgeValuedChannelsLists[0][-1] + 1, edgeValuedChannelsLists[-1][0]))
            if len(idx) > 0:
                allBaselineXChannels = idx
                myspectrum = spectrum[idx]
                casalogPost('Avoided %d edge channels of %d when computing min channels. nBaselineChannels is now %d' % (len(spectrum)-len(myspectrum), len(spectrum), nBaselineChannels),debug=True)
                if len(spectrum)-len(myspectrum) > 0:
                    casalogPost("using channels %d-%d" % (idx[0],idx[-1]),debug=True)
            else:
                casalogPost('*************************  len(idx) == 0')

        allBaselineXChannelsOriginal = allBaselineXChannels

        # Sort by intensity: myspectrum is often only a subset of spectrum, so
        # the channel numbers in original spectrum must be tracked separately
        # in the variable: allBaselineXChannels
        idx = np.argsort(myspectrum)
        if idxNonZeroMAD is not None:
            casalogPost('Invoked fix for PIPE-3107: based on earlier iteration (usefulChannels=%d)' % (len(idxNonZeroMAD)))
            idx = idx[np.isin(idx, idxNonZeroMAD)]  # preserve the order

        intensityAllBaselineChannels = myspectrum[idx[:nBaselineChannels]] 
        allBaselineOriginalChannels = originalSpectrum[idx[:nBaselineChannels]]  # pick the lowest nBaseline channels
        highestChannels = myspectrum[idx[-nBaselineChannels:]]  
        medianOfAllChannels = nanmedian(myspectrum)
        mad0 = MAD(intensityAllBaselineChannels)
        mad1 = MAD(highestChannels)
        middleChannels = myspectrum[idx[nBaselineChannels:-nBaselineChannels]]
        madMiddleChannels = MAD(middleChannels)
        # added useBaseline as a top-level parameter on June 20, 2023
        if useBaseline == 'auto' and amendMaskIterationName.find('xtra') < 0 and amendMaskIterationName.find('autoLower') < 0: # to not conflict with 46 lines down
            # Introduced the lowHighBaselineThreshold factor on Aug 31, 2016 for CAS-8938
            whichBaseline = np.argmin([mad0, lowHighBaselineThreshold*mad1, lowHighBaselineThreshold*madMiddleChannels])
            if whichBaseline == 0 and mad0 <= 0.0:
                # fix for PIPE-3107
                idxNonZeroMAD = np.where(myspectrum != 0.0)
                casalogPost('Invoked fix for PIPE-3107: when MAD=0, ignore those (%d) zero-level channels thereafter: %s' % (len(idxNonZeroMAD[0]),idxNonZeroMAD))
                idx = idx[np.isin(idx, idxNonZeroMAD)]  # preserve the order of idx (np.intersect1d does an internal sort)
                intensityAllBaselineChannels = myspectrum[idx[:nBaselineChannels]] 
                allBaselineOriginalChannels = originalSpectrum[idx[:nBaselineChannels]]  # pick the lowest nBaseline channels
                highestChannels = myspectrum[idx[-nBaselineChannels:]]  
                medianOfAllChannels = nanmedian(myspectrum[idx])
                mad0 = MAD(intensityAllBaselineChannels)
                mad1 = MAD(highestChannels)
                middleChannels = myspectrum[idx[nBaselineChannels:-nBaselineChannels]]
                madMiddleChannels = MAD(middleChannels)
                whichBaseline = np.argmin([mad0, lowHighBaselineThreshold*mad1, lowHighBaselineThreshold*madMiddleChannels])
            casalogPost('Using automatic heuristic of useBaseline = %s, which picked whichBaseline=%d because mads: %f, %f, %f' % (useBaseline,whichBaseline,mad0, lowHighBaselineThreshold*mad1, lowHighBaselineThreshold*madMiddleChannels))
            leftmostHighChannel = np.min(idx[-nBaselineChannels:])  # pick the lowest-numbered channel from group of highest intensity channels
            rightmostHighChannel = np.max(idx[-nBaselineChannels:]) # pick the highest-numbered channel from group of highest intensity channels
            noBluePointInterruptsHighestChannels = len(np.intersect1d(range(leftmostHighChannel,rightmostHighChannel), idx[:nBaselineChannels])) == 0
            if (whichBaseline == 0) or noBluePointInterruptsHighestChannels:
                if noBluePointInterruptsHighestChannels and whichBaseline != 0:
                    casalogPost('No blue point interrupts the sequence of highest channels, so this signifies an emission line, and we will useBaseline=low instead of high.')
                useBaseline = 'low'
            elif (whichBaseline == 1):
                useBaseline = 'high'
            else:
                if madMiddleChannels > 0:
                    useBaseline = 'middle'
                else: # fix for PRTSPR-50321
                    useBaseline = 'low'
        else:
            casalogPost('***** Using manual setting of useBaseline = %s.' %(useBaseline))
            
        # In the following if/elif/else, we convert the tri-valued variable useBaseline into two
        # Booleans for legacy purposes (i.e before useMiddleChannels was considered as an option).
        if (useBaseline == 'high'):
            # This is the absorption line case
            casalogPost("%s Using highest %d channels as baseline because low:mid:high = %g:%g:%g" % (projectCode,nBaselineChannels,mad0,madMiddleChannels,mad1), debug=True)
            intensityAllBaselineChannels = highestChannels[::-1] # reversed it so that first channel is highest value
            allBaselineXChannels = allBaselineXChannels[idx][-nBaselineChannels:]
            allBaselineXChannels = allBaselineXChannels[::-1] # reversed it so that first channel is highest value
            mad0 = MAD(intensityAllBaselineChannels)
            useLowBaseline = False
            useMiddleChannels = False
        elif useBaseline == 'middle':
            # This case is needed when there is a mix of emission and absorption lines, 
            # as in Band 10 Orion = 2016.1.00970.S spw25,31.
            casalogPost("%s Using middle %d channels as baseline because low:mid:high = %g:%g:%g, setting dropBaselineChannels to 0" % (projectCode, len(middleChannels),mad0,madMiddleChannels,mad1), debug=True)
            intensityAllBaselineChannels = middleChannels
            allBaselineXChannels = allBaselineXChannels[idx][nBaselineChannels:-nBaselineChannels]
            dropBaselineChannels = 0
            mad0 = MAD(intensityAllBaselineChannels)
            useLowBaseline = False
            useMiddleChannels = True
        else:
            # This is the emission line case
            casalogPost("%s Using lowest %d channels as baseline because low:mid:high = %g:%g:%g" % (projectCode,nBaselineChannels,mad0,madMiddleChannels,mad1), debug=True)
            useLowBaseline = True
            useMiddleChannels = False
            allBaselineXChannels = allBaselineXChannels[idx][:nBaselineChannels]

        if (amendMaskIterationName.find('xtra') >= 0 or amendMaskIterationName.find('autoLower') >= 0) and not useMiddleChannels and not np.isnan(madMiddleChannels): # fix for PIPE-3107 is to add 'and not np.isnan'
            intensityAllBaselineChannels = middleChannels
            allBaselineXChannels = allBaselineXChannelsOriginal[idx][nBaselineChannels:-nBaselineChannels]
            dropBaselineChannels = 0
            mad0 = MAD(intensityAllBaselineChannels)
            useLowBaseline = False
            useMiddleChannels = True
            casalogPost('***** PL2024: Forcing use of middle channels in %s' % (amendMaskIterationName))
            # This was found to work better on the extra mask construction as positive
            # features (emission) and negative features (missing flux) are often both present.
            if returnBluePoints:
                casalogPost('***** PL2024: Forcing returnBluePoints=False')
                returnBluePoints = False
        casalogPost("Median of all channels = %f,  MAD of selected baseline channels = %f" % (medianOfAllChannels,mad0))
        madRatio = None
        if dropBaselineChannels <= 0:
            casalogPost('dropBaselineChannels is zero, so madRatio will not be computed, and instead set to None')
        else:
            dropExtremeChannels = int(int(len(idx)*dropBaselineChannels)/100)
            casalogPost('dropExtremeChannels = %d' % (dropExtremeChannels))
            if dropExtremeChannels > 0:
                intensityAllBaselineChannelsDropExtremeChannels = myspectrum[idx[dropExtremeChannels:nBaselineChannels+dropExtremeChannels]] 
                allBaselineXChannelsDropExtremeChannels = allBaselineXChannelsOriginal[idx[dropExtremeChannels:nBaselineChannels+dropExtremeChannels]] 
#                casalogPost('intensityAllBaselineChannelsDropExtremeChannels = %s' % str(intensityAllBaselineChannelsDropExtremeChannels))
                mad0_dropExtremeChannels = MAD(intensityAllBaselineChannelsDropExtremeChannels)
                if mad0_dropExtremeChannels <= 0:
                    casalogPost('mad0_dropExtremeChannels = 0, will set madRatio to None and will not drop any baseline channels.')
                else:
                    # prevent division by zero error
                    madRatio = mad0/mad0_dropExtremeChannels
                    if madRatioLowerLimit < madRatio < madRatioUpperLimit:
                        # more than 1.2 means there was a significant improvement; more than 1.5 means something unexpected about the statistics
                        casalogPost("****** Dropping most extreme %d = %.1f%% of channels when computing the MAD, since it reduces the mad by a factor of x=%.2f (%.2f<x<%.2f)" % (dropExtremeChannels, dropBaselineChannels, madRatio, madRatioLowerLimit, madRatioUpperLimit))
                        intensityAllBaselineChannels = intensityAllBaselineChannelsDropExtremeChannels
#                        print("len(allBaselineXChannels)=%d, len(idx)=%d, dropExtremeChannels=%d, nBaselineChannels+dropExtremeChannels=%d" % (len(allBaselineXChannels), len(idx), dropExtremeChannels, nBaselineChannels+dropExtremeChannels))
                        allBaselineXChannels = allBaselineXChannelsDropExtremeChannels
                        allBaselineOriginalChannels = originalSpectrum[allBaselineXChannelsDropExtremeChannels] # allBaselineXChannels[idx][dropExtremeChannels:nBaselineChannels+dropExtremeChannels]]
                    else:
                        casalogPost("**** Not dropping most extreme channels when computing the MAD, since the change in MAD of %.2f is not within %.2f<x<%.2f" % (madRatio, madRatioLowerLimit, madRatioUpperLimit))
            

        casalogPost("min method: computing MAD and median of %d channels used as the baseline" % (len(intensityAllBaselineChannels)))
        mad = MAD(intensityAllBaselineChannels)
        madOriginal = MAD(allBaselineOriginalChannels)
        casalogPost("MAD of all baseline channels = %f" % (mad))

        if (fitResult is not None):
            casalogPost("MAD of original baseline channels (before removal of fit) = %f" % (madOriginal))
        if (mad < 1e-17 or madOriginal < 1e-17): 
            casalogPost("min method: avoiding blocks of identical-valued channels")
            if (len(originalSpectrum) > 10):
                # first avoid values that propagate from either edge
                myspectrum = spectrum[np.where((originalSpectrum != originalSpectrum[0]) * (originalSpectrum != originalSpectrum[-1]))]
                # next avoid identical zeroes  (PIPE-1213)
                myspectrum = myspectrum[np.where(myspectrum != 0.0)] # possible fix for PIPE-1213
            else: # original logic, prior to linear fit removal
                idx = np.where(spectrum != intensityAllBaselineChannels[0])
                myspectrum = spectrum[idx]
            idx = np.argsort(myspectrum)
            intensityAllBaselineChannels = myspectrum[idx[:nBaselineChannels]] 
            allBaselineXChannels = idx[:nBaselineChannels]  # was commented out in PL2020
#            for i in range(len(idx)):
#                casalogPost("idx[%d] = %d has %.3f" % (i,idx[i],myspectrum[idx[i]]))
#            casalogPost("len(allBaselineXChannels)=%d, shape(idx)=%s, nBaselineChannels=%d" % (len(allBaselineXChannels), str(np.shape(idx)), nBaselineChannels))
#            allBaselineXChannels = allBaselineXChannels[idx][:nBaselineChannels]  # was used in PL2020, caused crash PIPE-1213
            casalogPost("            computing MAD and median of %d channels used as the baseline" % (len(intensityAllBaselineChannels)))
        mad = MAD(intensityAllBaselineChannels)
        median = nanmedian(intensityAllBaselineChannels)
        casalogPost("min method: median intensity of %d channels used as the baseline: %f, mad: %f" % (len(intensityAllBaselineChannels), median, mad))
        
        if removeOutlierBaselineChannels in ['low', 'high', 'both', True]:  
            # Try this for PL2023
            rOBCsigma = 3
            if removeOutlierBaselineChannels in ['both', True]:  
                idx = np.where(np.abs(intensityAllBaselineChannels - median) / mad < rOBCsigma)[0]
            elif removeOutlierBaselineChannels in ['low']:  
                idx = np.where((intensityAllBaselineChannels - median) / mad > -rOBCsigma)[0]
            elif removeOutlierBaselineChannels in ['high']:  
                idx = np.where((intensityAllBaselineChannels - median) / mad < rOBCsigma)[0]
            if len(idx) < len(allBaselineXChannels)//2:
                remainingChannels = len(allBaselineXChannels) - len(idx)
                casalogPost("not removing outlier baseline channels as it would remove over half of them (%d/%d)" % (remainingChannels,len(allBaselineXChannels)))
            else:
                removedChannels = len(allBaselineXChannels)-len(idx)
                casalogPost("removing %d outlier baseline channels" % (removedChannels))
                allBaselineXChannels = allBaselineXChannels[idx]
                intensityAllBaselineChannels = intensityAllBaselineChannels[idx]
            mad = MAD(intensityAllBaselineChannels)
            median = nanmedian(intensityAllBaselineChannels)
            casalogPost("min method: after outlier removal: median intensity of %d channels used as the baseline: %f, mad: %f" % (len(intensityAllBaselineChannels), median, mad))
    # endif baselineMode=='edge' else 'min'
    # signalRatio will be 1.0 if no lines present and 0.25 if half the channels have lines, etc.
    signalRatio = (1.0 - 1.0*len(np.where(np.abs(spectrum-median)>(sigmaEffective*mad*2))[0]) / len(spectrum))**2
    if returnBluePoints:
        channels = sorted(allBaselineXChannels)
        medianTrue = median # prevent crash for undefined variable
        threshold = sigmaEffective*mad + medianTrue # prevent crash for undefined variable
        negativeThreshold = -negativeThresholdFactor*sigmaEffective*mad + medianTrue # prevent crash for undefined variable
        lineStrengthFactor = 1.0/signalRatio # prevent crash for undefined variable
        singleChannelPeaksAboveSFC = 0 # prevent crash for undefined variable
        allGroupsAboveSFC = 0 # prevent crash for undefined variable
        spectralDiff = 0 # prevent crash for undefined variable
        spectralDiff2 = 0 # prevent crash for undefined variable
        rangesDropped = 0 # prevent crash for undefined variable
    else: # not returning blue points
        if signalRatio >= 0.99 or ((signalRatio > 0.98) and (15 > peakOverMad > 10)):
            # If nearly no lines are found, then look a bit deeper to allow the possibility of signalRatio to drop enough to 
            # go below 0.965, and thereby prevent the desensitizing that comes later (see line 2542).
            weakLineFactor = 1.35 #1.39 gives 0.982 on 305  # 1.5 gives 1.0 on 305
            casalogPost("Initial signalRatio: %f, peakOverMad=%f,  Re-assessing." % (signalRatio,peakOverMad))
            signalRatio = (1.0 - 1.0*len(np.where(np.abs(spectrum-median)>(sigmaEffective*mad*weakLineFactor))[0]) / len(spectrum))**2
            maxdiff = np.max(np.abs(spectrum-median))
            casalogPost("max(abs(spectrum-median)) = %f while sigmaEffective(%f)*mad(%f)*%.2f=%f" % (maxdiff,sigmaEffective,mad,weakLineFactor,sigmaEffective*mad*weakLineFactor))
        originalMedian = np.median(originalSpectrum)
        # Should not divide by post-baseline-fit median since it may be close to 0
        spectralDiff = 100*np.median(np.abs(np.diff(spectrum)))/originalMedian
        spectralDiff2 = 100*np.median(np.abs(np.diff(spectrum,n=2)))/originalMedian
        casalogPost("signalRatio=%f, spectralDiff = %f and spectralDiff2=%f percent of the median" % (signalRatio, spectralDiff,spectralDiff2))
        lineStrengthFactor = 1.0/signalRatio
        if (spectralDiff2 < 0.65 and npts > 192 and signalRatio<signalRatioTier2):  # used to be <0.95 but regression 321 has 0.949 so use 0.94
            # This appears to be a channel-averaged FDM spectrum with lots of real line emission.
            # So, don't allow the median to be raised, and reduce the mad to lower the threshold.
            # page 15: G11.92_B7.ms_spw3 yields spectralDiff2=0.6027
            # We can also get into here if there is a large slope in the mean spectrum, so we
            # counter that by removing a linear slope before evaluating lineSNR.
            casalogPost('The spectral difference (n=2) is rather small, so set signalRatio=0 to reduce the baseline level.',debug=True)
            signalRatio = 0
            if True:
                # note: this slope removal differs from the one in runFindContinuum because
                # it always acts upon the whole spectrum, not just the potential baseline windows.
                print("Removing linear slope for purposes of computing lineSNR.")
                x = np.arange(len(spectrum))
                slope, intercept = linfit(x, spectrum, MAD(spectrum))
                newspectrum = spectrum - x*slope
                newmad = MAD(newspectrum)
                # Avoid high edge channels from creating false high SNR
                lineSNR = (np.max(newspectrum[1:-1])-np.median(newspectrum))/newmad
            else:
                # Avoid high edge channels from creating false high SNR
                lineSNR = (np.max(spectrum[1:-1])-median)/mad
            casalogPost('lineSNR = %f' % lineSNR)
            if (lineSNR > lineSNRThreshold):
                casalogPost('The lineSNR > %d, so scaling the mad by 1/3 to reduce the threshold.' % lineSNRThreshold, debug=True)
                mad *= 0.33
                if (trimChannels == 'auto'): 
                    trimChannels = 6
                    casalogPost('Setting trimChannels to %d.' % (trimChannels))
        else:
            casalogPost('Not reducing mad by 1/3: npts=%d, signalRatio=%.3f, spectralDiff2=%.2f' % (npts,signalRatio,spectralDiff2),debug=True)
        if useMiddleChannels or useUncorrectedMedian:
            medianTrue = median
        else:
            # Do not use signalRatio here as the 5th parameter anymore, because we may have set it to zero just above, 
            # but 1/lineStrengthFactor is the same value as before it was set to zero.  Setting it to zero causes
            # problems on non-hot core FDM, like spw29 of 2015.1.00581.S, uid://A001/X2f6/X16b (CAS-11671).  Not
            # setting it to zero changes the selection on hot cores, some giving more channels, some giving fewer,
            # but not drastically either direction.  Largest change is on spw22 of uid://A001/X2f6/X265 (page 245).
            medianTrue = medianCorrected(baselineMode, percentile, median, mad, 
                                         1.0/lineStrengthFactor, useLowBaseline)
            if medianTrue < np.min(spectrum):
                casalogPost("**** PIPE-525: Corrected median is less than all data points.  Using true median instead.")
                medianTrue = np.median(spectrum)

        peakFeatureSigma = (np.max(spectrum)-medianTrue)/mad 
        # De-sensitize the threshold in order to avoid small differences in noise levels 
        # (e.g. that occur between serial and parallel tclean cubes)
        # from producing drastically different continuum selections.
        if ((signalRatio > signalRatioTier1) or (signalRatio > signalRatioTier2 and peakOverMad < 5)) and (sigmaFindContinuumMode in ['auto','autolower']):
            casalogPost('findContinuumChannels: desensitizing because %f>%f or (%f>%f and %f<5)' % (signalRatio,signalRatioTier1,signalRatio,signalRatioTier2,peakOverMad))
            maxPeakFeatureSigma = (np.max(spectrum) - np.min(spectrum))/MAD(spectrum)
            if peakFeatureSigma > maxPeakFeatureSigma:
                oldmad = 1*mad
                mad *= peakFeatureSigma/maxPeakFeatureSigma
                casalogPost('findContinuumChannels: Limiting peakFeatureSigma from %f to %f, reset MAD from %f to %f' % (peakFeatureSigma,maxPeakFeatureSigma,oldmad,mad))
                peakFeatureSigma = maxPeakFeatureSigma
            else:
                casalogPost('findContinuumChannels: not limiting peakFeatureSigma because %f > %f' % (peakFeatureSigma,maxPeakFeatureSigma))
        threshold = sigmaEffective*mad + medianTrue
        # Use a (default=15%) lower negative threshold to help prevent false identification of absorption features.
        negativeThreshold = -negativeThresholdFactor*sigmaEffective*mad + medianTrue
        casalogPost("MAD = %f, median = %f, trueMedian=%f, signalRatio=%f" % (mad, median, medianTrue, signalRatio))
        casalogPost("findContinuumChannels: computed threshold = %f, medianTrue=%f" % (threshold, medianTrue))
        channels = np.where(spectrum < threshold)[0]
        if (negativeThreshold is not None):
            channels2 = np.where(spectrum > negativeThreshold)[0]
            channels = np.intersect1d(channels,channels2)

        # for CAS-8059: remove channels that are equal to the minimum if all 
        # channels from it toward the nearest edge are also equal to the minimum: 
        channels = list(channels)
        if (abs(originalSpectrum[np.min(channels)] - np.min(originalSpectrum)) < abs(1e-10*np.min(originalSpectrum))):
            lastmin = np.min(channels)
            channels.remove(lastmin)
            removed = 1
            casalogPost("Checking channels %d-%d" % (np.min(channels),np.max(channels)))
            for c in range(np.min(channels),np.max(channels)):
                mydiff = abs(originalSpectrum[c] - np.min(originalSpectrum))
                mycrit = abs(1e-10*np.min(originalSpectrum))
                if (mydiff > mycrit):
                    break
                if c in channels:
                    channels.remove(c)
                    removed += 1
            casalogPost("Removed %d channels on low channel edge that were at the minimum." % (removed))
        # Now come in from the upper side
        if (abs(originalSpectrum[np.max(channels)] - np.min(originalSpectrum)) < abs(1e-10*np.min(originalSpectrum))):
            lastmin = np.max(channels)
            channels.remove(lastmin)
            removed = 1
            casalog.post("Checking channels %d-%d" % (np.max(channels),np.min(channels)))
            for c in range(np.max(channels),np.min(channels)-1,-1):
                mydiff = abs(originalSpectrum[c] - np.min(originalSpectrum))
                mycrit = abs(1e-10*np.min(originalSpectrum))
                if (mydiff > mycrit):
                    break
                if c in channels:
                    channels.remove(c)
                    removed += 1
            casalogPost("Removed %d channels on high channel edge that were at the minimum." % (removed))
        peakChannels = np.where(spectrum > threshold)[0]
        peakChannelsLists = splitListIntoContiguousLists(peakChannels)
        widthOfWidestFeature = maxLengthOfLists(peakChannelsLists)
        casalogPost("Width of widest feature = %d (length spectrum = %d)" % (widthOfWidestFeature, len(spectrum)))
        # C4R2 had signalRatio < 0.6 and spectralDiff2 < 1.2 but this yielded only 1 channel 
        # of continuum on NGC6334I spw25 when memory expanded to 256GB.
        if (signalRatio > 0 and signalRatio < 0.925 and spectralDiff2 < 1.3 and 
            len(spectrum) > 1000  and trimChannels=='auto' and 
            widthOfWidestFeature < len(spectrum)/8):
            # This is meant to prevent rich hot cores from returning only 1
            # or 2 channels of continuum.  signalRatio>0 is to avoid conflict
            # with the earlier heuristic above where it is set to zero.
            trimChannels = 13
            if autoNarrow:
                narrow = 2
            casalogPost('Setting trimChannels=%d, narrow=%s since many lines appear to be present (signalRatio=%f).' % (trimChannels,str(narrow), signalRatio))
        else:
            casalogPost('Not changing trimChannels from %s: signalRatio=%f, spectralDiff2=%f' % (str(trimChannels), signalRatio, spectralDiff2))
            

        peakMultiChannelsLists = splitListIntoContiguousListsAndRejectNarrow(peakChannels, narrow=2)
        allGroupsAboveSFC = len(peakChannelsLists)
        singleChannelPeaksAboveSFC = allGroupsAboveSFC - len(peakMultiChannelsLists)
        selection = convertChannelListIntoSelection(channels)
        casalogPost("Found %d potential continuum channels: %s" % (len(channels), str(selection)))
        if (len(channels) == 0):
            selection = ''
            groups = 0
        else:
            channels = splitListIntoContiguousListsAndRejectZeroStd(channels, spectrum, nanmin, verbose=verbose)
            if verbose: 
                print("channels = ", channels)
            selection = convertChannelListIntoSelection(channels,separator=separator)
            groups = len(selection.split(separator))
            casalogPost("Found %d channels in %d groups after rejecting zero std: %s" % (len(channels), groups, str(selection)))
            if (len(channels) == 0):
                selection = ''
            else:
                casalogPost("Calling splitListIntoContiguousListsAndTrim(totalChannels=%s, channels=%s, trimChannels=%s, maxTrim=%d, maxTrimFraction=%f)" % (str(npts), str(channels), str(trimChannels), maxTrim, maxTrimFraction))
                originalChannels = channels[:]
                channels = splitListIntoContiguousListsAndTrim(npts, channels, 
                             trimChannels, maxTrim, maxTrimFraction, verbose)
                selection = convertChannelListIntoSelection(channels)
                if selection == '':  # fix for PIPE-359
                    maxTrim = 0.1; trimChannels = 2
                    casalogPost("All potential continuum channels trimmed, which is quite rare.  Reverting to maxTrim=%.2f and trimChannels=%d." % (maxTrim,trimChannels), priority='WARN')
                    channels = splitListIntoContiguousListsAndTrim(npts, originalChannels, 
                                  trimChannels, maxTrim, maxTrimFraction, verbose)
                    selection = convertChannelListIntoSelection(channels)
                    if selection == '':  
                        casalogPost('Selection is still blank, skipping trim, originalChannels=%s, length=%d' %(originalChannels,len(originalChannels) ), priority='WARN')
                        channels = originalChannels  # add this line to fix PIPE-2374
                        selection = convertChannelListIntoSelection(channels)
                groups = len(selection.split(separator))
                if (groups > maxGroupsForMaxTrimAdjustment and trimChannels=='auto'
                    and maxTrim>maxTrimDefault):
                    maxTrim = maxTrimDefault
                    casalogPost("Restoring maxTrim=%d because groups now > %d" % (maxTrim,maxGroupsForMaxTrimAdjustment))
                    if verbose:
                        casalogPost("Calling splitListIntoContiguousListsAndTrim(totalChannels=%s, channels=%s, trimChannels=%s, maxTrim=%d, maxTrimFraction=%f)" % (str(npts), str(channels), str(trimChannels), maxTrim, maxTrimFraction))
                    trialChannels = splitListIntoContiguousListsAndTrim(npts, channels, 
                                 trimChannels, maxTrim, maxTrimFraction, verbose)
                    if len(trialChannels) == 0:
                        casalogPost("Zero channels left after trimming. Reverting to prior selection.")
                    else:
                        channels = trialChannels
                        if verbose:
                            print("channels = ", channels)
                        selection = convertChannelListIntoSelection(channels)
                        groups = len(selection.split(separator))

                if verbose:
                    print("Found %d groups of channels = " % (groups), channels)
                if (groups > 1):
    #                if verbose:
    #                    casalogPost("Calling splitListIntoContiguousListsAndRejectNarrow(channels=%s, narrow=%s)" % (str(channels), str(narrow)))
    #                else:
    #                    casalogPost("Calling splitListIntoContiguousListsAndRejectNarrow(narrow=%s)" % (str(narrow)))
                    trialChannels = splitListIntoContiguousListsAndRejectNarrow(channels, narrow)
                    if (len(trialChannels) > 0):
                        channels = trialChannels
                        casalogPost("Found %d channels after trimming %s channels and rejecting narrow groups." % (len(channels),str(trimChannels)))
                        selection = convertChannelListIntoSelection(channels)
                        groups = len(selection.split(separator))
                else:
                    casalogPost("Not rejecting narrow groups since there is only %d group!" % (groups))
        casalogPost("Found %d continuum channels in %d groups: %s" % (len(channels), groups, selection))
        potentialChannels, messages, rangesDropped = rejectNarrowInnerWindowsChannels(channels, fCCiteration)
        if enableRejectNarrowInnerWindows:
            channels = potentialChannels
            for message in messages:
                casalogPost(message)
        else:
            if rangesDropped > 0:
                casalogPost("rejectNarrowInnerWindows wanted to remove %d ranges, but it was not enabled" % (rangesDropped))
    # endif for the else (i.e., end of returnBluePoints==False)
    selection = convertChannelListIntoSelection(channels)
    groups = len(selection.split(separator))
    casalogPost("Final: found %d continuum channels (sFC=%.2f) in %d groups: %s" % (len(channels), sigmaFindContinuum, groups, selection))
    return(channels, selection, threshold, median, groups, correctionFactor, 
           medianTrue, mad, computeMedianCorrectionFactor(baselineMode, percentile)*signalRatio,
           negativeThreshold, lineStrengthFactor, singleChannelPeaksAboveSFC, 
           allGroupsAboveSFC, [spectralDiff, spectralDiff2], trimChannels, 
           useLowBaseline, narrow, [allBaselineXChannels,intensityAllBaselineChannels], 
           madRatio, useMiddleChannels, signalRatio, rangesDropped, idxNonZeroMAD)

def rejectNarrowInnerWindowsChannels(channels, fCCiteration=0):
    """
    This function is called by findContinuumChannels.
    If there are 3-15 groups of channels, then remove any inner window 
    that is narrower than both edge windows.
    Returns: a list of channels
    """
    mylists = splitListIntoContiguousLists(channels)
    groups = len(mylists)
    messages = []
    rangesDropped = 0
#    if (groups > 2 and groups < 8):  C4R2 heuristic
    if (groups > 2 and groups < 16):  # Cycle 5 onward
        channels = []
        lenFirstGroup = len(mylists[0])
        lenLastGroup = len(mylists[-1])
        channels += mylists[0]  # start with the first window
        if (groups < 8):
            widthFactor = 1
        else:
            # this is to prevent unnecessarily trimming narrow windows, e.g. in a hot core case
            widthFactor = 0.2
        # The final group needs to be included in the output channels list, and it is safe to
        # simply included it in the loop because it will always pass the test.
        for group in range(1,groups): 
#            if (len(mylists[group]) >= np.mean([lenFirstGroup,lenLastGroup])*widthFactor): #  or len(mylists[group]) >= lenLastGroup*widthFactor):
            widthThreshold = np.min([lenFirstGroup,lenLastGroup])
            if (len(mylists[group]) >= (widthThreshold*widthFactor)): 
                channels += mylists[group]
                if group < groups-1:
                    # Don't report the final group because it is not really under consideration here
                    messages.append('fCCiter%d:  Keeping channel group %d because it is wider than %.1f*width of both edge groups' % (fCCiteration,group,widthFactor))
            else:
                rangesDropped += 1
                messages.append('fCCiter%d: Dropping channel group %d because it is narrower (%d) than %.1f*width (%d) of both edge groups' % (fCCiteration, group, len(mylists[group]), widthFactor, widthThreshold*widthFactor))
        if rangesDropped == 0:
            messages.append('fCCiter%d: No channel groups dropped by rejectNarrowInnerWindowsChannels' % (fCCiteration))
    else:
        messages.append('fCCiter%d: Number of channel groups (%d) disqualifies it for rejectNarrowInnerWindowsChannels' % (fCCiteration,groups))
    return(channels, messages, rangesDropped)

def splitListIntoContiguousListsAndRejectNarrow(channels, narrow=3):
    """
    This function is called by findContinuumChannels.
    Split a list of channels into contiguous lists, and reject those that
    have a small number of channels.
    narrow: if >=1, then interpret it as the minimum number of channels that
                  a group must have in order to survive
            if <1, then interpret it as the minimum fraction of channels that
                  a group must have relative to the total number of channels
    Returns: a new single list (as an array)
    Example:  [1,2,3,4,6,7,9,10,11] --> [ 1,  2,  3,  4,  9, 10, 11]
    """
    length = len(channels)
    mylists = splitListIntoContiguousLists(channels)
    channels = []
    for mylist in mylists:
        if (narrow < 1):
            if (len(mylist) <= narrow*length):  # PIPE-1865 fraction->narrow
                continue
        elif (len(mylist) < narrow):
            continue
        channels += mylist
    return(np.array(channels))

def splitListIntoContiguousListsAndTrim(totalChannels, channels, trimChannels=0.1, 
                                        maxTrim=maxTrimDefault, 
                                        maxTrimFraction=1.0, verbose=False):
    """
    This function is called by findContinuumChannels.
    Split a list of channels into contiguous lists, and trim some number
    of channels from each edge.
    totalChannels: number of channels in the spectrum
    channels: a list of channels selected for potential continuum
    trimChannels: if integer, use that number of channels.  If float between
        0 and 1, then use that fraction of channels in each contiguous list
        (rounding up). If 'auto', then use 0.1 but not more than maxTrim 
        channels and not more than maxTrimFraction of channels.
    maxTrim and maxTrimChannels: used in 'auto' mode
    Returns: a new single list
    """
    if not isinstance(trimChannels, (str, np.bytes_)):
        if (trimChannels <= 0):
            return(np.array(channels))
    length = len(channels)
    trimChannelsMode = trimChannels
    medianWidthOfRanges = np.median(countChannelsInRanges(convertChannelListIntoSelection(channels)))
    if (trimChannels == 'auto'):
        trimChannels = pickAutoTrimChannels(totalChannels, length, maxTrim, medianWidthOfRanges)
        casalogPost('Picked trimChannels = %s.' % (str(trimChannels)))
    mylists = splitListIntoContiguousLists(channels)
    channels = []
    if totalChannels < 75:
        trimLimitForEdgeRegion = 2
    else:
        trimLimitForEdgeRegion = 3
    for i,mylist in enumerate(mylists):
        trimChan = trimChannels
        if verbose:
            print("trimChan=%d, Checking list = " % (trimChan), mylist)
        if (trimChannels < 1):
            trimChan = int(np.ceil(len(mylist)*trimChannels))
            if verbose:
                print("since trimChannels=%s<1; reset trimChan=%d" % (str(trimChannels),trimChan))
        if (trimChannelsMode == 'auto' and 1.0*trimChan/len(mylist) > maxTrimFraction):
            trimChan = int(np.floor(maxTrimFraction*len(mylist)))
        if verbose:
            print("trimChan for this window = %d" % (trimChan))
        if (len(mylist) < 1+trimChan*2):
            if (len(mylists) == 1):
                # If there was only one list of 1 or 2 channels, then don't trim it away!
                channels += mylist[:1]
            continue
        # Limit the trimming of the edge closest to the edge of the spw to trimLimitForEdgeRegion (i.e., 2 or 3)
        # in order to preserve bandwidth.
        if (i==0 and trimChan > trimLimitForEdgeRegion):
            if (len(mylists)==1):
                # It is the only window so limit the trim on the far edge too
#                print("trim case 0: %d-%d --> %d-%d" % (mylist[0], mylist[-1], mylist[trimLimitForEdgeRegion],mylist[-trimLimitForEdgeRegion]))
                channels += mylist[trimLimitForEdgeRegion:-trimLimitForEdgeRegion]
            else:
                channels += mylist[trimLimitForEdgeRegion:-trimChan]
        elif (i==len(mylists)-1 and trimChan > trimLimitForEdgeRegion):
            channels += mylist[trimChan:-trimLimitForEdgeRegion]
        else:
            # It is either not an edge window, or it is an edge window and trimChan<=trimLimitForEdgeRegion
            channels += mylist[trimChan:-trimChan]
    return(np.array(channels))

def maxLengthOfLists(lists):
    """
    This function is called by findContinuumChannels.
    lists: a list if lists
    Returns: an integer that is the length of the longest list
    """
    maxLength = 0
    for a in lists:
        if (len(a) > maxLength):
            maxLength = len(a)
    return maxLength

def roundFigures(value, digits):
    """
    This function is called by runFindContinuum and drawYlabel.
    This function rounds a floating point value to a number of significant 
    figures.  Was originally taken from analysisUtils.
    value: value to be rounded (between 1e-20 and 1e+20)
    digits: number of significant digits, both before or after decimal point
    Returns: a floating point value
    """
    if (value != 0.0):
        if (np.log10(np.abs(value)) % 1 < np.log10(5)):
            digits -= 1
    for r in range(-20,20):
        if (round(value,r) != 0.0):
            value = round(value,r+digits)
            break
    return(value)

def pickAutoTrimChannels(totalChannels, length, maxTrim, medianWidthOfRanges):
    """
    This function is called by splitListIntoContiguousListsAndTrim and pickTrimString.
    Automatic choice of number of trimChannels as a function of the number 
    of potential continuum channels in an spw.
    length: number of channels in the list of potential continuum channels
    Returns: an integer or floating point value
    """
    trimChannels = 0.1
    casalogPost('median width of potential continuum ranges = %.1f' % (medianWidthOfRanges))
    if (length*trimChannels > maxTrim and medianWidthOfRanges > maxTrim*1.5 and totalChannels > 256):
        casalogPost("pickAutoTrimChannels(): set trimChannels = %d because %.0f > %d and totalChannels=%d>256" % (maxTrim,length*trimChannels,maxTrim,totalChannels))
        trimChannels = maxTrim
    else:
        casalogPost("pickAutoTrimChannels(): set trimChannels = %.1f because %.0f <= %d or %.1f <= %f or totalChannels=%d <= 256" % (trimChannels,length*trimChannels,maxTrim,medianWidthOfRanges,maxTrim*1.5,totalChannels))
    return(trimChannels)

def pickTrimString(totalChannels, trimChannels, length, maxTrim, selection):
    """
    This function is called by runFindContinuum.
    Generate a string describing the setting of the trimChannels parameter.
    totalChannels: total number of points in spectrum
    trimChannels: string (e.g. 'auto') or integer (e.g. 20) or float (e.g. 0.1)
    length: number of channels in the spectrum  (now unused)
    selection: continuum channel selection string
    Returns: a string
    """
    if (trimChannels=='auto'):
        contSelectionLength = np.sum(countChannelsInRanges(selection))
        medianWidthOfRanges = np.median(countChannelsInRanges(selection))
#        trimString = 'auto_max=%g' % pickAutoTrimChannels(length, maxTrim, medianWidthOfRanges)
        trimString = 'auto_max=%g' % pickAutoTrimChannels(totalChannels, contSelectionLength, maxTrim, medianWidthOfRanges)
    else:
        trimString = '%g' % trimChannels
    return(trimString)

def pickNarrowString(narrow, length, narrowValueModified=None):
    """
    This function is called by runFindContinuum.
    Generate a string describing the setting of the narrow parameter.
    Returns: a string
    """
    if (narrow=='auto'):
        myNarrow = pickNarrow(length)
        if (narrowValueModified is None or myNarrow == narrowValueModified):
            narrowString = 'auto=%g' % myNarrow
        else:
            narrowString = 'auto=%g' % narrowValueModified
    else:
        narrowString = '%g' % narrow
    return(narrowString)

def pickNarrow(length):
    """
    This function is called by pickNarrowString and findContinuumChannels.
    Automatically picks a setting for the narrow parameter of 
    findContinuumChannels() based on the number of channels in the spectrum.  
    Returns: an integer
    Examples: This formula results in the following values:
    length: 64,128,240,480,960,1920,3840,7680:
    return:  2,  3,  3,  3,  3,   4,   4,   4  (ceil(log10)) **** current function ****
             2,  2,  2,  3,  3,   3,   4,   4  (round_half_up(log10))
             1,  2,  2,  2,  2,   3,   3,   3  (floor(log10))
             5,  5,  6,  7,  7,   8,   9,   9  (ceil(log))
             4,  5,  5,  6,  7,   8,   8,   9  (round_half_up(log))
             4,  4,  5,  6,  6,   7,   8,   8  (floor(log))
    """
    return(int(np.ceil(np.log10(length))))

def sigmaCorrectionFactor(baselineMode, npts, percentile):
    """
    This function is called by findContinuumChannels.
    Computes the correction factor for the fact that the measured rms (or MAD) 
    on the N%ile lowest points of a datastream will be less than the true rms 
    (or MAD) of all points.  
    npts: nchan in the spw
    Returns: a value between 0 and 1, which will be used to reduce the 
             estimate of the population sigma based on the sample sigma
    """
    edgeValue = (npts/128.)**0.08
    if (baselineMode == 'edge'):
        return(edgeValue)
    value = edgeValue*2.8*(percentile/10.)**-0.25
    casalogPost("sigmaCorrectionFactor using percentile = %g to get sCF=%g" % (percentile, value), debug=True)
    return(value)

def medianCorrected(baselineMode, percentile, median, mad, signalRatio, 
                    useLowBaseline):
    """
    This function is called by findContinuumChannels.
    Computes the true median of a datastream from the observed median and MAD 
    of the lowest Nth percentile points.
    signalRatio: when there is a lot of signal, we need to reduce the 
                 correction factor because it is less like Gaussian noise
                 It is 1.0 if no lines are present, 0.25 if half the channels 
                 have signal, etc.
    """
    casalogPost('medianCorrected using signalRatio=%.2f' % (signalRatio),debug=True)
    if useLowBaseline:
        corrected = median + computeMedianCorrectionFactor(baselineMode, percentile)*mad*signalRatio
    else:
        corrected = median - computeMedianCorrectionFactor(baselineMode, percentile)*mad*signalRatio
    return(corrected)

def computeMedianCorrectionFactor(baselineMode, percentile):
    """
    This function is called by findContinuumChannels and medianCorrected.
    Computes the effect due to the fact that taking the median of the
    N%ile lowest points of a Gaussian noise datastream will be lower than the true 
    median of all the points.  This is (TrueMedian-ObservedMedian)/ObservedMAD
    """
    casalogPost('computeMedianCorrectionFactor using percentile=%.2f' % (percentile),debug=True)
    if (baselineMode == 'edge'):
        return(0)
    return(6.3*(5.0/percentile)**0.5)

def getImageInfo(img, returnBeamAreaPerChannel=False):
    """
    This function is called by findContinuum and meanSpectrum.
    Extract the beam and pixel information from a CASA image.
    This was copied from getFitsBeam in analysisUtils.py.
    Returns: A list of 12 things: bmaj, bmin, bpa, cdelt1, cdelt2, 
                 naxis1, naxis2, frequency, shape, crval1, crval2, maxBaseline,
                 telescope name
       Beam angles are in arcseconds (bpa in degrees), crvals are in radians
       Frequency is in GHz and is the central frequency
       MaxBaseline is in meters, inferred from freq and beamsize (will be zero for .residual images)
    """
    ARCSEC_PER_RAD = 206264.80624709636
    c_mks = 2.99792458e8
    if (os.path.exists(img) == False):
        print("image not found: ", img)
        return
    myia = iatool()
    myia.open(img)
    mydict = myia.restoringbeam()
    if 'major' in mydict or 'beams' in mydict:
        myqa = qatool()
        if 'major' in mydict:
            # single beam case
            bmaj = myqa.convert(mydict['major'], 'arcsec')['value']
            bmin = myqa.convert(mydict['minor'], 'arcsec')['value']
            bpa = myqa.convert(mydict['positionangle'], 'deg')['value']
        elif 'beams' in mydict:
            # perplane beams
            beams = mydict['beams']
            major = []
            minor = []
            sinpa = []
            cospa = []
            for chan_beam in beams.values():
                for chan_pol_beam in chan_beam.values():
                    major.append(myqa.convert(chan_pol_beam['major'], 'arcsec')['value'])
                    minor.append(myqa.convert(chan_pol_beam['minor'], 'arcsec')['value'])
                    sinpa.append(np.sin(myqa.convert(chan_pol_beam['positionangle'], 'rad')['value']))
                    cospa.append(np.cos(myqa.convert(chan_pol_beam['positionangle'], 'rad')['value']))
            if returnBeamAreaPerChannel:
                return np.array(major)*np.array(minor)
            bmaj = np.median(major)
            bmin = np.median(minor)
            bpa = np.degrees(np.arctan2(np.median(sinpa), np.median(cospa)))
        else:
            print("Unrecognized beam dictionary.")
            return
    else:
        bmaj = 0
        bmin = 0
        bpa = 0
#        if 'mask' not in img:
#            print("Warning: No beam found in header.")
    naxis1 = myia.shape()[0]
    naxis2 = myia.shape()[1]
    axis = findSpectralAxis(myia)
    mycs = myia.coordsys()
    myqa = qatool()
    restfreq = myqa.convert(mycs.restfrequency(), 'Hz')['value'][0]
    cdelt1 = mycs.increment()['numeric'][0] * ARCSEC_PER_RAD  # arcsec
    cdelt2 = mycs.increment()['numeric'][1] * ARCSEC_PER_RAD  # arcsec
    crval1 = mycs.referencevalue()['numeric'][0] # radian
    crval2 = mycs.referencevalue()['numeric'][1] # radian
    deltaFreq = mycs.increment()['numeric'][axis]
    frequency = mycs.referencevalue()['numeric'][axis]
    frequencyGHz =  frequency * 1e-9
    mycs.done()
    bunit = myia.brightnessunit()
    velocityWidth = abs(c_mks * 0.001 * deltaFreq / frequency)
    shape = myia.shape()
    telescope = getTelescope(img, myia)
    myia.close()
    myqa.done()
    if bmaj > 0:
        maxBaseline = c_mks  / ((1e9*frequencyGHz*(bmaj*bmin)**0.5)/ARCSEC_PER_RAD)
    else:
        maxBaseline = 0
    return([bmaj,bmin,bpa,cdelt1,cdelt2,naxis1,naxis2,frequencyGHz,shape,crval1,crval2,maxBaseline,telescope])
                                                            
def numberOfChannelsInCube(img, returnFreqs=False, returnChannelWidth=False, 
                           verbose=False, myia=None):
    """
    This function is called by findContinuum, cubeFrameToTopo, 
    computeStatisticalSpectrumFromMask, and meanSpectrum.
    Finds the number of channels in a CASA image cube.
    returnFreqs: if True, then also return the frequency of the center of the
           first and last channel (in Hz)
    returnChannelWidth: if True, then also return the channel width (in Hz)
    verbose: if True, then print the frequencies of first and last channel
    -Todd Hunter
    """
    if (not os.path.exists(img)):
        print("Image not found.")
        return
    needToClose = False
    if myia is None or myia == '':
        myia = iatool()
        myia.open(img)
        needToClose = True
    axis = findSpectralAxis(myia)
    naxes = len(myia.shape())
    nchan = myia.shape()[axis]
    mycs = myia.coordsys()
    cdelt = mycs.increment()['numeric'][axis]
    pixel = [0]*naxes
    firstFreq = mycs.toworld(pixel, format='n')['numeric'][axis]
    pixel[axis] = nchan-1
    lastFreq = mycs.toworld(pixel, format='n')['numeric'][axis]
    mycs.done()
    if needToClose:
        myia.close()
    if (returnFreqs):
        if (returnChannelWidth):
            return(nchan,firstFreq,lastFreq,cdelt)
        else:
            return(nchan,firstFreq,lastFreq)
    else:
        if (returnChannelWidth):
            return(nchan,cdelt)
        else:
            return(nchan)

def nanmean(a, axis=0):
    """
    This function is called by findContinuumChannels, runFindContinuum, and avgOverCube.
    Takes the mean of an array, ignoring the nan entries
    """
    if list(map(int, np.__version__.split('.')[:3])) < [1, 8, 1]:
        return scipy_nanmean(a, axis)
    else:
        return np.nanmean(a, axis)

def _nanmedian(arr1d, preop=None):  # This only works on 1d arrays
    """
    Private function for rank a arrays. Compute the median ignoring Nan.
    This function is called by nanmedian(), which is in turn called by MAD.

    Parameters
    ----------
    arr1d : ndarray
        Input array, of rank 1.

    Results
    -------
    m : float
        The median.
    """
    x = arr1d.copy()
    c = np.isnan(x)
    s = np.where(c)[0]
    if s.size == x.size:
        warnings.warn("All-NaN slice encountered (length=%d)" % (x.size), RuntimeWarning)
        return np.nan
    elif s.size != 0:
        # select non-nans at end of array
        enonan = x[-s.size:][~c[-s.size:]]
        # fill nans in beginning of array with non-nans of end
        x[s[:enonan.size]] = enonan
        # slice nans away
        x = x[:-s.size]
    if preop:
        x = preop(x)
    return np.median(x, overwrite_input=True)

def nanmedian(x, axis=0, preop=None):
    """
    This function is called by MAD, avgOverCube, and meanSpectrum.
    Compute the median along the given axis ignoring nan values.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or None, optional
        Axis along which the median is computed. Default is 0.
        If None, compute over the whole array `x`.
    preop : function
        function to apply on 1d slice after removing the NaNs and before
        computing median

    Returns
    -------
    m : float
        The median of `x` along `axis`.

    See Also
    --------
    nanstd, nanmean, numpy.nanmedian

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([0, 3, 1, 5, 5, np.nan])
    >>> stats.nanmedian(a)
    array(3.0)

    >>> b = np.array([0, 3, 1, 5, 5, np.nan, 5])
    >>> stats.nanmedian(b)
    array(4.0)

    Example with axis:

    >>> c = np.arange(30.).reshape(5,6)
    >>> idx = np.array([False, False, False, True, False] * 6).reshape(5,6)
    >>> c[idx] = np.nan
    >>> c
    array([[  0.,   1.,   2.,  nan,   4.,   5.],
           [  6.,   7.,  nan,   9.,  10.,  11.],
           [ 12.,  nan,  14.,  15.,  16.,  17.],
           [ nan,  19.,  20.,  21.,  22.,  nan],
           [ 24.,  25.,  26.,  27.,  nan,  29.]])
    >>> stats.nanmedian(c, axis=1)
    array([  2. ,   9. ,  15. ,  20.5,  26. ])

    """
    x = np.asarray(x)
    if axis is None:
        x = x.ravel()
        axis = 0
    if x.ndim == 0:
        return float(x.item())
    if preop is None and hasattr(np, 'nanmedian'):
        return np.nanmedian(x, axis)
    x = np.apply_along_axis(_nanmedian, axis, x, preop)
    if x.ndim == 0:
        x = float(x.item())
    return x

def findSpectralAxis(img):
    """
    This function is called by computeStatisticalSpectrumFromMask, getImageInfo, 
    findContinuum and numberOfChannelsInCube.
    Finds the spectral axis number of an image tool instance, or an image.
    img: string or iatool instance
    """
    if (type(img) == str):
        myia = iatool()
        myia.open(img)
        needToClose = True
    else:
        myia = img
        needToClose = False
    mycs = myia.coordsys()
    try:
        iax = mycs.findaxisbyname("spectral")
    except:
        print("ERROR: can't find spectral axis.  Assuming it is 3.")
        iax = 3
    mycs.done()
    if needToClose: myia.close()
    return iax

def countUnmaskedPixels(img, useImstat=True, outdir='.'):
    """
    This function is called by meanSpectrumFromMom0Mom8JointMask.
    Returns number of unmasked pixels in an multi-dimensional image, i.e. 
    where the internal mask is True.
    Total pixels: spatial * spectral * Stokes
    Todd Hunter
    """
    if useImstat:
        stats = imstat(img, listit=imstatListit, verbose=imstatVerbose)
        if True:
            # append the statistics of dirty cube to a text file for this MOUS (PIPE-2171)
            mous = os.path.basename(img).split('.')[0]
            if outdir != '':
                filename = os.path.join(outdir,mous + '_dirtyCubeStats_CASA%s.txt'%(casaVersion))
            else:
                filename = os.path.join(os.path.dirname(img), mous + '_dirtyCubeStats_CASA%s.txt'%(casaVersion))
            alreadyExists = os.path.exists(filename)
            f = open(filename,'a')
            if not alreadyExists:
                f.write('# dirtyCubeName min max rms medabsdevmed\n')
            mymin = stats['min']; mymax = stats['max']; mystd = stats['rms']; mymad = stats['medabsdevmed']
            mymin = pickFirstElementIfNecessary(mymin)
            mymax = pickFirstElementIfNecessary(mymax)
            mystd = pickFirstElementIfNecessary(mystd)
            mymad = pickFirstElementIfNecessary(mymad)
            f.write('%s %.7f %.7f %.7f %.7f\n' % (img,mymin,mymax,mystd,mymad))
            f.close()
        
        npix = stats['npts']
        if type(npix) == list or type(npix) == np.ndarray:
            if len(npix) == 0:
                npix = 0
            else:
                npix = int(npix[0])
        return npix
    else:
        myia = iatool()
        myia.open(img)
        maskdata = myia.getregion(getmask=True)
        myia.close()
        idx = np.where(maskdata==0)[0]
        maskedPixels = len(idx)
        pixels = np.prod(np.shape(maskdata))
        return pixels-maskedPixels

def countPixelsAboveZero(img, pbmom=None, useImstat=True, value=0):
    """
    This function is called by meanSpectrumFromMom0Mom8JointMask.
    Returns number of pixels > a specified threshold.
    pbmom: if specified, and useImstat==True, then also require pb>0.21
    Note that the value of 0.21 is also used in findOuterAnnulusForPBCube and should be changed in parallel
    value: threshold (default=0)
    Total pixels: spatial * spectral * Stokes
    Todd Hunter
    """
    if not os.path.exists(img):
        print("Could not find image: ", img)
        return
    if useImstat:
        if pbmom is None:
            mask = '"%s">0'%(img)
        else:
            mask = '"%s">0 && "%s">0.21' % (img,pbmom)
        npix = imstat(img, mask=mask, listit=imstatListit, verbose=imstatVerbose)['npts']
        if type(npix) == list or type(npix) == np.ndarray:
            if len(npix) == 0:
                npix = 0
            else:
                npix = int(npix[0])
        return npix
    else:
        myia = iatool()
        myia.open(img)
        data = myia.getregion()
        myia.close()
        idx = np.where(data>value)[0]
        pixels = len(idx)
        return pixels

def flattenMask(mask, outfile='', overwrite=True):
    """
    Takes a multi-channel CASA image mask and propagates all pixels to a 
    new single plane image mask
    outfile: default name = <mask>.flattened
    - Todd Hunter
    """
    if (not os.path.exists(mask)):
        print("Could not find mask image.")
        return
    if (outfile == ''):
        outfile = mask + '.flattened'
    axis = findSpectralAxis(mask)
    imcollapse(mask, outfile=outfile, function='max', axes=axis, 
               overwrite=overwrite)
    return outfile

def imagePercentileNoMask(img, percentiles):
    """
    percentiles: a single value or a list
    Returns: a single value or a list
    """
    myia = iatool()
    myia.open(img)
    mymask = myia.getregion(getmask=True)
    pixels = myia.getregion(getmask=False)
    myia.close()
    if not isinstance(percentiles, (list, np.ndarray)): # type(percentiles) != list and type(percentiles) != np.ndarray:
        value = scoreatpercentile(pixels[np.where(mymask > 0.5)], percentiles)
        return value
    value = []
    for percentile in percentiles:
        value.append(scoreatpercentile(pixels[np.where(mymask > 0.5)], percentile))
    return value

def meanSpectrumFromMom0Mom8JointMask(cube, imageInfo, nchan, pbcube=None, psfcube=None, minbeamfrac=0.3, 
                                      projectCode='', overwriteMoments=False, 
                                      overwriteMasks=True, phase2=True, 
                                      normalizeByMAD=True, minPixelsInJointMask=3,
                                      initialQuadraticImprovementThreshold=1.6, userJointMask='', 
                                      snrThreshold=23, mom0minsnr=MOM0MINSNR_DEFAULT, 
                                      mom8minsnr=MOM8MINSNR_DEFAULT, rmStatContQuadratic=True,
                                      bidirectionalMaskPhase2=False, outdir='', 
                                      avoidExtremaInNoiseCalcForJointMask=False, momentdir='',
                                      statistic='mean', pbmom=None, nbin=1, window='flat', 
                                      maxBaseline=150, subimage=False, mom08simultaneous=True,
                                      enableInitialQuadratic=False):
    """
    This function is called by runFindContinuum when meanSpectrumMethod='mom0mom8jointMask'.
    This is the new heuristic for Cycle 6 which creates the moment 0 and moment 8 images
    for a cube, takes their union and determines the mean spectrum by calling 
    computeStatisticalSpectrumFromMask(), which uses ia.getprofile.
    pbcube: if not specified, then assume '.residual' should be replaced in the name by '.pb'
    pbmom: if not specified, then assume '.pb' should be replaced in the name by '.pbmom'
    overwriteMoments: rebuild the mom0 and mom8 images even if they already exist
    overwriteMasks: rebuild the mom0 and mom8 mask images even if they already exist
    phase2: if True, then run a second phase if SNR is high
    minPixelsInJointMask: if fewer than these pixels are found, then use all pixels above pb=0.3
    userJointMask: if specified, use this joint mask instead of computing one; if it is a cube mask,
       as from tclean, then this function will form a flattened version first, and then use that
    snrThreshold: if SNR is higher than this, and phase2==True, then run a phase 2 mask calculation
    mom0minsnr: sets the threshold for the mom0 image (i.e. median + mom0minsnr*scaledMAD of the whole image)
    mom8minsnr: sets the threshold for the mom8 image (i.e. median + mom8minsnr*scaledMAD of thw whole image)
    enableInitialQuadratic: if True, assess the need for quadratic removal, and use the 
        old-style method to remove it
    rmStatContQuadratic: if True, then do not do the initialQuadratic removal (if enabled
       and deemed necessary), but instead always perform the new-style quadratic removal
    bidirectionalMaskPhase2: True=extend mask to negative values beyond threshold; 
            False=Cycle6+ behavior (True was tried but gives worse results in some cases)
    outdir: directory to write the mom0, mom8, masks, .dat and mean spectrum text files
    avoidExtremaInNoiseCalcForJointMask: experimental Boolean to avoid pixels <5%ile and >95%ile
             in the chauvenet imstat of mom0 and mom8 images
    momentdir: alternate directory to look for existing mom0 and mom8 images
    mom08simultaneous: if True, set moments=[0,8] in one call to immoments instead of two
    Returns: 17 things: 
       * 1) the meanSpectrum
       * 2) a Boolean which states whether normalization was applied
       * 3) the number of pixels in the mask
       * 4) a Boolean for whether the mask reverted to pb-based
       * 5) a Boolean for whether a quadratic was removed
       * 6) the initialQuadraticImprovementRatio
       * 7) the list of 3 mom0snrs
       * 8) the list of 3 mom8snrs 
       * 9) the number of regions pruned
       * 10) number of pixels in moment 8 mask
       * 11) mom0peak (Jy*km/s)
       * 12) mom8peak (Jy)
       * 13) name of jointMask file
       * 14) nbin
       * 15) initialPeakOverMad
       * 16) mom0peakOverMad
       * 17) mom8peakOverMad
    """
    print("momentdir = ", momentdir)
    overwritePhase2 = True
    # Look for the .pb image if it was not specified
    if pbcube is None:
        pbcube = locatePBCube(cube)
    if pbcube is None:
        print("meanSpectrumFromMom0Mom8JointMask(): Could not find pbcube")
        return
    if pbmom is None:
        pbmom = os.path.join(outdir,os.path.basename(pbcube)) + 'mom'
        if not os.path.exists(pbmom):
            if nchan < 16:  # PIPE-1657
                pbmomChans = ''
            else:
                pbmomChans = '%d,%d,%d,%d,%d' % (nchan//8, nchan//4, nchan//2, 3*nchan//4, 7*nchan//8)
            casalogPost("Running immoments('%s', moments=[-1], outfile='%s', chans='%s')" % (pbcube,pbmom,pbmomChans))
            immoments(pbcube, moments=[-1], outfile=pbmom, chans=pbmomChans)
        else:
            casalogPost("Re-using existing pbmom: %s" % (pbmom))
    # Look for the .psf image if it was not specified
    if psfcube is None:
        if cube.find('.residual') >= 0: 
            if os.path.islink(cube):
                psfcube = os.readlink(cube).replace('.residual','.psf')
            else:
                psfcube = cube.replace('.residual','.psf')
            if not os.path.exists(psfcube):
                psfcube = None
    casalogPost('psfcube = %s' % (psfcube))
    if mom08simultaneous:
        mom0suffix = '.integrated'
        mom8suffix = '.maximum'
    else:
        mom0suffix = '.mom0'
        mom8suffix = '.mom8'
    if momentdir == '':
        if outdir == '':
            momRoot = cube +'.mom'
        else:
            momRoot = os.path.join(outdir,os.path.basename(cube)+'.mom')
        mom0 = momRoot + mom0suffix
        mom8 = momRoot + mom8suffix
    else:
        # Look first in the output directory
        momRoot = os.path.join(outdir,os.path.basename(cube)+'.mom')
        mom0 = momRoot + mom0suffix
        mom8 = momRoot + mom8suffix
        if not os.path.exists(mom0):
            # now look in the momentdir directory
            firstParentDir = os.path.dirname(cube).split()[-1]
            if momentdir[0] == '.':
                momentdir = os.path.join(os.getcwd(),momentdir)
            mom0src = os.path.join(momentdir, os.path.basename(cube)+mom0suffix)
            print("Looking for mom0 at: ", mom0src)
            if os.path.exists(mom0src):
                casalogPost('Creating symlink to moment 0')
                os.symlink(mom0src, mom0)
        if not os.path.exists(mom8):
            mom8src = os.path.join(momentdir, os.path.basename(cube)+mom8suffix)
            print("Looking for mom8 at: ", mom8src)
            if os.path.exists(mom8src):
                casalogPost('Creating symlink to moment 8')
                os.symlink(mom8src, mom8)
    if os.path.exists(mom0):
        if overwriteMoments:
            os.system('rm -rf ' + mom0)
        else:
            mom0info = getImageInfo(mom0)
            # check if RA/Dec images sizes match
            if imageInfo[5] == mom0info[5] and imageInfo[6] == mom0info[6]:
                print("Re-using existing moment0 image at %s" % (mom0))
            else:
                print("Mismatch between cube and mom0 image, rebuilding mom0")
                os.system('rm -rf ' + mom0)
    if os.path.exists(mom8):
        if overwriteMoments:
            os.system('rm -rf ' + mom8)
        else:
            mom8info = getImageInfo(mom8)
            # check if RA/Dec images sizes match
            if imageInfo[5] == mom8info[5] and imageInfo[6] == mom8info[6]:
                print("Re-using existing moment8 image at %s" % (mom8))
            else:
                print("Mismatch between cube and mom8 image, rebuilding mom8")
                os.system('rm -rf ' + mom8)
    if mom08simultaneous:
        if not os.path.exists(mom0) and not os.path.exists(mom8):
            casalogPost("Running immoments('%s', moments=[0,8], outfile='%s', mask='%s'!=0" % (cube,momRoot,cube))
            immoments(cube, moments=[0,8], outfile=momRoot, mask='"%s"!=0'%(cube)) # PL use case
        elif not os.path.exists(mom0):
            immoments(cube, moments=[0], outfile=momRoot+'.integrated', mask='"%s"!=0'%(cube))
        elif not os.path.exists(mom8): 
            immoments(cube, moments=[8], outfile=momRoot+'.maximum', mask='"%s"!=0'%(cube))
        moment0name = momRoot+'.integrated'
        moment8name = momRoot+'.maximum'
    else:
        if not os.path.exists(mom0):
            print("Could not find: ", mom0)
            casalogPost("Running immoments('%s', moments=[0], outfile='%s')" % (cube, mom0))
    #        immoments(cube, moments=[8], outfile=mom8) # PL2022
            immoments(cube, moments=[0], outfile=mom0, mask='"%s"!=0'%(cube))
        if not os.path.exists(mom8):
            casalogPost("Running immoments('%s', moments=[8], outfile='%s')" % (cube, mom8))
            immoments(cube, moments=[8], outfile=mom8, mask='"%s"!=0'%(cube))
        moment0name = mom0
        moment8name = mom8

    # PL2024: PIPE-1992 if dynamic range of mom0 is high, raise the mom0minsnr mask threshold
    imstatResults = imstat(moment0name)
    mom0peakOverMad = imstatResults['max']/imstatResults['medabsdevmed']
    mom0peakOverMad = pickFirstElementIfNecessary(mom0peakOverMad)   # fix for PIPE-2673
    imstatResults = imstat(moment8name)
    mom8peakOverMad = imstatResults['max']/imstatResults['medabsdevmed']
    mom8peakOverMad = pickFirstElementIfNecessary(mom8peakOverMad)  # fix for PIPE-2673
    if mom0peakOverMad > 90:    # 2019.1.01184 needs 93
        new_mom0minsnr = np.max([mom0minsnr, 8])
        casalogPost('Raising mom0minsnr from %f to %f because mom0peakOverMad=%.1f > 90' % (mom0minsnr,new_mom0minsnr,mom0peakOverMad))
        mom0minsnr = new_mom0minsnr
    
    pbBasedMask = False
    mom0mask = mom0+'.mask_bi'
    mom0mask2 = mom0+'.mask2_bi'
    mom8mask = mom8+'.mask_bi'
    mom8mask2 = mom8+'.mask2_bi'
    if outdir == '':
        jointMask = cube+'.joint.mask'
        jointMask2 = cube+'.joint.mask2'
    else:
        jointMask = os.path.join(outdir, os.path.basename(cube)+'.joint.mask')
        jointMask2 = os.path.join(outdir, os.path.basename(cube)+'.joint.mask2')
        
    lowerAnnulusLevel = None
    higherAnnulusLevel = None
#    sevenMeter = False
#    if sevenMeter:
#        snrThreshold = 20
#    else:
#        snrThreshold = 25
    regionsPruned = 0
#    if overwriteMasks or not os.path.exists(mom0mask) or not os.path.exists(mom8mask):
    if (overwriteMasks or not os.path.exists(mom0mask) or not os.path.exists(mom8mask)) and userJointMask == '':
        #####################
        # Build Moment 0 mask
        #####################
        nptsInCube = countUnmaskedPixels(cube, outdir=outdir)
        cmd = 'rm -rf %s.mask*' % (mom0)
        print("running: %s" % cmd)
        os.system(cmd)
        classicResult = imstat(mom0, listit=imstatListit, verbose=imstatVerbose)
        if avoidExtremaInNoiseCalcForJointMask:
            avoidLow, avoidHigh = imagePercentileNoMask(mom0, [5,95])
            mask = '"%s" > %.9f && "%s" < %.9f' % (mom0, avoidLow, mom0, avoidHigh)
            print("running imstat('%s', algorithm='chauvenet', maxiter=5, mask='%s')" % (mom0,mask))
            result = imstat(mom0, algorithm='chauvenet', maxiter=5, listit=imstatListit, verbose=imstatVerbose,
                            mask=mask)
        else:
            print("running imstat('%s', algorithm='chauvenet', maxiter=5)" % (mom0))
            result = imstat(mom0, algorithm='chauvenet', maxiter=5, listit=imstatListit, verbose=imstatVerbose)
        mom0peak = classicResult['max'][0]
        mom0snr = classicResult['max'][0]/result['medabsdevmed'][0]
        scaledMAD = result['medabsdevmed'][0]*1.4826
        if False:
            mom0sigma = mom0minsnr
        else:
            if mom0minsnr == MOM0MINSNR_DEFAULT:
                mom0sigma = np.max([mom0minsnr,oneEvent(nptsInCube,1)])
            else:
                mom0sigma = mom0minsnr
        casalogPost("+++ nptsInCube=%d, initial mom0sigma=%f, scaledMAD=%f,  mom0median=%f" % (nptsInCube, mom0sigma, scaledMAD, result['median'][0]))
        mom0threshold = mom0sigma*scaledMAD + result['median'][0]
        mom0min = classicResult['min'][0]
        mom0max = classicResult['max'][0]
        # the test against mom0max is to prevent infinite loop
        while (mom0min >= mom0threshold and mom0threshold < mom0max):
            # Then there will be no points that satisfy subsequent mask, so raise the threshold so that at least some points are considered to be signal-free
            mom0sigma += 1
            print("   %e < %e: increasing mom0sigma to %d = %f" % (mom0min,mom0threshold,mom0sigma,mom0sigma*scaledMAD))
            mom0threshold = mom0sigma*scaledMAD + result['median'][0]

        mask = '"%s" > %.9f || "%s" < -%.9f' % (mom0, mom0threshold, mom0, mom0threshold)
        print("applying mask to mom0: ", mask)
        # mom0.mask_chauv_bi: this image will have the true values of the image where it is not masked
        imsubimage(mom0, mask=mask, outfile=mom0+'.mask_chauv_bi')
        # mom0.mask_bi:       this image will have the value of 1.0 where it is not masked
        makemask(mode='copy', inpimage=mom0+'.mask_chauv_bi',
                 inpmask=mom0+'.mask_chauv_bi:mask0', output=mom0mask)

        #####################
        # Build Moment 8 mask
        #####################
        cmd = 'rm -rf %s.mask*' % (mom8)
        print("running: %s" % cmd)
        os.system(cmd)
        classicResult = imstat(mom8, listit=imstatListit, verbose=imstatVerbose)
        if avoidExtremaInNoiseCalcForJointMask:
            avoidLow8, avoidHigh8 = imagePercentileNoMask(mom8, [5,95])
            mask = '"%s" > %.9f && "%s" < %.9f' % (mom8, avoidLow8, mom8, avoidHigh8)
            print("running imstat('%s', algorithm='chauvenet', maxiter=5, mask='%s')" % (mom8,mask))
            result = imstat(mom8, algorithm='chauvenet', maxiter=5, listit=imstatListit, verbose=imstatVerbose,
                            mask=mask)
        else:
            result = imstat(mom8, algorithm='chauvenet', maxiter=5, listit=imstatListit, verbose=imstatVerbose)
        mom8peak = classicResult['max'][0]
        myMAD8 = result['medabsdevmed'][0]
        scaledMAD = result['medabsdevmed'][0]*1.4826
        print("+++mom8 scaled MAD = %f" % (scaledMAD))
        if scaledMAD <= 0:
            scaledMAD = result['sigma'][0]
            myMAD8 = scaledMAD / 1.4826
            casalogPost("Using sigma = %g instead of scaled MAD, which was zero." % (scaledMAD))
        mom8snr = classicResult['max'][0]/myMAD8
        if False:
            oneEventPoints = 0.05
            if mom8minsnr == MOM8MINSNR_DEFAULT:
                mom8sigma = np.max([mom8minsnr,oneEvent(nptsInCube//nchan,oneEventPoints)])
            else:
                mom8sigma = mom8minsnr
            casalogPost("Choosing %f from mom8minsnr=%f, oneEvent=%f" % (mom8sigma, mom8minsnr, oneEvent(nptsInCube//nchan,oneEventPoints)))
        else:
            # Cycle 7
            oneEventPoints = 1
            if mom8minsnr == MOM8MINSNR_DEFAULT:
                mom8sigma = np.max([mom8minsnr,oneEvent(nptsInCube,oneEventPoints)])
            else:
                mom8sigma = mom8minsnr
            casalogPost("Choosing %f from mom8minsnr=%f, oneEvent=%f" % (mom8sigma, mom8minsnr, oneEvent(nptsInCube,oneEventPoints)))
        mom8min = classicResult['min'][0]
        mom8max = classicResult['max'][0]
        mom8threshold = mom8sigma*scaledMAD + result['median'][0]
        # the test against mom8max is to prevent infinite loop
        casalogPost('++++++ Initial mom8sigma = %f, scaledMAD=%f,  mom8median=%f, mom8min=%g, mom8max=%g, mom8threshold=%g' % (mom8sigma, scaledMAD, result['median'][0], mom8min, mom8max, mom8threshold))
        while (mom8min >= mom8threshold and mom8threshold < mom8max):
            # Then there will be no points that satisfy subsequent mask, so raise the threshold so that at least some points are considered to be signal-free
            mom8sigma += 1
            mom8threshold = mom8sigma*scaledMAD + result['median'][0]
            casalogPost("  %e>%e:  increasing mom8sigma to %d: %f" % (mom8min,mom8threshold,mom8sigma,mom8sigma*scaledMAD))

        mask = '"%s" > %.9f || "%s" < -%.9f' % (mom8, mom8threshold, mom8, mom8threshold)
        print("applying mask to mom8: ", mask)
        # mom8.mask_chauv_bi: this image will have the true values of the image where it is not masked
        imsubimage(mom8, mask=mask, outfile=mom8+'.mask_chauv_bi')
        # mom8.mask_bi: this image will have the true values of the image where it is not masked
        makemask(mode='copy', inpimage=mom8+'.mask_chauv_bi',
                 inpmask=mom8+'.mask_chauv_bi:mask0', output=mom8mask)
        numberPixelsInMom8Mask = countPixelsAboveZero(mom8mask, pbmom)

        ####################
        # Build joint mask
        ####################
        os.system('rm -rf %s' % (jointMask))
        casalogPost("Running makemask(inpimage='%s', mode='copy', inpmask=['%s','%s'], output='%s')" % (mom0, mom0mask, mom8mask, jointMask))
        jointMaskTemp = jointMask + '.tmp'
        removeIfNecessary(jointMaskTemp)
        makemask(inpimage=mom0, mode='copy', inpmask=[mom0mask, mom8mask], 
                 output=jointMaskTemp)
        if minbeamfrac > 0:
            regionsPruned = pruneMask(jointMaskTemp, psfcube, minbeamfrac)

        # try to get pb mask onto the joint.mask = PIPE-1629
        casalogPost("Running imsubimage('%s', mask='%s'>0, outfile='%s')" % (jointMaskTemp,pbmom,jointMask))
        # looks like you need to protect mask with quotes if name includes a '/' character
        imsubimage(jointMaskTemp, mask="'%s'>0"%pbmom, outfile=jointMask)  # try adding > 0, yes we need it!
        removeIfNecessary(jointMaskTemp)
            
        pixelsInMask = imstat(jointMask, listit=imstatListit, verbose=imstatVerbose)['max'][0] > 0.5
        jointMask1 = jointMask

        snr = np.max([mom0snr,mom8snr])
        mom0snr2 = None
        mom8snr2 = None
        mom0snr3 = None
        mom8snr3 = None
        mom0sigma2 = None
        mom8sigma2 = None
        if os.path.exists(jointMask2):
            # remove any old version to avoid confusion if phase2 is not run
            os.system('rm -rf %s*' % (jointMask2))
        if phase2 and snr > snrThreshold:
            casalogPost("Doing phase 2 mask calculation because one or both SNR > %.0f (mom0=%f,mom8=%f)" % (snrThreshold,mom0snr,mom8snr))
            if (overwritePhase2 or not os.path.exists(mom0mask2) 
                or not os.path.exists(mom8mask2)) and pixelsInMask:
                ####################################################################
                # Recompute statistics using pixels outside the initial coarse mask
                # because it will likely yield a lower MAD value.
                #################################################
                # Build Mom 0 mask
                ##################
                os.system('rm -rf %s.mask2*' % (mom0))
                print("running imstat('%s', mask='%s'<0.5)" % (mom0,jointMask))
                classicResult = imstat(mom0, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                print("running imstat('%s', algorithm='chauvenet', maxiter=5, mask='%s'<0.5)" % (mom0,jointMask))
                result = imstat(mom0, algorithm='chauvenet', maxiter=5, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                # Compute SNR (peak/MAD) outside of phase 1 mask
                mom0snr2 = classicResult['max'][0]/result['medabsdevmed'][0]
                scaledMAD = result['medabsdevmed'][0]*1.4826
                resultPositive = imstat(mom0, algorithm='chauvenet', maxiter=5, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                if len(resultPositive['medabsdevmed']) == 0:
                    casalogPost("WARNING: zero-length results from imstat(mom0)", priority='WARN')
                # reduce the sigma somewhat: 
                if False:
                    mom0sigma2 = mom0minsnr-1
                else:  # Cycle 7
                    if mom0minsnr == MOM0MINSNR_DEFAULT:
                        mom0sigma2 = np.max([mom0minsnr-1,oneEvent(nptsInCube,0.5)])
                    else:
                        mom0sigma2 = mom0minsnr-1
                mom0threshold = mom0sigma2*scaledMAD + result['median'][0]
                casalogPost('++++++ phase 2 mom0sigma2=%f, npts=%d, scaledMAD=%f, median=%f' % (mom0sigma2, result['npts'][0], scaledMAD, result['median'][0]))
                casalogPost('++++++ phase 2 mom0threshold = %f' % mom0threshold)
                if bidirectionalMaskPhase2:
                    mask = '"%s" > %.9f || "%s" < -%.9f' % (mom0, mom0threshold, mom0, mom0threshold)
                else:
                    mask = '"%s" > %f' % (mom0, mom0threshold)  # Cycle 6 release
                imsubimage(mom0, mask=mask, outfile=mom0+'.mask2_chauv')
                makemask(mode='copy', inpimage=mom0+'.mask2_chauv',
                         inpmask=mom0+'.mask2_chauv:mask0', output=mom0mask2)

                ##################
                # Build Mom 8 mask
                ##################
                cmd = 'rm -rf %s.mask2*' % (mom8)
                print("Running: ", cmd)
                os.system(cmd)
                classicResult = imstat(mom8, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                print("Running imstat('%s',algorithm='chauvenet',maxiter=5,mask='\"%s\"<0.5, listit=%s, verbose=%s)" % (mom8, jointMask, imstatListit, imstatVerbose))
                result = imstat(mom8, algorithm='chauvenet', maxiter=5, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                myMAD8 = result['medabsdevmed'][0]
                if myMAD8 <= 0.0:
                    # mom8 images can potentially have a majority of identical 0, if immoments(mask) does not avoid 0.0
                    myMAD8 = result['sigma'][0]/1.4826
                mom8snr2 = classicResult['max'][0]/myMAD8
                if len(result['medabsdevmed']) == 0:
                    print("WARNING: zero-length results from mom8")
                scaledMAD = result['medabsdevmed'][0]*1.4826
                if mom8minsnr == MOM8MINSNR_DEFAULT:
                    mom8sigma2 = np.max([mom8minsnr,oneEvent(nptsInCube,0.5)])
#                    mom8sigma2 = np.max([mom8minsnr,oneEvent(nptsInCube//nchan,oneEventPoints)]) # was 0.5
                else:
                    mom8sigma2 = mom8minsnr
                mom8threshold = mom8sigma2*scaledMAD + result['median'][0]
                casalogPost('++++++ phase 2 mom8sigma2=%f, npts=%d, scaledMAD=%f, median=%f' % (mom8sigma2, result['npts'][0], scaledMAD, result['median'][0]))
                casalogPost('++++++ phase 2 mom8threshold = %f' % mom8threshold)
                if bidirectionalMaskPhase2:
                    mask = '"%s" > %.9f || "%s" < -%.9f' % (mom8, mom8threshold, mom8, mom8threshold)
                else:
                    mask = '"%s" > %f' % (mom8, mom8threshold)  # Cycle 6 release
                print("Running imsubimage('%s', mask='%s', outfile='%s')" % (mom8, mask, mom8+'.mask2_chauv'))
                imsubimage(mom8, mask=mask, outfile=mom8+'.mask2_chauv')
                print("Running makemask(mode='copy', inpimage='%s', inpmask='%s', output='%s')" % (mom8+'.mask2_chauv', mom8+'.mask2_chauv:mask0', mom8mask2))
                myia = iatool()
                myia.open(mom8+'.mask2_chauv')
                if myia.maskhandler('default')[0] != '':
                    mom8mask2exists = True
                else:
                    casalogPost('++++++ phase 2 mom8.mask2_chauv has no mask (no qualifying pixels).')
                    mom8mask2exists = False
                myia.close()
                if mom8mask2exists:
                    makemask(mode='copy', inpimage=mom8+'.mask2_chauv',
                           inpmask=mom8+'.mask2_chauv:mask0', output=mom8mask2)
                    inpmask = [mom0mask2, mom8mask2]
                else:
                    inpmask = [mom0mask2]

                ##########################
                # Build second joint mask
                ##########################
                jointMask2temp = jointMask2+'.tmp'
                makemask(inpimage=mom0, mode='copy', inpmask=inpmask, 
                         output=jointMask2temp)
                if minbeamfrac > 0:
                    regionsPruned = pruneMask(jointMask2temp, psfcube, minbeamfrac)

                # try to get pb mask onto the joint.mask2 = PIPE-1629
                casalogPost("Running imsubimage('%s', mask='%s>0', outfile='%s')" % (jointMask2temp,pbmom,jointMask2))
                # looks like you need to protect mask with quotes if name includes a '/' character
                imsubimage(jointMask2temp, mask="'%s'>0"%pbmom, outfile=jointMask2)  # try adding > 0, yes we need it!
                removeIfNecessary(jointMask2temp)

                jointMask = jointMask2
                pixelsInMask = imstat(jointMask2, listit=imstatListit, verbose=imstatVerbose)['max'][0] > 0.5

                # Compute SNR (peak/MAD) outside of phase 2 mask
                classicResult = imstat(mom0, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                result = imstat(mom0, algorithm='chauvenet', maxiter=5, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                mom0snr3 = classicResult['max'][0]/result['medabsdevmed'][0]
                classicResult = imstat(mom8, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                result = imstat(mom8, algorithm='chauvenet', maxiter=5, mask='"%s"<0.5'%jointMask, listit=imstatListit, verbose=imstatVerbose)
                myMAD8 = result['medabsdevmed'][0]
                if myMAD8 < 0.0:
                    # mom8 images can potentially have a majority of identical 0, if immoments(mask) does not avoid 0.0
                    myMAD8 = result['sigma'][0] / 1.4826
                mom8snr3 = classicResult['max'][0] / myMAD8
        else:
            print("Not doing phase 2 because both SNR < %.0f (%f,%f)" % (snrThreshold,mom0snr,mom8snr))
            os.system('rm -rf %s.mask2*' % (mom0))
            os.system('rm -rf %s.mask2*' % (mom8))
        if outdir == '':
            meanSpectrumFile = cube+'.meanSpectrum.mom0mom8jointMask'
        else:
            meanSpectrumFile = os.path.join(outdir,os.path.basename(cube)+'.meanSpectrum.mom0mom8jointMask')
        mom0snrs = [mom0snr, mom0snr2, mom0snr3]
        mom8snrs = [mom8snr, mom8snr2, mom8snr3]
        # if no pixels were found in the mask, then build one from PB annulus, or use whole image if no PB is available.
        numberPixelsInMask = countPixelsAboveZero(jointMask, pbmom)
        if not pixelsInMask or numberPixelsInMask < minPixelsInJointMask:
            pbBasedMask = True
            os.system('rm -rf %s' % (jointMask)) 
            myMask1chan = jointMask + '.1chan'
            if pbcube is None:
                casalogPost("No pb was passed into %s and .residual does not appear in cube name. Using whole image." % (__file__),debug=True)
#                imsubimage(cube, chans='1', mask='"%s">-1000'%(cube), outfile=myMask1chan, overwrite=True)
                immath(cube, chans='1', mode='evalexpr', expr='iif(IM0>-1000, 1.0, 0.0)', outfile=jointMask)
            else:
                casalogPost("Because less than %d pixels were in the joint mask, building pb-based mask from %s" % (minPixelsInJointMask,pbcube), debug=True)
                lowerAnnulusLevel, higherAnnulusLevel = findOuterAnnulusForPBCube(pbcube, imstatListit, imstatVerbose, subimage, imageInfo)
                
#                imsubimage(pbcube, chans='1', mask='"%s">%f' % (pbcube,higherAnnulusLevel), outfile=myMask1chan, overwrite=True) 
                imsubimage(pbcube, chans='1', mask='"%s">%f && "%s">0' % (pbcube,higherAnnulusLevel,pbmom), outfile=myMask1chan, overwrite=True)  # PIPE-1629 for the pb-based jointmask case
                makemask(mode='copy', inpimage=myMask1chan, overwrite=True,
                         inpmask=myMask1chan+':mask0', output=jointMaskTemp)
                imsubimage(jointMaskTemp, mask='"%s">0' % (pbmom), outfile=jointMask, overwrite=True) # PIPE-1629 for the pb-based jointmask case
                removeIfNecessary(jointMaskTemp)
    else:
        classicResult = imstat(mom0, listit=imstatListit, verbose=imstatVerbose)
        mom0peak = classicResult['max'][0]
        result = imstat(mom0, algorithm='chauvenet', maxiter=5, listit=imstatListit, verbose=imstatVerbose)
        mom0snr = classicResult['max'][0]/result['medabsdevmed'][0]
        mom0snrs = [mom0snr,None,None]
        mom0threshold = 0
        classicResult = imstat(mom8, listit=imstatListit, verbose=imstatVerbose)
        mom8peak = classicResult['max'][0]
        myMAD8 = result['medabsdevmed'][0]
        mom8snr = classicResult['max'][0]/myMAD8
        mom8snrs = [mom8snr,None,None]
        mom8threshold = 0
        if os.path.exists(mom8mask):
            numberPixelsInMom8Mask = countPixelsAboveZero(mom8mask, pbmom)
        elif os.path.exists(userJointMask):
            numberPixelsInMom8Mask = countPixelsAboveZero(userJointMask, pbmom)
            casalogPost("Using userJointMask to set numberPixelsInMom8Mask = %d" % (numberPixelsInMom8Mask))
        else:
            numberPixelsInMom8Mask = 0
            casalogPost("Neither the mom8mask nor the userJointMask exists. Setting numberPixelsInMom8Mask = %d" % (numberPixelsInMom8Mask))
    if userJointMask != '':
        if numberOfChannelsInCube(userJointMask) > 1:
            jointMask = userJointMask + '.flattened'
            if os.path.exists(jointMask):
                print("Using existing flattened version of userJointMask")
            else:
                print("Running flattenMask('%s', '%s')" % (userJointMask, jointMask))
                flattenMask(userJointMask, jointMask)
        else:
            casalogPost("Setting jointMask = userJointMask")
            jointMask = userJointMask
        if outdir == '':
            if userJointMask.find('.amendedJointMask') > 0:  # special case name
                meanSpectrumFile = cube+'.meanSpectrum.amendedJointMask'
            else:
                meanSpectrumFile = cube+'.meanSpectrum.userJointMask'
        else:
            if userJointMask.find('.amendedJointMask') > 0:  # special case name
                meanSpectrumFile = os.path.join(outdir,os.path.basename(cube)+'.meanSpectrum.amendedJointMask')
            else:
                meanSpectrumFile = os.path.join(outdir,os.path.basename(cube)+'.meanSpectrum.userJointMask')
#        print("++++++++++++++++ Setting meanSpectrumFile = %s" % (meanSpectrumFile))
    computeFirstSpectrum = False
    if computeFirstSpectrum:
        channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(cube, jointMask1, pbcube, imageInfo, statistic, normalizeByMAD, projectCode, higherAnnulusLevel, lowerAnnulusLevel, outdir, jointMask, subimage)
        writeMeanSpectrum(meanSpectrumFile+'_bidirectional', frequency, intensity, 
                          intensity, mom0threshold, 
                          nchan, numberPixelsInMask, mom8threshold, centralArcsec='mom0mom8jointMask', 
                          mask=False, iteration=0)
    channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(cube, jointMask, pbcube, imageInfo, statistic, normalizeByMAD, projectCode, higherAnnulusLevel, lowerAnnulusLevel, outdir, jointMask, subimage)

    numberPixelsInMask = countPixelsAboveZero(jointMask, pbmom)
    if MAD(intensity) == 0.0:
        # Fix for CAS-11960
        pbBasedMask = True
        os.system('rm -rf %s' % (jointMask))
        myMask1chan = jointMask + '.1chan'
        casalogPost("Because less than %d pixels were in the joint mask, building pb-based mask from %s" % (minPixelsInJointMask,pbcube), debug=True)
        lowerAnnulusLevel, higherAnnulusLevel = findOuterAnnulusForPBCube(pbcube, imstatListit, imstatVerbose, subimage, imageInfo)
        
#        imsubimage(pbcube, chans='1', mask='"%s">%f' % (pbcube,higherAnnulusLevel), outfile=myMask1chan, overwrite=True)  # PL2023
        imsubimage(pbcube, chans='1', mask='"%s">%f && "%s">0' % (pbcube,higherAnnulusLevel,pbmom), outfile=myMask1chan, overwrite=True)  # PIPE-1629 for pb-based jointmask case
        jointMaskTemp = jointMask + '.tmp' # in case it was not defined earlier, which happened in PIPEREQ-368
        makemask(mode='copy', inpimage=myMask1chan, overwrite=True,
                 inpmask=myMask1chan+':mask0', output=jointMaskTemp)
        imsubimage(jointMaskTemp, mask='"%s">0' % (pbmom), outfile=jointMask, overwrite=True) # PIPE-1629 for the pb-based jointmask case
        removeIfNecessary(jointMaskTemp)

        channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(cube, jointMask, pbcube, imageInfo, statistic, normalizeByMAD, projectCode, higherAnnulusLevel, lowerAnnulusLevel, outdir, jointMask, subimage)
        numberPixelsInMask = countPixelsAboveZero(jointMask, pbmom)
        initialQuadraticRemoved = False
        initialQuadraticImprovementRatio = 1.0
    elif not rmStatContQuadratic and enableInitialQuadratic:
        intensity, initialQuadraticRemoved, initialQuadraticImprovementRatio = removeInitialQuadraticIfNeeded(intensity,initialQuadraticImprovementThreshold)
    else:
        initialQuadraticRemoved = False
        initialQuadraticImprovementRatio = 1.0
    initialPeakOverMad = (np.nanmax(intensity) - np.nanmedian(intensity)) / MAD(intensity)
    if nbin >= NBIN_THRESHOLD: # PIPE-848  need to add a maximum peak/MAD
        # smooth by nbin and re-run findContinuumChannels
        MAX_PEAK_OVER_MAD = 10
        if initialPeakOverMad > MAX_PEAK_OVER_MAD:
            # do not allow large nbin when strong lines are present
            if maxBaseline <= ACA_MAX_BASELINE:  # ACA
                nbin = 3 # np.min([3,NBIN_THRESHOLD])
            else:
                nbin = 2 # np.min([2,NBIN_THRESHOLD])
            casalogPost('Limiting nbin to %d because initialPeakOverMad = %.2f > %d' % (nbin,initialPeakOverMad,MAX_PEAK_OVER_MAD))
        casalogPost('Applying nbin of %d' % (nbin))
        intensity = smooth(intensity, nbin, window)  # original heuristic
    writeMeanSpectrum(meanSpectrumFile, frequency, intensity, 
                      intensity, mom0threshold, 
                      nchan, numberPixelsInMask, mom8threshold, centralArcsec='mom0mom8jointMask', 
                      mask=initialQuadraticRemoved, iteration=0)
    return intensity, normalized, numberPixelsInMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad, mom0peakOverMad, mom8peakOverMad
# immoment

def locatePBCube(cube):
    if cube.find('.residual') >= 0: 
        if os.path.islink(cube):
            pbcube = os.readlink(cube).replace('.residual','.pb')
        else:
            pbcube = cube.replace('.residual','.pb')
        if not os.path.exists(pbcube):
            pbcube = None
    elif cube.find('.image') >= 0:
        # need this section for it to work on a clean cube instead of dirty cube
        if os.path.islink(cube):
            pbcube = os.readlink(cube).replace('.image','.pb')
        else:
            pbcube = cube.replace('.image','.pb')
        if not os.path.exists(pbcube):
            pbcube = None
    else:
        pbcube = None
    return pbcube

def oneEvent(npts, events=1.0, positive=True, verbose=False):
    """
    This function is called by meanSpectrumFromMom0Mom8JointMask.
    For a specified size of a Gaussian population of data, compute the sigma 
    that gives just less than one event, by using scipy.special.erfinv.
    Inputs:
    positive: if True, then only considers positive events.
    events: how many events to allow
    verbose: passed to sigmaEvent
    Return:
    sigma: floating point value
    """
    odds = events/float(npts)
    if positive:
        odds *= 2
    sigma = (2**0.5)*(scipy.special.erfinv(1-odds))
    return(sigma)

def cornersEmptyEdgesNotEmpty(pbcube, imageInfo=None, middleChannel=None):
    """
    Checks if the four corner pixels of an image are masked while the four
    middle edge pixels are not masked --> True, otherwise False
    """
    if imageInfo is None:
        imageInfo = getImageInfo(pbcube)
    if middleChannel is None:
        middleChannel = numberOfChannelsInCube(pbcube)//2
    bmaj, bmin, bpa, cdelt1, cdelt2, naxis1, naxis2, freq, imgShape, crval1, crval2, maxBaselineIgnore, telescope = imageInfo
    response = []
    # corners
    box1 = '%d,%d,%d,%d' % (       0,        0,        0,        0) 
    box2 = '%d,%d,%d,%d' % (       0, naxis2-1,        0, naxis2-1)
    box3 = '%d,%d,%d,%d' % (naxis1-1,        0, naxis1-1,        0) 
    box4 = '%d,%d,%d,%d' % (naxis1-1, naxis2-1, naxis1-1, naxis2-1)
    corners = 0
    for box in [box1,box2,box3,box4]:
        corners += len(imstat(pbcube,chans=str(middleChannel), box=box)['npts'])
    # edges
    box1 = '%d,%d,%d,%d' % (        0, naxis2//2,        0, naxis2//2) 
    box2 = '%d,%d,%d,%d' % ( naxis1-1, naxis2//2, naxis1-1, naxis2//2)
    box3 = '%d,%d,%d,%d' % (naxis1//2,         0, naxis1/2,         0) 
    box4 = '%d,%d,%d,%d' % (naxis1//2,  naxis2-1, naxis1/2,  naxis2-1)
    edges = 0
    for box in [box1,box2,box3,box4]:
        edges += len(imstat(pbcube, chans=str(middleChannel), box=box)['npts'])
    return (corners == 0 and edges == 4)
    
def findOuterAnnulusForPBCube(pbcube, imstatListit=False, imstatVerbose=False, 
                              subimage=False, imageInfo=None):
    """
    Given a PB cube, finds the minimum sensitivity value, then computes the
    corresponding higher value to form an annulus.  Returns 0.2-0.3 for normal
    images that have not been mitigated.  The factor of 1.15 below will 
    effectively mimic a corresponding higher range.  For example:
    CASA <247>: au.gaussianBeamResponse(au.gaussianBeamOffset(0.5)/1.15, fwhm=1)
    Out[247]: 0.59207684245045789
    CASA <248>: au.gaussianBeamResponse(au.gaussianBeamOffset(0.8)/1.15, fwhm=1)
    Out[248]: 0.8447381483786558
    subimage: if True, then add 0.2 to both radii of the annulus
    """
    lowerAnnulusLevel = imstat(pbcube, listit=imstatListit, verbose=imstatVerbose)['min'][0]
    casalogPost('lowerAnnulusLevel = %f' % (lowerAnnulusLevel))
    if imageInfo is None:
        imageInfo = getImageInfo(pbcube)
    middleChannel = numberOfChannelsInCube(pbcube)//2
    if lowerAnnulusLevel > 0.22 or cornersEmptyEdgesNotEmpty(pbcube,imageInfo,middleChannel):
        # fix for PIPE-1579
        bmaj, bmin, bpa, cdelt1, cdelt2, naxis1, naxis2, freq, imgShape, crval1, crval2, maxBaselineIgnore, telescope = imageInfo
        response = []
        box1 = '%d,%d,%d,%d' % (       0,        0,        0, naxis2-1) 
        box2 = '%d,%d,%d,%d' % (       0, naxis2-1, naxis1-1, naxis2-1)
        box3 = '%d,%d,%d,%d' % (naxis1-1,        0, naxis1-1, naxis2-1) 
        box4 = '%d,%d,%d,%d' % (       0,        0, naxis1-1,        0) 
        for box in [box1,box2,box3,box4]:
            response.append(imstat(pbcube, chans=str(middleChannel), box=box)['max'])
        newLowerAnnulusLevel = np.nanmax(response)
        casalogPost('FOV mitigated, maximum pb along image edges: %f' % (newLowerAnnulusLevel))
        # compute a higher annulus level but keep the original lowerAnnulusLevel
        # in order to include the unmasked part of the corners in the annulus
        higherAnnulusLevel = gaussianBeamResponse(gaussianBeamOffset(newLowerAnnulusLevel)/1.15, fwhm=1)
    else:
        higherAnnulusLevel = gaussianBeamResponse(gaussianBeamOffset(lowerAnnulusLevel)/1.15, fwhm=1)
    casalogPost('higherAnnulusLevel = %f' % higherAnnulusLevel)
    # Note that the value of 0.21 is also used in countPixelsAboveZero and should be changed in parallel
    if subimage:
        increase = 0.2
        return np.max([0.21,lowerAnnulusLevel])+increase, higherAnnulusLevel+increase
    else:
        return np.max([0.21,lowerAnnulusLevel]), higherAnnulusLevel

def computeStatisticalSpectrumFromMask(cube, jointmask, pbimg=None, imageInfo=None,
                                       statistic='mean', normalizeByMAD=False,
                                       projectCode='', higherAnnulusLevel=None, lowerAnnulusLevel=None, 
                                       outdir='', jointMaskForNormalize=None, subimage=False):
    """
    New heuristic for Cycle 6 pipeline.  It is called by meanSpectrumFromMom0Mom8JointMask
    Uses ia.getprofile to compute the mean spectrum of a cube within a 
    masked area.
    jointmask: a 2D or 3D mask image indicating which pixels shall be used
               or an expression with < or > along with a 2D or 3D mask image
    jointMaskForNormalize: a 2D or 3D mask image to be used when normalizeByMAD==True
    pbimg: the path to the primary beam of this cube, or a 2D primary beam with an expression
    statistic: passed to ia.getprofile via the 'function' parameter
    normalizeByMAD: if True, then create the inverse of the jointmask and 
      normalize the spectrum by the spectrum of 'xmadm' (scaled MAD) 
      computed on the inverse mask
    subimage: if True, then raise the pb level annulus by 0.2
    Returns: three arrays: channels, freqs(Hz), intensities, and a Boolean 
       which says if normalization was applied
    """
    chanInfo = numberOfChannelsInCube(cube, returnChannelWidth=True, returnFreqs=True) 
    nchan, firstFreq, lastFreq, channelWidth = chanInfo # freqs are in Hz
    frequency = np.linspace(firstFreq, lastFreq, nchan)  # lsrk

    myia = iatool()
    myia.open(cube)
    casalogPost("ia.open('%s')" % (cube))
    axis = findSpectralAxis(myia)
    casalogPost("Using jointmask = %s" % (jointmask))
    if jointmask == '' or jointmask.find('>') > 0 or jointmask.find('<') > 0:
        # an expression was given
        jointmaskQuoted = jointmask
    else:
        # a filename was given
        jointmaskQuoted = '"'+jointmask+'">0'
    if pbimg != '':
        if pbimg is not None:
            if pbimg.find('"') < 0:
                # a plain pbimg was given
                # Note that the value of 0.21 is also used in countPixelsAboveZero and should be changed in parallel
                pbimgExpression = '"%s">0.21' % (pbimg)
            else:
                pbimgExpression = pbimg
            jointmaskQuoted += ' && ' + pbimgExpression
    casalogPost("Running ia.getprofile(axis=%d, function='%s', mask='%s', stretch=True)" % (axis, statistic, jointmaskQuoted))
    casalogPost(" on the cube: %s" % (cube))
    avgIntensity = myia.getprofile(axis=axis, function=statistic, mask=jointmaskQuoted, stretch=True)['values']
    myia.close()

#   Note: I tried putting this stage here, but a strong line at spw center (2016.1.00484.L) 
#   CS5-4 in spw21, will cause it to over-remove the quadratic, leaving a bowl which prevents
#   identifying the line in the later stage, and puts it in the continuum selection.
#    avgIntensity, initialQuadraticRemoved, initialQuadraticImprovementRatio = removeInitialQuadraticIfNeeded(avgIntensity,initialQuadraticImprovementThreshold)

    normalized = False  # start off by assuming it won't get normalized
    if normalizeByMAD:
        if pbimg is None:
            casalogPost("pbimg is None, will not attempt to use the sensitivity annulus")
            pbimgExists = False
        elif not os.path.exists(pbimg.lstrip('"').split('"')[0]):  # remove any > < expression
            casalogPost("Could not find primary beam cube: %s" % (pbimg))
            pbimgExists = False
        else:
            pbimgExists = True
        if not pbimgExists:
            if jointMaskForNormalize in ['',None]:
                outsideMaskQuoted = ''
            else:
                casalogPost("Computing potential normalization spectrum from outside the joint mask (to remove atmospheric features). jointmask=%s." % (jointMaskForNormalize))
                casalogPost("**** Number of unmasked pixels in jointmask = %s" % (countUnmaskedPixels(jointMaskForNormalize)))
                if jointMaskForNormalize.find('.joint.') > 0:
                    outsideMask = jointMaskForNormalize.replace('.joint.','.inverseJoint.')
                else:
                    outsideMask = jointMaskForNormalize + '.inverseJoint'
                    print("Generating outsideMask = %s" % (outsideMask))
                mask = '"' + jointMaskForNormalize + '"<0.5' 
                print("Running imsubimage('%s', mask='%s', outfile='%s')" % (jointMaskForNormalize, mask, outsideMask))
                os.system('rm -rf '+outsideMask)
                imsubimage(jointMaskForNormalize, mask=mask, outfile=outsideMask)
                print("**** Using unmasked pixels outside = ", countUnmaskedPixels(outsideMask))
                outsideMaskQuoted = '"'+outsideMask+'"'
        else:
            # Use the annulus region from 0.21-0.3, unless image does not go out that far, in 
            # which case we use a comparable annulus that begins at a higher level
            if higherAnnulusLevel is None:
                pbimg = pbimg.lstrip('"').split('"')[0]
                print("Running fc.findOuterAnnulusForPbcube('%s',False,False)" % (pbimg))
                lowerAnnulusLevel, higherAnnulusLevel = findOuterAnnulusForPBCube(pbimg, imstatListit, imstatVerbose, subimage, imageInfo)
            casalogPost("Computing potential normalization spectrum from MAD of an outer annulus <%.2f (to remove atmospheric features)." % (higherAnnulusLevel))
            if jointMaskForNormalize not in ['', None]:
                outsideMaskQuoted = '"%s">%g && "%s"<%g && "%s"<0.5'%(pbimg,lowerAnnulusLevel,pbimg,higherAnnulusLevel,jointMaskForNormalize)
            else:
                outsideMaskQuoted = '"%s">%g && "%s"<%g'%(pbimg,lowerAnnulusLevel,pbimg,higherAnnulusLevel)
        myia.open(cube)
        casalogPost("Running ia.getprofile(axis=%d, function='xmadm', mask='%s', stretch=True)" % (axis, outsideMaskQuoted))
        xmadm = myia.getprofile(axis=axis, function='xmadm', mask=outsideMaskQuoted, stretch=True)['values']
        # The following is only needed for debugging
        if True:
            avgIntensityOutsideMask = myia.getprofile(axis=axis, function='mean', mask=outsideMaskQuoted, stretch=True)['values']
        else:
            avgIntensityOutsideMask = avgIntensity*0.0
        myia.close()
        originalMAD = MAD(avgIntensity)
        originalMedian = np.median(avgIntensity)
        offset = np.max([0,-np.min(avgIntensity)])
        avgIntensityOffset = avgIntensity + offset
        avgIntensityNormToZero = np.max(avgIntensityOffset)/np.median(avgIntensityOffset) - 1  # this is a scalar
        xmadmNormToZero = xmadm - np.min(xmadm) # this is a vector
        normalizationFactor = 1 + xmadmNormToZero * avgIntensityNormToZero/np.max(xmadmNormToZero)  # this is a vector
        mymin = np.min(normalizationFactor)
        mymax = np.max(normalizationFactor)
        # 2018-04-05: avoid dividing by numbers near zero and thus inserting spikes
        if mymin < 0.05*mymax:
            normalizationFactor += 0.05*mymax - mymin # 1*np.median(normalizationFactor)
#        if mymin < 0.5:
#            normalizationFactor += 0.5-mymin
        myMAD = MAD(normalizationFactor)

        avgIntensityNormalized = avgIntensityOffset / normalizationFactor - offset

        # Proposed modification 2018-04-05 to avoid producing spectra with negative medians:
        myMedian = np.median(avgIntensityNormalized)
        newMADestimate = MAD(avgIntensityNormalized) # could raise this to account for processing
        avgIntensityNormalized = (avgIntensityNormalized-myMedian)*originalMAD/newMADestimate + originalMedian

        # Here is where you would put writeXmadmFile()

        # Here we use the highest common MAD because the number of pixels may 
        # be drastically different between the mask area and the 
        # outside-the-mask area.  We only need to try to remove the effect 
        # atmospheric lines if they dominate the signal spectrum.
        highestMAD = np.max([MAD(avgIntensity), MAD(xmadm)])
        casalogPost("peak(avgIntensity)=%f, MAD(avgIntensity)=%f, MAD(xmadm)=%f" % (np.max(avgIntensity),MAD(avgIntensity),MAD(xmadm)),debug=True)
        # This definition of peakOverMad will be high if there is strong continuum
        # or if there is a strong line. Removing the median from the peak would 
        # eliminate the sensitivity to continuum emission.
        peakOverMAD_signal = np.max(avgIntensity) / highestMAD
        peakOverMAD_xmadm = np.max(xmadm) / highestMAD
        applyNormalizationThreshold = 3.2 # was initially 3.5
        if False:
            # This method was attempted to try to resolve the 308 vs. 312, 355, 372 
            # discrepancy but caused too many other poor results.
            peakOverMAD_signal = (np.max(avgIntensity)-np.median(avgIntensity)) / MAD(avgIntensity)
            peakOverMAD_xmadm = (np.max(xmadm)-np.median(xmadm)) / MAD(xmadm)
            applyNormalizationThreshold = 1.4
            if tdmSpectrum(channelWidth,nchan):
                applyNormalizationThreshold *= 4
                
        projectCode = projectCode + ' '
        if peakOverMAD_xmadm > peakOverMAD_signal/applyNormalizationThreshold:
            avgIntensity = avgIntensityNormalized
            casalogPost('%sApplying normalization because peak/MAD of xmadm spectrum %f > (peak/MAD of signal %.3f/%.2f=%.3f)' % (projectCode, peakOverMAD_xmadm,peakOverMAD_signal,applyNormalizationThreshold,peakOverMAD_signal/applyNormalizationThreshold))
            normalized = True
        else:
            casalogPost('%sRejecting normalization because peak/MAD of xmadm spectrum %f <= (peak/MAD of signal %.3f/%.2f=%.3f)' % (projectCode, peakOverMAD_xmadm, peakOverMAD_signal, applyNormalizationThreshold, peakOverMAD_signal/applyNormalizationThreshold))
    else:
            casalogPost('Not-computing normalization because atmospheric variation considered too small')
    channels = list(range(len(avgIntensity)))
    if nchan != len(channels):
        print("Discrepant number of channels!")
    return np.array(channels), frequency, avgIntensity, normalized

def create_casa_quantity(myqatool,value,unit):
    """
    This function is called by CalcAtmTransmissionForImage, frames, and lsrkToRest.
    A wrapper to handle the changing ways in which casa quantities are invoked.
    Todd Hunter
    """
    if 'casac' in locals():
        if (type(casac.Quantity) != type):  # casa 4.x
            myqa = myqatool.quantity(value, unit)
        else:  # casa 3.x
            myqa = casac.Quantity(value, unit)
    else:
        # This is CASA 6 (same as 4.x and 5.x)
        myqa = myqatool.quantity(value,unit)
    return(myqa)

def MAD(a, c=0.6745, axis=0):
    """
    This function is called by removeInitialQuadraticIfNeeded, findContinuumChannels,
    meanSpectrum, computeStatisticalSpectrumFromMask, plotStatisticalSpectrumFromMask
    and runFindContinuum.
    Median Absolute Deviation along given axis of an array:
         median(abs(a - median(a))) / c
    c = 0.6745 is the constant to convert from MAD to std 
    """
    a = np.asarray(a, np.float64)
    m = nanmedian(a, axis=axis,
                  preop=lambda x:
                        np.fabs(np.subtract(x, np.median(x, axis=None), out=x), out=x))
    return m / c

def splitListIntoContiguousLists(mylist):
    """
    This function is called by findContinuumChannels.
    Called by copyweights. See also splitListIntoHomogenousLists.
    Converts [1,2,3,5,6,7] into [[1,2,3],[5,6,7]], etc.
    -Todd Hunter
    """
    mylists = []
    if (len(mylist) < 1):
        return(mylists)
    newlist = [int(mylist[0])]  # convert np.int64 to int
    for i in range(1,len(mylist)):
        if (mylist[i-1] != mylist[i]-1):
            mylists.append(newlist)
            newlist = [int(mylist[i])] # convert np.int64 to int
        else:
            newlist.append(int(mylist[i]))  # convert np.int64 to int
    mylists.append(newlist)
    return(mylists)

def splitListIntoContiguousListsAndRejectZeroStd(channels, values, nanmin=None, verbose=False):
    """
    This function is called by findContinuumChannels.
    Takes a list of numerical values, splits into contiguous lists and 
    removes those that have zero standard deviation in their associated values.
    Note that values *must* hold the values of the whole array, while channels 
    can be a subset.
    If nanmin is specified, then reject lists that contain more than 3 
    appearances of this value.
    """
    if verbose:
        print("splitListIntoContiguousListsAndRejectZeroStd:  len(values)=%d, len(channels)=%d" % (len(values), len(channels)))
    values = np.array(values)
    mylists = splitListIntoContiguousLists(channels)
    channels = []
    for i,mylist in enumerate(mylists):
        mystd = np.std(values[mylist])
        if (mystd > 1e-17):  # avoid blocks of identically-zero values
            if (nanmin is not None):
                minvalues = np.sum(values[i] == nanmin)             # fix for PIPE-2544
                if (float(minvalues)/len(mylist) > 0.1 and minvalues > 3):
                    print("Rejecting list %d with multiple min values (%d)" % (i,minvalues))
                    continue
            channels += mylist
    return(np.array(channels))

def convertChannelListIntoSelection(channels, trim=0, separator=';'):
    """
    This function is called by findContinuumChannels.
    Converts a list of channels into casa selection string syntax.  It first calls
    splitListIntoContiguousLists().
    channels: a list of channels
    trim: the number of channels to trim off of each edge of a block of channels
    """
    selection = ''
    firstList = True
    if (len(channels) > 0):
        mylists = splitListIntoContiguousLists(channels)
        for mylist in mylists:
            if (mylist[0]+trim <= mylist[-1]-trim):
                if (not firstList):
                    selection += separator
                selection += '%d~%d' % (mylist[0]+trim, mylist[-1]-trim)
                firstList = False
    return(selection)

def widenSelections(datfiles, eachside=1, minwidth=0, verbose=True, nchan=None):
    """
    Read a list of _findContinuum.dat files, and widen each channel range
    of the selection string by a number of channels on each side.
    datfiles: python list of datfiles, or a comma-delimited or wildcard string
    eachside: how many channels to add on each side
    minwidth: width of narrowest range to expand
    verbose: passed to widenSelection to print the new selection strings
    nchan: if specified, then avoid going past the final channel (nchan-1)
         can be an int if single file is passed (or all spws have same nchan);  
         or a list of ints
    Returns: if only 1 file passed, it returns the new selection string; if  
       multiple files are passed, then return the bandwidth increase factor
    """
    newBW = 0; oldBW = 0
    if isinstance(datfiles, (str, np.bytes_)):
        if datfiles.find('*') >= 0:
            datfiles = sorted(glob.glob(datfiles))
        else:
            datfiles = datfiles.split(',')
    print("Reading %d files" % (len(datfiles)))
    for i, datfile in enumerate(datfiles):
        if not os.path.exists(datfile):
            print("Could not find datfile: ", datfile)
            return
        selection = readContDat(datfile)
        chanwidth = readContDatChanWidth(datfile) * 1e9 # convert GHz to Hz
        oldBW += computeBandwidth(selection, chanwidth)
        if type(nchan) in [list, np.ndarray]:
            mynchan = nchan[i]
        else:
            mynchan = nchan
        newSelection = widenSelection(selection, eachside, minwidth, chanwidth, verbose, mynchan)
        newBW += computeBandwidth(newSelection, chanwidth)
    factor = newBW/oldBW
    if len(datfiles) > 1:
        print("Total bandwidth increase factor = %f" % factor)
        return factor
    else:
        print("Bandwidth increase factor = %f" % factor)
        return newSelection

def widenSelection(selection, eachside=1, minwidth=0, channelWidth=1.0, 
                   verbose=False, nchan=None):
    """
    Widen each component of a channel selection string on each side.
    selection: channel selection string
    eachside: how many channels to add on each side
    minwidth: width of narrowest range to expand
    channelWidth: width of channels in Hz
    verbose: if True, print the new selection
    nchan: if specified, then avoid going past the final channel (nchan-1)
    Returns: new selection string
    """
    channelList = convertSelectionIntoChannelList(selection)
    mylists = splitListIntoContiguousLists(channelList)
    newlist = []
    for mylist in mylists:
        if (mylist[-1]-mylist[0] >= minwidth):
            if nchan is None:
                newlist += range(np.max([0,mylist[0]-eachside]), mylist[-1]+eachside+1)
            else:
                newlist += range(np.max([0,mylist[0]-eachside]), np.min([mylist[-1]+eachside, nchan-1])+1)
        else:
            newlist += range(mylist[0], mylist[-1]+1)
    # the sorted(unique()) is needed to avoid producing adjacent selections 
    # that have no channel gap, i.e. they are merged into one wider selection,
    # in order to prevent problems downstream when converting to topo, etc.
    newSelection = convertChannelListIntoSelection(sorted(np.unique(newlist)))
    if verbose:
        print("Old selection: ", selection)
        print("New selection: ", newSelection)
    if channelWidth is not None:
        oldBW = computeBandwidth(selection, channelWidth)
        newBW = computeBandwidth(newSelection, channelWidth)
        factor = newBW/oldBW
        print("Bandwidth increase factor = %f" % factor)
    return newSelection

def convertSelectionIntoChannelList(selection):
    """
    Converts an arbitrary CASA selection string into an array of channel numbers
    '0~1;4~6' --> np.array([0,4,5])
    Ignores any initial spw number that preceeds a colon.  Will fail if there is more than one colon.
    """
    if selection.find(':') >= 0:
        spw, selection = selection.split(':')
    selections = selection.split(';')
    mylist = []
    for myrange in selections:
        if myrange.find('~') > 0:
            [c0,c1] = myrange.split('~')
            mylist += range(int(c0),int(c1)+1)
        else: # single value
            mylist += [int(myrange)]
    return np.array(mylist)
    
def CalcAtmTransmissionForImage(img, imageInfo, chanInfo='', airmass=1.5, pwv=-1,
                                spectralaxis=-1, value='transmission', P=-1, H=-1, 
                                T=-1, altitude=-1, vis='', source='', spw=''):
    """
    This function is called by atmosphereVariation.
    Supported telescopes are VLA and ALMA (needed for default weather and PWV)
    img: name of CASA image
    value: 'transmission' or 'tsky'
    chanInfo: a list containing nchan, firstFreqHz, lastFreqHz, channelWidthHz
    pwv: in mm
    P: in mbar
    H: in percent
    T: in Kelvin
    Returns:
    2 arrays: frequencies (in GHz) and values (Kelvin, or transmission: 0..1)
    """
    if not os.path.isdir(img):
        # Input was a spectrum rather than an image
        print("chanInfo: ", chanInfo)
        if (chanInfo[1] > 60e9):
            telescopeName = 'ALMA'
        else:
            telescopeName = 'VLA'
    else:
        telescopeName = getTelescope(img)
    freqs = np.linspace(chanInfo[1]*1e-9, chanInfo[2]*1e-9, chanInfo[0])
    from astroviper.distributed_applications.imaging.find_continuum.casa_image_tasks import atmosphere_spectrum
    return freqs, atmosphere_spectrum(img, value)
    numchan = len(freqs)
    lsrkwidth = (chanInfo[2] - chanInfo[1])/(numchan-1)
    result, fromFrame = cubeFrameToTopo(img, imageInfo, nchan=numchan, f0=chanInfo[1], f1=chanInfo[2], chanwidth=lsrkwidth, vis=vis, source=source, spw=spw)
    if (result is None):
        topofreqs = freqs
    else:
        topoWidth = (result[1]-result[0])/(numchan-1)
        topofreqs = np.linspace(result[0], result[1], chanInfo[0]) * 1e-9
        if fromFrame is not None:
            casalogPost("Converted %s range, width (%f-%f, %f) to TOPO (%f-%f, %f) over %d channels" % (fromFrame, chanInfo[1]*1e-9, chanInfo[2]*1e-9,lsrkwidth,topofreqs[0],topofreqs[-1],topoWidth,numchan))
    P0 = 1000.0 # mbar
    H0 = 20.0   # percent
    T0 = 273.0  # Kelvin
    if (telescopeName.find('ALMA') >= 0 or telescopeName.find('ACA') >= 0):
        pwv0 = 1.0   
        P0 = 563.0
        H0 = 20.0
        T0 = 273.0
        altitude0 = 5059
    elif (telescopeName.find('VLA') >= 0):
        P0 = 786.0
        pwv0 = 5.0  
        altitude0 = 2124
    else:
        pwv0 = 10.0  
        altitude0 = 0
    if (pwv < 0):
        pwv = pwv0
    if (T < 0):
        T = T0
    if (H < 0):
        H = H0
    if (P < 0):
        P = P0
    if (altitude < 0):
        altitude = altitude0
    tropical = 1
    midLatitudeSummer = 2
    midLatitudeWinter = 3
    reffreq = 0.5*(topofreqs[int(numchan/2)-1]+topofreqs[int(numchan/2)])
#    reffreq = np.mean(topofreqs)
    numchanModel = numchan*1
    chansepModel = (topofreqs[-1]-topofreqs[0])/(numchanModel-1)
    nbands = 1
    myqa = qatool()
    fCenter = create_casa_quantity(myqa, reffreq, 'GHz')
    fResolution = create_casa_quantity(myqa, chansepModel, 'GHz')
    fWidth = create_casa_quantity(myqa, numchanModel*chansepModel, 'GHz')
    myat = attool()
    myat.initAtmProfile(humidity=H, temperature=create_casa_quantity(myqa,T,"K"),
                        altitude=create_casa_quantity(myqa,altitude,"m"),
                        pressure=create_casa_quantity(myqa,P,'mbar'),atmType=midLatitudeWinter)
    myat.initSpectralWindow(nbands, fCenter, fWidth, fResolution)
    myat.setUserWH2O(create_casa_quantity(myqa, pwv, 'mm'))
#    myat.setAirMass()  # This does not affect the opacity, but it does effect TebbSky, so do it manually.
    myqa.done()

    dry = np.array(myat.getDryOpacitySpec(0)[1])
    wet = np.array(myat.getWetOpacitySpec(0)[1]['value'])
    TebbSky = myat.getTebbSkySpec(spwid=0)[1]['value']
    # readback the values to be sure they got set
    
    if (myat.getRefFreq()['unit'] != 'GHz'):
        casalogPost("There is a unit mismatch for refFreq in the atm code.")
    if (myat.getChanSep()['unit'] != 'MHz'):
        casalogPost("There is a unit mismatch for chanSep in the atm code.")
    numchanModel = myat.getNumChan()
    freq0 = myat.getChanFreq(0)['value']
    freq1 = myat.getChanFreq(numchanModel-1)['value']
    # We keep the original LSRK freqs for overlay on the LSRK spectrum, but associate
    # the transmission values from the equivalent TOPO freqs
    newfreqs = np.linspace(freqs[0], freqs[-1], numchanModel)  # fix for SCOPS-4815
    transmission = np.exp(-airmass*(wet+dry))
    TebbSky *= (1-np.exp(-airmass*(wet+dry)))/(1-np.exp(-wet-dry))
    if value=='transmission':
        values = transmission
    else:
        values = TebbSky
    del myat
    return(newfreqs, values)

def mjdToUT(mjd, use_metool=True, prec=6):
    """
    This function is called by getDateObs.
    Converts an MJD value to a UT date and time string
    such as '2012-03-14 00:00:00 UT'
    use_metool: whether or not to use the CASA measures tool if running from CASA.
         This parameter is simply for testing the non-casa calculation.
    -Todd Hunter
    """
    mjdsec = mjd*86400
    utstring = mjdSecondsToMJDandUT(mjdsec, use_metool, prec=prec)[1]
    return(utstring)
        
def mjdSecondsToMJDandUT(mjdsec, debug=False, prec=6, delimiter='-'):
    """
    This function is called by mjdToUT.
    Converts a value of MJD seconds into MJD, and into a UT date/time string.
    prec: 6 means HH:MM:SS,  7 means HH:MM:SS.S
    example: (56000.0, '2012-03-14 00:00:00 UT')
    Caveat: only works for a scalar input value
    """
    myme = metool()
    today = myme.epoch('utc','today')
    mjd = np.array(mjdsec) / 86400.
    today['m0']['value'] =  mjd
    myqa = qatool()
    hhmmss = myqa.time(today['m0'], prec=prec)[0]
    date = myqa.splitdate(today['m0'])
    myqa.done()
    myme.done()
    utstring = "%s%s%02d%s%02d %s UT" % (date['year'],delimiter,date['month'],delimiter,
                                             date['monthday'],hhmmss)
    return(mjd, utstring)

def cubeFrameToTopo(img, imageInfo, freqrange='', prec=4, verbose=False, 
                    nchan=None, f0=None, f1=None, chanwidth=None,
                    vis='', source='', spw=''):
    """
    This function is called by CalcAtmTransmissionForImage and writeContDat.
    Reads the date of observation, central RA and Dec,
    and observatory from an image cube and then calls lsrkToTopo to
    return the specified frequency range in TOPO, and the name of the original 
    frame that was converted (which will not be a REST-frame cube's frame if 
    vis not specified).
    If the cube is in REST frame, it calls casaRestToTopo(), and if the cube is
    in LSRK frame, it calls lsrkToTopo().
    freqrange: desired range of frequencies (empty string or list = whole cube)
          floating point list of two frequencies, or a delimited string
          (delimiter = ',', '~' or space)
    prec: in fractions of Hz (only used to display the value when verbose=True)
    chanwidth: unused
    vis: read date of observation from the specified measurement set (instead of from img)
    """
#    print("cubeFrameToTopo received spw = %s.  type=%s" % (str(spw), type(spw)))
    if img != '':
        if getFreqType(img).upper() == 'TOPO':
            print("This cube is purportedly already in TOPO.")
            return None, None
    if (nchan is None or f0 is None or f1 is None or chanwidth is None):
        if img != '':
            nchan,f0,f1,chanwidth = numberOfChannelsInCube(img, returnFreqs=True, returnChannelWidth=True)
    if len(freqrange) == 0:
        startFreq = f0 
        stopFreq = f1
    elif (type(freqrange) == str):
        if (freqrange.find(',') > 0):
            freqrange = [parseFrequencyArgument(i) for i in freqrange.split(',')]
        elif (freqrange.find('~') > 0):
            freqrange = [parseFrequencyArgument(i) for i in freqrange.split('~')]
        else:
            freqrange = [parseFrequencyArgument(i) for i in freqrange.split()]
        startFreq, stopFreq = freqrange
    else:
        startFreq, stopFreq = freqrange
    ra,dec = rad2radec(imageInfo[9], imageInfo[10], delimiter=' ', verbose=False).split()
    myia = iatool()
    myia.open(img)
    equinox = getEquinox(img,myia)
    observatory = getTelescope(img,myia)
    if len(vis) == 0:
        datestring = getDateObs(img,myia)
    else:
        if type(vis) == list or type(vis) == np.ndarray:
            vis = vis[0]
        else:
            vis = vis.split(',')[0]
        if not os.path.exists(vis):
            print("Measurement set does not exist!")
            return None, None
        datestring = getObservationStartDate(vis)
    myia.close()
    myType = getFreqType(img).upper()
    if myType == 'LSRK' or casaVersion < '6.2' or vis == '':
        if myType == 'REST':
            if casaVersion < '6.2':
                casalogPost('This CASA version is unable to convert from REST to TOPO. Converting from LSKR to TOPO instead.  Atmospheric overlay will not be quite right.', priority='WARN')
            elif vis == '':
                casalogPost('This cube is in REST frame, but the vis parameter was not supplied. Converting from LSKR to TOPO instead.  Atmospheric overlay will not be quite right.', priority='WARN')
        f0 = lsrkToTopo(startFreq, datestring, ra, dec, equinox, observatory, prec, verbose)
        f1 = lsrkToTopo(stopFreq, datestring, ra, dec, equinox, observatory, prec, verbose) 
        fromFrame = 'LSRK'
    elif myType == 'REST':
        fromFrame = 'REST'
        if spw == '':
            spw = getSpwFromPipelineImageName(img)
            casalogPost("Read spw %d from image name, type=%s" % (spw, type(spw)))
        else:
            spw = int(spw)
        mymsmd = msmdtool()
        mymsmd.open(vis)
        fieldid = fieldIDForName(mymsmd, source)
        chanfreqs = mymsmd.chanfreqs(spw)
        mymsmd.close()
        casalogPost("    Calling fc.casaRestToTopo(%f, %f, '%s', %s, %d)" % (startFreq, stopFreq, vis, str(spw), fieldid))
        c0, c1 = casaRestToTopo(startFreq, stopFreq, vis, spw, fieldid)
        # convert TOPO channel to TOPO frequency
        if chanfreqs[1] > chanfreqs[0]:  # USB
            f0 = chanfreqs[c0]
            f1 = chanfreqs[c1]
        else:  # LSB
            f0 = chanfreqs[c1]
            f1 = chanfreqs[c0]
    else:
        casalogPost('Unrecognized frequency frame type: %s.  Skipping frame conversion.' % (myType))
        f0 = startFreq
        f1 = stopFreq
        fromFrame = None
    return(np.array([f0,f1]), fromFrame)

def fieldIDForName(mymsmd, source):
    """
    Returns field ID for source name, ignoring those fields which are not used
    to observe the main target and calibrator intents (when those intents are
    also observed).
    Note: the following simple mechanism will fail if there is an ATMOSPHERE-only field
    fieldnames = mymsmd.fieldnames()
    fieldid = fieldnames.index(source)
    """
    fieldids = mymsmd.fieldsforname(source)
    intents = mymsmd.intentsforfield(source)
    intentfields = []
    for intent in ['OBSERVE_TARGET#ON_SOURCE','CALIBRATE_BANDPASS#ON_SOURCE',
                   'CALIBRATE_PHASE#ON_SOURCE','CALIBRATE_FLUX#ON_SOURCE',
                   'CALIBRATE_POLARIZATION#ON_SOURCE','OBSERVE_CHECK_SOURCE#ON_SOURCE']:
        if intent in intents:
            intentfields = mymsmd.fieldsforintent(intent)
            break
    if len(intentfields) > 0:
        fieldid = np.intersect1d(intentfields,fieldids)[0]
        casalogPost('    Picked field ID %d for intent %s' % (fieldid,intent))
    else:
        fieldid = fieldids[0]
        casalogPost('    Picked field ID %d from %s' % (fieldid,str(fieldids)))
    return fieldid

def getObservationStartDate(vis, obsid=0, delimiter='-', measuresToolFormat=True):
    """
    Uses the tb tool to read the start time of the observation and reports the date.
    See also getObservationStartDay.
    Returns: '2013-01-31 07:36:01 UT'
    measuresToolFormat: if True, then return '2013/01/31/07:36:01', suitable for lsrkToTopo
    -Todd Hunter
    """
    mjdsec = getObservationStart(vis, obsid)
    if (mjdsec is None):
        return
    obsdateString = mjdToUT(mjdsec/86400.)
    if (delimiter != '-'):
        obsdateString = obsdateString.replace('-', delimiter)
    if measuresToolFormat:
        return(obsdateString.replace(' UT','').replace(delimiter,'/').replace(' ','/'))
    else:
        return(obsdateString)

def rad2radec(ra=0,dec=0, prec=5, verbose=True, component=0,
              replaceDecDotsWithColons=True, hmsdms=False, delimiter=', ',
              prependEquinox=False, hmdm=False):
    """
    This function is called by cubeFrameToTopo.
    Convert a position in RA/Dec from radians to sexagesimal string which
    is comma-delimited, e.g. '20:10:49.01, +057:17:44.806'.
    The position can either be entered as scalars via the 'ra' and 'dec' 
    parameters, as a tuple via the 'ra' parameter, or as an array of shape (2,1)
    via the 'ra' parameter.
    replaceDecDotsWithColons: replace dots with colons as the Declination d/m/s delimiter
    hmsdms: produce output of format: '20h10m49.01s, +057d17m44.806s'
    hmdm: produce output of format: '20h10m49.01, +057d17m44.806' (for simobserve)
    delimiter: the character to use to delimit the RA and Dec strings output
    prependEquinox: if True, insert "J2000" before coordinates (i.e. for clean or simobserve)
    """
    if (type(ra) == tuple or type(ra) == list or type(ra) == np.ndarray):
        if (len(ra) == 2):
            dec = ra[1] # must come first before ra is redefined
            ra = ra[0]
        else:
            ra = ra[0]
            dec = dec[0]
    if (np.shape(ra) == (2,1)):
        dec = ra[1][0]
        ra = ra[0][0]
    myqa = qatool()
    myra = myqa.formxxx('%.12frad'%ra,format='hms',prec=prec+1)
    mydec = myqa.formxxx('%.12frad'%dec,format='dms',prec=prec-1)
    if replaceDecDotsWithColons:
        mydec = mydec.replace('.',':',2)
    if (len(mydec.split(':')[0]) > 3):
        mydec = mydec[0] + mydec[2:]
    mystring = '%s, %s' % (myra, mydec)
    myqa.done()
    if (hmsdms):
        mystring = convertColonDelimitersToHMSDMS(mystring)
        if (prependEquinox):
            mystring = "J2000 " + mystring
    elif (hmdm):
        mystring = convertColonDelimitersToHMSDMS(mystring, s=False)
        if (prependEquinox):
            mystring = "J2000 " + mystring
    if (delimiter != ', '):
        mystring = mystring.replace(', ', delimiter)
    if (verbose):
        print(mystring)
    return(mystring)

def convertColonDelimitersToHMSDMS(mystring, s=True, usePeriodsForDeclination=False):
    """
    This function is called by rad2radec.
    Converts HH:MM:SS.SSS, +DD:MM:SS.SSS  to  HHhMMmSS.SSSs, +DDdMMmSS.SSSs
          or HH:MM:SS.SSS +DD:MM:SS.SSS   to  HHhMMmSS.SSSs +DDdMMmSS.SSSs
          or HH:MM:SS.SSS, +DD:MM:SS.SSS  to  HHhMMmSS.SSSs, +DD.MM.SS.SSS
          or HH:MM:SS.SSS +DD:MM:SS.SSS   to  HHhMMmSS.SSSs +DD.MM.SS.SSS
    s: whether or not to include the trailing 's' in both axes
    """
    colons = len(mystring.split(':'))
    if (colons < 5 and (mystring.strip().find(' ')>0 or mystring.find(',')>0)):
        print("Insufficient number of colons (%d) to proceed (need 4)" % (colons-1))
        return
    if (usePeriodsForDeclination):
        decdeg = '.'
        decmin = '.'
        decsec = ''
    else:
        decdeg = 'd'
        decmin = 'm'
        decsec = 's'
    if (s):
        outstring = mystring.strip(' ').replace(':','h',1).replace(':','m',1).replace(',','s,',1).replace(':',decdeg,1).replace(':',decmin,1) + decsec
        if (',' not in mystring):
            outstring = outstring.replace(' ', 's ',1)
    else:
        outstring = mystring.strip(' ').replace(':','h',1).replace(':','m',1).replace(':',decdeg,1).replace(':',decmin,1)
    return(outstring)
    
def casaRestToTopo(restFrequency, restFrequency2, msname, spw, fieldid):
    """
    Converts a range of SOURCE frame frequencies to TOPO channels in a 
    specified measurement set.  The input frequencies should be the 
    center frequencies of the edge channels in the cube.
    """
    # needs to use new tool in CASA 6.2
    from casatools import synthesisutils
    # Figure out which channels in the ms were used to make the SOURCE frame
    # cube
    spw = int(spw)
    su = synthesisutils()
    # this function expects the center frequency of each edge channel and returns channel number in ms/spw
    mydict = su.advisechansel(msname=msname, freqframe='SOURCE',
                              ephemtable='TRACKFIELD', fieldid=fieldid,
                              freqstart='%fHz'%(restFrequency),
                              freqend='%fHz'%(restFrequency2))
    idx = np.where(mydict['spw'] == spw)[0]
    startChan = int(pickFirstElementIfNecessary(mydict['start'][idx])) # fix for PIPE-2673
    nchan = int(pickFirstElementIfNecessary(mydict['nchan'][idx])) # fix for PIPE-2673
    stopChan = startChan + nchan - 1
    print("    TOPO channel range in ms: %d-%d" % (startChan,stopChan))
    return startChan, stopChan

def lsrkToTopo(lsrkFrequency, datestring, ra, dec, equinox='J2000', 
               observatory='ALMA', prec=4, verbose=False):
    """
    This function is called by cubeFrameToTopo.
    Converts an LSRKfrequency and observing date/direction
    to the corresponding frequency in the TOPO frame.
    Inputs:
    lsrkFrequency: floating point value in Hz or GHz, or a string with units
    datestring:  "YYYY/MM/DD/HH:MM:SS" (format = image header keyword 'date-obs')
    ra: string "HH:MM:SS.SSSS"
    dec: string "DD.MM.SS.SSSS" or "DD:MM:SS.SSSS" (colons will be replaced with .)
    prec: only used to display the value when verbose=True
    Returns: the TOPO frequency in Hz
    """
    velocityLSRK = 0  # does not matter what it is, just needs to be same in both calls
    restFreqHz = lsrkToRest(lsrkFrequency, velocityLSRK, datestring, ra, dec, equinox,
                            observatory, prec, verbose)
    topoFrequencyHz = restToTopo(restFreqHz, velocityLSRK, datestring, ra, dec, equinox, 
                                observatory, verbose=verbose)
    return topoFrequencyHz

def lsrkToRest(lsrkFrequency, velocityLSRK, datestring, ra, dec, 
               equinox='J2000', observatory='ALMA', prec=4, verbose=True):
    """
    This function is called by lsrkToTopo.
    Converts an LSRK frequency, LSRK velocity, and observing date/direction
    to the corresponding frequency in the rest frame.
    Inputs:
    lsrkFrequency: floating point value in Hz or GHz, or a string with units
    velocityLSRK: floating point value in km/s
    datestring:  "YYYY/MM/DD/HH:MM:SS" (format = image header keyword 'date-obs')
    ra: string "HH:MM:SS.SSSS"
    dec: string "DD.MM.SS.SSSS" or "DD:MM:SS.SSSS" (colons will be replaced with .)
    prec: only used to display the value when verbose=True
    Returns: the Rest frequency in Hz
    """
    if (dec.find(':') >= 0):
        dec = dec.replace(':','.')
        if verbose:
            print("Warning: replacing colons with decimals in the dec field.")
    freqGHz = parseFrequencyArgumentToGHz(lsrkFrequency)
    myqa = qatool()
    myme = metool()
    velocityRadio = create_casa_quantity(myqa,velocityLSRK,"km/s")
    position = myme.direction(equinox, ra, dec)
    obstime = myme.epoch('TAI', datestring)
    dopp = myme.doppler("RADIO",velocityRadio)
    radialVelocityLSRK = myme.toradialvelocity("LSRK",dopp)
    myme.doframe(position)
    myme.doframe(myme.observatory(observatory))
    myme.doframe(obstime)
    rvelRad = myme.measure(radialVelocityLSRK,'LSRK')
    doppRad = myme.todoppler('RADIO', rvelRad)
    freqRad = myme.torestfrequency(myme.frequency('LSRK',str(freqGHz)+'GHz'), dopp)
    myqa.done()
    myme.done()
    return freqRad['m0']['value']

def restToTopo(restFrequency, velocityLSRK, datestring, ra, dec, equinox='J2000', 
               observatory='ALMA', veltype='radio', verbose=False):
    """
    This function is called by lsrkToTopo.
    Converts a rest frequency, LSRK velocity, and observing date/direction
    to the corresponding frequency in the TOPO frame.
    Inputs:
    restFrequency: floating point value in Hz or GHz, or a string with units
    velocityLSRK: floating point value in km/s
    datestring:  "YYYY/MM/DD/HH:MM:SS"
    ra: string "HH:MM:SS.SSSS"
    dec: string "DD.MM.SS.SSSS" or "DD:MM:SS.SSSS" (colons will be replaced with .)
    prec: only used to display the value when verbose=True
    Returns: the TOPO frequency in Hz
    """
    topoFreqHz, diff1, diff2 = frames(velocityLSRK, datestring, ra, dec, equinox, 
                                      observatory, verbose=verbose,
                                      restFreq=restFrequency, veltype=veltype)
    return topoFreqHz

def frames(velocity=286.7, datestring="2005/11/01/00:00:00",
           ra="05:35:28.105", dec="-069.16.10.99", equinox="J2000", 
           observatory="ALMA", prec=4, verbose=True, myme='', myqa='',
           restFreq=345.79599, veltype='optical'):
    """
    This function is called by restToTopo.
    Converts an optical velocity into barycentric, LSRK and TOPO frames.
    Converts a radio LSRK velocity into TOPO frame.
    Inputs:
    velocity: in km/s
    datestring:  "YYYY/MM/DD/HH:MM:SS"
    ra: "05:35:28.105"
    dec: "-069.16.10.99"
    equinox: "J2000" 
    observatory: "ALMA"
    prec: precision to display (digits to the right of the decimal point)
    veltype: 'radio' or 'optical'
    restFreq: in Hz, GHz or a string with units
    Returns: 
    * TOPO frequency in Hz
    * difference between LSRK-TOPO in km/sec
    * difference between LSRK-TOPO in Hz
    """
    localme = False
    localqa = False
    if (myme == ''):
        myme = metool()
        localme = True
    if (myqa == ''):
        myqa = qatool()
        localqa = True
    if (dec.find(':') >= 0):
        dec = dec.replace(':','.')
    position = myme.direction(equinox, ra, dec)
    obstime = myme.epoch('TAI', datestring)

    if (veltype.lower().find('opt') == 0):
        velOpt = create_casa_quantity(myqa,velocity,"km/s")
        dopp = myme.doppler("OPTICAL",velOpt)
        # CASA doesn't do Helio, but difference to Bary is hopefully small
        rvelOpt = myme.toradialvelocity("BARY",dopp)
    elif (veltype.lower().find('rad') == 0):
        rvelOpt = myme.radialvelocity('LSRK',str(velocity)+'km/s')
    else:
        print("veltype must be 'rad'io or 'opt'ical")
        return

    myme.doframe(position)
    myme.doframe(myme.observatory(observatory))
    myme.doframe(obstime)
    myme.showframe()

    rvelRad = myme.measure(rvelOpt,'LSRK')
    doppRad = myme.todoppler("RADIO",rvelRad)       
    restFreq = parseFrequencyArgumentToGHz(restFreq)
    freqRad = myme.tofrequency('LSRK',doppRad, myme.frequency('rest',str(restFreq)+'GHz'))
    myqa = qatool()
    lsrk = myqa.tos(rvelRad['m0'],prec=prec)
    rvelTop = myme.measure(rvelOpt,'TOPO')
    doppTop = myme.todoppler("RADIO",rvelTop)       
    freqTop = myme.tofrequency('TOPO', doppTop, myme.frequency('rest',str(restFreq)+'GHz'))
    if (localme):
        myme.done()
    topo = myqa.tos(rvelTop['m0'],prec=prec)
    if (localqa):
        myqa.done()
    velocityDifference = 0.001*(rvelRad['m0']['value']-rvelTop['m0']['value'])
    frequencyDifference = freqRad['m0']['value'] - freqTop['m0']['value']
    return(freqTop['m0']['value'], velocityDifference, frequencyDifference)

def parseFrequencyArgumentToGHz(bandwidth):
    """
    This function is called by frames and lsrkToRest.
    Converts a frequency string into floating point in GHz, based on the units.
    If the units are not present, then the value is assumed to be GHz if less
    than 1000.
    """
    value = parseFrequencyArgument(bandwidth)
    if (value > 1000):
        value *= 1e-9
    return(value)

def parseFrequencyArgumentToHz(bandwidth):
    """
    Converts a frequency string into floating point in Hz, based on the units.
    If the units are not present, then the value is assumed to be GHz if less
    than 1000 (in contrast to parseFrequencyArgument).
    -Todd Hunter
    """
    value = parseFrequencyArgumentToGHz(bandwidth) * 1e9
    return(value)

def parseFrequencyArgument(bandwidth):
    """
    This function is called by parseFrequencyArgumentToGHz, topoFreqToChannel and cubeFrameToTopo.
    Converts a string frequency into floating point in Hz, based on the units.
    If the units are not present, then the value is simply converted to float.
    """
    bandwidth = str(bandwidth)
    ghz = bandwidth.lower().find('ghz')
    mhz = bandwidth.lower().find('mhz')
    khz = bandwidth.lower().find('khz')
    hz = bandwidth.lower().find('hz')
    if (ghz>0):
        bandwidth = 1e9*float(bandwidth[:ghz])
    elif (mhz>0):
        bandwidth = 1e6*float(bandwidth[:mhz])
    elif (khz>0):
        bandwidth = 1e3*float(bandwidth[:khz])
    elif (hz>0):
        bandwidth = float(bandwidth[:hz])
    else:
        bandwidth = float(bandwidth)
    return(bandwidth)

def intersectChannelSelections(selection1, selection2, separator=';'):
    """
    Take the intersection of two ranges.
    e.g. '3~8;10~11', '5~9'  ->  '5~8'
    """
    list1 = channelSelectionRangesToIndexArray(selection1, separator)
    list2 = channelSelectionRangesToIndexArray(selection2, separator)
    return convertChannelListIntoSelection(np.intersect1d(list1,list2))

def channelSelectionRangesToIndexArray(selection, separator=';'):
    """
    This function is called by runFindContinuum.
    Convert a channel selection range string to integer array of indices.
    e.g.:  '3~8;10~11' -> [3,4,5,6,7,8,10,11]
    """
    index = []
    for s in selection.split(separator):
        a,b = s.split('~')
        index += list(range(int(a), int(b)+1, 1))
    return np.array(index)

def linfit(x, y, yerror, pinit=[0,0]):
    """
    This function is called by atmosphereVariation in Cycle 4+5+6 and by
    runFindContinuum in Cycle 4+5.
    Basic linear function fitter with error bars in y-axis data points.
    Uses scipy.optimize.leastsq().  Accepts either lists or arrays.
    Example:
         lf = au.linfit()
         lf.linfit(x, y, yerror, pinit)
    Input:
         x, y: x and y axis values
         yerror: uncertainty in the y-axis values (vector or scalar)
         pinit contains the initial guess of [slope, intercept]
    Output:
       The fit result as: [slope, y-intercept]
    """
    x = np.array(x)
    y = np.array(y)
    if (type(yerror) != list and type(yerror) != np.ndarray):
        yerror = np.ones(len(x)) * yerror
    fitfunc = lambda p, x: p[1] + p[0]*x
    errfunc = lambda p,x,y,err: (y-fitfunc(p,x))/(err**2)
    out = scipy.optimize.leastsq(errfunc, pinit, args=(x,y,yerror/y), full_output=1)
    p = out[0]
    covar = out[1]
    return(p)

def polyfit(x, y, yerror, pinit=[0,0,0,0]):
    """
    This function is called by removeInitialQuadraticIfNeeded in Cycle 6, and by
    runFindContinuum in Cycle 4+5.
    Basic second-order polynomial function fitter with error bars in y-axis data points.
    Uses scipy.optimize.leastsq().  Accepts either lists or arrays.
    Input:
         x, y: x and y axis values
         yerror: uncertainty in the y-axis values (vector or scalar)
         pinit contains the initial guess of [slope, intercept]
         y = order2coeff*(x-xoffset)**2 + slope*(x-xoffset) + y-intercept
    Output:
       The fit result as: [xoffset, order2coeff, slope, y-intercept]
    """
    x = np.array(x)
    y = np.array(y)
    pinit[2] = np.mean(y)
    pinit[3] = x[int(len(x)/2)]
    if (type(yerror) != list and type(yerror) != np.ndarray):
        yerror = np.ones(len(x)) * yerror
    fitfunc = lambda p, x: p[2] + p[1]*(x-p[3]) + p[0]*(x-p[3])**2
    errfunc = lambda p,x,y,err: (y-fitfunc(p,x))/(err**2)
    out = scipy.optimize.leastsq(errfunc, pinit, args=(x,y,yerror/y), full_output=1)
    p = out[0]
    covar = out[1]
    return(p)

def channelsInLargestGroup(selection):
    """
    This function is called by runFindContinuum.
    Returns the number of channels in the largest group of channels in a
    CASA selection string.
    """
    if (selection == ''):
        return 0
    ranges = selection.split(';')
    largest = 0
    for r in ranges:
        c0 = int(r.split('~')[0])
        c1 = int(r.split('~')[1])
        channels = c1-c0+1
        if (channels > largest):
            largest = channels
    return largest

def countChannelsInRanges(channels, separator=';'):
    """
    This function is called by runFindContinuum.
    Counts the number of channels in one spw of a CASA channel selection string
    and return a list of numbers.
    e.g. "5~20;30~40"  yields [16,11]
    """
    tokens = channels.split(separator)
    count = []
    for i in range(len(tokens)):
        c0 = int(tokens[i].split('~')[0])
        c1 = int(tokens[i].split('~')[1])
        count.append(c1-c0+1)
    return count

def countChannels(channels):
    """
    This function is called by runFindContinuum.
    Counts the number of channels in a CASA channel selection string.
    If multiple spws are found, then it returns a dictionary of counts:
    e.g. "1:5~20;30~40"  yields 27; or  '6~30' yields 25
         "1:5~20;30~40,2:6~30" yields {1:27, 2:25}
    """
    if (channels == ''):
        return 0
    tokens = channels.split(',')
    nspw = len(tokens)
    count = {}
    for i in range(nspw):
        string = tokens[i].split(':')
        if (len(string) == 2):
            spw,string = string
        else:
            string = string[0]
            spw = 0
        ranges = string.split(';')
        for r in ranges:
            c0 = int(r.split('~')[0])
            c1 = int(r.split('~')[1])
            if (c0 > c1):
                casalogPost("Invalid channel range: c0 > c1 (%d > %d)" % (c0,c1))
                return
        channels = [1+int(r.split('~')[1])-int(r.split('~')[0]) for r in ranges]
        count[spw] = np.sum(channels)
    if (nspw == 1):
        count = count[spw]
    return(count)

def grep(filename, arg):
    """
    This function is called only by maskArgumentMismatch and centralArcsecArgumentMismatch.
    Runs grep for string <arg> in a subprocess and reports stdout and stderr.
    """
    process = subprocess.Popen(['grep', '-n', arg, filename], stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr

def topoFreqToChannel(freqlist, vis, spw, mymsmd=''):
    """
    ++++ This function is used by pipeline in writeContDat()
    Converts a python floating point list of topocentric frequency ranges to 
    channel ranges in the specified ms.
    freqlist:  [150.45e9,151.67e9]  or [150.45, 151.67] 
    spw: integer ID
    vis: reads the channel frequencies from this measurement set for the specified spw
    Returns: a python list of channels
    """
    needToClose = False
    if mymsmd == '':
        needToClose = True
        mymsmd = msmdtool()
        mymsmd.open(vis)
    chanfreqs = mymsmd.chanfreqs(spw)
    width = np.abs(mymsmd.chanwidths(spw))[0]
    if needToClose:
        mymsmd.close()
    if type(freqlist) != list and type(freqlist) != np.ndarray:
        freqlist = [freqlist]
    channels = []
    for freq in freqlist:
        freq = parseFrequencyArgument(freq)
        f0 = np.min(chanfreqs) - 0.5*width
        f1 = np.max(chanfreqs) + 0.5*width
        if freq < f0  or freq > f1:
            chanoff = np.min([np.abs(f0-freq),np.abs(freq-f1)]) / width
            print("frequency %.6f GHz not within spw %d: %.6f - %.6f GHz (off by %.2f channels)" % (freq*1e-9, spw, f0*1e-9, f1*1e-9, chanoff))
            if freq < f0:
                freq = f0
            elif freq > f1:
                freq = f1
            print("Setting frequency to last channel: %f" % (freq))
        diffs = np.abs(chanfreqs-freq)
        channels.append(np.argmin(diffs))
    return channels
    
def topoFreqRangeListToChannel(contdatline='', vispath='./', spw=-1, freqlist='', vis='', mymsmd='', 
                               returnFlatlist=False, writeChannelsInIncreasingOrder=True):
    """
    ++++ This function is used by pipeline in writeContDat()
    Converts a semicolon-delimited string list of topocentric frequency ranges to 
    channel ranges in the specified ms.
    contdatline: line cut-and-pasted from .dat file  (e.g. 'my.ms 150.45~151.67GHz;151.45~152.67GHz') 
    vispath: set the path to the measurement set whose name was read from contdatline
    freqlist:  '150.45~151.67GHz;151.45~152.67GHz'
    spw: integer ID or string ID
    writeChannelsInIncreasingOrder: if True, then ensure that the list is in increasing order
    Returns: a string like:  '29:134~136;200~204'
    if flatlist==True, then return  [134,136,200,204]
    """
    if contdatline == '' and freqlist == '':
        print("Need to specify either contdatline or freqlist.")
        return
    if contdatline == '' and vis == '':
        print("Need to specify either contdatline or vis.")
        return
    if contdatline != '':
        vis, freqlist = contdatline.split()
        if vispath != './':
            vis = os.path.join(vispath,vis)
    if (spw == -1 or spw == ''):
        print("spw must be specified")
        return
    freqlist = freqlist.split(';')
    chanlist = ''
    myqa = qatool()
    flatlist = []
    spw = int(spw)
    mymsmd = msmdtool()
    mymsmd.open(vis)
    for r in freqlist:
        freqs = r.split('~')
        if len(freqs) == 2:
            myfreq = myqa.quantity(freqs[1])
            f1 = myqa.convert(myfreq, 'Hz')['value']
            unit = myfreq['unit'] 
            f0 = myqa.convert(myqa.quantity(freqs[0]+unit), 'Hz')['value']
            print("Running topoFreqToChannel([%f,%f], '%s', %d)" % (f0,f1,vis,spw))
            chans = topoFreqToChannel([f0,f1], vis, spw, mymsmd)
            chanlist += '%d~%d' % (np.min(chans), np.max(chans))
            if r != freqlist[-1]:
                chanlist += ';'
            flatlist.append(np.min(chans))
            flatlist.append(np.max(chans))
    myqa.done()
    mymsmd.close()
    if returnFlatlist:
        return flatlist
    else:
        if writeChannelsInIncreasingOrder:
            cselections = chanlist.split(';')
            if len(cselections) > 1:
                if int(cselections[0].split('~')[0]) > int(cselections[1].split('~')[0]):
                    print("Reversing order of output channel list (due to lower sideband spw).")
                    cselections.reverse()
                    chanlist = ';'.join(cselections)
        chanlist = '%d:' % (spw) + chanlist
        return chanlist

def gaussianBeamOffset(response=0.5, fwhm=1.0):
    """
    ++++ This function is used by pipeline in computeStatisticalSpectrumFromMask().
    Computes the radius at which the response of a Gaussian
    beam drops to the specified level.  For the inverse
    function, see gaussianBeamResponse.
    """
    if (response <= 0 or response > 1):
        print("response must be between 0..1")
        return
    radius = (fwhm/2.3548)*(-2*np.log(response))**0.5 
    return(radius)
    
def gaussianBeamResponse(radius, fwhm):
    """
    ++++ This function is used by pipeline in computeStatisticalSpectrumFromMask().
    Computes the gain at the specified radial offset from the center
    of a Gaussian beam. For the inverse function, see gaussianBeamOffset.
    Required inputs:
    radius: in arcseconds
    fwhm: in arcseconds
    """
    sigma = fwhm/2.3548
    gain = np.exp(-((radius/sigma)**2)*0.5)
    return(gain)

def imageSNRAnnulus(mymomDiff, jointMaskTest, jointMaskTestWithAnnulus, applyMaskToAll=True):                   
    # Need 2 calls to get peak anywhere outside the joint mask, but median from annulus, and MAD the lower of the two possibilities
    ignore2, momDiffPeakOutside, ignore0, ignore1 = imageSNR(mymomDiff, mask=jointMaskTest,
                                                             useAnnulus=False, returnAllStats=True, applyMaskToAll=applyMaskToAll)
    ignore1, ignore2, medianForSNR, madForSNR = imageSNR(mymomDiff, mask=jointMaskTest, maskWithAnnulus=jointMaskTestWithAnnulus, 
                                                         useAnnulus=True, returnAllStats=True, applyMaskToAll=applyMaskToAll)
    momDiffSNROutside = (momDiffPeakOutside - medianForSNR) / madForSNR
    return momDiffSNROutside, momDiffPeakOutside

def imageSNR(img, axes=[], mask='', maskWithAnnulus='', useAnnulus=False, 
             verbose=False, applyMaskToAll=False, returnAllStats=True,
             usepb='', chans=''):
    """
    Uses imstat (which automatically applies the internal mask, i.e avoids the 
    black edges of the image) to compute (max-median)/scaledMAD
    axes: passed to imstat (use [0,1,2] to get a vector SNR per channel)
    img: a 2D image or a cube
    mask: additional mask image (or expression) to apply when computing everything but max
    maskWithAnnulus: additional mask image (or expression) to use for median and MAD
            if useAnnulus==True (but always take the smaller of the 2 possible MADs)
    applyMaskToAll: also apply the provided mask when computing the max (default=False=anywhere)
    returnAllStats: if True, then also return the peak, median and scaledMAD where the
          peak has *not* had the median subtracted (in contrast to the SNR)
    usepb: (optional) name of a single-plane pb image to include in the mask (at >0.23 level)
    -ToddHunter
    """
    if usepb != '':
        mask += ' && "%s">0.23' % (usepb)
    stats = imstat(img, axes=axes, mask=mask, listit=imstatListit, chans=chans)
    for myval in ['median','max','medabsdevmed','min','sigma','npts']: # fix for PIPE-2673
        stats[myval] = pickFirstElementIfNecessary(stats[myval])       # fix for PIPE-2673
    if useAnnulus:
        statsAnnulus = imstat(img, axes=axes, mask=maskWithAnnulus, listit=imstatListit, chans=chans)
        for myval in ['median','max','medabsdevmed','min','sigma','npts']: # fix for PIPE-2673
            statsAnnulus[myval] = pickFirstElementIfNecessary(statsAnnulus[myval])       # fix for PIPE-2673
    if not applyMaskToAll:
        stats_nomask = imstat(img, listit=imstatListit, axes=axes, chans=chans)
        # replace max with the max over the whole image
        stats['max'] = pickFirstElementIfNecessary(stats_nomask['max'])
    if useAnnulus:
        # use the median from the annulus
        casalogPost("median = %f, median from annulus = %f;  scMAD = %f, scMAD from annulus = %f" % (stats['median'],statsAnnulus['median'],stats['medabsdevmed']/.6745,statsAnnulus['medabsdevmed']/.6745))
        stats['median'] = statsAnnulus['median']
        # pick the lower of the two possible MADs; otherwise, do not use the rest of the statsAnnulus results
        if statsAnnulus['medabsdevmed'] > 0 and stats['medabsdevmed'] > 0:
            # keep it as an array of length 1
#            stats['medabsdevmed'] = np.array([np.min([statsAnnulus['medabsdevmed'], stats['medabsdevmed']])])
            stats['medabsdevmed'] = np.min([statsAnnulus['medabsdevmed'], stats['medabsdevmed']])
        elif statsAnnulus['medabsdevmed'] > 0: # this means that stats['medabsdevmed'] is zero, so we replace it
            stats['medabsdevmed'] = statsAnnulus['medabsdevmed']
    if stats['medabsdevmed'] > 0:
        snr = (stats['max']-stats['median'])/(stats['medabsdevmed']/.6745)
        casalogPost('snr = (%f-%f)/%f = %f (npts=%d)' % (stats['max'],stats['median'],stats['medabsdevmed']/.6745,snr,stats['npts']))
    else:
        casalogPost('Because the MAD is zero, using the sigma instead in the calculation of SNR')
        snr = (stats['max']-stats['median'])/stats['sigma']
        casalogPost('snr = (%f-%f)/%f = %f (npts=%d)' % (stats['max'],stats['median'],stats['sigma'],snr, stats['npts']))
    mymax = stats['max'] # fix for PIPE-2673
    mymedian = stats['median'] # fix for PIPE-2673
    myscaledMAD = stats['medabsdevmed']/.6745 # fix for PIPE-2673
    if type(snr) == list or type(snr) == np.ndarray:
        if len(snr) == 1:
            snr = snr[0]
            mymax = stats['max'][0]
            mymedian = stats['median'][0]
            myscaledMAD = stats['medabsdevmed'][0]/.6745
    if returnAllStats:
        return snr, mymax, mymedian, myscaledMAD
    else:
        return snr

def getSpwFromPipelineImageName(img, verbose=False):
    """
    Extracts the spw ID from the pipeline image file name.
    -Todd Hunter
    """
    sourceName = os.path.basename(img.rstrip('/'))
    if verbose: print("A = ", sourceName)
    if (sourceName.find('.mfs.') < 0):
        if (sourceName.find('.cube.') < 0):
            return None
        else:
            sourceName = sourceName.split('.cube.')[0]
    else:
        sourceName = sourceName.split('.mfs.')[0]
    if verbose: print("B = ", sourceName)
    sourceName = sourceName.split('.spw')[1]
    if verbose: print("spw = ", sourceName)
    if sourceName.isdigit():
        return int(sourceName)
    else:
        return None

def representativeSpwBandwidth(vis, intent='TARGET', mymsmd=None, verbose=False):
    """
    Uses updateSBSummary to learn the name of the representativeWindow, then 
    translates that name to a science spw ID using msmd.spwsfornames, and 
    finally a bandwidth.  
    Returns: spw ID, spw bandwidth in Hz (or 0 if no rep spw can be discerned),
       minimum science spw bandwidth, maximum science spw bandwidth
    -Todd Hunter
    """
    if not os.path.exists(vis):
        print("Could not find vis: ", vis)
        return
    mytb = tbtool()
    mytb.open(vis+'/ASDM_SBSUMMARY', nomodify=True)
    scienceGoal = mytb.getcol('scienceGoal')
    # Read the existing values for those that were not specified
    info = {}
    representativeSPW = None
    for i,args in enumerate(scienceGoal):
        # args will look like: 
        # array(['representativeFrequency = 219.55647641503566 GHz'])
        for arg in args:
            # arg will look like: 
            #  'representativeFrequency = 219.55647641503566 GHz'
            loc = arg.find('representativeWindow')
            if (loc >= 0 and representativeSPW is None):
                tokens = arg[loc:].split()
                representativeSPW = str(tokens[2])
                representativeSpwPresent = True
                info['representativeWindow'] = ' '.join(tokens[2:])
    if 'representativeWindow' not in info:
        print("SBSummary does not contain the representativeWindow key")
        print(info)
        return
    if mymsmd is None:
        mymsmd = msmdtool()
        mymsmd.open(vis)
        needToClose = True
    else:
        needToClose = False
    spwname = info['representativeWindow']
    myspws = mymsmd.spwsfornames(spwname)
    if spwname not in myspws:
        print("SBSummary does not contain a valid representativeWindow name")
        return
    spw = np.intersect1d(myspws[spwname], mymsmd.spwsforintent('*'+intent+'*'))
#    spw = np.intersect1d(mymsmd.spwsfornames(info['representativeWindow'])[spwname], mymsmd.spwsforintent('*'+intent+'*'))
    if len(spw) == 1:
        spw = int(spw[0])  # intersection of uint64 with int64 is a float!
        bandwidth = mymsmd.bandwidths()[spw]
        nchan = mymsmd.nchan(spw)
    else:
        bandwidth = 0
        nchan = 0
    bws = getScienceSpwBandwidths(vis, mymsmd=mymsmd)
    if needToClose:
        mymsmd.close()
    return spw, bandwidth, nchan, np.min(bws), np.max(bws)
    
def updateSBSummary(vis, representativeFrequency=None, 
                    minAcceptableAngResolution=None, 
                    maxAcceptableAngResolution=None,
                    dynamicRange=None, representativeBandwidth=None,
                    representativeSource=None, representativeSPW=None,
                    maxAllowedBeamAxialRatio=None,
                    spectralDynamicRangeBandWidth=None, verbose=True):
    """
    Updates the ASDM_SBSUMMARY table of a measurement set with one or more new
    values.  If a value is not present in the existing table and also not 
    specified, then it will remain not present in the updated table.
    representativeFrequency: float value in typical units (GHz), 
            or string with units (space before units is optional)
    minAcceptableAngResolution: float value in typical units (arcsec), 
            or string with units (space before units is required)
    maxAcceptableAngResolution: float value in typical units (arcsec), 
            or string with units (space before units is required)
    dynamicRange: float value
    representativeBandwidth: float value in typical units (GHz), 
            or string with units (space before units is optional)
    representativeSource: string
    representativeSPW: string
    maxAllowedBeamAxialRatio: float
    spectralDynamicRangeBandWidth: float value in typical units (MHz), 
            or string with units (space before units is optional)
    Returns: dictionary keyed by parameter name
    -Todd Hunter
    """
    if not os.path.exists(vis):
        print("Could not find ms.")
        return
    t = vis+'/ASDM_SBSUMMARY'
    if not os.path.exists(t):
        print("Could not find ASDM_SBSUMMARY table for this ms.  Was it imported?")
        return
    if (representativeFrequency is not None or 
        minAcceptableAngResolution is not None or 
        maxAcceptableAngResolution is not None or maxAllowedBeamAxialRatio is not None or
        dynamicRange is not None or representativeBandwidth is not None or
        representativeSource is not None or representativeSPW is not None or
        spectralDynamicRangeBandWidth is not None):
        update = True
    else:
        update = False
    mytb = tbtool()
    nomodify = not update
    mytb.open(vis+'/ASDM_SBSUMMARY', nomodify=nomodify)
    scienceGoal = mytb.getcol('scienceGoal')
    numScienceGoal = mytb.getcol('numScienceGoal')[0]
    representativeSpwPresent = False
    axialRatioPresent = False
    mydict = {}
    # Read the existing values for those that were not specified
    for i,args in enumerate(scienceGoal):
        # args will look like: 
        # array(['representativeFrequency = 219.55647641503566 GHz'])
        for arg in args:
            # arg will look like: 
            #  'representativeFrequency = 219.55647641503566 GHz'
            loc = arg.find('representativeFrequency')
            if (loc >= 0 and representativeFrequency is None):
                tokens = arg[loc:].split()
                representativeFrequency = float(tokens[2])
                freqUnits = tokens[3]
                mydict['representativeFrequency'] = ' '.join(tokens[2:])
            loc = arg.find('minAcceptableAngResolution')
            if (loc >= 0 and minAcceptableAngResolution is None):
                tokens = arg[loc:].split()
                minAcceptableAngResolution = float(tokens[2])
                minUnits = tokens[3]
                mydict['minAcceptableAngResolution'] = ' '.join(tokens[2:])
            loc = arg.find('maxAcceptableAngResolution')
            if (loc >= 0 and maxAcceptableAngResolution is None):
                tokens = arg[loc:].split()
                maxAcceptableAngResolution = float(tokens[2])
                maxUnits = tokens[3]
                mydict['maxAcceptableAngResolution'] = ' '.join(tokens[2:])
            loc = arg.find('dynamicRange')
            if (loc >= 0 and dynamicRange is None):
                tokens = arg[loc:].split()
                dynamicRange = float(tokens[2])
                mydict['dynamicRange'] = ' '.join(tokens[2:])
            loc = arg.find('representativeBandwidth')
            if (loc >= 0 and representativeBandwidth is None):
                tokens = arg[loc:].split()
                representativeBandwidth = float(tokens[2])
                bwUnits = tokens[3]
                mydict['representativeBandwidth'] = ' '.join(tokens[2:])
            loc = arg.find('spectralDynamicRangeBandWidth')
            if (loc >= 0 and spectralDynamicRangeBandWidth is None):
                tokens = arg[loc:].split()
                spectralDynamicRangeBandWidth = float(tokens[2])
                bwUnits = tokens[3]
                mydict['spectralDynamicRangeBandWidth'] = ' '.join(tokens[2:])
            loc = arg.find('representativeSource')
            if (loc >= 0 and representativeSource is None):
                tokens = arg[loc:].split()
                representativeSource = str(tokens[2])
                mydict['representativeSource'] = ' '.join(tokens[2:])
            loc = arg.find('representativeWindow')
            if (loc >= 0 and representativeSPW is None):
                tokens = arg[loc:].split()
                representativeSPW = str(tokens[2])
                representativeSpwPresent = True
                mydict['representativeWindow'] = ' '.join(tokens[2:])
            loc = arg.find('maxAllowedBeamAxialRatio')
            if (loc >= 0 and maxAllowedBeamAxialRatio is None):
                tokens = arg[loc:].split()
                maxAllowedBeamAxialRatio = float(tokens[2])
                axialRatioPresent = True
                mydict['maxAllowedBeamAxialRatio'] = ' '.join(tokens[2:])
            loc = arg.find('SBName')
            if (loc >= 0):
                tokens = arg[loc:].split()
                sbname = tokens[2]
                mydict['SBName'] = sbname
            loc = arg.find('sensitivityGoal')
            if (loc >= 0):
                tokens = arg[loc:].split()
                sbname = tokens[2]
                mydict['sensitivityGoal'] = sbname
    if update:
        # convert any command-line arguments from string to value and units
        if type(representativeFrequency) is str:
            representativeFrequency = parseFrequencyArgumentToGHz(representativeFrequency)
            freqUnits = 'GHz'
        if type(representativeBandwidth) is str:
            representativeBandwidth = parseFrequencyArgumentToGHz(representativeBandwidth) * 1000
            bwUnits = 'MHz'
        if type(spectralDynamicRangeBandWidth) is str:
            spectralDynamicRangeBandWidth = parseFrequencyArgumentToGHz(spectralDynamicRangeBandWidth) * 1000
            bwUnits = 'MHz'
        if type(dynamicRange) is str:
            dynamicRange = float(dynamicRange)
        if type(minAcceptableAngResolution) is str:
            result = minAcceptableAngResolution.split()
            if len(result) == 1:
                value = result
                minUnits = 'arcsec'
            else:
                value, minUnits = result
            minAcceptableAngResolution = float(value)
        if type(maxAcceptableAngResolution) is str:
            result = maxAcceptableAngResolution.split()
            if len(result) == 1:
                value = result
                maxUnits = 'arcsec'
            else:
                value, maxUnits = result
            maxAcceptableAngResolution = float(value)
        newvalues = []
        if representativeFrequency is not None:
            newvalues += [['representativeFrequency = %f %s'%(representativeFrequency,freqUnits)]]
        if minAcceptableAngResolution is not None:
            newvalues += [['minAcceptableAngResolution = %f %s'%(minAcceptableAngResolution, minUnits)]]
        if maxAcceptableAngResolution is not None:
            newvalues += [['maxAcceptableAngResolution = %f %s'%(maxAcceptableAngResolution, maxUnits)]]
        if dynamicRange is not None:
            newvalues += [['dynamicRange = %f'%(dynamicRange)]]
        if representativeBandwidth is not None:
            newvalues += [['representativeBandwidth = %f %s'%(representativeBandwidth, bwUnits)]]
        if spectralDynamicRangeBandWidth is not None:
            newvalues += [['spectralDynamicRangeBandWidth = %f %s'%(spectralDynamicRangeBandWidth, bwUnits)]]
        if representativeFrequency is not None:
            newvalues += [['representativeSource = %s'%representativeSource]]
        if representativeSPW is not None:
            newvalues += [['representativeWindow = %s'%str(representativeSPW)]]
        if maxAllowedBeamAxialRatio is not None:
            newvalues += [['maxAllowedBeamAxialRatio = %f'%maxAllowedBeamAxialRatio]]
        newvalues = np.array(newvalues, dtype=str)
        if len(newvalues) != numScienceGoal:
            print("Updating numScienceGoal to %d" % (len(newvalues)))
            mytb.putcol('numScienceGoal',[len(newvalues)])
            casalog.post('Wrote new value of numScienceGoal to %s/ASDM_SBSUMMARY: %d'%(vis,len(newvalues)))
        print("Putting new values:\n", newvalues)
        mytb.putcol('scienceGoal',newvalues)
        casalog.post('Wrote new values to %s/ASDM_SBSUMMARY: %s'%(vis,str(newvalues)))
    else:
        if verbose:
            print("Current values: shape=%s\n" % (str(np.shape(scienceGoal))), scienceGoal)
        if not representativeSpwPresent:
            print("Looking for spw that contains the representative frequency...")
            spw, repBW, minBW, maxBW = representativeSpwBandwidth(vis,verbose=False)
            if spw is not None:
                print("spw = ", spw)
    mytb.close()
    return mydict

def representativeFrequency(vis, verbose=True):
    """
    Get the representative frequency from the ASDM_SBSUMMARY table of a
    measurement set, if it has been imported with asis.
    e.g. [representativeFrequency = 230.0348592858192 GHz, ...] 
    verbose: if True, then also print the min/max acceptable angular resolutions
    Returns the value in GHz.
    """
    if (not os.path.exists(vis)):
        print("Could not find measurement set.")
        return
    mytb = tbtool()
    if (not os.path.exists(vis+'/ASDM_SBSUMMARY')):
        print("Could not find ASDM_SBSUMMARY table.  Did you not import it with asis='SBSummary'?")
        return
    mytb.open(vis+'/ASDM_SBSUMMARY')
    scienceGoal = mytb.getcol('scienceGoal')
    mytb.close()
    freq = 0
    minAcceptableResolution = 0
    maxAcceptableResolution = 0
    bw = None
    for args in scienceGoal:
        for arg in args:
            loc = arg.find('representativeFrequency')
            if (loc >= 0):
                tokens = arg[loc:].split()
                freq = parseFrequencyArgumentToGHz(tokens[2]+tokens[3])
            loc = arg.find('representativeBandwidth')
            if (loc >= 0):
                tokens = arg[loc:].split()
                bw = parseFrequencyArgumentToGHz(tokens[2]+tokens[3])
            loc = arg.find('minAcceptableAngResolution')
            if (loc >= 0):
                tokens = arg[loc:].split()
                minAcceptableResolution = float(tokens[2])
                minUnits = tokens[3]
            loc = arg.find('maxAcceptableAngResolution')
            if (loc >= 0):
                tokens = arg[loc:].split()
                maxAcceptableResolution = float(tokens[2])
                maxUnits = tokens[3]
    if verbose:
        print("minAcceptableResolution = %f %s" % (minAcceptableResolution, minUnits))
        print("maxAcceptableResolution = %f %s" % (maxAcceptableResolution, maxUnits))
        if bw is not None:
            print("representativeBandwidth = %f GHz" % (bw))
    return(freq)

def surmiseRepresentativeSpw(vis, verbose=True):
    """
    Reads the representative frequency from the measurement set, then computes 
    which science spw(s) contains it.
    Returns: spwID, repBW, repNchan, np.min(bandwidths), np.max(bandwidths)
    -Todd Hunter
    """
    freq = representativeFrequency(vis, verbose)
    mymsmd = msmdtool()
    mymsmd.open(vis)
    spws = getScienceSpwsForFrequency(vis, freq, mymsmd=mymsmd)
    scienceSpws = getScienceSpws(vis, mymsmd=mymsmd, returnString=False)
    if (len(spws) == 1):
        value = spws[0]
    elif (len(spws) == 0):
        print("No spws cover the representative frequency (%g GHz)" % (freq))
        spws = scienceSpws
        print("Spw central frequencies in GHz: ", np.array([mymsmd.meanfreq(spw) for spw in spws]) * 1e-9)
        value = None
    else:
        print("Multiple spws (%s) cover the representative frequency (%g GHz)" % (str(spws),freq))
        print("Returning the one nearest to the center.")
        spw = getScienceSpwsForFrequency(vis, freq, nearestOnly=True, mymsmd=mymsmd)
        value = spw
    repBW = mymsmd.bandwidths(value)
    repNchan = mymsmd.nchan(value)
    bandwidths = mymsmd.bandwidths(scienceSpws)
    mymsmd.close()
    return value, repBW, repNchan, np.min(bandwidths), np.max(bandwidths)

def getScienceSpwsForFrequency(vis, frequency, nearestOnly=False, mymsmd=None):
    """
    Returns a list of science spws that cover a given frequency.
    vis: name of measurement set
    frequency: in Hz, GHz, or a string with units
    nearestOnly: if True, the return only one spw (nearest to center)
    -Todd Hunter
    """
    needToClose = False
    if mymsmd is None:
        mymsmd = msmdtool()
        mymsmd.open(vis)
        needToClose = True
    spws = getScienceSpws(vis, returnString=False, mymsmd=mymsmd)
    frequency = parseFrequencyArgumentToHz(frequency)
    spws2 = []
    delta = []
    for spw in spws:
        freqs = mymsmd.chanfreqs(spw)
        if (np.min(freqs) <= frequency and np.max(freqs) >= frequency):
            spws2.append(spw)
            delta.append(abs(frequency-mymsmd.meanfreq(spw)))
    if needToClose:
        mymsmd.close()
    if nearestOnly:
        return(spws2[np.argmin(delta)])
    else:
        return(spws2)

def getScienceSpwBandwidths(vis, intent='OBSERVE_TARGET#ON_SOURCE', 
                             tdm=True, fdm=True, mymsmd=None, sqld=False, 
                             verbose=False, returnDict=False, returnMHz=False):
    """
    Returns: an array of bandwidths (in Hz) in order sorted by spw ID
    returnDict: if True, then return a dictionary keyed by spw ID
    -Todd Hunter
    """
    if mymsmd is None:
        mymsmd = msmdtool()
        mymsmd.open(vis)
        needToClose = True
    else:
        needToClose = False
    spws = sorted(getScienceSpws(vis, intent, False, False, tdm, fdm, mymsmd, sqld, verbose))
    bandwidths = mymsmd.bandwidths(spws)
    if needToClose:
        mymsmd.close()
    mymsmd.close()
    if returnMHz:
        bandwidths *= 1e-6
    if returnDict:
        mydict = {}
        for i, spw in enumerate(spws):
            mydict[spw] = bandwidths[i]
        return mydict
    else:
        return bandwidths

def getScienceSpws(vis, intent='OBSERVE_TARGET#ON_SOURCE', returnString=True, 
                   returnListOfStrings=False, tdm=True, fdm=True, mymsmd=None, 
                   sqld=False, verbose=False, returnFreqRanges=False):
    """
    Return a list of spws with the specified intent.  For ALMA data,
    it ignores channel-averaged and SQLD spws.
    intent: either full intent name including #subIntent, or an abbreviated key, like 'PHASE'
    returnString: if True, return '1,2,3'
                  if False, return [1,2,3]
    returnListOfStrings: if True, return ['1','2','3']  (amenable to tclean spw parameter)
                         if False, return [1,2,3]
    returnFreqRanges: if True, returns a dictionary keyed by spw ID, with values
          equal to the frequency of the middle of the min and max channel (Hz)
    -- Todd Hunter
    """
    if returnString and returnListOfStrings:
        print("You can only specify one of: returnString, returnListOfStrings")
        return
    needToClose = False
    if (mymsmd is None):
        mymsmd = msmdtool()
        mymsmd.open(vis)
        needToClose = True
    allIntents = mymsmd.intents()
    if (intent not in allIntents and intent != ''):
        for i in allIntents:
            if i.find(intent) >= 0:
                intent = i
                print("Translated intent to ", i)
                break
    # minimum match OBSERVE_TARGET to OBSERVE_TARGET#UNSPECIFIED
    value = [i.find(intent.replace('*','')) for i in allIntents]
    # If any intent gives a match, the mean value of the location list will be > -1
    if np.mean(value) == -1 and intent != '':
        print("%s not found in this dataset. Available intents: " % (intent), allIntents)
        if needToClose: 
            mymsmd.close()
        if returnString:
            return ''
        else:
            return []
    if intent == '':
        spws = mymsmd.spwsforintent('*')
    else:
        spws = mymsmd.spwsforintent(intent)
    if (getObservatoryName(vis).find('ALMA') >= 0 or getObservatoryName(vis).find('OSF') >= 0):
        almaspws = mymsmd.almaspws(tdm=tdm,fdm=fdm,sqld=sqld)
        if (len(spws) == 0 or len(almaspws) == 0):
            scienceSpws = []
        else:
            scienceSpws = np.intersect1d(spws,almaspws)
    else:
        scienceSpws = spws
    mydict = {}
    for spw in scienceSpws:
        mydict[spw] = sorted([mymsmd.chanfreqs(spw)[0],mymsmd.chanfreqs(spw)[-1]])
    if needToClose:
        mymsmd.close()
    if returnFreqRanges:
        return mydict
    if returnString:
        return(','.join(str(i) for i in scienceSpws))
    elif returnListOfStrings:
        return list([str(i) for i in scienceSpws])
    else:
        return(list(scienceSpws))

def pickFirstElementIfNecessary(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        return value[0]
    else:
        return value
    
################################################################################
# Functions below this point are not used by the Cycle 6 or 7 pipeline or PL2020
################################################################################

def readContDat(filename):
    """
    Reads the channel selection for a _findContinuum.dat file
    """
    f = open(filename,'r')
    line = f.readlines()
    selection = line[0].split()[0]
    f.close()
    return selection

def readContDatPNG(filename):
    """
    Reads the png name from a _findContinuum.dat file
    """
    f = open(filename,'r')
    line = f.readlines()
    png = line[0].split()[1]
    f.close()
    return png

def readContDatAggregateContinuum(filename):
    """
    Reads the aggregate continuum from a _findContinuum.dat file.
    Returns: value in GHz
    """
    f = open(filename,'r')
    line = f.readlines()
    selection = float(line[0].split()[-1])
    f.close()
    return selection

def readContDatChanWidth(filename):
    """
    Converts aggBW and channel number to chanWidth (in GHz).
    This will differ slightly from fc.numberOfChannelsInCube(jointMask,returnChannelWidth=True)
    due to precision of the frequency display in the file.
    """
    aggBW = readContDatAggregateContinuum(filename)
    nchan = len(convertSelectionIntoChannelList(readContDat(filename)))
    chanWidth = float(aggBW)/nchan
    return chanWidth

def readContDatLSRKRanges(filename):
    """
    Reads the LSRK ranges from a _findContinuum.dat file and returns one
    string of format: 'XXX.YYYYYYYYYGHz~XXX.YYYYYYYYYGHz;...'
    """
    f = open(filename,'r')
    lines = f.readlines()
    channelString, png, aggregateContinuum = lines[0].split()
    lsrkRangesString = lines[1].strip().replace(' ','').replace('LSRK',' ').split()
    lsrkRanges = ';'.join(lsrkRangesString)
    return lsrkRanges

def imageChannelFrequency(img, channel=0, myia=None):
    """
    Returns the frequency (in GHz) of the specified channel of any CASA or FITS
    image cube with a spectral axis.
    channel: can be a scalar or a list/vector, if None then return list for all channels, 
             if -1 then return the final channel
    Returns: value in GHz; scalar (if one channel) or array (if more than 1 channel)
    -Todd Hunter
    """
    if not os.path.exists(img):
        print("Could not find image")
        return
    pixel = [0,0,0,channel]
    if myia is None or myia == '':
        myia = iatool()
        try:
            myia.open(img)
        except:
            print("Could not open image: %s, check permissions" % (img))
            return
        needToClose = True
    else:
        needToClose = False
    maxchan = numberOfChannelsInCube(img, myia=myia) - 1
    if type(channel) not in [list, range, np.ndarray]:
        if channel is not None:
            if (channel < -1 or channel > maxchan):
                print("Invalid channel %d, max=%d" % (channel,maxchan))
                if needToClose:
                    myia.close()
                return
    elif type(channel) == range:
        channel = list(channel)
    spectralAxis = findSpectralAxis(myia)
    freq = []
    nchan = myia.shape()[spectralAxis]
    if type(channel) != list and type(channel) != np.ndarray:
        if channel is None:
            channels = range(nchan)
        elif channel == -1:
            channels = [nchan-1]
        else:
            channels = [channel]
    else:
        channels = channel
    for channel in channels:
        pixel[spectralAxis] = channel
        mydict = myia.toworld(pixel)
        if (len(mydict['numeric']) <= spectralAxis):
            print("Not a cube?")
            return
        freq.append(mydict['numeric'][spectralAxis])
    if needToClose:
        myia.close()
    freq = np.array(freq)
#    print("type(freq) = ", type(freq[0]))
    if len(freq) == 1:
        # the first channel is all we have
        freq = freq[0]
    return freq * 1e-9
             
def recalcMomDiffSNR(priorValuesFile, img='', intersectRanges='', 
                     subimage=False, useAnnulus=True, datfile='',
                     outdir='', includeMom10=True, imageInfo=None):
    """
    Allows rapid feedback for adjusting the channel ranges to use for momDiffSNR
    priorValuesFile: a pickle file generated by findContinuum()
    img: the cube (in case it has been moved and cannot be found automatically)
      1) The pbmom image is required to be present, either in the same directory
         as the img, or the current working directory.
      2) The joint mask listed in the priorValuesFile must also exist in the
         current working directory.
    intersectRanges: the selection of channels to use;  either a single string,
        or a python list of strings to be intersected before using.  If you 
        would like to use LSRK frequency ranges instead, you can first use the 
        function au.imageFrequencyRangesToChannelRanges() to do the conversion 
        for your cube.
    subimage: setting this True will raise the pb levels of the annulus by 0.2
    useAnnulus: passed to fc.imageSNR
    datfile: if present, then generate new datfile with the missing ranges 
        removed
    outdir: if blank, then put it where the cube is
    includeMom10: also compute moment 10 and mom10DiffSNR
    Returns: new momDiffSNR, new AggBW (GHz), and (if includeMom10=True) the
             new mom10DiffSNR
    """
    if datfile != '':
        if not os.path.exists(datfile):
            print("Could not find datfile: ", datfile)
            return
    open_file = open(priorValuesFile, "rb")
    result = pickle.load(open_file)
    print("loaded %d values from pickle file: %s" % (len(result), priorValuesFile))
    open_file.close()
    if len(result) < 14:  # PL2021
        [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask] = result
        if img == '':
            loc = jointMask.find('.findcont.residual')
            cube = jointMask[:loc]+'.findcont.residual'
        else:
            cube = img
    elif len(result) == 15:  # PL2022+23
        [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad] = result
        if img == '':
            loc = jointMask.find('.findcont.residual')
            cube = jointMask[:loc]+'.findcont.residual'
        else:
            cube = img
    elif len(result) == 17:  # PL2024
        [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad, mom0peakOverMad, mom8peakOverMad] = result
        if img == '':
            loc = jointMask.find('.findcont.residual')
            cube = jointMask[:loc]+'.findcont.residual'
        else:
            cube = img
    else:
        print("Unrecognized len(result) = %d" % (len(result)))
        return
    if not os.path.exists(cube):
        print("Could not find the cube.")
        return
    chanInfo = numberOfChannelsInCube(cube, returnChannelWidth=True, returnFreqs=True) 
    nchan, firstFreq, lastFreq, channelWidth = chanInfo
    if isinstance(intersectRanges, (str, np.bytes_)):
        selection = intersectRanges
    elif isinstance(intersectRanges, (list, np.ndarray)):
        selection = intersectRanges[0]
        if len(intersectRanges) > 1:
            for r in intersectRanges[1:]:
                print("Intersecting %s with %s" % (selection,r))
                selection = intersectChannelSelections(selection,r)
            print("Intersected range: ", selection)
    else:
        print("Invalid type for intersectRanges: %s" % (str(type(intersectRanges))))
        return
    if selection == '':
        selection = '%d~%d' % (0,nchan-1)
    fcChannels = len(convertSelectionIntoChannelList(selection))

    # build mom8fc and mom0fc images
    if outdir == '':
        momRoot = cube+'.recalcMomDiffSNR.fc'
    else:
        momRoot = os.path.join(outdir,os.path.basename(cube)+'.recalcMomDiffSNR.fc')
    mom8fc = momRoot + '.maximum' # cube+'.recalcMomDiffSNR.mom8fc'
    removeIfNecessary(mom8fc)  # in case there was a prior run
    mom0fc = momRoot + '.integrated'
    removeIfNecessary(mom0fc)  # in case there was a prior run
    if includeMom10:
        moments = [0,8,10]
        mom10fc = momRoot + '.minimum'
        removeIfNecessary(mom10fc)  # in case there was a prior run
    else:
        moments = [0,8]
    immoments(cube, moments=moments, chans=selection, outfile=momRoot)
        
    # create scaled version of mom0fc
    if outdir == '':
        mom0fcScaled = mom0fc + '.scaled'
    else:
        mom0fcScaled = os.path.join(outdir, os.path.basename(mom0fc) + '.scaled')
    removeIfNecessary(mom0fcScaled)  # in case there was a prior run
    meanFreqHz = np.mean([firstFreq, lastFreq])
    channelWidth = (lastFreq-firstFreq)/(nchan-1)
    chanWidthKms = 299792.458*channelWidth/meanFreqHz 
    factor = 1.0/chanWidthKms/fcChannels
    print("scale factor = %f, chanWidthKms=%f, fcChannels=%d" % (factor, chanWidthKms, fcChannels))
    immath(mom0fc, mode='evalexpr', expr='IM0*%f'%(factor), outfile=mom0fcScaled)
    # Create momDiff 
    if outdir == '':
        momDiff = cube+'.recalcMomDiffSNR.momDiff'
    else:
        momDiff = os.path.join(outdir, os.path.basename(cube)+'.recalcMomDiffSNR.momDiff')
    removeIfNecessary(momDiff)  # in case there was a prior run
    pbcube = locatePBCube(cube)
    pbmom = pbcube+'mom'
    if not os.path.exists(pbmom):
        print("pbmom is not in the same directory as the cube, looking in current directory instead")
        # look in current directory
        pbmom = os.path.basename(pbcube)+'mom'
        if not os.path.exists(pbmom):
            print("recalcMomDiffSNR: Could not find pbmom image, neither with the cube, nor in the current working directory.")
            return
    immath([mom8fc, mom0fcScaled], mode='evalexpr', expr='IM0-IM1', 
           mask='"%s">0.23'%(pbmom), outfile=momDiff)
    if includeMom10:
        if outdir == '':
            mom10Diff = cube+'.recalcMomDiffSNR.mom10Diff'
        else:
            mom10Diff = os.path.join(outdir, os.path.basename(cube)+'.recalcMomDiffSNR.mom10Diff')
        removeIfNecessary(mom10Diff)  # in case there was a prior run
        immath([mom0fcScaled, mom10fc], mode='evalexpr', expr='IM0-IM1', 
               mask='"%s">0.23'%(pbmom), outfile=mom10Diff)
        
    if True:
        if jointMask is None:
            jointMaskTest = None
        else:
            jointMaskTest = '"' + jointMask + '"==0'
        if useAnnulus and pbcube is not None:
            lowerAnnulusLevel, higherAnnulusLevel = findOuterAnnulusForPBCube(pbcube, False, False, subimage, imageInfo)
            if jointMaskTest is None:  # this happens when meanSpectrumFile is specified
                jointMaskTestAnnulus = '"%s">%f && "%s"<%f' % (pbmom,lowerAnnulusLevel,pbmom,higherAnnulusLevel)
            else:  # this is the pipeline use case
                jointMaskTestAnnulus = jointMaskTest + ' && "%s">%f && "%s"<%f' % (pbmom,lowerAnnulusLevel,pbmom,higherAnnulusLevel)
        else:
            jointMaskTestAnnulus = ''
    print("jointMaskTest = ", jointMaskTest)
    print("jointMaskTestAnnulus = ", jointMaskTestAnnulus)
    momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD = imageSNR(momDiff, mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus, 
                                                                  useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
    if includeMom10:
        mom10DiffSNR, mom10DiffPeak, mom10DiffMedian, mom10DiffMAD = imageSNR(mom10Diff, mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus, 
                                                                  useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)

    chanwidth = numberOfChannelsInCube(jointMask, returnChannelWidth=True)[1] # Hz
    newAggBW = computeBandwidth(selection, chanwidth)
    print("aggregate bandwidth = %.4f GHz" % (newAggBW))
    print("momDiffSNR = %.4f" % (momDiffSNR))
    if includeMom10:
        print("mom10DiffSNR = %.4f" % (mom10DiffSNR))
    if datfile != '':
        originalSelection = readContDat(datfile)
        originalPNG = readContDatPNG(datfile)
        originalAggBW = readContDatAggregateContinuum(datfile)
        originalLSRK = readContDatLSRKRanges(datfile)
        ranges = selection.split(';')
        originalRanges = originalSelection.split(';')
        originalLSRKRanges = originalLSRK.split(';')
        LSRKranges = []
        for i,r in enumerate(ranges):
            if r not in originalRanges:
                print("Warning: You have changed a range (%d), not merely removed some.  I will recalculate the LSRKrange and overwrite the datfile anyway." % (i))
                freqs = imageChannelFrequency(cube, channel=None)
                chanwidth = freqs[1]-freqs[0]
                newrange = '%.9fGHz~%.9fGHz' % (freqs[int(r.split('~')[0])]-0.5*chanwidth, freqs[int(r.split('~')[-1])]+0.5*chanwidth)
                LSRKranges.append(newrange)
            else:
                loc = originalRanges.index(r)
                LSRKranges.append(originalLSRKRanges[loc])
        LSRKranges = ' LSRK '.join(LSRKranges) + ' LSRK'
        foutput = datfile+'.recalc.dat'
        f = open(foutput, 'w')
        f.write('%s %s %.5f\n' % (selection, originalPNG, newAggBW))
        f.write('%s\n' % (LSRKranges))
        f.close()
        print("Wrote ", foutput)
    if includeMom10:
        return momDiffSNR, newAggBW, mom10DiffSNR
    else:
        return momDiffSNR, newAggBW

def printPickleFile(priorValuesFile):
    """
    Prints the contents of the specified pickle file of prior values
    """
    open_file = open(priorValuesFile, "rb")
    result = pickle.load(open_file)
    print("loaded %d values from pickle file: %s" % (len(result), priorValuesFile))
    open_file.close()
    if len(result) < 14:  # PL2021
        [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask] = result
    elif len(result) == 15:  # PL2022
        [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad] = result
    for i,variable in enumerate(['avgSpectrumNansReplaced', 'normalized', 'numberPixelsInJointMask', 'pbBasedMask', 'initialQuadraticRemoved', 'initialQuadraticImprovementRatio', 'mom0snrs', 'mom8snrs', 'regionsPruned', 'numberPixelsInMom8Mask', 'mom0peak', 'mom8peak', 'jointMask']):
        print("%2d) %s = %s" % (i, variable, eval(variable)))
    if len(result) == 15:  # PL2022
        for i,variable in enumerate(['nbin','initialPeakOverMad']):
            print("%2d) %s = %s" % (i+13, variable, eval(variable)))

def modifyContinuumChannels(avgSpectrumNansReplaced, minlevel, maxlevel, intersectRanges='', 
                            trimChannels=0.2, narrow=2, maxTrim=maxTrimDefault, maxTrimFraction=1.0,
                            avoidance='', verbose=False):
    """
    Re-assess spectrum using new minlevel and maxlevel
    verbose: print the list of channel ranges after each stage of processing
    """
    nchan = len(avgSpectrumNansReplaced)
    avgSpectrumNansReplaced = np.array(avgSpectrumNansReplaced)
    idx = np.where((minlevel < avgSpectrumNansReplaced) * (avgSpectrumNansReplaced < maxlevel))
#    print('qualifying channels: ', idx[0])
    if len(intersectRanges) > 0:
        finalList = np.intersect1d(convertSelectionIntoChannelList(intersectRanges), idx[0])
    else:
        finalList = idx[0]
    if verbose:
        print("finalList after intersect = ", finalList)
    if avoidance != '':
        finalList = np.intersect1d(finalList, list(set(range(nchan))-set(convertSelectionIntoChannelList(avoidance))))
        if verbose:
            print("finalList after avoidance = ", finalList)
    finalList = splitListIntoContiguousListsAndTrim(nchan, finalList, trimChannels, maxTrim, maxTrimFraction, verbose)
    if verbose:
        print("finalList after trim = ", finalList)
    finalList = splitListIntoContiguousListsAndRejectNarrow(finalList, narrow)
    if verbose:
        print("finalList after narrow = ", finalList)
    mad = MAD(avgSpectrumNansReplaced[finalList])
    median = np.median(avgSpectrumNansReplaced[finalList])
    return finalList, mad, median

def plotPickleFile(priorValuesFile, sigmaFindContinuum=0.5, plotfile='', contdat='', 
                   narrow=2, trimChannels=0, img='', intersectRanges='', 
                   computeMomDiffSNR=False, useAnnulus=True, subimage=False, 
                   sigmaFindContinuumMode='fixed',
                   minlevel=None, maxlevel=None, median=None, fraction=1.0,
                   maxTrim=maxTrimDefault, maxTrimFraction=1.0, avoidance='',
                   grid=False, verbose=False, imageInfo=None):
    """
    Allows rapid feedback from adjusting the cut level of the pickle file 
    spectrum created by a previous run of findContinuum, and produces a 
    new _findContinuum.dat file.
    priorValuesFile: pickle file from prior run of findContinuum
         The following things are used: the spectrum, jointMask.
    plotfile: if not defined, set it to jointMask+'_adjusted.png'
             if it is a directory path, then prepend it
    contdat: name of file to write; if not defined, set it to 
        jointMask+'_adjusted_findContinuum.dat'; 
        if it is a directory path, then prepend it to this default filename
    img: full path to cube, in case it has been moved since creating the pickle
    intersectRanges: if specified, don't allow selections outside of these ranges, 
         example: '60~80;200~300'
    computeMomDiffSNR: if specified, compute and return the momDiffSNR
    useAnnulus: passed to fc.imageSNR
    trimChannels: use 0 to avoid re-trimming ranges that were already trimmed
         in the initial run of findContinuum
    subimage: passed to findOuterAnnulusForPBCube() to raise annulus by 0.2
    sigmaFindContinuumMode: passed to findContinuumChannels()
    To invoke alternative method, set:
    minlevel: level of lower dotted line of prior findContinuum run
    maxlevel: level of upper dotted line of prior findContinuum run
    median: level of solid line of prior findContinuum run (only needed to
        draw this level on the plot, or if fraction is specified)
    fraction: scale the median-to-min distance and median-to-max distance by this
       fraction prior to running; if >1.0, then also ignore intersectRanges
    avoidance: passed to modifyContinuumChannels; thus, only used if minlevel/maxlevel/median
        are set
    grid: if True, draw dotted grid lines at the major tick marks
    Note: only minlevel and maxlevel are ultimately passed to modifyContinuumChannels;
          fraction and median are merely a way to adjust them automatically.
    """
    open_file = open(priorValuesFile, "rb")
    result = pickle.load(open_file)
    print("loaded %d values from pickle file: %s" % (len(result), priorValuesFile))
    open_file.close()
    if len(result) < 14:  # PL2021
        [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask] = result
        if img == '':
            loc = jointMask.find('.findcont.residual')
            cube = jointMask[:loc]+'.findcont.residual'
        else:
            cube = img
        chanInfo = numberOfChannelsInCube(cube, returnChannelWidth=True, returnFreqs=True) 
        imageInfo = getImageInfo(cube)
    elif len(result) == 15:  # PL2022
        [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad] = result
        if img == '':
            loc = jointMask.find('.findcont.residual')
            cube = jointMask[:loc]+'.findcont.residual'
        else:
            cube = img
        chanInfo = numberOfChannelsInCube(cube, returnChannelWidth=True, returnFreqs=True) 
        imageInfo = getImageInfo(cube)
    else: # if I expand the contents of the pkl file someday to include chanInfo and imageInfo:
        [avgSpectrumNansReplaced, normalized, numberPixelsInJointMask, pbBasedMask, initialQuadraticRemoved, initialQuadraticImprovementRatio, mom0snrs, mom8snrs, regionsPruned, numberPixelsInMom8Mask, mom0peak, mom8peak, jointMask, nbin, initialPeakOverMad, cube, chanInfo, imageInfo] = result
    nchan, firstFreq, lastFreq, channelWidth = chanInfo
    if plotfile == '':
        plotfile = jointMask + '_adjusted.png'
    elif os.path.isdir(plotfile):
        plotfile = os.path.join(plotfile, jointMask + '_adjusted.png')
    if contdat == '':
        contdat = jointMask + '_adjusted'
    elif os.path.isdir(contdat):
        contdat = os.path.join(contdat, jointMask + '_adjusted')
    pl.clf()
    fig = pl.gcf()
    if casaVersion >= '5.9': 
        fig.set_size_inches(8, 6, forward=True)
    desc = pl.subplot(111)
    pl.plot(range(len(avgSpectrumNansReplaced)), avgSpectrumNansReplaced)
    pl.xlabel('Channels')
    pl.grid(grid)
    jointMask = jointMask.replace('./','')
    print("jointMask = ", jointMask)
    pl.title(os.path.basename(priorValuesFile), size=10)
    nchan = len(avgSpectrumNansReplaced)
    nBaselineChannels = int(round(0.19*nchan))
    medianCyan = None
    if minlevel is not None and maxlevel is not None and median is not None:
        # This method was added in August 2022
        if maxlevel < median or minlevel > median and fraction >= 1:
            print("non-sensical min,median,max values")
            return
        if fraction != 1.0:
            minlevel = median - fraction*(median-minlevel)
            maxlevel = median + fraction*(maxlevel-median)
            if fraction > 1:
                intersectRanges = ''
        casalogPost('CALLING modifyContinuumChannels() instead of findContinuumChannels()')
        finalList,mad,medianCyan = modifyContinuumChannels(avgSpectrumNansReplaced, minlevel, maxlevel, 
                                                           intersectRanges, trimChannels, narrow, maxTrim, 
                                                           maxTrimFraction, avoidance, verbose)
        positiveThreshold = maxlevel
        negativeThreshold = minlevel
        medianTrue = median
    elif minlevel is None and maxlevel is None and median is None:
        result = findContinuumChannels(avgSpectrumNansReplaced, nBaselineChannels, 
                                       sigmaFindContinuum, 
                                       sigmaFindContinuumMode=sigmaFindContinuumMode, 
                                       narrow=narrow, trimChannels=trimChannels)
        initialList = result[0]
        positiveThreshold = result[2]
        median = result[3]
        medianTrue = result[6]
        mad = result[7]
        negativeThreshold = result[9]
        if intersectRanges != '':
            finalList = np.intersect1d(initialList, convertSelectionIntoChannelList(intersectRanges))
            reduction = len(initialList) - len(finalList)
            if reduction > 0:
                print("intersectRanges reduced the channel list by %d channels to %d channels." % (reduction,len(finalList)))
        else:
            finalList = result[0]
        
    else:
        print("You must specify all 3 parameters or none of them: minlevel, maxlevel, median")
        return
    myChannelLists = splitListIntoContiguousLists(finalList)
    selection = convertChannelListIntoSelection(finalList)
    print("MAD = ", MAD(avgSpectrumNansReplaced))
    print("selection: ", selection)
    ylevel = medianTrue + mad
    print("medianTrue + MAD = ", ylevel)
    pl.plot([0,nchan-1], 2*[positiveThreshold], 'k:')
    pl.plot([0,nchan-1], 2*[negativeThreshold], 'k:')
    pl.plot([0,nchan-1], 2*[median], 'k--')
    pl.plot([0,nchan-1], 2*[medianTrue], 'k-')
    if medianCyan is not None:
        pl.plot([0,nchan-1], 2*[medianCyan], 'c:')
        pl.text(0.9,0.95,'fraction=%.3f' % (fraction), ha='right', transform=desc.transAxes)
    for myChannelList in myChannelLists:
        if len(myChannelList) == 1: # [0] == myChannelList[1]:
            myChannelList[0] = myChannelList[0]-0.5
            myChannelList.append(myChannelList[0]+1)
        pl.plot(myChannelList, len(myChannelList)*[positiveThreshold+3*mad], 'c-', lw=2)
    aggBW = computeBandwidth(selection, channelWidth)  # in GHz
    pl.text(0.9,0.9,'aggBW=%.3f MHz' % (aggBW*1000), ha='right', transform=desc.transAxes)
    pl.draw()
    if pl.get_backend() == 'TkAgg' and casaVersion >= '5.9':
        fig.canvas.flush_events() # critical for TkAgg in CASA 6, or else set_fig_size never takes effect!
    pl.savefig(plotfile, dpi=dpiDefault)
    print("Wrote ", plotfile)
    writeContDat(contdat, selection, plotfile, aggBW, firstFreq, lastFreq, channelWidth, cube, imageInfo)
    if computeMomDiffSNR:
        # build mom8fc and mom0fc images
        momRoot = cube+'.plotPickleFile.fc'
        mom8fc = momRoot + '.maximum' 
        removeIfNecessary(mom8fc)  # in case there was a prior run
        mom0fc = momRoot + '.integrated'
        removeIfNecessary(mom0fc)  # in case there was a prior run
        mom10fc = momRoot + '.minimum'
        removeIfNecessary(mom10fc)  # in case there was a prior run
        immoments(cube, moments=[0,8,10], chans=selection, outfile=momRoot)

        # create scaled version of mom0fc
        mom0fcScaled = mom0fc + '.scaled'
        removeIfNecessary(mom0fcScaled)  # in case there was a prior run
        meanFreqHz = np.mean([firstFreq, lastFreq])
        channelWidth = (lastFreq-firstFreq)/(nchan-1)
        chanWidthKms = 299792.458*channelWidth/meanFreqHz 
        fcChannels = len(finalList)
        factor = 1.0/chanWidthKms/fcChannels
        print("scale factor = %f, chanWidthKms=%f, fcChannels=%d" % (factor, chanWidthKms, fcChannels))
        immath(mom0fc, mode='evalexpr', expr='IM0*%f'%(factor), outfile=mom0fcScaled)
        # Create momDiff 
        momDiff = cube+'.plotPickleFile.momDiff'
        removeIfNecessary(momDiff)  # in case there was a prior run
        mom10Diff = cube+'.plotPickleFile.mom10Diff'
        removeIfNecessary(mom10Diff)  # in case there was a prior run
        pbcube = locatePBCube(cube)
        pbmom = pbcube+'mom'
        if not os.path.exists(pbmom):
            print("pbmom is not in the same directory as the cube, looking in current directory instead")
            # look in current directory
            pbmom = os.path.basename(pbcube)+'mom'
            if not os.path.exists(pbmom):
                print("plotPickleFile: Could not find pbmom image, neither with the cube, nor in the current working directory.")
                return
        immath([mom8fc, mom0fcScaled], mode='evalexpr', expr='IM0-IM1', 
               mask='"%s">0.23'%(pbmom), outfile=momDiff)
        immath([mom0fcScaled, mom10fc], mode='evalexpr', expr='IM0-IM1', 
               mask='"%s">0.23'%(pbmom), outfile=mom10Diff)
        if True:
            if jointMask is None:
                jointMaskTest = None
            else:
                jointMaskTest = '"' + jointMask + '"==0'
            if useAnnulus and pbcube is not None:
                lowerAnnulusLevel, higherAnnulusLevel = findOuterAnnulusForPBCube(pbcube, False, False, subimage, imageInfo)
                if jointMaskTest is None:  # this happens when meanSpectrumFile is specified
                    jointMaskTestAnnulus = '"%s">%f && "%s"<%f' % (pbmom,lowerAnnulusLevel,pbmom,higherAnnulusLevel)
                else:  # this is the pipeline use case
                    jointMaskTestAnnulus = jointMaskTest + ' && "%s">%f && "%s"<%f' % (pbmom,lowerAnnulusLevel,pbmom,higherAnnulusLevel)
            else:
                jointMaskTestAnnulus = ''
        print("jointMaskTest = ", jointMaskTest)
        print("jointMaskTestAnnulus = ", jointMaskTestAnnulus)
        momDiffSNR, momDiffPeak, momDiffMedian, momDiffMAD = imageSNR(momDiff, 
              mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus,
              useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
        mom10DiffSNR, mom10DiffPeak, mom10DiffMedian, mom10DiffMAD = imageSNR(mom10Diff, 
              mask=jointMaskTest, maskWithAnnulus=jointMaskTestAnnulus,
              useAnnulus=useAnnulus, returnAllStats=True, applyMaskToAll=False)
        print("momDiffSNR = %f" % (momDiffSNR))
        print("mom10DiffSNR = %f" % (mom10DiffSNR))
        return momDiffSNR, mom10DiffSNR

def meanSpectrum(img, nBaselineChannels=16, sigmaCube=3, verbose=False, 
                 nanBufferChannels=2, useAbsoluteValue=False,
                 baselineMode='edge', percentile=20, continuumThreshold=None,
                 meanSpectrumFile='', centralArcsec=-1, imageInfo=[], 
                 chanInfo=[], mask='', meanSpectrumMethod='peakOverRms', 
                 peakFilterFWHM=15, iteration=0, 
                 applyMaskToMask=False, useIAGetProfile=True, 
                 useThresholdWithMask=False, overwrite=False, nbin=1):
    """
    This function is not used by Cycle 6 pipeline, but remains as a manual user
    option. Computes a threshold and then uses it to compute the average 
    spectrum across a CASA image cube, via the specified method.
    Inputs:
    nBaselineChannels: number of channels to use as the baseline in 
                       the 'meanAboveThreshold' methods.
    baselineMode: how to select the channels to use as the baseline:
              'edge': use an equal number of channels on each edge of the spw
               'min': use the percentile channels with the lowest absolute intensity
    sigmaCube: multiply this value by the rms to get the threshold above which a pixel
               is included in the mean spectrum
    nanBufferChannels: when removing or replacing NaNs, do this many extra channels
                       beyond their actual extent
    useAbsoluteValue: passed to avgOverCube
    percentile: used with baselineMode='min'
    continuumThreshold: if specified, only use pixels above this intensity level
    meanSpectrumFile: name of ASCII file to produce to speed up future runs 
    centralArcsec: default=-1 means whole field, or a floating point value in arcsec
    mask: a mask image (e.g. from clean); restrict calculations to pixels with mask=1
    chanInfo: the list returned by numberOfChannelsInCube
    meanSpectrumMethod: 'peakOverMad', 'peakOverRms', 'meanAboveThreshold', 
                  'meanAboveThresholdOverRms', or 'meanAboveThresholdOverMad' 
    * 'meanAboveThreshold': uses a selection of baseline channels to compute the 
        rms to be used as a threshold value (similar to constructing a moment map).
    * 'meanAboveThresholdOverRms/Mad': scale spectrum by RMS or MAD        
    * 'peakOverRms/Mad' computes the ratio of the peak in a spatially-smoothed 
        version of cube to the RMS or MAD.  Smoothing is set by peakFilterFWHM.
    peakFilterFWHM: value used by 'peakOverRms' and 'peakOverMad' to presmooth 
        each plane with a Gaussian kernel of this width (in pixels)
        Set to 1 to not do any smoothing.
    useIAGetProfile: if True, then for peakOverMad, or meanAboveThreshold with 
        baselineMode='min', then use ia.getprofile instead of ia.getregion 
        and the subsequent arithmetic (because it should be faster)
    useThresholdWithMask: if False, then just take mean rather than meanAboveThreshold
        when a mask has been specified
    nbin: this argument is accepted but not used (not yet implemented for this older mode)
    Returns 8 items:
       * avgspectrum (vector)
       * avgspectrumAboveThresholdNansRemoved (vector)
       * avgspectrumAboveThresholdNansReplaced (vector)
       * threshold (scalar) 
       * edgesUsed: 0=lower, 1=upper, 2=both
       * nchan (scalar)
       * nanmin (the value used to replace NaNs)
       * percentagePixelsNotMasked
    """
    if meanSpectrumFile == '':
        meanSpectrumFile = img + '.meanSpectrum.' + meanSpectrumMethod
    if meanSpectrumMethod == 'auto':
        print("Invalid meanSpectrumMethod: cannot be 'auto' at this point.")
        return
    if (not os.path.exists(img)):
        casalogPost("Could not find image = %s" % (img))
        return
    myia = iatool()
    usermaskdata = ''
    if (len(mask) > 0):
        # This is the user-specified mask (not the minpb mask inside the cube).
        myia.open(mask)
        maskAxis = findSpectralAxis(myia)
        usermaskShape = myia.shape()
        if useIAGetProfile:
            myshape = myia.shape()
            if myshape[maskAxis] > 1:
                singlePlaneUserMask = False
            else:
                singlePlaneUserMask = True
        else:
            print("Calling myia.getregion(axes=2)")
            usermaskdata = myia.getregion(axes=2)
            if (verbose): print("shape(usermaskdata) = ", np.array(np.shape(usermaskdata)))
            if applyMaskToMask:
                usermaskmask = myia.getregion(getmask=True)
                idx = np.where(usermaskmask==False)
                if len(idx) > 0:
                    casalogPost('applyMaskToMask has zeroed out %d pixels.' % (len(idx[0])))
                    usermaskdata[idx] = 0
            if (np.shape(usermaskdata)[maskAxis] > 1):
                singlePlaneUserMask = False
            else:
                singlePlaneUserMask = True
        if singlePlaneUserMask:
            if (meanSpectrumMethod.find('meanAboveThreshold') >= 0):
                casalogPost("single plane user masks are not supported by meanSpectrumMethod='meanAboveThreshold', try peakOverMad.")
                myia.close()
                return
        myia.close()
    myia.open(img)
    axis = findSpectralAxis(myia)
    nchan = myia.shape()[axis]
    if verbose: print("Found spectral axis = ", axis)
    if len(imageInfo) == 0:
        imageInfo = getImageInfo(img)
    myrg = None
    if True:
        myrg = rgtool()
        if (centralArcsec < 0 or centralArcsec == 'auto' or useIAGetProfile):
            if not useIAGetProfile:
                centralArcsec = -1
            if ((len(mask) > 0 or meanSpectrumMethod != 'peakOverMad') and not useIAGetProfile):
                print("Running ia.getregion() useIAGetProfile=", useIAGetProfile)
                pixels = myia.getregion()
                print("Running ia.getregion(getmask=True)")
                maskdata = myia.getregion(getmask=True)
            blc = [0,0,0,0]
            trc = [myia.shape()[0]-1, myia.shape()[1]-1, 0, 0]
        else:
            bmaj, bmin, bpa, cdelt1, cdelt2, naxis1, naxis2, freq, imgShape, crval1, crval2, maxBaseline, telescope = imageInfo
            nchan = chanInfo[0]
            x0 = int(round_half_up(naxis1*0.5 - centralArcsec*0.5/np.abs(cdelt1)))
            x1 = int(round_half_up(naxis1*0.5 + centralArcsec*0.5/np.abs(cdelt1)))
            y0 = int(round_half_up(naxis2*0.5 - centralArcsec*0.5/cdelt2))
            y1 = int(round_half_up(naxis2*0.5 + centralArcsec*0.5/cdelt2))
            # avoid going off the edge of non-square images
            if (x0 < 0): x0 = 0
            if (y0 < 0): y0 = 0
            if (x0 >= naxis1): x0 = naxis1 - 1
            if (y0 >= naxis2): y0 = naxis2 - 1
            blc = [x0,y0,0,0]
            trc = [x1,y1,0,0]
            trc[axis] = nchan
            region = myrg.box(blc=blc, trc=trc)
            print("Running ia.getregion(region=region)")
            pixels = myia.getregion(region=region)
            print("Running ia.getregion(region=region, getmask=True)")
            maskdata = myia.getregion(region=region,getmask=True)
            if (len(mask) > 0):
                casalogPost("Taking submask for central area of image: blc=%s, trc=%s" % (str(blc),str(trc)))
                usermaskdata = submask(usermaskdata, region)
            if verbose:
                print("shape of pixels = ", np.array(np.shape(pixels)))

    if len(mask) > 0:
        if useIAGetProfile:
            if not (myia.shape() == usermaskShape).all():
                casalogPost("Mismatch in shape between image (%s) and mask (%s)" % (myia.shape(), usermaskShape))
                if myrg is not None: myrg.done()
                return
        else:
            # in principle, the 'if' branch could be used for both cases, but it is not yet tested so we keep the old method.
            if not (np.array(np.shape(pixels)) == np.array(np.shape(usermaskdata))).all():
                casalogPost("Mismatch in shape between image (%s) and mask (%s)" % (np.shape(pixels),np.shape(usermaskdata)))
                if myrg is not None: myrg.done()
                return

    if ((casaVersionCompare('<','5.3.0-22') or not useIAGetProfile) and 
        (meanSpectrumMethod.find('OverRms') > 0 or meanSpectrumMethod.find('OverMad') > 0)):
        # compute myrms or mymad, ignoring masked values and usermasked values
        if (meanSpectrumMethod.find('OverMad') < 0):
            casalogPost("Computing std on each plane")
        else:
            casalogPost("Computing mad on each plane")
        myvalue = []
        for a in range(nchan):
            if ((a+1)%100 == 0): 
                print("Done %d/%d" % (a+1, nchan))
            # Extract this one channel
            if (axis == 2):
                if len(mask) > 0:
                    mypixels = pixels[:,:,a,0]
                    mymask = maskdata[:,:,a,0]
                    if (singlePlaneUserMask):
                        myusermask = usermaskdata[:,:,0,0]
                    else:
                        myusermask = usermaskdata[:,:,a,0]
                else:
                    blc[axis] = a
                    trc[axis] = a
                    myregion = myrg.box(blc=blc,trc=trc)
                    mypixels = myia.getregion(region=myregion)
                    mymask = myia.getregion(region=myregion,getmask=True)
            elif (axis == 3):
                if (len(mask) > 0):
                    mypixels = pixels[:,:,0,a]
                    mymask = maskdata[:,:,0,a]
                    if (singlePlaneUserMask):
                        myusermask = usermaskdata[:,:,0,0]
                    else:
                        myusermask = usermaskdata[:,:,0,a]
                else:
                    blc[axis] = a
                    trc[axis] = a
                    myregion = myrg.box(blc=blc,trc=trc)
                    mypixels = myia.getregion(region=myregion)
                    mymask = myia.getregion(region=myregion,getmask=True)
                    
            if (len(mask) > 0):
                # user mask is typically a clean mask, so we want to use the region outside the
                # clean mask for computing the MAD, but also avoiding the masked edges of the image,
                # which are generally masked to False
                pixelsForStd = mypixels[np.where((myusermask<1) * (mymask==True))]
            else: 
                # avoid the masked (typically outer) edges of the image using the built-in mask
                pixelsForStd = mypixels[np.where(mymask==True)]
            if (meanSpectrumMethod.find('OverMad') < 0):
                myvalue.append(np.std(pixelsForStd))
            else:
                myvalue.append(MAD(pixelsForStd))

        if (meanSpectrumMethod.find('OverMad') < 0):
            myrms = np.array(myvalue)
        else:
            mymad = np.array(myvalue)

    percentagePixelsNotMasked = 100
    threshold = 0  # perhaps it should be None, but this would require modification to print statement in writeMeanSpectrum
    edgesUsed = 0

    if (meanSpectrumMethod.find('peakOver') == 0):
        mybox = '%d,%d,%d,%d' % (blc[0],blc[1],trc[0],trc[1])
        threshold = 0
        edgesUsed = 0
        gaussianSigma = peakFilterFWHM/(2*np.sqrt(2*np.log(2))) # 2.355
        if casaVersionCompare('>=','5.3.0-22') and useIAGetProfile:
            if (gaussianSigma > 1.1/(2*np.sqrt(2*np.log(2)))):
                myia.close()
                smoothimg = img+'.imsmooth'
                if not os.path.exists(smoothimg):
                    casalogPost('Creating smoothed cube for extracting peak/mad over box=%s'%(mybox))
                    imsmooth(imagename=img, kernel='gauss', major='%fpix'%(peakFilterFWHM), 
                             minor='%fpix'%(peakFilterFWHM), pa='0deg',
                             box=mybox, outfile=smoothimg)
                else:
                    casalogPost('Using existing smoothed cube')
                smoothia = iatool()
                smoothia.open(smoothimg)
                avgspectrum = smoothia.getprofile(axis=axis, function='max/xmadm', mask=mask)['values']
#               Old method, before ratios were implemented in toolkit:
#                mymax = smoothia.getprofile(axis=axis, function='max', mask=mask)['values']
#                mymad = smoothia.getprofile(axis=axis, function='xmadm', mask=mask)['values']
#                avgspectrum = mymax / mymad
                smoothia.close()
            else:
                avgspectrum = myia.getprofile(axis=axis, function='max/xmadm', mask=mask)['values']
                mymax = myia.getprofile(axis=axis, function='max', mask=mask)['values']
                mymad = myia.getprofile(axis=axis, function='xmadm', mask=mask)['values']
        else:
            # compute mymax (an array of channel maxima), then divide by either myrms or mymad array
            # which already exist from above.
            myvalue = []
            casalogPost("Smoothing and computing peak on each plane over box=%s." % mybox)
            if (len(mask) > 0):
                pixels[np.where(usermaskdata==0)] = np.nan
            for a in range(nchan):
                if ((a+1)%100 == 0): 
                        print("Done %d/%d" % (a+1, nchan))
                if (axis == 2):
                    if len(mask) > 0:
                        mypixels = pixels[:,:,a,0]
                    else:
                        blc[axis] = a
                        trc[axis] = a
                        myregion = myrg.box(blc=blc,trc=trc)
                        mypixels = myia.getregion(region=myregion)
                elif (axis == 3):
                    if len(mask) > 0:
                        mypixels = pixels[:,:,0,a]
                    else:
                        blc[axis] = a
                        trc[axis] = a
                        myregion = myrg.box(blc=blc,trc=trc)
                        mypixels = myia.getregion(region=myregion)
                if (gaussianSigma > 1.1/(2*np.sqrt(2*np.log(2)))):
                    if (len(mask) > 0):
                        # taken from stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
                        V = mypixels.copy()
                        V[mypixels!=mypixels] = 0
                        VV = gaussian_filter(V,sigma=gaussianSigma)
                        W = 0*mypixels.copy() + 1   # the lack of a zero here was a long-standing bug found on Oct 12, 2017
                        W[mypixels!=mypixels] = 0
                        WW = gaussian_filter(W,sigma=gaussianSigma)
                        mypixels = VV/WW
                        myvalue.append(np.nanmax(mypixels))
                    else:
                        myvalue.append(np.nanmax(gaussian_filter(mypixels,sigma=gaussianSigma)))
                else:
                    myvalue.append(np.nanmax(mypixels))
            print("finished")
            mymax = np.array(myvalue)
            if (meanSpectrumMethod == 'peakOverRms'):
                avgspectrum = mymax/myrms
            elif (meanSpectrumMethod == 'peakOverMad'):
                avgspectrum = mymax/mymad
        nansRemoved = removeNaNs(avgspectrum, verbose=True)
        nansReplaced,nanmin = removeNaNs(avgspectrum, replaceWithMin=True, 
                                         nanBufferChannels=nanBufferChannels, verbose=True)
    elif (meanSpectrumMethod.find('meanAboveThreshold') == 0):
        if (continuumThreshold is not None):
            belowThreshold = np.where(pixels < continuumThreshold)
            if verbose:
                print("shape of belowThreshold = ", np.shape(belowThreshold))
            pixels[belowThreshold] = 0.0
        naxes = len(myia.shape()) # len(np.shape(pixels))
        nchan = myia.shape()[axis] # np.shape(pixels)[axis]
        if (baselineMode == 'edge'):  # not currently used by pipeline
            # Method #1: Use the two edges of the spw to find the line-free rms of the spectrum
            nEdgeChannels = nBaselineChannels/2
            # lower edge
            blc = np.zeros(naxes)
            trc = [i-1 for i in list(np.shape(pixels))]
            trc[axis] = nEdgeChannels
            myrg = rgtool()
            region = myrg.box(blc=blc, trc=trc)
            print("Running ia.getregion(blc=%s,trc=%s)" % (blc,trc))
            lowerEdgePixels = myia.getregion(region=region)
            # drop all floating point zeros (which will drop pixels outside the mosaic image mask)
            lowerEdgePixels = lowerEdgePixels[np.where(lowerEdgePixels!=0.0)]
            stdLowerEdge = MAD(lowerEdgePixels)
            medianLowerEdge = nanmedian(lowerEdgePixels)
            if verbose: print("MAD of %d channels on lower edge = %f" % (nBaselineChannels, stdLowerEdge))

            # upper edge
            blc = np.zeros(naxes)
            trc = [i-1 for i in list(np.shape(pixels))]
            blc[axis] = trc[axis] - nEdgeChannels
            region = myrg.box(blc=blc, trc=trc)
            print("Running ia.getregion(blc=%s,trc=%s)" % (blc,trc))
            upperEdgePixels = myia.getregion(region=region)
#            myrg.done()
            # drop all floating point zeros
            upperEdgePixels = upperEdgePixels[np.where(upperEdgePixels!=0.0)]
            stdUpperEdge = MAD(upperEdgePixels)
            medianUpperEdge = nanmedian(upperEdgePixels)
            casalogPost("meanSpectrum(): edge medians: lower=%.10f, upper=%.10f" % (medianLowerEdge, medianUpperEdge))

            if verbose: 
                print("MAD of %d channels on upper edge = %f" % (nEdgeChannels, stdUpperEdge))
            if (stdLowerEdge <= 0.0):
                edgesUsed = 1
                stdEdge = stdUpperEdge
                medianEdge = medianUpperEdge
            elif (stdUpperEdge <= 0.0):
                edgesUsed = 0
                stdEdge = stdLowerEdge
                medianEdge = medianLowerEdge
            else:
                edgesUsed = 2
                stdEdge = np.mean([stdLowerEdge,stdUpperEdge])
                medianEdge = np.mean([medianLowerEdge,medianUpperEdge])

        if (baselineMode != 'edge'):  # of the meanAboveThreshold case
            # Method #2: pick the N channels with the lowest absolute values (to avoid
            #            confusion from absorption lines and negative bowls of missing flux)
            #            However, for the case of significant absorption lines, this only 
            #            works if the continuum has been subtracted, but in
            #            the pipeline case, it has not been.  It should probably be changed
            #            to first determine if the median is closer to the min or the max, 
            #            and if the latter, then use the maximum absolute values.
            if useIAGetProfile:
                if mask == '' or useThresholdWithMask:
                    if mask == '':
                        maskArgument = ''
                    else:
                        maskArgument = "'%s'>0" % (mask)
                    profile = myia.getprofile(axis=axis, function='min', mask=maskArgument)
                    absPixels = np.abs(profile['values'])
                    # Find the lowest pixel values
                    percentileThreshold = scoreatpercentile(absPixels, percentile)
                    idx = np.where(absPixels < percentileThreshold)
                    # Take their statistics
                    # In the original implementation, the percentile min is computed over all pixels, but 
                    # here I take aggregated over all spatial pixels into a spectral profile, because that
                    # is what is available from getprofile.
                    stdMin = MAD(absPixels[idx])
                    medianMin = np.median(absPixels[idx])
            else:
                if (centralArcsec < 0):
                    print("Running ia.getregion")
                    allPixels = myia.getregion()
                else:
                    allPixels = pixels
                # Convert all NaNs to zero
                allPixels[np.isnan(allPixels)] = 0
                # Drop all floating point zeros and internally-masked pixels from calculation
                if (mask == ''):
                    allPixels = allPixels[np.where((allPixels != 0) * (maskdata==True))]
                else:
                    # avoid identical zeros and clean mask when looking for lowest pixels
                    allPixels = allPixels[np.where((allPixels != 0) * (maskdata==True) * (usermaskdata<1))]
                # Take absolute value
                absPixels = np.abs(allPixels)
                # Find the lowest pixel values by percentile
                percentileThreshold = scoreatpercentile(absPixels, percentile)
                idx = np.where(absPixels < percentileThreshold)
                # Take their statistics
                stdMin = MAD(allPixels[idx])
                medianMin = nanmedian(allPixels[idx])

        if (baselineMode == 'edge'):
            std = stdEdge
            median = medianEdge
            casalogPost("meanSpectrum(): edge mode:  median=%f  MAD=%f  threshold=%f (edgesUsed=%d)" % (medianEdge, stdEdge, medianEdge+stdEdge*sigmaCube, edgesUsed))
            threshold = median + sigmaCube*std
        elif (not useIAGetProfile or mask == '' or useThresholdWithMask):
            std = stdMin
            median = medianMin
            edgesUsed = 0
            casalogPost("**** meanSpectrum(useIAGetProfile=%s): min mode:  median=%f  MAD=%f  threshold=%f" % (useIAGetProfile, medianMin, stdMin, medianMin+stdMin*sigmaCube))
            threshold = median + sigmaCube*std
        
        if useIAGetProfile:
            if mask != '':
                maskArgument = "'%s'>0" % mask
            else:
                maskArgument = ''
            casalogPost("running ia.getprofile(mean, mask='%s')" % maskArgument)
            profile = myia.getprofile(axis=axis, function='mean', mask=maskArgument)
            avgspectrum = profile['values']
            percentagePixelsNotMasked = sum(profile['npix'])*100. / np.prod(myia.shape())
        else:
            if (axis == 2 and naxes == 4):
                # drop the degenerate axis so that avgOverCube will work with nanmean(axis=0)
                pixels = pixels[:,:,:,0]
            if (len(mask) > 0):
                maskdata = propagateMaskToAllChannels(maskdata, axis)
            else:
                maskdata = ''
            avgspectrum, percentagePixelsNotMasked = avgOverCube(pixels, useAbsoluteValue, mask=maskdata, usermask=usermaskdata)
        if meanSpectrumMethod.find('OverRms') > 0:
            avgspectrum /= myrms
        elif meanSpectrumMethod.find('OverMad') > 0:
            avgspectrum /= mymad
        if useIAGetProfile:
            if mask == '':
                casalogPost("running ia.getprofile(mean above threshold: mask='%s>%f')" % (img, threshold))
                profile = myia.getprofile(axis=axis, function='mean', mask="'%s'>%f"%(img,threshold))
            else:
                if useThresholdWithMask:
                    profile = myia.getprofile(axis=axis, function='mean', mask="'%s'>0 && '%s'>%f"%(mask,img,threshold))
                else:
                    # we just keep the prior result for profile which did not use a threshold
                    pass
            avgspectrumAboveThreshold = profile['values']
            percentagePixelsNotMasked = sum(profile['npix'])*100. / np.prod(myia.shape())
        else:
            casalogPost("Using threshold above which to compute mean spectrum = %f" % (threshold), debug=True)
            pixels[np.where(pixels < threshold)] = 0.0
            casalogPost("Running avgOverCube")
            avgspectrumAboveThreshold, percentagePixelsNotMasked = avgOverCube(pixels, useAbsoluteValue, threshold, mask=maskdata, usermask=usermaskdata)

        if meanSpectrumMethod.find('OverRms') > 0:
            avgspectrumAboveThreshold /= myrms
        elif meanSpectrumMethod.find('OverMad') > 0:
            avgspectrumAboveThreshold /= mymad
        if useIAGetProfile:
            nansRemoved = removeZeros(avgspectrumAboveThreshold)
            nansReplaced = nansRemoved
            nanmin = 0
        else:
            if verbose: 
                print("Running removeNaNs (len(avgspectrumAboveThreshold)=%d)" % (len(avgspectrumAboveThreshold)))
            nansRemoved = removeNaNs(avgspectrumAboveThreshold)
            nansReplaced,nanmin = removeNaNs(avgspectrumAboveThreshold, replaceWithMin=True, 
                                             nanBufferChannels=nanBufferChannels)

    myia.close()
    if len(chanInfo) == 0:
        chanInfo = numberOfChannelsInCube(img, returnChannelWidth=True, returnFreqs=True) 
    nchan, firstFreq, lastFreq, channelWidth = chanInfo
    frequency = np.linspace(firstFreq, lastFreq, nchan)
    writeMeanSpectrum(meanSpectrumFile, frequency, avgspectrum, nansReplaced, threshold,
                      nchan, edgesUsed, nanmin, centralArcsec, mask, iteration)
    if (myrg is not None): myrg.done()
    return(avgspectrum, nansRemoved, nansReplaced, threshold, 
           edgesUsed, nchan, nanmin, percentagePixelsNotMasked)

def getMaxpixpos(img):
    """
    This function is called by findContinuum, but only in Cycle 4+5 heuristic.
    """
    myia = iatool()
    myia.open(img)
    maxpixpos = myia.statistics()['maxpos']
    myia.close()
    return maxpixpos

def submask(mask, region):
    """
    This function is called by meanSpectrum, but only in Cycle 4+5 heuristic.
    Takes a spectral mask array, and picks out a subcube defined by a 
    blc,trc-defined region
    region: dictionary containing keys 'blc' and 'trc'
    """
    mask = mask[region['blc'][0]:region['trc'][0]+1, region['blc'][1]:region['trc'][1]+1]
    return mask

def avgOverCube(pixels, useAbsoluteValue=False, threshold=None, median=False, 
                mask='', usermask='', useIAGetProfile=True):
    """
    This function was used by Cycle 4 and 5 pipeline from meanSpectrum(), 
    but will not be used in the Cycle 6 default heuristic of 
    'mom0mom8jointMask'.
    Computes the average spectrum across a multi-dimensional
    array read from an image cube, ignoring any NaN values.
    Inputs:
    pixels: a 3D array (with degenerate Stokes axis dropped)
    threshold: value in Jy
    median: if True, compute the median instead of the mean
    mask: use pixels inside this spatial mask (i.e. with value==1 or True) to compute 
          the average (the spectral mask is taken as the union of all channels)
          This is the mask located inside the image cube specified in findContinuum(img='')
    usermask: this array results from the mask specified in findContinuum(mask='') parameter
    If threshold is specified, then it only includes pixels
    with an intensity above that value.
    Returns:
    * average spectrum
    * percentage of pixels not masked
    Note: This function assumes that the spectral axis is the final axis.
        If it is not, there is no single setting of the axis parameter that 
        can make it work.
    """
    if (useAbsoluteValue):
        pixels = np.abs(pixels)
    if (len(mask) > 0):
        npixels = np.prod(np.shape(pixels))
        before = np.count_nonzero(np.isnan(pixels))
        pixels[np.where(mask==False)] = np.nan
        after = np.count_nonzero(np.isnan(pixels))
        maskfalse = after-before
        print("Ignoring %d/%d pixels (%.2f%%) where mask is False" % (maskfalse, npixels, 100.*maskfalse/npixels))
    if (len(usermask) > 0):
        npixels = np.prod(np.shape(pixels))
        before = np.count_nonzero(np.isnan(pixels))
        pixels[np.where(usermask==0)] = np.nan
        after = np.count_nonzero(np.isnan(pixels))
        maskfalse = after-before
        print("Ignoring %d/%d pixels (%.2f%%) where usermask is 0" % (maskfalse, npixels, 100.*maskfalse/npixels))
    if (len(mask) > 0 or len(usermask) > 0):
        nonnan = np.prod(np.shape(pixels)) - np.count_nonzero(np.isnan(pixels))
        percentageUsed = 100.*nonnan/npixels
        print("Using %d/%d pixels (%.2f%%)" % (nonnan, npixels, percentageUsed))
    else:
        percentageUsed = 100
    nchan = np.shape(pixels)[-1]
    # Check if each channel has an intensity contribution from at least one spatial pixel.
    for i in range(nchan):
        if (len(np.shape(pixels)) == 4):
            channel = pixels[:,:,0,i].flatten()
        else:
            channel = pixels[:,:,i].flatten()
        validChan = len(np.where(channel >= threshold)[0])
        if (validChan < 1):
            casalogPost("ch%4d: none of the %d pixels meet the threshold in this channel" % (i,len(channel)))
    # Compute the mean (or median) spectrum by averaging over one spatial dimension followed by the next,
    # replacing the pixels array with an array that is smaller by one dimension after the first averaging step.
    for i in range(len(np.shape(pixels))-1):
        if (median):
            pixels = nanmedian(pixels, axis=0)
        else:
            if (threshold is not None):
                idx = np.where(pixels < threshold)
                if len(idx[0]) > 0:
                    pixels[idx] = np.nan
            pixels = nanmean(pixels, axis=0)
    return(pixels, percentageUsed)

def propagateMaskToAllChannels(mask, axis=3):
    """
    This function is called by meanSpectrum.
    Takes a spectral mask array, and propagates every masked spatial pixel to 
    all spectral channels of the array.
    Returns: a 2D mask (of the same spatial dimension as the input, 
              but with only 1 channel)
    """
    casalogPost("Propagating image mask to all channels")
    startTime = timeUtilities.time()
    newmask = np.sum(mask,axis=axis)
    newmask[np.where(newmask>0)] = 1
    casalogPost("  elapsed time = %.1f sec" % (timeUtilities.time()-startTime))
    return(newmask)

def widthOfMaskArcsec(mask, maskInfo):
    """
    ++++ This function is not used by pipeline when meanSpectrumMethod='mom0mom8jointMask' or if mask=''
    Finds width of the smallest central square that circumscribes all masked 
    pixels.  Returns the value in arcsec.
    """
    print("Opening mask: ", mask)
    myia = iatool()
    myia.open(mask)
    pixels = myia.getregion(axes=[2,3])
    #  ia.getregion()           yields np.shape(pixels) = (1,138119016)
    #  ia.getregion(axes=[0,1]) yields np.shape(pixels) = (1,1,1,1918)
    #  ia.getregion(axes=[2,3]) yields np.shape(pixels) = (1960,1960,1,1)
    myia.close()
    idx = np.where(pixels==True)
    leftRadius = np.shape(pixels)[0]/2 - np.min(idx[0])
    rightRadius = np.max(idx[0]) - np.shape(pixels)[0]/2
    width = 2*np.max(np.abs([leftRadius,rightRadius]))
    topRadius = np.max(idx[1]) - np.shape(pixels)[1]/2
    bottomRadius = np.shape(pixels)[1]/2 - np.min(idx[1]) 
    height = 2*np.max(np.abs([topRadius,bottomRadius]))
    cdelt1 = maskInfo[3]
    width = np.abs(cdelt1)*(np.max([width,height])+1)
    return width

def checkForTriangularWavePattern(img, triangleFraction=0.83, pad=20):
    """
    +++++++ This function is not used for meanSpectrumMethod == 'mom0mom8jointMask'.
    Fit and remove linear slopes to each half of the spectrum, then comparse
    the MAD of the residual to the MAD of the original spectrum
    pad: fraction of total channels to ignore on each edge (e.g. 20: 1/20th)
    triangleFraction: MAD of dual-linfit residual must be less than this fraction*originalMAD
    Returns: 
    * Boolean: whether triangular pattern meets the threshold
    * value: either False (if slope test failed) or a float (the ratio of the MADs)
    """
    casalogPost('Checking for triangular wave pattern in the noise')
    chanlist, mad = computeMadSpectrum(img)
    nchan = len(chanlist)
    n0 = nchan/2 - nchan/pad
    n1 = nchan/2 + nchan/pad
    slope = 0
    intercept = np.mean(mad[nchan/pad:-nchan/pad])
    slope0, intercept0 = linfit(chanlist[nchan/pad:n0], mad[nchan/pad:n0], MAD(mad[nchan/pad:n0]))
    slope1, intercept1 = linfit(chanlist[n1:-nchan/pad], mad[n1:-nchan/pad], MAD(mad[n1:-nchan/pad]))
    casalogPost("checkForTriangularWavePattern: slope0=%+g, slope1=%+g, %s" % (slope0,slope1,os.path.basename(img)))
    slopeTest = slope0 > 0 and slope1 < 0 
    slopeDiff = abs(abs(slope1)-abs(slope0))
    slopeSum = (abs(slope1)+abs(slope0))
    similarSlopeThreshold = 0.70 # 0.40
    print("slopeSum = ", slopeSum)
    similarSlopes = slopeDiff/slopeSum < similarSlopeThreshold 
    # if the slope is sufficiently high, then it is probably a real feature, not a noise feature
    slopeSignPattern = slope0 > 0 and slope1 < 0
    largeSlopes = abs(slope0)>1e-5 or abs(slope1)>1e-5
    differentSlopeThreshold = 5
    differentSlopes = (abs(slope0/slope1) > differentSlopeThreshold) or (abs(slope0/slope1) < 1.0/differentSlopeThreshold)
    slopeTest = (slopeSignPattern and similarSlopes and not largeSlopes) or (differentSlopes and not largeSlopes)
    if slopeTest:
        residual = mad - (chanlist*slope + intercept)
        madOfResidualSingleLine = MAD(residual)  
        residual0 = mad[:nchan/2] - (chanlist[:nchan/2]*slope0 + intercept0)
        residual1 = mad[nchan/2:] - (chanlist[nchan/2:]*slope1 + intercept1)
        madOfResidual = MAD(list(residual0) + list(residual1))
        madRatio = madOfResidual/madOfResidualSingleLine
        casalogPost("checkForTriangularWavePattern: MAD_of_residual=%e, thresholdFraction=%.2f, ratio_of_MADs=%.3f, slopeDiff/slopeSum=%.2f, slope0/slope1=%.2f, signPattern=%s similarSlopes=%s largeSlopes=%s differentSlopes=%s" % (madOfResidual, triangleFraction, madRatio, slopeDiff/slopeSum,slope0/slope1,slopeSignPattern,similarSlopes,largeSlopes,differentSlopes))
        if (madRatio < triangleFraction):
            return True, madRatio
        else:
            return False, madRatio
    else:
        casalogPost("checkForTriangularWavePattern: slopeDiff/slopeSum=%.2f signPattern=%s similarSlopes=%s largeSlopes=%s  %s" % (slopeDiff/slopeSum,slopeSignPattern,similarSlopes,largeSlopes,os.path.basename(img)))
    return False, slopeTest
    
def computeMadSpectrum(img, box=''):
    """
    +++++++ This function is not used for meanSpectrumMethod == 'mom0mom8jointMask'.
            Only used by checkForTriangularWavePattern.
    Uses the imstat task to compute an array containing the MAD spectrum of a cube.
    """
    axis = findSpectralAxis(img)
    myia = iatool()
    myia.open(img)
    myshape = myia.shape()
    myia.close()
    axes = list(range(len(myshape)))
    axes.remove(axis)
    nchan = myshape[axis]
    chanlist = np.array(range(nchan))
    casalogPost("Computing MAD spectrum with imstat.")
    print("imstat", end=' ') 
    mydict = imstat(img, axes=axes, box=box, listit=imstatListit, verbose=imstatVerbose)
    return(chanlist, mydict['medabsdevmed'])

def isSingleContinuum(vis, spw='', source='', intent='OBSERVE_TARGET', 
                      verbose=False, mymsmd=None):
    """
    +++++++ This function is not used by pipeline, rather it is an expected input parameter.
    Checks whether an spw was defined as single continuum in the OT by
    looking at the transition name in the SOURCE table of a measurement set.
    vis: a single measurement set, a comma-delimited list, or a python list 
         (only the first one will be used)
    Optional parameters:
    spw: can be integer ID or string integer ID; if not specified, then it 
         will use the first science spw
    source: passed to transitions(); if not specified, it will use the first one
    intent: if source is blank then use first one with matching intent and spw
    mymsmd: an existing instance of the msmd tool
    """
    if vis=='': return False
    if type(vis) == list or type(vis) == np.ndarray:
        vis = vis[0]
    else:
        vis = vis.split(',')[0]
    if not os.path.exists(vis): return False
    needToClose = False
    if spw=='':
        if mymsmd is None:
            needToClose = True
            mymsmd = msmdtool()
            mymsmd.open(vis)
        spw = getScienceSpws(vis, returnString=False, mymsmd=mymsmd)[0]
    info = transitions(vis, spw, source, intent, verbose, mymsmd)
    if needToClose:
        mymsmd.close()
    if len(info) > 0:
        if info[0].find('Single_Continuum') >= 0:
            casalogPost("Identified spectral setup as Single_Continuum from transition name.")
            return True
    return False
    
def sanitizeNames(names, newchar='_'):
    """
    Replaces various special characters with specified character.
    +++++++ This function is not used by pipeline, because it is only used by transitions() which is only used by isSingleContinuum.
    -Todd Hunter
    """
    if (type(names) == str):
        names = names.split(',')
    newnames = []
    for name in names:
        newname = name
        for ch in [')','(','/']:
            newname = newname.replace(ch,newchar)
        newnames.append(newname)
    return newnames

def transitions(vis, spw, source='', intent='OBSERVE_TARGET', 
                verbose=True, mymsmd=None):
    """
    +++++++ This function is not used by pipeline, because it is only used by 
    isSingleContinuum() and the pipeline passes the Boolean singleContinuum.
    However, it is used by my regression, so it is important to pass the vis
    which has the virtual spw == real spw, which is normally the first spw
    observed.
    Returns the list of transitions for specified spw (and source).
    vis: measurement set
    spw: can be integer ID or string integer ID
    Optional parameters:
    source: can be integer ID or string name; if not specified, use the first one
    intent: if source is blank then use first one with matching intent and spw
    """
    if (not os.path.exists(vis)):
        print("Could not find measurement set")
        return
    needToClose = False
    if mymsmd is None:
        needToClose = True
        mymsmd = msmdtool()
        mymsmd.open(vis)
    spw = int(spw)
    if (spw >= mymsmd.nspw()):
        print("spw not in the dataset")
        if needToClose:
            mymsmd.close()
        return
    mytb = tbtool()
    mytb.open(vis+'/SOURCE')
    spws = mytb.getcol('SPECTRAL_WINDOW_ID')
    sourceIDs = mytb.getcol('SOURCE_ID')
    names = mytb.getcol('NAME')
    spw = int(spw)
    if (type(source) == str):
        if (source.isdigit()):
            source = int(source)
        elif (source == ''):
            # pick source
            fields1 = mymsmd.fieldsforintent(intent+'*')
            fields2 = mymsmd.fieldsforspw(spw)
            fields = np.intersect1d(fields1,fields2)
            print("spw=%d, intent=%s, fields1=%s, fields2=%s, Fields=%s" % (spw,intent,fields1,fields2,fields))
            source = mymsmd.namesforfields(fields[0])[0]
            if verbose:
                print("For spw %d, picked source: " % (spw), source)
    if isinstance(source, (str, np.bytes_)):
        sourcerows = np.where(names==source)[0]
        if (len(sourcerows) == 0):
            # look for characters ()/ and replace with underscore
            names = np.array(sanitizeNames(names))
            sourcerows = np.where(source==names)[0]
    else:
        sourcerows = np.where(sourceIDs==source)[0]
        
    spwrows = np.where(spws==spw)[0]
    row = np.intersect1d(spwrows, sourcerows)
    if (len(row) > 0):
        if (mytb.iscelldefined('TRANSITION',row[0])):
            transitions = mytb.getcell('TRANSITION',row[0])
        else:
            transitions = []
    else:
        transitions = []
    if (len(transitions) == 0):
        print("No transition value found for this source/spw (row=%s)." % row)
    mytb.close()
    if needToClose:
        mymsmd.close()
    return(transitions)

def getObservationStart(vis, obsid=-1, verbose=False):
    """
    Reads the start time of the observation from the OBSERVATION table (using tb tool)
    and reports it in MJD seconds.
    obsid: if -1, return the start time of the earliest obsID
    -Todd Hunter
    """
    if (os.path.exists(vis) == False):
        print("vis does not exist = %s" % (vis))
        return
    if (os.path.exists(vis+'/table.dat') == False):
        print("No table.dat.  This does not appear to be an ms.")
        print("Use au.getObservationStartDateFromASDM().")
        return
    mytb = tbtool()
    try:
        mytb.open(vis+'/OBSERVATION')
    except:
        print("ERROR: failed to open OBSERVATION table on file "+vis)
        return(3)
    time_range = mytb.getcol('TIME_RANGE')
    mytb.close()
    if verbose:  print("time_range: ", str(time_range))
    # the first index is whether it is starttime(0) or stoptime(1) 
    time_range = time_range[0]
    if verbose:  print("time_range[0]: ", str(time_range))
    if (obsid >= len(time_range)):
        print("Invalid obsid")
        return
    if obsid >= 0:
        time_range = time_range[obsid]
    elif (type(time_range) == np.ndarray):
        time_range = np.min(time_range)
    return(time_range)

def getObservatoryName(vis):
    """
    +++++++ This function is not used by pipeline, because it is only used by getScienceSpws.
    Returns the observatory name in the specified ms.
    """
    antTable = vis+'/OBSERVATION'
    try:
        mytb = tbtool()
        mytb.open(antTable)
        myName = mytb.getcell('TELESCOPE_NAME')
        mytb.close()
    except:
        casalogPost("Could not open OBSERVATION table to get the telescope name: %s" % (antTable))
        myName = ''
    return(myName)

def makeUvcontsub(files='*.dat', fitorder=1, useFrequency=False):
    """
    +++++++ This function is not used anywhere else in this file.
    Takes a list of output files from findContinuum and builds an spw selection
    for uvcontsub, then prints the resulting commands to the screen.
    files: a list of files, either a comma-delimited string, a python list, or a wildcard string
    fitorder: passed through to the uvcontsub
    useFrequency: if True, then produce selection string in frequency; otherwise use channels
      Note: frequencies in the findContinuum .dat file are topo, which is what uvcontsub wants.
    """
    if files.find('*') >= 0:
        resultFiles = sorted(glob.glob(files))
        uids = []
        for resultFile in resultFiles:
            uid = resultFile.split('.')[0]
        if len(np.unique(uids)) > 1:
            print("There are results for more than one OUS in this directory.  Be more specific.")
            return
    elif type(files) == str:
        resultFiles = sorted(files.split(','))
    else:
        resultFiles = sorted(files)
    freqRanges = {}
    spws = []
    for rf in resultFiles:
        spw = rf.split('spw')[1].split('.')[0]
        spws.append(spw)
        uid = rf.split('.')[0]
        field = rf[len(uid)+6:].split('_')[1]
        f = open(rf,'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            tokens = line.split()
            if line == lines[0]:
                channelRanges = tokens[0]
            if len(tokens) == 2:
                uid = tokens[0]
                if uid not in freqRanges:
                    freqRanges[uid] = {}
                if field not in freqRanges[uid]:
                    freqRanges[uid][field] = ''
                else:
                    freqRanges[uid][field] += ','
                if useFrequency:
                    freqRanges[uid][field] += '%s:%s' % (spw,tokens[1])
                else:
                    freqRanges[uid][field] += '%s:%s' % (spw,channelRanges)
    spws = ','.join(np.unique(spws))
    if freqRanges == {} and useFrequency:
        print("There are no frequency ranges in the *.dat files.  You need to run findContinuum with the 'vis' parameter specified.")
        return
    for uid in freqRanges:
        for field in freqRanges[uid]:
            print("uvcontsub('%s', field='%s', fitorder=%d, spw='%s', fitspw='%s')\n" % (uid, field, fitorder, spws, freqRanges[uid][field]))

def getcube(i, filename='cubelist.txt'):
    """
    ++++ This function is not used by pipeline.
    Translates a PDF page number to a cube name, for regression purposes,
    by reading the specified file.
    filename: a file containing 2-column lines like:  '1 mycube.residual'
    """
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        token = line.split()
        if len(token) == 2:
            if i == int(token[0]):
                cube = token[1]
    f.close()         
    casalogPost('Translated cube %d into %s' % (i, cube))
    return cube

def readViewerOutputFile(lines, debug=False):
    """
    ++++++ This function is not used by the pipeline.
    Reads an ASCII spectrum file produced by the CASA viewer or by tt.ispec.
    Returns: 4 items: 2 arrays and 2 strings:
    * array for x-axis, array for y-axis 
    * x-axis units, intensity units
    """
    x = []; y = []
    pixel = ''
    xunits = 'unknown'
    intensityUnits = 'unknown'
    for line in lines:
        if (line[0] == '#'): 
            if (line.find('pixel') > 0):
                pixel = line.split('[[')[1].split(']]')[0]
            elif (line.find('xLabel') > 0):
                xunits = line.split()[-1]
                if (debug):
                    print("Read xunits = ", xunits)
            elif (line.find('yLabel') > 0):
                tokens = line.split()
                if (len(tokens) == 2):
                    intensityUnits = tokens[1]
                else:
                    intensityUnits = tokens[1] + ' (' + tokens[2] + ')'
            continue
        tokens = line.split()
        if (len(tokens) < 2): 
            continue
        x.append(float(tokens[0]))
        y.append(float(tokens[1]))
    return(np.array(x), np.array(y), xunits, intensityUnits)

def readMeanSpectrumFITSFile(meanSpectrumFile, unit=0, nanBufferChannels=1):
    """
    ++++++ This function is not used by the pipeline.
    Reads a spectrum from a FITS table, such as one output by the spectralcube
    python module: https://spectral-cube.readthedocs.io/en/latest/index.html
    Returns: 8 items
    * average spectrum
    * average spectrum with the NaNs replaced with the minimum value
    * threshold used (currently hardcoded to 0.0)
    * edges used (currently hardcoded to 2)
    * number of channels
    * the minimum value
    * first frequency
    * last frequency
    """
    f = pyfits.open(meanSpectrumFile)
    tbheader = f[unit].header
    tbdata = f[unit].data
    nchan = len(tbdata)
    crpix = tbheader['CRPIX1']
    crval = tbheader['CRVAL1']
    cdelt = tbheader['CDELT1']
    units = tbheader['CUNIT1']
    if (units.lower() != ('hz')):
        print("Warning: frequency units are not Hz = ", units.lower())
        return
    firstFreq = crval - (crpix-1)*cdelt
    lastFreq = firstFreq + cdelt*(nchan-1)
    avgspectrum = tbdata
    edgesUsed = 2
    threshold = 0
#    print("Found %d channels" % (len(tbdata)), np.shape(tbdata))
    avgSpectrumNansReplaced,nanmin = removeNaNs(tbdata, replaceWithMin=True, 
                                     nanBufferChannels=nanBufferChannels)
    return(avgspectrum, avgSpectrumNansReplaced, threshold,
           edgesUsed, nchan, nanmin, firstFreq, lastFreq)

def readPreviousMeanSpectrum(meanSpectrumFile, verbose=False, invert=False):
    """
    ++++++ This function is not used by the pipeline.
    Read a previous calculated mean spectrum and its parameters,
    or an ASCII file created by the CASA viewer (or tt.ispec).
    Note: only the intensity column(s) are returned, not the 
       channel/frequency columns.
    This function will not typically be used by the pipeline, only manual users.
    Returns: 11 things:  
           avgspectrum, avgSpectrumNansReplaced, threshold,
           edgesUsed, nchan, nanmin, firstFreq, lastFreq, centralArcsec,
           mask, percentagePixelsNotMasked
    """
    f = open(meanSpectrumFile, 'r')
    lines  = f.readlines()
    f.close()
    if (len(lines) < 3):
        return None
    i = 0
    # Detect file type:
    if (lines[0].find('title: Spectral profile') > 0):
        # CASA viewer/au.ispec output file
        print("Reading CASA viewer output file")
        freq, intensity, freqUnits, intensityUnits = readViewerOutputFile(lines)
        if (freqUnits.lower().find('ghz')>=0):
            freq *= 1e9
        elif (freqUnits.lower().find('mhz')>=0):
            freq *= 1e6
        elif (freqUnits.lower().find('[hz')<0):
            print("Spectral axis of viewer output file must be in Hz, MHz or GHz, not %s." % freqUnits)
            return
        threshold = 0
        edgesUsed = 2
        return(intensity, intensity, threshold,
               edgesUsed, len(intensity), np.nanmin(intensity), freq[0], freq[-1], None, None, None)
    centralArcsec = ''
    mask = ''
    percentagePixelsNotMasked = -1
    while (lines[i][0] == '#'):
        if (lines[i].find('centralArcsec=') > 0):
            if (lines[i].find('=auto') > 0):
                centralArcsec = 'auto'
            elif (lines[i].find('=mom0mom8jointMask') > 0):
                centralArcsec = 'mom0mom8jointMask'
            else:
                centralArcsec = float(lines[i].split('centralArcsec=')[1].split()[0])
            token = lines[i].split()
            if (len(token) > 6):
                mask = token[6]
                if (len(token) > 7):
                    percentagePixelsNotMasked = int(token[7])
        i += 1
    token = lines[i].split()
    if len(token) == 4:
        a,b,c,d = token
        threshold = float(a)
        edgesUsed = int(b)
        nchan = int(c)
        nanmin = float(d)
    else:
        threshold = 0
        edgesUsed = 0
        nchan = 0
        nanmin = 0
    avgspectrum = []
    avgSpectrumNansReplaced = []
    freqs = []
    channels = []
    for line in lines[i+1:]:
        if (line[0] == '#'): continue
        tokens = line.split()
        if (len(tokens) == 2):
            # continue to support the old 2-column format
            freq,a = line.split()
            b = a
            nchan += 1
            freq = float(freq)
            if freq < 1e6:
                freq *= 1e6 # convert MHz to Hz
            freqs.append(freq)
        elif len(tokens) > 2:
            chan,freq,a,b = line.split()
            channels.append(int(chan))
            freqs.append(float(freq))
        else:
            print("len(tokens) = ", len(tokens))
        if invert:
            a = -float(a)
            b = -float(b)
        avgspectrum.append(float(a))
        avgSpectrumNansReplaced.append(float(b))
    avgspectrum = np.array(avgspectrum)    
    avgSpectrumNansReplaced = np.array(avgSpectrumNansReplaced)
    if invert:
        avgspectrum += abs(np.min(avgspectrum))
        avgSpectrumNansReplaced += abs(np.min(avgSpectrumNansReplaced))
        
    if (len(freqs) > 0):
        firstFreq = freqs[0]
        lastFreq = freqs[-1]
    else:
        firstFreq = 0
        lastFreq = 0
    casalogPost("Read previous mean spectrum with %d channels, (%d freqs: %f-%f)" % (len(avgspectrum),len(freqs),firstFreq,lastFreq),verbose)
    return(avgspectrum, avgSpectrumNansReplaced, threshold,
           edgesUsed, nchan, nanmin, firstFreq, lastFreq, centralArcsec,
           mask, percentagePixelsNotMasked)

def removeNaNs(a, replaceWithMin=False, verbose=False, nanBufferChannels=0, 
               replaceWithZero=False):
    """
    +++++++ This function is not used for meanSpectrumMethod == 'mom0mom8jointMask' by pipeline,
            But it is called by readMeanSpectrumFITSFile available to manual users.
    Remove or replace the nan values from an array.
    replaceWithMin: if True, then replace NaNs with np.nanmin of array
                    if False, then simply remove the NaNs
    replaceWithZero: if True, then replace NaNs with np.nanmin of array
                    if False, then simply remove the NaNs
    nanBufferChannels: only active if replaceWithMin=True
    Returns:
    * new array
    * if replaceWithMin=True, then also return the value used to replace NaNs
    """
    a = np.array(a) 
    if (len(a) < 1): return(a)
    startLength = len(a)
    if (replaceWithMin or replaceWithZero):
        idx = np.isnan(a)
        print("Found %d nan channels: %s" % (len(np.where(idx==True)[0]), np.where(idx==True)[0]))
        idx2 = np.isinf(a)
        print("Found %d inf channels" % (len(np.where(idx2==True)[0])))
        a_nanmin = np.nanmin(a)
        if (nanBufferChannels > 0):
            idxlist = splitListIntoHomogeneousLists(idx)
            idx = []
            for i,mylist in enumerate(idxlist):
                if (mylist[0]):
                    # if this is a sublist of NaN channels, then apply list as is
                    idx += mylist
                else:
                    # if this is a sublist of non-NaN channels, then flag
                    # an additional nanBufferChannels after the prior NaN 
                    # sublist, but only if there was a prior sublist
                    if i==0:
                        newSubString = []
                    else:
                        newSubString = nanBufferChannels*[True] 
                    if (i < len(idxlist)-1):
                        # we are not on the final sublist
                        newSubString += mylist[nanBufferChannels:-nanBufferChannels] + nanBufferChannels*[True]
                    else:
                        newSubString += mylist[-nanBufferChannels:]
                    # In case the channel block was less than 2*nanBufferChannels wide, then only insert up to its width
                    idx += newSubString[:len(mylist)]
            idx = np.where(np.array(idx) == True)
        if (verbose):
            print("Replaced %d NaNs" % (len(idx)))
            print("Replaced %d infs" % (len(idx2)))
        if (replaceWithMin):
            a[idx] = a_nanmin
            a[idx2] = a_nanmin
            return(a, a_nanmin)
        elif (replaceWithZero):
            a[idx] = 0
            a[idx2] = 0
            return(a, 0)
    else:
        a = a[np.where(np.isnan(a) == False)]
        if (verbose):
            print("Removed %d NaNs" % (startLength-len(a)))
        startLength = len(a)
        a = a[np.where(np.isinf(a) == False)]
        if (verbose):
            print("Removed %d infs" % (startLength-len(a)))
        return(a)

def splitListIntoHomogeneousLists(mylist):
    """
    This function is called only by removeNaNs, and hence is not used in Cycle 6-onward pipeline.
    Converts [1,1,1,2,2,3] into [[1,1,1],[2,2],[3]], etc.
    -Todd Hunter
    """
    mylists = []
    newlist = [mylist[0]]
    for i in range(1,len(mylist)):
        if (mylist[i-1] != mylist[i]):
            mylists.append(newlist)
            newlist = [mylist[i]]
        else:
            newlist.append(mylist[i])
    mylists.append(newlist)
    return(mylists)
    
def removeZeros(a):
    """
    +++++ This function is not used for meanSpectrumMethod='mom0mom8jointMask'.
    Takes an array, and replaces all appearances of 0.0 with the minimum value
    of all channels not equal to 0.0.  This is used to reduce the bias caused
    by the edge channels being 0.0 in the profiles returned by ia.getprofile.
    If the absolute value of the minimum is greater than the absolute value of
    the maximum, then set the channels to the maximum value.
    """
    idx = np.where(a == 0.0)
    idx2 = np.where(a != 0.0)
    a_nanmin = np.nanmin(a[idx2])
    a_nanmax = np.nanmax(a[idx2])
    if np.abs(a_nanmin) < np.abs(a_nanmax):
        # emission spectrum
        a[idx] = a_nanmin
    else:
        # absorption spectrum
        a[idx] = a_nanmax
    print("Replaced %d zeros with %f" % (len(idx[0]), a_nanmin))
    return a

def plotStatisticalSpectrumFromMask(cube, jointMask='', pbcube=None, 
                                    statistic='mean', normalizeByMAD=False, 
                                    png='', jointMaskForNormalize='', subimage=False):
    """
    Simple wrapper to call computeStatisticalSpectrumFromMask and plot it.
    jointMask: either a mask image, or an expression with < or > on an mask 
          image
    """
    if not os.path.exists(cube):
        print("Could not find cube")
        return
    statistics = statistic.split(',')
    jointMasks = jointMask.split(',')
    for jointMask in jointMasks:
        if jointMask != '' and jointMask.find('>') < 0 and jointMask.find('>') < 0:
            if not os.path.exists(jointMask):
                print("Could not find jointmask")
                return
    pl.clf()
    colors = ['r','k','b']
    for i,statistic in enumerate(statistics):
        if len(jointMasks) == 1:
            jointMask = jointMasks[0]
        else:
            jointMask = jointMasks[i]
        if jointMaskForNormalize == '':
            myJointMaskForNormalize = jointMask
        else:
            myJointMaskForNormalize = jointMaskForNormalize
        channels, frequency, intensity, normalized = computeStatisticalSpectrumFromMask(cube, jointMask, pbcube, statistic=statistic, normalizeByMAD=normalizeByMAD, jointMaskForNormalize=myJointMaskForNormalize, subimage=subimage)
        pl.plot(channels,intensity,'-',color=colors[i])
        pl.xlabel('Channel')
        pl.ylabel(statistic)
        print( "peak over MAD = ", np.max(intensity)/MAD(intensity))
    pl.title(','.join([os.path.basename(cube), os.path.basename(jointMask)]))
    if png == '':
        png = cube + '.' + jointMask + '.' + statistic + '.png'
    pl.savefig(png)
    print("Wrote ", png)
    pl.draw()
    
def pruneMask(mymask, psfcube=None, minbeamfrac=0.3, prunesize=6.0, nchan=1, 
              overwrite=True, pbmom=None):
    """
    available in CASA >= 5.4.0-X  (see CAS-11335)
    Removes any regions smaller than prunesize contiguous pixels.
    psfcube: if specified, then use minbeamfrac * pixelsPerBeam, otherwise use prunesize
    prunesize: value used if psfcube is None; use 6 because pipeline size mitigation could
        produce images that have only 3 pixels across the beam, meaning only ~7-8 pts/beam
    nchan: number of channels to process, will always be 1 in findContinuum's use case
    overwrite: if True, replaces mymask with pruned mask, putting original in mymask.prepruned
               if False, is simply creates: mymask.pruned, but only if it pruned anything
    If all regions are pruned, it checks if it is ACA7m (maxBaseline < 60m) and if so, tries
    again with minbeamfrac*0.5.
    """
    npruned = 0
    if not os.path.exists(mymask):
        print("Could not find mask image")
        return npruned
    casalogPost('Pruning the joint mask')
    if psfcube is not None:
        myinfo = getImageInfo(psfcube)
        # np.log(2) standard factor for solid angle was not in Cycles0..7
#        pixelsPerBeam = myinfo[0]*myinfo[1]*np.pi/(4.*np.log(2))/np.abs(myinfo[3]*myinfo[4])
        pixelsPerBeam = myinfo[0]*myinfo[1]*np.pi/4./np.abs(myinfo[3]*myinfo[4])
        maxBaseline = myinfo[11] # in meters
        prunesize = np.max([4,int(pixelsPerBeam*minbeamfrac)])
        casalogPost('Beam = %.3fx%.3f, pixels per beam = %.2f, using prunesize = %.2f' % (myinfo[0],myinfo[1],pixelsPerBeam, prunesize))
    try:
        maskhandler = synthesismaskhandler()
    except:
        casalogPost("This casa does not contain the tool synthesismaskhandler(), so the joint mask cannot be pruned.")
        return npruned
    chanflag = np.zeros(nchan, dtype=bool)
    mymask = mymask.rstrip('/')
    try:
        mydict = maskhandler.pruneregions(mymask, prunesize, chanflag)
    except:
        casalogPost("pruneregions failed, skipping the prune step")
        return npruned
    npruned = mydict['N_reg_pruned'][0]
    newmask = mymask + '.pruned'
    if npruned > 0:
        casalogPost('Pruned %d regions.' % (npruned))
        remainingPixels = countPixelsAboveZero(newmask, pbmom)
        casalogPost('Remaining pixels = %s.' % (str(remainingPixels)))
        if remainingPixels < 1 and psfcube is not None:
            # Everything got pruned away, so check if ACA and reduce minbeamfrac by factor of 2
            casalogPost('All regions were pruned.')
            if maxBaseline <= ACA_MAX_BASELINE:
                newprunesize = np.max([4,int(pixelsPerBeam*minbeamfrac*0.5)])
                casalogPost('New prunesize = %d.' % (npruned))
                if prunesize != newprunesize:
                    casalogPost('All regions pruned and this looks like a 7m image (maxBaseline ~ %.0fm), so setting new prunesize=%d' % (maxBaseline,newprunesize))
                    shutil.rmtree(newmask)
                    mydict = maskhandler.pruneregions(mymask, newprunesize, chanflag)
                    npruned = mydict['N_reg_pruned'][0]
    maskhandler.done()
    if overwrite:
        if npruned > 0:
            removeIfNecessary(mymask+'.prepruned')
            os.rename(mymask, mymask+'.prepruned')
            os.rename(newmask, mymask)
            casalogPost("Updated joint mask after pruning %d region(s)" % (npruned))
        else:
            # Remove newly-created file because it is identical to original
            shutil.rmtree(newmask)
    else:
        print("Pruned %d regions" % (npruned))
    return npruned

def sortRangesByMax(cube, selection, box=''):
    """
    Runs imstat on each individual channel range of a selection string, sorts the
    result by 'max' and returns a sorted list of tuples: ['maxpos', 'max']
    """
    ranges = selection.split(';')
    statsval = []
    for chans in ranges:
        statsval.append(imstat(cube, chans=chans, box=box))
    sortedStats = sorted(statsval, key=lambda x: x['max'], reverse=True)
    result = []
    for stat in sortedStats:
        result.append([stat['maxpos'],stat['max']])
        print('%s %.4f' % (' '.join(['%4d'%i for i in stat['maxpos']]), stat['max']))
    return result

def plotSigmaCorrectionFactor(percentile=19):
    """
    Makes a plot of sigmaCorrectionFactor for the principle options
    for number of channels in an ALMA cube
    """
    values = []
    nchans = [64,128,240,356,480,512,960,1024,1920,2048,3840,4096,7680,8192]
    for nchan in nchans:
        values.append(sigmaCorrectionFactor('min', nchan, percentile))
    pl.clf()
    pl.plot(nchans, values, 'ko')
    x = range(60,nchans[-1])
    y = []
    for i in x:
        y.append(2.8*(i/128.)**0.08 * (1.9)**-0.25)
    size = 16
    pl.plot(x,y,'k-')
    pl.xlim([0,8300])
    pl.xlabel('Channels in spw',size=size)
    pl.ylabel('Sigma correction factor',size=size)
    pl.draw()
    pl.savefig('sigmaCorrectionFactor.png')

def countMaskedPixels(img):
    """
    The mask internal to an image will have a value of False where masked, 
    and True where not masked.
    To count non-zero pixels in a clean 1/0 mask, use au.nonzeroPixels.
    axes: set to [0,1,2] to get a per-channel count (only works for useImstat)
    -Todd Hunter
    """
    myia = iatool()
    myia.open(img)
    maskdata = myia.getregion(getmask=True)
    myia.close()
    idx = np.where(maskdata==0)[0]
    maskedPixels = len(idx)
    pixels = np.prod(np.shape(maskdata))
    print("%d/%d pixels (%f%%) are masked" % (maskedPixels,pixels,100.*maskedPixels/pixels))
    return maskedPixels
    
