#!python
#cython: boundscheck=False
#cython: wraparound=False

'''
Created on Aug 5, 2013

This file mostly contains access utility for BWTs that are already created on disk.

@author: holtjma
'''

import bisect
import gc
import gzip
import heapq
import math
import numpy as np
cimport numpy as np
import os
import pickle
import pysam#@UnresolvedImport
import shutil
import sys
import time

#from MUSCython import MSBWTGenCython as MSBWTGen
import MSBWTGenCython as MSBWTGen

from cython.operator cimport preincrement as inc

#flags for samtools
REVERSE_COMPLEMENTED_FLAG = 1 << 4#0x10
FIRST_SEGMENT_FLAG = 1 << 6#0x40
#SECOND_SEGMENT_FLAG = 1 << 7#0x80

cdef class BasicBWT(object):
    '''
    This class is the root class for ANY msbwt created by this code regardless of it being compressed or no.
    Shared Functions:
    __init__
    constructIndexing
    countOccurrencesOfSeq
    findIndicesOfStr
    getSequenceDollarID
    recoverString
    loadMsbwt
    constructTotalCounts
    constructFMIndex
    getCharAtIndex
    getBWTRange
    getOccurrenceOfCharAtIndex
    getFullFMAtIndex
    '''
    cdef np.ndarray numToChar
    cdef unsigned char [:] numToChar_view
    cdef np.ndarray charToNum
    cdef unsigned char [:] charToNum_view
    cdef unsigned int vcLen
    cdef unsigned int cacheDepth
    
    cdef char * dirName
    cdef np.ndarray bwt
    cdef np.uint8_t [:] bwt_view
    
    cdef unsigned long totalSize
    cdef np.ndarray totalCounts
    cdef np.uint64_t [:] totalCounts_view
    
    cdef np.ndarray startIndex
    cdef np.uint64_t [:] startIndex_view
    cdef np.ndarray endIndex
    cdef np.uint64_t [:] endIndex_view
    
    cdef dict searchCache
    cdef unsigned long bitPower
    cdef unsigned long binSize
    cdef np.ndarray partialFM
    cdef np.uint64_t[:, :] partialFM_view
    
    cdef unsigned long iterIndex
    cdef unsigned long iterCount
    cdef unsigned long iterPower
    cdef np.uint8_t iterCurrChar
    cdef np.uint8_t iterCurrCount
    cdef unsigned long fileSize
    
    def __init__(BasicBWT self):
        '''
        Constructor
        Nothing special, use this for all at the start
        '''
        cdef unsigned long i
        
        #valid characters are hard-coded for now
        self.numToChar = np.array([ord(c) for c in sorted(['$', 'A', 'C', 'G', 'N', 'T'])], dtype='<u1')
        self.numToChar_view = self.numToChar
        self.vcLen = len(self.numToChar)
        
        #construct a reverse map from a character to a number
        self.charToNum = np.zeros(dtype='<u1', shape=(256,))
        self.charToNum_view = self.charToNum
        for i in range(0, self.vcLen):
            self.charToNum_view[self.numToChar_view[i]] = i
        
        #this is purely for querying and determines how big our cache will be to shorten query times
        #TODO: experiment with this number
        self.cacheDepth = 6
    
    cdef void constructIndexing(BasicBWT self):
        '''
        This helper function calculates the start and end index for each character in the BWT.  Basically, the information
        generated here is for quickly finding offsets.  This is run AFTER self.constructTotalCounts(...)
        '''
        #mark starts and ends of key elements
        self.startIndex = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        self.startIndex_view = self.startIndex
        self.endIndex = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        self.endIndex_view = self.endIndex
        
        cdef unsigned long pos = 0
        cdef unsigned long i
        with nogil:
            for i in range(0, self.vcLen):
                self.startIndex_view[i] = pos
                pos += self.totalCounts_view[i]
                self.endIndex_view[i] = pos
    
    def getTotalSize(BasicBWT self):
        return self.totalSize
    
    def getSymbolCount(BasicBWT self, unsigned int symbol):
        '''
        @param symbol - this is an integer from [0, 6)
        '''
        ret = int(self.totalCounts_view[symbol])
        return ret
    
    def countOccurrencesOfSeq(BasicBWT self, bytes seq, givenRange=None):
        '''
        This function counts the number of occurrences of the given sequence
        @param seq - the sequence to search for
        @param givenRange - the range to start from (if a partial search has already been run), default=whole range
        @return - an integer count of the number of times seq occurred in this BWT
        '''
        cdef unsigned long l, h
        cdef long x
        cdef unsigned long s
        cdef unsigned int c
        
        if givenRange == None:
            #initialize our search to the whole BWT
            l = 0
            h = self.totalSize
        else:
            l = givenRange[0]
            h = givenRange[1]
        s = len(seq)
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        
        #with nogil:
        #for x in range(0, s):
        for x in range(s-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            l = self.getOccurrenceOfCharAtIndex(c, l)
            h = self.getOccurrenceOfCharAtIndex(c, h)
            
            #early exit for counts
            if l == h:
                break
        
        #return the difference
        return h - l
    
    def findIndicesOfStr(BasicBWT self, bytes seq, givenRange=None):
        '''
        This function will search for a string and find the location of that string OR the last index less than it. It also
        will start its search within a given range instead of the whole structure
        @param seq - the sequence to search for
        @param givenRange - the range to search for, whole range by default
        @return - a python range representing the start and end of the sequence in the bwt
        '''
        cdef unsigned long l, h
        cdef unsigned long s
        cdef long x
        cdef unsigned int c
        
        #initialize our search to the whole BWT
        if givenRange == None:
            #initialize our search to the whole BWT
            l = 0
            h = self.totalSize
        else:
            l = givenRange[0]
            h = givenRange[1]
        s = len(seq)
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        
        #with nogil:
        for x in range(s-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            l = self.getOccurrenceOfCharAtIndex(c, l)
            h = self.getOccurrenceOfCharAtIndex(c, h)
        
        #return the difference
        return (l, h)
    
    def findIndicesOfRegex(BasicBWT self, bytes seq, givenRange=None):
        '''
        This function will search for a string and find the location of that string OR the last index less than it. It also
        will start its search within a given range instead of the whole structure
        @param seq - the sequence to search for with valid symbols [$, A, C, G, N, T, *, ?]
            $, A, C, G, N, T - exact match of specific symbol
            * - matches 0 or more of any non-$ symbols (may be different symbols)
            ? - matches exactly one of any non-$ symbol
        @param givenRange - the range to search for, whole range by default
        @return - a python list of ranges representing the start and end of the sequence in the bwt
        '''
        cdef list ret = []
        
        cdef unsigned long l, h
        cdef unsigned long lc, hc
        cdef unsigned long s
        cdef long x
        cdef unsigned int c
        
        #initialize our search to the whole BWT
        if givenRange == None:
            #initialize our search to the whole BWT
            l = 0
            h = self.totalSize
        else:
            l = givenRange[0]
            h = givenRange[1]
        s = len(seq)
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        cdef bint recursed = False
        
        for x in range(s-1, -1, -1):
            if seq_view[x] == ord('*'):
                #handle the asterisk
                recursed = True
                
                if x > 0:
                    ret += self.findIndicesOfRegex(seq[0:x], (l, h))
                
                #skip that end in '$'
                for c in range(1, self.vcLen):
                    lc = self.getOccurrenceOfCharAtIndex(c, l)
                    hc = self.getOccurrenceOfCharAtIndex(c, h)
                    if hc > lc:
                        ret += self.findIndicesOfRegex(seq[0:x+1], (lc, hc))
                    
            elif seq_view[x] == ord('?'):
                #handle the single character match
                recursed = True
                
                #don't allow '$' as a ? symbol
                for c in range(1, self.vcLen):
                    lc = self.getOccurrenceOfCharAtIndex(c, l)
                    hc = self.getOccurrenceOfCharAtIndex(c, h)
                    if hc > lc:
                        ret += self.findIndicesOfRegex(seq[0:x], (lc, hc))
                
            else:
                #get the character from the sequence, then search at both high and low
                c = self.charToNum_view[seq_view[x]]
                l = self.getOccurrenceOfCharAtIndex(c, l)
                h = self.getOccurrenceOfCharAtIndex(c, h)
        
        #return the difference
        cdef list finalRet = []
        cdef unsigned long currStart, currEnd, nextStart, nextEnd
        
        if (not recursed) and (h-l > 0):
            #normal search, no variable symbols like * or ?
            finalRet += [(l, h)]
        elif recursed and len(ret) > 1:
            #we had to recurse, may need to condense groups that overlap
            ret.sort()
            
            currStart = ret[0][0]
            currEnd = ret[0][1]
            
            for x in range(0, len(ret)):
                nextStart = ret[x][0]
                nextEnd = ret[x][1]
                if nextStart <= currEnd:
                    if nextEnd <= currEnd:
                        #this range is totally enclosed by the current range, do nothing
                        pass
                    else:
                        #this range overlaps, so we need to extend our current end
                        currEnd = nextEnd
                else:
                    #this range starts past our current end, so add the current end and then set the new currs
                    finalRet.append((currStart, currEnd))
                    currStart = nextStart
                    currEnd = nextEnd
            
            #add the range that still remains
            finalRet.append((currStart, currEnd))
                    
        else:
            finalRet = ret
        
        return finalRet
    
    cdef unsigned int getCharAtIndex(BasicBWT self, unsigned long index) nogil:
        '''
        dummy function, shouldn't be called
        '''
        cdef unsigned int ret = 0
        return ret
    
    cpdef unsigned long getOccurrenceOfCharAtIndex(BasicBWT self, unsigned int sym, unsigned long index):# nogil:
        '''
        dummy function, shouldn't be called
        '''
        cdef unsigned long ret = 0
        return ret
    
    def iterInit(BasicBWT self):
        '''
        this function must be called to reset the iterator to the beginning, used for both normal and
        compressed data structures since it's so simple
        '''
        self.iterIndex = 0
        self.iterCount = 0
        self.iterPower = 0
        self.fileSize = self.bwt.shape[0]
        self.iterCurrChar = 255
        self.iterCurrCount = 0
        return self
    
    def iterNext(BasicBWT self):
        return self.iterNext_cython()
    
    cdef np.uint8_t iterNext_cython(BasicBWT self) nogil:
        '''
        dummy function, override in all subclasses
        '''
        return 255
    
    def getSequenceDollarID(BasicBWT self, unsigned long strIndex, bint returnOffset=False):
        '''
        This will take a given index and work backwards until it encounters a '$' indicating which dollar ID is
        associated with this read
        @param strIndex - the index of the character to start with
        @return - an integer indicating the dollar ID of the string the given character belongs to
        '''
        #figure out the first hop backwards
        cdef unsigned long currIndex = strIndex
        cdef unsigned int prevChar
        cdef unsigned long i
        
        prevChar = self.getCharAtIndex(currIndex)
        currIndex = self.getOccurrenceOfCharAtIndex(prevChar, currIndex)
        i = 0
    
        #while we haven't looped back to the start
        while prevChar != 0:
            #figure out where to go from here
            prevChar = self.getCharAtIndex(currIndex)
            currIndex = self.getOccurrenceOfCharAtIndex(prevChar, currIndex)
            inc(i)
        
        if returnOffset:
            return (currIndex, i)
        else:
            return currIndex
    
    def recoverString(BasicBWT self, unsigned long strIndex, bint withIndex=False):
        '''
        This will return the string that starts at the given index
        @param strIndex - the index of the string we want to recover
        @return - string that we found starting at the specified '$' index
        '''
        cdef list retNums = []
        cdef list indices = []
        
        #figure out the first hop backwards
        cdef unsigned int prevChar = self.getCharAtIndex(strIndex)
        cdef unsigned long currIndex = self.getOccurrenceOfCharAtIndex(prevChar, strIndex)
        
        #while we haven't looped back to the start
        while currIndex != strIndex:
            #update the string
            retNums.append(prevChar)
            if withIndex:
                indices.append(currIndex)
        
            #figure out where to go from here
            prevChar = self.getCharAtIndex(currIndex)
            currIndex = self.getOccurrenceOfCharAtIndex(prevChar, currIndex)
        
        for i in xrange(0, self.vcLen):
            if strIndex < self.endIndex[i]:
                retNums.append(i)
                break
                
        if withIndex:
            indices.append(strIndex)
        
        #reverse the numbers, convert to characters, and join them in to a single sequence
        #ret = ''.join([chr(x) for x in self.numToChar[retNums[::-1]]])
        
        #build up all the converted numbers in their encoding
        cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] retConvert = np.zeros(dtype='<u1', shape=(len(retNums, )))
        cdef np.uint8_t [:] retConvert_arr = retConvert
        cdef unsigned long retVal
        cdef unsigned long x
        cdef unsigned long retNumPos = len(retNums)-1
        for x in range(0, len(retNums)):
            retVal = retNums[retNumPos]
            retNumPos -= 1
            retConvert_arr[x] = self.numToChar_view[retVal]
        
        #have to make this view to do a bytes conversion
        cdef char * retConvert_view = <char *>&retConvert_arr[0]
        cdef bytes ret = retConvert_view[0:len(retNums)]
        
        #return what we found
        if withIndex:
            return (ret, indices[::-1])
        else:
            return ret

cdef class MultiStringBWT(BasicBWT):
    '''
    This class is a BWT capable of hosting multiple strings inside one structure.  Basically, this would allow you to
    search for a given string across several strings simultaneously.  Note: this class is for the non-compressed version,
    for general purposes use the function loadBWT(...) which automatically detects whether this class or CompressedMSBWT 
    is correct
    '''
    def loadMsbwt(MultiStringBWT self, char * dirName, logger):
        '''
        This functions loads a BWT file and constructs total counts, indexes start positions, and constructs an FM index
        on disk if it doesn't already exist
        @param dirName - the filename to load
        '''
        #open the file with our BWT in it
        self.dirName = dirName
        self.bwt = np.load(self.dirName+'/msbwt.npy', 'r+')
        self.bwt_view = self.bwt
        
        #build auxiliary structures
        self.constructTotalCounts(logger)
        self.constructIndexing()
        self.constructFMIndex(logger)
    
    def constructTotalCounts(MultiStringBWT self, logger):
        '''
        This function constructs the total count for each valid character in the array or loads them if they already exist.
        These will always be stored in '<DIR>/totalCounts.p', a pickled file
        '''
        cdef unsigned long i
        self.totalSize = self.bwt.shape[0]
        abtFN = self.dirName+'/totalCounts.npy'
        if os.path.exists(abtFN):
            self.totalCounts = np.load(abtFN, 'r+')
            self.totalCounts_view = self.totalCounts
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % abtFN)
            
            self.totalCounts = np.zeros(dtype='<u8', shape=(self.vcLen, ))
            self.totalCounts_view = self.totalCounts
            with nogil:
                for i in range(0, self.totalSize):
                    inc(self.totalCounts_view[self.bwt_view[i]])
            np.save(abtFN, self.totalCounts)
    
    def constructFMIndex(MultiStringBWT self, logger):
        '''
        This function iterates through the BWT and counts the letters as it goes to create the FM index.  For example, the string 'ACC$' would have BWT
        'C$CA'.  The FM index would iterate over this and count the occurence of the letter it found so you'd end up with this:
        BWT    FM-index
        C    0    0    0
        $    0    0    1
        C    1    0    1
        A    1    0    2
             1    1    2
        This is necessary for finding the occurrence of a letter using the getOccurrenceOfCharAtIndex(...) function.
        In reality, this function creates a sampled FM-index so only one index every 2048 bases is filled in.
        This file is always stored in '<DIR>/fmIndex.npy'
        '''
        #sampling method
        self.searchCache = {}
        self.bitPower = 11
        self.binSize = 2**self.bitPower
        fmIndexFN = self.dirName+'/fmIndex.npy'
        
        cdef np.ndarray[np.uint64_t, ndim=1] counts
        cdef np.uint64_t [:] counts_view
        cdef unsigned long i, j
        cdef unsigned long samplingSize
        
        if os.path.exists(fmIndexFN):
            self.partialFM = np.load(fmIndexFN, 'r+')
            self.partialFM_view = self.partialFM
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % fmIndexFN)
            
            #pre-allocate space
            samplingSize = int(math.ceil(float(self.totalSize+1)/self.binSize))
            #self.partialFM = np.lib.format.open_memmap(fmIndexFN, 'w+', '<u8', (self.bwt.shape[0]/self.binSize+1, self.vcLen))
            self.partialFM = np.lib.format.open_memmap(fmIndexFN, 'w+', '<u8', (samplingSize, self.vcLen))
            self.partialFM_view = self.partialFM
            
            #now perform each count and store it to disk
            counts = np.zeros(dtype='<u8', shape=(self.vcLen,))
            counts_view = counts
            counts[:] = self.startIndex
            self.partialFM[0] = self.startIndex
            with nogil:
                for i in range(1, self.partialFM.shape[0]):
                    for j in range(self.binSize*(i-1), self.binSize*i):
                        inc(counts_view[self.bwt_view[j]])
                        
                    for j in range(0, self.vcLen):
                        self.partialFM_view[i][j] = counts_view[j]
            
    cdef unsigned int getCharAtIndex(MultiStringBWT self, unsigned long index) nogil:
        '''
        This function is only necessary for other functions which perform searches generically without knowing if the 
        underlying structure is compressed or not
        @param index - the index to retrieve the character from
        '''
        return self.bwt_view[index]
    
    def getBWTRange(MultiStringBWT self, unsigned long start, unsigned long end):
        '''
        This function is only necessary for other functions which perform searches generically without knowing if the 
        underlying structure is compressed or not
        @param start - the beginning of the range to retrieve
        @param end - the end of the range in normal python notation (bwt[end] is not part of the return)
        '''
        return self.bwt[start:end]
        
    cpdef unsigned long getOccurrenceOfCharAtIndex(MultiStringBWT self, unsigned int sym, unsigned long index):# nogil:
        '''
        This functions gets the FM-index value of a character at the specified position
        @param sym - the character to find the occurrence level
        @param index - the index we want to find the occurrence level at
        @return - the number of occurrences of char before the specified index
        '''
        #sampling method
        #get the bin we occupy
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long ret = self.partialFM_view[binID][sym]
        cdef unsigned long start = binID << self.bitPower
        cdef unsigned long i
        
        for i in range(start, index):
            if self.bwt_view[i] == sym:
                inc(ret)
        
        return ret
        
    def getFullFMAtIndex(MultiStringBWT self, unsigned long index):
        '''
        This function creates a complete FM-index for a specific position in the BWT.  Example using the above example:
        BWT    Full FM-index
                 $ A C G T
        C        0 1 2 4 4
        $        0 1 3 4 4
        C        1 1 3 4 4
        A        1 1 4 4 4
                 1 2 4 4 4
        @return - the above information in the form of an array that already incorporates the offset value into the counts
        '''
        
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.empty(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] ret_view = ret
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long bwtInd = binID * self.binSize
        cdef unsigned long i
        
        for i in range(0, self.vcLen):
            ret_view[i] = self.partialFM_view[binID][i]
        
        for i in range(bwtInd, index):
            inc(ret_view[self.bwt_view[i]])
        
        return ret
        
    cdef np.uint8_t iterNext_cython(MultiStringBWT self) nogil:
        '''
        returns each character one at a time
        '''
        cdef np.uint8_t ret 
        
        if self.iterIndex < self.fileSize:
            ret = self.bwt_view[self.iterIndex]
            inc(self.iterIndex)
        else:
            ret = 255
        
        return ret
    
cdef class CompressedMSBWT(BasicBWT):
    '''
    This structure inherits from the BasicBWT and includes several functions with identical functionality to the MultiStringBWT
    class.  However, the implementations are different as this class represents a version of the BWT that is stored in a 
    compressed format.  Generally speaking, this class is slower due to partial decompressions and more complicated routines.
    For understanding the compression, refer to MSBWTGen.compressBWT(...).
    '''
    cdef unsigned long letterBits
    cdef unsigned long numberBits
    cdef unsigned long numPower
    cdef np.uint8_t mask
    
    cdef np.ndarray refFM
    cdef np.uint64_t [:] refFM_view
    cdef unsigned long offsetSum
    
    def loadMsbwt(CompressedMSBWT self, char * dirName, logger):
        '''
        This functions loads a BWT file and constructs total counts, indexes start positions, and constructs an FM index in memory
        @param dirName - the directory to load, inside should be '<DIR>/comp_msbwt.npy' or it will fail
        '''
        #open the file with our BWT in it
        self.dirName = dirName
        self.bwt = np.load(self.dirName+'/comp_msbwt.npy', 'r+')
        self.bwt_view = self.bwt
        
        #build auxiliary structures
        self.constructTotalCounts(logger)
        self.constructIndexing()
        self.constructFMIndex(logger)
    
    def constructTotalCounts(CompressedMSBWT self, logger):
        '''
        This function constructs the total count for each valid character in the array and stores it under '<DIR>/totalCounts.p'
        since these values are independent of compression
        '''
        cdef unsigned long i
        cdef unsigned long numBytes
        cdef np.uint8_t currentChar
        cdef np.uint8_t prevChar
        cdef unsigned long currentCount
        cdef unsigned long powerMultiple
        
        self.letterBits = 3
        self.numberBits = 8-self.letterBits
        self.numPower = 2**self.numberBits
        self.mask = 255 >> self.numberBits
        
        abtFN = self.dirName+'/totalCounts.npy'
        if os.path.exists(abtFN):
            self.totalCounts = np.load(abtFN, 'r+')
            self.totalCounts_view = self.totalCounts
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % abtFN)
            
            self.totalCounts = np.zeros(dtype='<u8', shape=(self.vcLen, ))
            self.totalCounts_view = self.totalCounts
            numBytes = self.bwt.shape[0]
            prevChar = 255
            powerMultiple = 1
            
            with nogil:
                for i in range(0, numBytes):
                    currentChar = self.bwt_view[i] & self.mask
                    if currentChar == prevChar:
                        powerMultiple *= self.numPower
                    else:
                        powerMultiple = 1
                    prevChar = currentChar
                    
                    currentCount = (self.bwt_view[i] >> self.letterBits) * powerMultiple
                    self.totalCounts_view[currentChar] += currentCount
            
            np.save(abtFN, self.totalCounts)
        
        self.totalSize = int(np.sum(self.totalCounts))
    
    def constructFMIndex(CompressedMSBWT self, logger):
        '''
        This function iterates through the BWT and counts the letters as it goes to create the FM index.  For example, the string 'ACC$' would have BWT
        'C$CA'.  The FM index would iterate over this and count the occurence of the letter it found so you'd end up with this:
        BWT    FM-index
        C    0    0    0
        $    0    0    1
        C    1    0    1
        A    1    0    2
             1    1    2
        This is necessary for finding the occurrence of a letter using the getOccurrenceOfCharAtIndex(...) function.
        In reality, this function creates a sampled FM-index more complicated than the uncompressed counter-part.  This is 
        because the 2048 size bins don't fall evenly all the time.  A second data structure is used to tell you where to start
        a particular FM-index count.  The two files necessary are '<DIR>/comp_fmIndex.npy' and '<DIR>/comp_refIndex.npy'
        '''
        #sampling method
        self.searchCache = {}
        self.bitPower = 11
        self.binSize = 2**self.bitPower
        
        cdef unsigned long i, j
        cdef unsigned long binID
        cdef unsigned long totalCharCount
        cdef unsigned long bwtIndex
        cdef np.uint8_t currentChar
        cdef np.uint8_t prevChar
        cdef unsigned long prevStart
        cdef unsigned long powerMultiple
        cdef unsigned long binEnd
        
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] countsSoFar
        cdef np.uint64_t [:] countsSoFar_view
        
        fmIndexFN = self.dirName+'/comp_fmIndex.npy'
        fmRefFN = self.dirName+'/comp_refIndex.npy'
        
        if os.path.exists(fmIndexFN) and os.path.exists(fmRefFN):
            #both exist, just memmap them
            self.partialFM = np.load(fmIndexFN, 'r+')
            self.partialFM_view = self.partialFM
            self.refFM = np.load(fmRefFN, 'r+')
            self.refFM_view = self.refFM
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % fmIndexFN)
            
            #pre-allocate space
            samplingSize = int(math.ceil(float(self.totalSize+1)/self.binSize))
            self.partialFM = np.lib.format.open_memmap(fmIndexFN, 'w+', '<u8', (samplingSize, self.vcLen))
            self.partialFM_view = self.partialFM
            self.refFM = np.lib.format.open_memmap(fmRefFN, 'w+', '<u8', (samplingSize,))
            self.refFM_view = self.refFM
            
            numBytes = self.bwt.shape[0]
            prevChar = 0
            totalCharCount = 0
            powerMultiple = 1
            binEnd = 0
            binID = 0
            bwtIndex = 0
            prevStart = 0
            
            countsSoFar = np.cumsum(self.totalCounts)-self.totalCounts
            countsSoFar_view = countsSoFar
            
            with nogil:
                #go through each byte, we'll determine when to save inside the loop
                for i in range(0, numBytes):
                    currentChar = self.bwt_view[i] & self.mask
                    if currentChar == prevChar:
                        totalCharCount += (self.bwt_view[i] >> self.letterBits) * powerMultiple
                        powerMultiple *= self.numPower
                    else:
                        #first save this count
                        while bwtIndex + totalCharCount >= binEnd:
                            self.refFM_view[binID] = prevStart
                            for j in range(0, self.vcLen):
                                self.partialFM_view[binID][j] = countsSoFar_view[j]
                            binEnd += self.binSize
                            inc(binID)
                        
                        #add the previous
                        countsSoFar_view[prevChar] += totalCharCount
                        bwtIndex += totalCharCount
                        
                        prevChar = currentChar
                        prevStart = i
                        totalCharCount = (self.bwt_view[i] >> self.letterBits)
                        powerMultiple = self.numPower
                
                while bwtIndex + totalCharCount >= binEnd:
                    self.refFM_view[binID] = prevStart
                    for j in range(0, self.vcLen):
                        self.partialFM_view[binID][j] = countsSoFar_view[j]
                    binEnd += self.binSize
                    inc(binID)
            
        #we'll use this later when we do lookups
        self.offsetSum = np.sum(self.partialFM[0])
        
    cdef unsigned int getCharAtIndex(CompressedMSBWT self, unsigned long index) nogil:
        '''
        Used for searching, this function masks the complexity behind retrieving a specific character at a specific index
        in our compressed BWT.
        @param index - the index to retrieve the character from
        @param return - return the character in our BWT that's at a particular index (integer format)
        '''
        
        #get the bin we should start from
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long bwtIndex = self.refFM_view[binID]
        
        #these are the values that indicate how far in we really are
        cdef unsigned long trueIndex = 0
        cdef unsigned long i
        for i in range(0, self.vcLen):
            trueIndex += self.partialFM_view[binID][i]
        trueIndex -= self.offsetSum
        
        cdef unsigned int prevChar = self.bwt_view[bwtIndex] & self.mask
        cdef unsigned int currentChar
        cdef unsigned long prevCount = self.bwt_view[bwtIndex] >> self.letterBits
        cdef unsigned powerMultiple = 1
        
        while trueIndex + prevCount <= index:
            trueIndex += prevCount
            inc(bwtIndex)
            
            currentChar = self.bwt_view[bwtIndex] & self.mask
            if currentChar == prevChar:
                powerMultiple *= self.numPower
                prevCount = (self.bwt_view[bwtIndex] >> self.letterBits) * powerMultiple
            else:
                powerMultiple = 1
                prevCount = self.bwt_view[bwtIndex] >> self.letterBits
            
            prevChar = currentChar
        
        return prevChar
        
    def getBWTRange(CompressedMSBWT self, unsigned long start, unsigned long end):
        '''
        TODO: cythonize
        This function masks the complexity of retrieving a chunk of the BWT from the compressed format
        @param start - the beginning of the range to retrieve
        @param end - the end of the range in normal python notation (bwt[end] is not part of the return)
        @return - a range of integers representing the characters in the bwt from start to end
        '''
        #set aside an array block to fill
        startBlockIndex = start >> self.bitPower
        endBlockIndex = int(math.floor(float(end)/self.binSize))
        trueStart = startBlockIndex*self.binSize
        
        #first we will extract the range of blocks
        return self.decompressBlocks(startBlockIndex, endBlockIndex)[start-trueStart:end-trueStart]
    
    def decompressBlocks(CompressedMSBWT self, unsigned long startBlock, unsigned long endBlock):
        '''
        TODO: cythonize
        This is mostly a helper function to get BWT range, but I wanted it to be a separate thing for use possibly in 
        decompression
        @param startBlock - the index of the start block we will decode
        @param endBlock - the index of the final block we will decode, if they are the same, we decode one block
        @return - an array of size blockSize*(endBlock-startBlock+1), interpreting that block is up to getBWTRange(...)
        '''
        expectedIndex = startBlock*self.binSize
        trueIndex = np.sum(self.partialFM[startBlock])-self.offsetSum
        dist = expectedIndex - trueIndex
        
        #find the end of the region of interest
        startRange = self.refFM[startBlock]
        if endBlock >= self.refFM.shape[0]-1:
            endRange = self.bwt.shape[0]
            returnSize = self.binSize*(endBlock-startBlock)+(self.totalSize % self.binSize)
        else:
            endRange = self.refFM[endBlock+1]+1
            returnSize = self.binSize*(endBlock-startBlock+1)
            while endRange < self.bwt.shape[0] and (self.bwt[endRange] & self.mask) == (self.bwt[endRange-1] & self.mask):
                endRange += 1
        
        ret = np.zeros(dtype='<u1', shape=(returnSize,))
        
        startRange = int(startRange)
        endRange = int(endRange)
        
        #split the letters and numbers in the compressed bwt
        letters = np.bitwise_and(self.bwt[startRange:endRange], self.mask)
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] counts = np.right_shift(self.bwt[startRange:endRange], self.letterBits, dtype='<u8')
        
        #multiply counts where needed
        i = 1
        #same = (letters[0:-1] == letters[1:])
        same = (letters[0:letters.shape[0]-1] == letters[1:])
        while np.count_nonzero(same) > 0:
            (counts[i:])[same] *= self.numPower
            i += 1
            #same = np.bitwise_and(same[0:-1], same[1:])
            same = np.bitwise_and(same[0:same.shape[0]-1], same[1:])
        
        #now I have letters and counts, time to fill in the array
        cdef unsigned long s = 0
        cdef unsigned long lInd = 0
        while dist > 0:
            if counts[lInd] < dist:
                dist -= counts[lInd]
                lInd += 1
            else:
                counts[lInd] -= dist
                dist = 0
        
        #we're at the correct letter index now
        while s < ret.shape[0]:
            if lInd >= letters.shape[0]:
                pass
            ret[s:s+counts[lInd]] = letters[lInd]
            s += counts[lInd]
            lInd += 1
        
        return ret
    
    cpdef unsigned long getOccurrenceOfCharAtIndex(CompressedMSBWT self, unsigned int sym, unsigned long index):# nogil:
        '''
        This functions gets the FM-index value of a character at the specified position
        @param sym - the character to find the occurrence level
        @param index - the index we want to find the occurrence level at
        @return - the number of occurrences of char before the specified index
        '''
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long compressedIndex = self.refFM_view[binID]
        cdef unsigned long bwtIndex = 0
        cdef unsigned long j
        for j in range(0, self.vcLen):
            bwtIndex += self.partialFM_view[binID][j]
        bwtIndex -= self.offsetSum
            
        cdef unsigned long ret = self.partialFM_view[binID][sym]
        
        cdef np.uint8_t prevChar = 255
        cdef np.uint8_t currentChar
        cdef unsigned long prevCount = 0
        cdef unsigned long powerMultiple = 1
        
        while bwtIndex + prevCount < index:
            currentChar = self.bwt_view[compressedIndex] & self.mask
            if currentChar == prevChar:
                prevCount += (self.bwt_view[compressedIndex] >> self.letterBits) * powerMultiple
                powerMultiple *= self.numPower
            else:
                if prevChar == sym:
                    ret += prevCount
                
                bwtIndex += prevCount
                prevCount = (self.bwt_view[compressedIndex] >> self.letterBits)
                prevChar = currentChar
                powerMultiple = self.numPower
                
            inc(compressedIndex)
        
        if prevChar == sym:
            ret += index-bwtIndex
        
        return ret
    
    def getFullFMAtIndex(CompressedMSBWT self, np.uint64_t index):
        '''
        This function creates a complete FM-index for a specific position in the BWT.  Example using the above example:
        BWT    Full FM-index
                 $ A C G T
        C        0 1 2 4 4
        $        0 1 3 4 4
        C        1 1 3 4 4
        A        1 1 4 4 4
                 1 2 4 4 4
        @return - the above information in the form of an array that already incorporates the offset value into the counts
        '''
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long compressedIndex = self.refFM_view[binID]
        cdef unsigned long bwtIndex = 0
        cdef unsigned long j
        
        #cdef unsigned long ret = self.partialFM_view[binID][sym]
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.empty(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] ret_view = ret
        
        for j in range(0, self.vcLen):
            bwtIndex += self.partialFM_view[binID][j]
            ret_view[j] = self.partialFM_view[binID][j]
        bwtIndex -= self.offsetSum
        
        cdef np.uint8_t prevChar = 255
        cdef np.uint8_t currentChar
        cdef unsigned long prevCount = 0
        cdef unsigned long powerMultiple = 1
        
        while bwtIndex + prevCount < index:
            currentChar = self.bwt_view[compressedIndex] & self.mask
            if currentChar == prevChar:
                prevCount += (self.bwt_view[compressedIndex] >> self.letterBits) * powerMultiple
                powerMultiple *= self.numPower
            else:
                ret_view[prevChar] += prevCount
                
                bwtIndex += prevCount
                prevCount = (self.bwt_view[compressedIndex] >> self.letterBits)
                prevChar = currentChar
                powerMultiple = self.numPower
                
            inc(compressedIndex)
        
        ret_view[prevChar] += index-bwtIndex
        
        return ret
    
    cdef np.uint8_t iterNext_cython(CompressedMSBWT self) nogil:
        '''
        returns each character, one at a time
        '''
        cdef np.uint8_t ret
        cdef np.uint8_t sym
        
        #print (self.iterCount, self.iterCurrCount, self.iterIndex, self.fileSize, sym)
        
        if self.iterCount < self.iterCurrCount:
            #we are still in the same byte of storage, increment and return the symbol there
            ret = self.iterCurrChar
            inc(self.iterCount)
        elif self.iterIndex < self.fileSize:
            #we have more bytes to process
            sym = self.bwt_view[self.iterIndex] & self.mask
            
            #increment our power if necessary
            if sym == self.iterCurrChar:
                inc(self.iterPower)
            else:
                self.iterCount = 0
                self.iterCurrCount = 0
                self.iterPower = 0
                self.iterCurrChar = sym
                
            #pull out the number of counts here and reset our counting
            self.iterCurrCount += (self.bwt_view[self.iterIndex] >> self.letterBits) * (self.numPower**self.iterPower)
            inc(self.iterCount) 
            ret = self.iterCurrChar
            inc(self.iterIndex)
        else:
            #we have run out of stuff
            ret = 255
        
        return ret
    
def loadBWT(bwtDir, logger=None):
    '''
    Generic load function, this is recommended for anyone wishing to use this code as it will automatically detect compression
    and assign the appropriate class preferring the decompressed version if both exist.
    @return - a MultiStringBWT, CompressedBWT, or none if neither can be instantiated
    '''
    if os.path.exists(bwtDir+'/msbwt.npy'):
        msbwt = MultiStringBWT()
        msbwt.loadMsbwt(bwtDir, logger)
        return msbwt
    elif os.path.exists(bwtDir+'/comp_msbwt.npy'):
        msbwt = CompressedMSBWT()
        msbwt.loadMsbwt(bwtDir, logger)
        return msbwt
    else:
        logger.error('Invalid BWT directory.')
        return None
    
def createMSBWTFromSeqs(seqArray, mergedDir, numProcs, areUniform, logger):
    '''
    This function takes a series of sequences and creates the BWT using the technique from Cox and Bauer
    @param seqArray - a list of '$'-terminated sequences to be in the MSBWT
    @param mergedFN - the final destination filename for the BWT
    @param numProcs - the number of processes it's allowed to use
    '''
    #wipe the auxiliary data stored here
    MSBWTGen.clearAuxiliaryData(mergedDir)
    
    #TODO: do we want a special case for N=1? there was one in early code, but we could just assume users aren't dumb
    seqFN = mergedDir+'/seqs.npy'
    offsetFN = mergedDir+'/offsets.npy'
    
    #sort the sequences
    seqCopy = sorted(seqArray)
    if areUniform:
        uniformLength = len(seqArray[0])
    else:
        uniformLength = 0
        #raise Exception('non-uniform is not handled anymore due to speed, consider building uniform and merging')
        
    #join into one massive string
    seqCopy = ''.join(seqCopy)
    
    #convert the sequences into uint8s and then save it
    seqCopy = np.fromstring(seqCopy, dtype='<u1')
    
    MSBWTGen.writeSeqsToFiles(seqCopy, seqFN, offsetFN, uniformLength)
    if areUniform:
        MSBWTGen.createMsbwtFromSeqs(mergedDir, numProcs, logger)
    else:
        MSBWTGen.createFromSeqs(seqFN, offsetFN, mergedDir+'/msbwt.npy', numProcs, areUniform, logger)
        
def createMSBWTFromFastq(fastqFNs, outputDir, numProcs, areUniform, logger):
    '''
    This function takes fasta filenames and creates the BWT using the technique from Cox and Bauer by simply loading
    all string prior to computation
    @param fastqFNs - a list of fastq filenames to extract sequences from
    @param outputDir - the directory for all of the bwt related data
    @param numProcs - the number of processes it's allowed to use
    @areUniform - true if all the sequences passed into the function are of equal length
    '''
    #generate the files we will reference and clear out the in memory array before making the BWT
    logger.info('Saving sorted sequences...')
    seqFN = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    bwtFN = outputDir+'/msbwt.npy'
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessFastqs(fastqFNs, seqFN, offsetFN, abtFN, areUniform, logger)
    #MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
    if areUniform:
        MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    else:
        MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
    
def createMSBWTFromBam(bamFNs, outputDir, numProcs, areUniform, logger):
    '''
    This function takes a fasta filename and creates the BWT using the technique from Cox and Bauer
    @param bamFNs - a list of BAM filenames to extract sequences from, READS MUST BE SORTED BY NAME
    @param outputDir - the directory for all of the bwt related data
    @param numProcs - the number of processes it's allowed to use
    @areUniform - true if all the sequences passed into the function are of equal length
    '''
    #generate the files we will reference and clear out the in memory array before making the BWT
    logger.info('Saving sorted sequences...')
    seqFN = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    bwtFN = outputDir+'/msbwt.npy'
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessBams(bamFNs, seqFN, offsetFN, abtFN, areUniform, logger)
    #MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
    if areUniform:
        MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    else:
        MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
        
def createMSBWTFromFasta(fastaFNs, outputDir, numProcs, areUniform, logger):
    logger.info('Saving sorted sequences...')
    seqFN = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    bwtFN = outputDir+'/msbwt.npy'
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessFastas(fastaFNs, seqFN, offsetFN, abtFN, areUniform, logger)
    if areUniform:
        MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    else:
        MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
    
def customiter(numpyArray):
    '''
    dummy iterator, for some reason numpy doesn't like to act like one by default
    '''
    for x in numpyArray:
        yield tuple(x)

def preprocessFastqs(fastqFNs, seqFNPrefix, offsetFN, abtFN, areUniform, logger):
    '''
    This function does the grunt work behind string extraction for fastq files
    @param fastqFNs - a list of .fq filenames for parsing
    @param seqFNPrefix - this is always of the form '<DIR>/seqs.npy'
    @param offsetFN - this is always of the form '<DIR>/offsets.npy'
    @param abtFN - this is always of the form '<DIR>/about.npy'
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    #create a seqArray
    seqArray = []
    
    tempFileId = 0
    seqsPerFile = 1000000
    maxSeqLen = -1
    numSeqs = 0
    
    subSortFNs = []
    
    for fnID, fn in enumerate(fastqFNs):
        #open the file and read in starting form the second, every 4th line
        logger.info('Loading \''+fn+'\'...')
        if fn.endswith('.gz'):
            fp = gzip.open(fn, 'r')
        else:
            fp = open(fn, 'r')
        i = -1
        
        #go through each line
        for line in fp:
            if i % 4 == 0:
                seqArray.append((line.strip('\n')+'$', fnID, i/4))
                if len(seqArray) == seqsPerFile:
                    if not areUniform or maxSeqLen == -1:
                        maxSeqLen = 0
                        for seq, fID, seqID in seqArray:
                            if len(seq) > maxSeqLen:
                                maxSeqLen = len(seq)
                    
                    tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
                    subSortFNs.append(tempFN)
                    
                    tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
                    tempArray[:] = sorted(seqArray)
                    numSeqs += len(seqArray)
                    del tempArray
                    tempFileId += 1
                    seqArray = []
            i += 1
                
        fp.close()
    
    if len(seqArray) > 0:
        if not areUniform or maxSeqLen == -1:
            maxSeqLen = 0
            for seq, fID, seqID in seqArray:
                if len(seq) > maxSeqLen:
                    maxSeqLen = len(seq)
        
        tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
        subSortFNs.append(tempFN)
        
        tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
        tempArray[:] = sorted(seqArray)
        numSeqs += len(seqArray)
        del tempArray
        tempFileId += 1
        seqArray = []
    
    logger.info('Pre-sorting '+str(numSeqs)+' sequences...')
    iters = []
    for fn in subSortFNs:
        iters.append(customiter(np.load(fn, 'r')))
    
    #save it
    tempFN = seqFNPrefix+'.temp.npy'
    fp = open(tempFN, 'w+')
    
    aboutFile = np.lib.format.open_memmap(abtFN, 'w+', '<u1,<u8', (numSeqs,))
    ind = 0
    
    for tup in heapq.merge(*iters):
        (seq, fID, seqID) = tup
        aboutFile[ind] = (fID, seqID)
        fp.write(seq)
        ind += 1
        
    fp.close()
    
    #clean up disk space
    for fn in subSortFNs:
        os.remove(fn)
    
    #convert the sequences into uint8s and then save it
    del seqArray
    seqArray = np.memmap(tempFN)
    
    if areUniform:
        uniformLength = maxSeqLen
    else:
        uniformLength = 0
    
    logger.info('Saving sorted sequences for BWT construction...')
    MSBWTGen.writeSeqsToFiles(seqArray, seqFNPrefix, offsetFN, uniformLength)
    
    #wipe this
    del seqArray
    os.remove(tempFN)

def preprocessBams(bamFNs, seqFNPrefix, offsetFN, abtFN, areUniform, logger):
    '''
    This does the grunt work behind read extraction from a name-sorted BAM file.  If it isn't sorted, this will not work
    as intended.
    @param bamFNs - a list of '.bam' filenames for parsing
    @param seqFNPrefix - this is always of the form '<DIR>/seqs.npy'
    @param offsetFN - this is always of the form '<DIR>/offsets.npy'
    @param abtFN - this is always of the form '<DIR>/about.npy'
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    #create a seqArray
    seqArray = []
    
    #prep to break this into several smaller sorted sequences
    tempFileId = 0
    seqsPerFile = 10000000
    maxSeqLen = -1
    numSeqs = 0
    
    subSortFNs = []
    
    for fnID, fn in enumerate(bamFNs):
        #open the file and read in starting form the second, every 4th line
        logger.info('Loading \''+fn+'\'...')
        
        bamFile = pysam.Samfile(fn, 'rb')
        i = 0
        
        nr = bamFile.next()
        constantSize = len(nr.seq)
        
        #go through each line
        while nr != None:
            #collect all reads that are the same as nr
            aligns = []
            nqname = nr.qname
            while nr != None and nr.qname == nqname:
                aligns.append(nr)
                try:
                    nr = bamFile.next()
                except:
                    nr = None
            
            #reduce this to a simpler set
            reads = [None, None]
            for a in aligns:
                if len(a.seq) != constantSize:
                    print 'DIFF SIZE='+str(len(a.seq))
                
                if a.flag & REVERSE_COMPLEMENTED_FLAG == 0:
                    #not reverse complemented
                    seq = a.seq
                else:
                    seq = reverseComplement(a.seq)
                
                if a.flag & FIRST_SEGMENT_FLAG == 0:
                    #second segment
                    if reads[1] == None:
                        reads[1] = seq
                    elif reads[1] != seq:
                        logger.warning('Two sequences with same flag and different seqs: '+reads[1]+'\n'+str(a))
                else:
                    if reads[0] == None:
                        reads[0] = seq
                    elif reads[0] != seq:
                        logger.warning('Two sequences with same flag and different seqs: '+reads[0]+'\n'+str(a))
            
            for j, r in enumerate(reads):
                if r == None:
                    continue
                
                if r[1-j] == None:
                    seqArray.append((r+'$', fnID, i, 0xFF, 0xFFFFFFFFFFFFFFFF))
                else:
                    seqArray.append((r+'$', fnID, i, fnID, i+1-2*j))
                
                if len(seqArray) == seqsPerFile:
                    if not areUniform or maxSeqLen == -1:
                        maxSeqLen = 0
                        for seq, fID, seqID, pfID, pseqID in seqArray:
                            if len(seq) > maxSeqLen:
                                maxSeqLen = len(seq)
                    
                    tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
                    subSortFNs.append(tempFN)
                    
                    sys.stdout.write('\rWriting file '+str(tempFileId))
                    sys.stdout.flush()
                    tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8,<u1,<u8', (len(seqArray),))
                    tempArray[:] = sorted(seqArray)
                    numSeqs += len(seqArray)
                    del tempArray
                    tempFileId += 1
                    seqArray = []
                i += 1
                
    sys.stdout.write('\n')
    if len(seqArray) > 0:
        if not areUniform or maxSeqLen == -1:
            maxSeqLen = 0
            for seq, fID, seqID, pfID, pseqID in seqArray:
                if len(seq) > maxSeqLen:
                    maxSeqLen = len(seq)
        
        tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
        subSortFNs.append(tempFN)
        
        tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8,<u1,<u8', (len(seqArray),))
        tempArray[:] = sorted(seqArray)
        numSeqs += len(seqArray)
        del tempArray
        tempFileId += 1
        seqArray = []
    
    logger.info('Pre-sorting '+str(numSeqs)+' sequences...')
    iters = []
    for fn in subSortFNs:
        iters.append(customiter(np.load(fn, 'r')))
    
    #save it
    tempFN = seqFNPrefix+'.temp.npy'
    fp = open(tempFN, 'w+')
    
    aboutFile = np.lib.format.open_memmap(abtFN, 'w+', '<u1,<u8,<u1,<u8', (numSeqs,))
    ind = 0
    
    for tup in heapq.merge(*iters):
        (seq, fID, seqID, pfID, pseqID) = tup
        aboutFile[ind] = (fID, seqID, pfID, pseqID)
        fp.write(seq)
        ind += 1
        
    fp.close()
    
    for fn in subSortFNs:
        os.remove(fn)
    
    #convert the sequences into uint8s and then save it
    del seqArray
    seqArray = np.memmap(tempFN)
    
    if areUniform:
        uniformLength = maxSeqLen
    else:
        uniformLength = 0
    
    logger.info('Saving sorted sequences for BWT construction...')
    MSBWTGen.writeSeqsToFiles(seqArray, seqFNPrefix, offsetFN, uniformLength)
    
    #wipe this
    del seqArray
    os.remove(tempFN)

def preprocessFastas(fastaFNs, seqFNPrefix, offsetFN, abtFN, areUniform, logger):
    '''
    This function does the grunt work behind string extraction for fasta files
    @param fastaFNs - a list of .fq filenames for parsing
    @param seqFNPrefix - this is always of the form '<DIR>/seqs.npy'
    @param offsetFN - this is always of the form '<DIR>/offsets.npy'
    @param abtFN - this is always of the form '<DIR>/about.npy'
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    #create a seqArray
    seqArray = []
    
    #TODO: make the seqPerFile work better for when they aren't uniform
    tempFileId = 0
    seqsPerFile = 1000000
    maxSeqLen = -1
    numSeqs = 0
    
    subSortFNs = []
    
    for fnID, fn in enumerate(fastaFNs):
        #open the file and read in starting form the second, every 4th line
        logger.info('Loading \''+fn+'\'...')
        fp = open(fn, 'r')
        i = 0
        
        currRead = ''
        
        #go through each line
        for line in fp:
            if line[0] == '>':
                if currRead == '':
                    continue
                
                #end of a read
                seqArray.append((currRead+'$', fnID, i))
                if len(seqArray) == seqsPerFile:
                    if not areUniform or maxSeqLen == -1:
                        maxSeqLen = 0
                        for seq, fID, seqID in seqArray:
                            if len(seq) > maxSeqLen:
                                maxSeqLen = len(seq)
                    
                    tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
                    subSortFNs.append(tempFN)
                    
                    tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
                    tempArray[:] = sorted(seqArray)
                    numSeqs += len(seqArray)
                    del tempArray
                    tempFileId += 1
                    seqArray = []
                
                #reset to empty read now
                currRead = ''
                i += 1
                
            else:
                currRead += line.strip('\n')
            
        fp.close()
    
    if len(seqArray) > 0 or currRead != '':
        seqArray.append((currRead+'$', fnID, i))
        
        if not areUniform or maxSeqLen == -1:
            maxSeqLen = 0
            for seq, fID, seqID in seqArray:
                if len(seq) > maxSeqLen:
                    maxSeqLen = len(seq)
        
        tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
        subSortFNs.append(tempFN)
        
        tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
        tempArray[:] = sorted(seqArray)
        numSeqs += len(seqArray)
        del tempArray
        tempFileId += 1
        seqArray = []
    
    logger.info('Pre-sorting '+str(numSeqs)+' sequences...')
    iters = []
    for fn in subSortFNs:
        iters.append(customiter(np.load(fn, 'r')))
    
    #save it
    tempFN = seqFNPrefix+'.temp.npy'
    fp = open(tempFN, 'w+')
    
    aboutFile = np.lib.format.open_memmap(abtFN, 'w+', '<u1,<u8', (numSeqs,))
    ind = 0
    
    for tup in heapq.merge(*iters):
        (seq, fID, seqID) = tup
        aboutFile[ind] = (fID, seqID)
        fp.write(seq)
        ind += 1
        
    fp.close()
    
    #clean up disk space
    for fn in subSortFNs:
        os.remove(fn)
    
    #convert the sequences into uint8s and then save it
    del seqArray
    seqArray = np.memmap(tempFN)
    
    if areUniform:
        uniformLength = maxSeqLen
    else:
        uniformLength = 0
    
    logger.info('Saving sorted sequences for BWT construction...')
    MSBWTGen.writeSeqsToFiles(seqArray, seqFNPrefix, offsetFN, uniformLength)
    
    #wipe this
    del seqArray
    os.remove(tempFN)

def mergeNewSeqs(seqArray, mergedDir, numProcs, areUniform, logger):
    '''
    This function takes a series of sequences and creates a big BWT by merging the smaller ones 
    Mostly a test function, no real purpose to the tool as of now
    @param seqArray - a list of '$'-terminated strings to be placed into the array
    @param mergedFN - the final destination filename for the merged BWT
    @param numProcs - the number of processors the merge is allowed to create
    '''
    #first wipe away any traces of old information for the case of overwriting a BWT at mergedFN
    MSBWTGen.clearAuxiliaryData(mergedDir)
    
    #create two smaller ones
    midPoint = len(seqArray)/3
    mergedDir1 = mergedDir+'0'
    mergedDir2 = mergedDir+'1'
    mergedDir3 = mergedDir+'2'
    
    try:
        shutil.rmtree(mergedDir1)
    except:
        pass
    try:
        shutil.rmtree(mergedDir2)
    except:
        pass
    try:
        shutil.rmtree(mergedDir3)
    except:
        pass
    os.makedirs(mergedDir1)
    os.makedirs(mergedDir2)
    os.makedirs(mergedDir3)
    
    createMSBWTFromSeqs(seqArray[0:midPoint], mergedDir1, numProcs, areUniform, logger)
    createMSBWTFromSeqs(seqArray[midPoint:2*midPoint], mergedDir2, numProcs, areUniform, logger)
    createMSBWTFromSeqs(seqArray[2*midPoint:], mergedDir3, numProcs, areUniform, logger)
    
    #now do the actual merging
    MSBWTGen.mergeNewMSBWT(mergedDir, [mergedDir1, mergedDir2, mergedDir3], numProcs, logger)

def compareKmerProfiles(profileFN1, profileFN2):
    '''
    This function takes two kmer profiles and compare them for similarity.
    @param profileFN1 - the first kmer-profile to compare to
    @param profileFN2 - the second kmer-profile to compare to
    @return - a tuple of the form (1-norm, 2-norm, sum of differences, normalized Dot product)
    '''
    fp1 = open(profileFN1, 'r')
    fp2 = open(profileFN2, 'r')
    
    oneNorm = 0
    twoNorm = 0
    sumDeltas = 0
    dotProduct = 0
    
    tot1 = float(fp1.readline().strip('\n').split(',')[1])
    tot2 = float(fp2.readline().strip('\n').split(',')[1])
    
    (seq1, count1) = parseProfileLine(fp1)
    (seq2, count2) = parseProfileLine(fp2)
    
    while seq1 != None or seq2 != None:
        if seq1 == seq2:
            delta = abs(count1/tot1-count2/tot2)
            dotProduct += (count1/tot1)*(count2/tot2)
            (seq1, count1) = parseProfileLine(fp1)
            (seq2, count2) = parseProfileLine(fp2)
        elif seq2 == None or (seq1 != None and seq1 < seq2):
            delta = count1/tot1
            (seq1, count1) = parseProfileLine(fp1)
        else:
            delta = count2/tot2
            (seq2, count2) = parseProfileLine(fp2)
        
        if delta > oneNorm:
            oneNorm = delta
        
        twoNorm += delta*delta
        sumDeltas += delta
    
    fp1.close()
    fp2.close()
    
    twoNorm = math.sqrt(twoNorm)
    #print '1-norm:\t\t'+str(oneNorm)
    #print '2-norm:\t\t'+str(twoNorm)
    #print 'Delta sum:\t'+str(sumDeltas)
    return (oneNorm, twoNorm, sumDeltas, dotProduct)
    
def parseProfileLine(fp):
    '''
    Helper function for profile parsing
    @param fp - the file pointer to get the next line from
    @return - (kmer, kmerCount) as (string, int)
    '''
    nextLine = fp.readline()
    if nextLine == None or nextLine == '':
        return (None, None)
    else:
        pieces = nextLine.strip('\n').split(',')
        return (pieces[0], int(pieces[1]))
    
def reverseComplement(seq):
    '''
    Helper function for generating reverse-complements
    '''
    revComp = ''
    complement = {'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N', '$':'$', '?':'?', '*':'*'}
    for c in reversed(seq):
        revComp += complement[c]
    return revComp
    