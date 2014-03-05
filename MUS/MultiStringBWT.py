'''
Created on Aug 5, 2013

@author: holtjma
'''

import bisect
import gc
import gzip
import heapq
import math
import numpy as np
import os
import pickle
import pysam#@UnresolvedImport
import shutil
import sys

import MSBWTGen

#flags for samtools
REVERSE_COMPLEMENTED_FLAG = 1 << 4#0x10
FIRST_SEGMENT_FLAG = 1 << 6#0x40
#SECOND_SEGMENT_FLAG = 1 << 7#0x80

class BasicBWT(object):
    def __init__(self):
        '''
        Constructor
        Nothing special, use this for all at the start
        '''
        #TODO: no hard coding of these values?
        self.numToChar = np.array(sorted(['$', 'A', 'C', 'G', 'N', 'T']))
        self.charToNum = {}
        for i, c in enumerate(self.numToChar):
            self.charToNum[c] = i
        self.vcLen = len(self.numToChar)
    
    def constructIndexing(self):
        '''
        This helper function calculates the start and end index for each character in the BWT.  Basically, the information
        generated here is for quickly finding offsets.
        '''
        #mark starts and ends of key elements
        self.startIndex = [None]*self.vcLen
        self.endIndex = [None]*self.vcLen
        pos = 0
        
        #go through the 1-mers
        for c in xrange(0, self.vcLen):
            #build start and end indexes
            self.startIndex[c] = pos
            pos += self.totalCounts[c]
            self.endIndex[c] = pos
    
    def countOccurrencesOfSeq(self, seq, givenRange=None):
        '''
        This function counts the number of occurrences of the given sequence
        @param seq - the sequence to search for
        @param givenRange - the range to start from (if a partial search has already been run), default=whole range
        @return - an integer count of the number of times seq occurred in this BWT
        '''
        #init the current range
        if givenRange == None:
            #'''
            if not self.searchCache.has_key(seq[-6:]):
                res = self.findIndicesOfStr(seq[-6:])
                self.searchCache[seq[-6:]] = (int(res[0]), int(res[1]))
            
            l, h = self.searchCache[seq[-6:]]
            seq = seq[0:-6]
            '''
            l = 0
            h = self.totalSize
            '''
        else:
            l = givenRange[0]
            h = givenRange[1]
            
        #reverse sequence and convert to ints so we can iterate through it
        #revSeq = [self.charToNum[c] for c in seq[::-1]]
        revSeq = [self.charToNum[c] for c in reversed(seq)]
        
        for c in revSeq:
            #get the start and end offsets
            l = self.getOccurrenceOfCharAtIndex(c, l)
            h = self.getOccurrenceOfCharAtIndex(c, h)
            
            #early exit for counts
            if l == h:
                return 0
        
        #return the difference
        return h - l
    
    def findIndicesOfStr(self, seq, givenRange=None):
        '''
        This function will search for a string and find the location of that string OR the last index less than it. It also
        will start its search within a given range instead of the whole structure
        @param seq - the sequence to search for
        @param givenRange - the range to search for, whole range by default
        @return - a python range representing the start and end of the sequence in the bwt
        '''
        #init the current range
        if givenRange == None:
            if not self.searchCache.has_key(seq[-6:]):
                res = self.findIndicesOfStr(seq[-6:], [0, self.totalSize])
                self.searchCache[seq[-6:]] = (int(res[0]), int(res[1]))
            
            l, h = self.searchCache[seq[-6:]]
            seq = seq[0:-6]
            '''
            l = 0
            h = self.totalSize
            '''
        else:
            l = givenRange[0]
            h = givenRange[1]
            
        #reverse sequence and convert to ints so we can iterate through it
        revSeq = [self.charToNum[c] for c in seq[::-1]]
        
        for c in revSeq:
            #get the start and end offsets
            l = self.getOccurrenceOfCharAtIndex(c, l)
            h = self.getOccurrenceOfCharAtIndex(c, h)
            
        return (l, h)
    
    def recoverString(self, strIndex, withIndex=False):
        '''
        This will return the string that starts at the given index
        @param strIndex - the index of the string we want to recover
        @return - string that we found starting at the specified '$' index
        '''
        retNums = []
        indices = []
        
        #figure out the first hop backwards
        currIndex = strIndex
        
        #TODO: the below info can be recovered with one function in the compressed BWT, will mean speedup
        prevChar = self.getCharAtIndex(currIndex)
        currIndex = self.getOccurrenceOfCharAtIndex(prevChar, currIndex)
        
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
        ret = ''.join(self.numToChar[retNums[::-1]])
        
        #return what we found
        if withIndex:
            return (ret, indices[::-1])
        else:
            return ret

class MultiStringBWT(BasicBWT):
    '''
    This class is a BWT capable of hosting multiple strings inside one structure.  Basically, this would allow you to
    search for a given string across several strings simultaneously.  Note: this class is for the non-compressed version
    '''
    def loadMsbwt(self, dirName, logger):
        '''
        This functions loads a BWT file and constructs total counts, indexes start positions, and constructs an FM index in memory
        @param dirName - the filename to load
        '''
        #open the file with our BWT in it
        self.dirName = dirName
        self.bwt = np.load(self.dirName+'/msbwt.npy', 'r')
        
        #build auxiliary structures
        self.constructTotalCounts(logger)
        self.constructIndexing()
        self.constructFMIndex(logger)
    
    def constructTotalCounts(self, logger):
        '''
        This function constructs the total count for each valid character in the array
        '''
        self.totalSize = self.bwt.shape[0]
        chunkSize = 2**20
        
        abtFN = self.dirName+'/totalCounts.p'
        if os.path.exists(abtFN):
            fp = open(abtFN, 'r')
            self.totalCounts = pickle.load(fp)
            fp.close()
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % abtFN)
            
            self.totalCounts = [0]*self.vcLen
            i = 0
            while i*chunkSize < self.bwt.shape[0]:
                self.totalCounts = np.add(self.totalCounts, np.bincount(self.bwt[i*chunkSize:(i+1)*chunkSize], minlength=self.vcLen))
                i += 1
            #self.totalCounts = np.bincount(self.bwt, minlength=self.vcLen)
            fp = open(abtFN, 'w+')
            pickle.dump(self.totalCounts, fp)
            fp.close()
    
    def constructFMIndex(self, logger):
        '''
        This function iterates through the BWT and counts the letters as it goes to create the FM index.  For example, the string 'ACC$' would have BWT
        'C$CA'.  The FM index would iterate over this and count the occurence of the letter it found so you'd end up with this:
        BWT    FM-index
        C        1
        $        1
        C        2
        A        1
        This is necessary for finding the occurrence of a letter using the getOccurrenceOfCharAtIndex(...) function.
        '''
        #sampling method
        self.searchCache = {}
        self.bitPower = 11
        self.binSize = 2**self.bitPower
        self.fmIndexFN = self.dirName+'/fmIndex.npy'
        
        if os.path.exists(self.fmIndexFN):
            self.partialFM = np.load(self.fmIndexFN, 'r')
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % self.fmIndexFN)
            
            self.partialFM = np.lib.format.open_memmap(self.fmIndexFN, 'w+', '<u8', (self.bwt.shape[0]/self.binSize+1, self.vcLen))
            
            counts = np.zeros(dtype='<u8', shape=(self.vcLen,))
            counts[:] = self.startIndex
            self.partialFM[0] = self.startIndex
            for j in xrange(1, self.partialFM.shape[0]):
                counts += np.bincount(self.bwt[self.binSize*(j-1):self.binSize*j], minlength=self.vcLen)
                self.partialFM[j] = counts
       
        '''
        self.fmIndexFN = self.fn+'.fmindex.npy'
        self.backRefsFN = self.fn+'.backrefs.npy'
        if os.path.exists(self.fmIndexFN) and os.path.exists(self.backRefsFN):
            self.partialFM = np.load(self.fmIndexFN, 'r')
            self.backRefs = np.load(self.backRefsFN, 'r')
        else:
            #exact + offsets method
            self.searchCache = {}
            #self.partialFM = np.zeros(dtype='<u8', shape=(self.bwt.shape[0]+1,))
            #self.backRefs = np.zeros(dtype='<u1', shape=(self.bwt.shape[0]+1, self.vcLen))
            self.partialFM = np.lib.format.open_memmap(self.fmIndexFN, 'w+', '<u8', (self.bwt.shape[0]+1,))
            self.backRefs = np.lib.format.open_memmap(self.backRefsFN, 'w+', '<u1', (self.bwt.shape[0]+1, self.vcLen))
            counts = np.zeros(dtype='<u8', shape=(self.vcLen,))
            counts[:] = self.startIndex
            
            dists = np.zeros(dtype='<u8', shape=(self.vcLen,))
            for j in xrange(1, self.bwt.shape[0]+1):
                c = self.bwt[j-1]
                counts[c] += 1
                self.partialFM[j] = counts[c]
                dists += 1
                dists[c] = 0
                self.backRefs[j][dists > 0] = np.floor(np.log2(dists[dists > 0]))+1
            
            #print 'Maxes:'+str(np.max(self.backRefs, axis=0))
            #print 'Average:'+str(np.average(self.backRefs, axis=0))
        '''
        
    def getSequenceDollarID(self, strIndex):
        #figure out the first hop backwards
        currIndex = strIndex
        prevChar = self.bwt[currIndex]
        currIndex = self.getOccurrenceOfCharAtIndex(prevChar, currIndex)
        
        #while we haven't looped back to the start
        while prevChar != 0:
            #figure out where to go from here
            prevChar = self.bwt[currIndex]
            currIndex = self.getOccurrenceOfCharAtIndex(prevChar, currIndex)
        
        return currIndex
        
    def recoverReversedInts(self, strIndex, withIndex=False):
        '''
        This will return the string that start at the strIndex'ed occurrence of '$' or None if it's out of bounds, it will recover them in 
        numpy array of uint8 format instead of string
        @param strIndex - the index of the string we want to recover
        @return - string that we found starting at the specified '$' index but in integer format and reversed
        '''
        #make sure we start in a valid position
        '''
        if strIndex >= self.totalCounts[0]:
            #this is out of bounds
            return None
        '''
        #init the string
        retNums = [0]
        if withIndex:
            indices = [strIndex]
        
        #figure out the first hop backwards
        currIndex = strIndex
        prevChar = self.bwt[currIndex]
        currIndex = self.getOccurrenceOfCharAtIndex(prevChar, currIndex)
        
        #while we haven't looped back to the start
        #while prevChar != 0:
        while currIndex != strIndex:
            #update the string
            retNums.append(prevChar)
            if withIndex:
                indices.append(currIndex)
        
            #figure out where to go from here
            prevChar = self.bwt[currIndex]
            currIndex = self.getOccurrenceOfCharAtIndex(prevChar, currIndex)
        
        #return what we found
        if withIndex:
            return (retNums, indices)
        else:
            return retNums
        
    def getCharAtIndex(self, index):
        return self.bwt[index]
    
    def getBWTRange(self, start, end):
        return self.bwt[start:end]
        
    def getOccurrenceOfCharAtIndex(self, char, index):
        '''
        This functions gets the FM-index value of a character at the specified position
        @param char - the character to find the occurrence level
        @param index - the index we want to find the occurrence level at
        @return - the number of occurrences of char before the specified index
        '''
        #'''
        #sampling method
        #get the bin we occupy
        binID = index >> self.bitPower
        
        #these two methods seem to have the same approximate run time
        if (binID << self.bitPower) == index:
            ret = self.partialFM[binID][char]
        else:
            ret = self.partialFM[binID][char] + np.bincount(self.bwt[binID << self.bitPower:index], minlength=6)[char]
        #ret = self.partialFM[binID][char] + np.where(self.bwt[binID << self.bitPower:index] == char)[0].shape[0]
        return int(ret)
        
        #this method is for only if binSize == 0
        #return int(self.partialFM[index][char])
        '''
        #exact + offsets method
        i = index
        while i != 0 and self.backRefs[i][char] != 0:
            i = i - (1 << (self.backRefs[i][char]-1))
        
        if i == 0:
            return self.startIndex[char]
        else:
            return int(self.partialFM[i])
        '''
        
    def getFullFMAtIndex(self, index):
        '''
        This function creates a complete FM-index for a specific position in the BWT.  Example using the above example:
        BWT    Full FM-index
                 $ A C G T
        C        0 0 1 0 0
        $        1 0 1 0 0
        C        1 0 2 0 0
        A        1 1 2 0 0
        @return - the above information in the form of an array
        '''
        #get the bin we occupy
        binID = index >> self.bitPower
        if binID << self.bitPower == index:
            ret = self.partialFM[binID]
        else:
            ret = self.partialFM[binID] + np.bincount(self.bwt[binID << self.bitPower:index], minlength=6)
        return ret
        
    def getSuffixStartAtIndex(self, i):
        '''
        This will get the starting character (in number form) of the entry at 'i' in the BWT
        @return - a int representing the character that is the start of the entry at 'i'
        '''
        if i >= self.totalSize:
            return None
        else:
            return bisect.bisect_right(self.startIndex, i)-1
    
    def findIndicesOfInt(self, reversedSeqAsInts, givenRange=None):
        '''
        This function will search for a string and find the location of that string OR the last index less than it. It also
        will start its search within a given range instead of the whole structure
        @param reversedSeqAsInts - the sequence to search for in integer form and already reversed
        @param givenRange - the range to search for, whole range by default
        @return - a python range representing the start and end of the sequence in the bwt
        '''
        #init the current range
        if givenRange == None:
            l = 0
            h = self.totalSize
        else:
            l = givenRange[0]
            h = givenRange[1]
            
        #reverse sequence of ints so we can iterate through it
        for c in reversedSeqAsInts:
            #get the start and end offsets
            l = self.getOccurrenceOfCharAtIndex(c, l)
            h = self.getOccurrenceOfCharAtIndex(c, h)
            
        return (l, h)
    
    def createKmerProfile(self, k, profileCsvFN):
        '''
        @param k - the length of the k-mers to profile
        @param profileCsvFN - the filename of the csv to create
        '''
        #'''
        searches = [('', 0, self.bwt.shape[0])]
        normTotal = 0
        lines = []
        while len(searches) > 0:
            (seq, start, end) = searches.pop(0)
            
            if len(seq) == k:
                lines.append(seq+','+str(end-start))
                normTotal += (end-start)**2
            else:
                '''
                for c in reversed(self.numToChar):
                    #TODO: use get fullFMIndex style function
                    nl = self.getOccurrenceOfCharAtIndex(self.charToNum[c], start)
                    nh = self.getOccurrenceOfCharAtIndex(self.charToNum[c], end)
                    if nl == nh:
                        #do nothing
                        pass
                    else:
                        newSeq = c+seq
                        searches.insert(0, (newSeq, nl, nh))
                '''
                nls = self.getFullFMAtIndex(start)
                nhs = self.getFullFMAtIndex(end)
                for c in xrange(self.vcLen-1, -1, -1):
                    if nls[c] == nhs[c]:
                        #do nothing
                        pass
                    else:
                        newSeq = self.numToChar[c]+seq
                        searches.insert(0, (newSeq, int(nls[c]), int(nhs[c])))
        
        fp = open(profileCsvFN, 'w+')
        fp.write('total,'+str(math.sqrt(normTotal))+'\n')
        for l in sorted(lines):
            fp.write(l+'\n')
        fp.close()

class CompressedMSBWT(BasicBWT):
    def loadMsbwt(self, dirName, logger):
        '''
        This functions loads a BWT file and constructs total counts, indexes start positions, and constructs an FM index in memory
        @param dirName - the filename to load
        '''
        #open the file with our BWT in it
        self.dirName = dirName
        self.bwt = np.load(self.dirName+'/comp_msbwt.npy', 'r')
        
        #build auxiliary structures
        self.constructTotalCounts(logger)
        self.constructIndexing()
        self.constructFMIndex(logger)
    
    def constructTotalCounts(self, logger):
        '''
        This function constructs the total count for each valid character in the array
        '''
        self.letterBits = 3
        self.numberBits = 8-self.letterBits
        self.numPower = 2**self.numberBits
        self.mask = 255 >> self.numberBits
        
        abtFN = self.dirName+'/totalCounts.p'
        if os.path.exists(abtFN):
            fp = open(abtFN, 'r')
            self.totalCounts = pickle.load(fp)
            fp.close()
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % abtFN)
            
            self.totalCounts = [0]*self.vcLen
            
            binSize = 2**15
            end = 0
            
            while end < self.bwt.shape[0]:
                start = end
                end = end + binSize
                if end > self.bwt.shape[0]:
                    end = self.bwt.shape[0]
                
                #find a clean break in the characters
                while end < self.bwt.shape[0] and ((self.bwt[end] & self.mask) == (self.bwt[end-1] & self.mask)):
                    end += 1
                
                letters = np.bitwise_and(self.bwt[start:end], self.mask)
                counts = np.right_shift(self.bwt[start:end], self.letterBits, dtype='<u8')
                powers = np.zeros(dtype='<u8', shape=(end-start,))
                
                #solve the actual powers
                i = 1
                same = (letters[0:-1] == letters[1:])
                while np.sum(same) > 0:
                    (powers[i:])[same] += 1
                    i += 1
                    same = np.bitwise_and(same[0:-1], same[1:])
                
                #each letter has a variable 'weight' which is the runlength of that region
                #self.totalCounts += np.bincount(letters, counts*(self.numPower**powers), minlength=self.vcLen)
                self.totalCounts += np.bincount(letters, np.multiply(counts, self.numPower**powers), minlength=self.vcLen)
                
            fp = open(abtFN, 'w+')
            pickle.dump(self.totalCounts, fp)
            fp.close()
        
        self.totalSize = int(np.sum(self.totalCounts))
    
    def constructFMIndex(self, logger):
        '''
        This function iterates through the BWT and counts the letters as it goes to create the FM index.  For example, the string 'ACC$' would have BWT
        'C$CA'.  The FM index would iterate over this and count the occurence of the letter it found so you'd end up with this:
        BWT    FM-index
        C        1
        $        1
        C        2
        A        1
        This is necessary for finding the occurrence of a letter using the getOccurrenceOfCharAtIndex(...) function.
        '''
        #sampling method
        self.searchCache = {}
        self.bitPower = 11
        self.binSize = 2**self.bitPower
        
        self.fmIndexFN = self.dirName+'/comp_fmIndex.npy'
        self.fmRefFN = self.dirName+'/comp_refIndex.npy'
        
        if os.path.exists(self.fmIndexFN) and os.path.exists(self.fmRefFN):
            self.partialFM = np.load(self.fmIndexFN, 'r')
            self.refFM = np.load(self.fmRefFN, 'r')
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % self.fmIndexFN)
            
            samplingSize = int(math.ceil(float(self.totalSize)/self.binSize))
            self.partialFM = np.lib.format.open_memmap(self.fmIndexFN, 'w+', '<u8', (samplingSize, self.vcLen))
            self.refFM = np.lib.format.open_memmap(self.fmRefFN, 'w+', '<u8', (samplingSize,))
            
            countsSoFar = np.cumsum(self.totalCounts)-self.totalCounts
            totalCounts = 0
            
            prevStart = 0
            bwtIndex = 0
            chunkSize = 10000
            
            samplingID = 0
            
            while bwtIndex < self.bwt.shape[0] and samplingID < samplingSize:
                letters = np.bitwise_and(self.bwt[bwtIndex:bwtIndex+chunkSize], self.mask)
                counts = np.right_shift(self.bwt[bwtIndex:bwtIndex+chunkSize], self.letterBits, dtype='<u8')
                
                #numpy methods for find the powers
                i = 1
                same = (letters[0:-1] == letters[1:])
                while np.count_nonzero(same) > 0:
                    (counts[i:])[same] *= self.numPower
                    i += 1
                    same = np.bitwise_and(same[0:-1], same[1:])
                
                offsets = np.cumsum(counts)
                
                moreToUpdate = True
                while moreToUpdate:
                    prevStart = np.searchsorted(offsets, samplingID*self.binSize-totalCounts, 'right')
                    if prevStart == letters.shape[0]:
                        prevStart -= 1
                        while prevStart > 0 and letters[prevStart] == letters[prevStart-1]:
                            prevStart -= 1
                        moreToUpdate = False
                    else:
                        while prevStart > 0 and letters[prevStart] == letters[prevStart-1]:
                            prevStart -= 1
                        
                        self.refFM[samplingID] = bwtIndex+prevStart
                        if prevStart > 0:
                            self.partialFM[samplingID][:] = np.add(countsSoFar, np.bincount(letters[0:prevStart], counts[0:prevStart], self.vcLen))
                        else:
                            self.partialFM[samplingID][:] = countsSoFar
                        #print self.refFM[samplingID], self.partialFM[samplingID], (countsSoFar-self.partialFM[0])
                        samplingID += 1
                        
                        
                bwtIndex += prevStart
                if prevStart > 0:
                    countsSoFar += np.bincount(letters[0:prevStart], counts[0:prevStart], self.vcLen)
                    totalCounts += np.sum(np.bincount(letters[0:prevStart], counts[0:prevStart], self.vcLen))
            
        #we'll use this later when we do lookups
        self.offsetSum = np.sum(self.partialFM[0])
        
    def getCharAtIndex(self, index):
        #get the bin we should start from
        binID = index >> self.bitPower
        bwtIndex = self.refFM[binID]
        
        #these are the values that indicate how far in we really are
        trueIndex = np.sum(self.partialFM[binID])-self.offsetSum
        dist = index-trueIndex
        
        #calculate how big of a region we actually need to 'decompress'
        if binID == self.refFM.shape[0]-1:
            endRange = self.bwt.shape[0]
        else:
            endRange = self.refFM[binID+1]+1
            while endRange < self.bwt.shape[0] and (self.bwt[endRange] & self.mask) == (self.bwt[endRange-1] & self.mask):
                endRange += 1
        
        #extract the symbols and counts associated with each byte
        letters = np.bitwise_and(self.bwt[bwtIndex:endRange], self.mask)
        counts = np.right_shift(self.bwt[bwtIndex:endRange], self.letterBits, dtype='<u8')
        
        #numpy methods for find the powers
        i = 1
        same = (letters[0:-1] == letters[1:])
        while np.count_nonzero(same) > 0:
            (counts[i:])[same] *= self.numPower
            i += 1
            same = np.bitwise_and(same[0:-1], same[1:])
        
        #these are the true counts after raising to the appropriate power
        #counts = np.multiply(numbers, (self.numPower ** powers), dtype='<u8')
        
        cs = np.cumsum(counts)
        x = np.searchsorted(cs, dist, 'right')
        return letters[x]
        '''
        ret += np.bincount(letters[0:x], counts[0:x], minlength=self.vcLen)
        
        if x == 0:
            ret[letters[x]] += dist
        elif x < letters.shape[0]:
            ret[letters[x]] += dist-cs[x-1]
        
        return ret
        '''
    
    def getBWTRange(self, start, end):
        #set aside an array block to fill
        startBlockIndex = start >> self.bitPower
        endBlockIndex = int(math.floor(float(end)/self.binSize))
        trueStart = startBlockIndex*self.binSize
        
        #first we will extract the range of blocks
        return self.decompressBlocks(startBlockIndex, endBlockIndex)[start-trueStart:end-trueStart]
    
    def decompressBlocks(self, startBlock, endBlock):
        '''
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
        
        #split the letters and numbers in the compressed bwt
        letters = np.bitwise_and(self.bwt[startRange:endRange], self.mask)
        counts = np.right_shift(self.bwt[startRange:endRange], self.letterBits, dtype='<u8')
        
        #multiply counts where needed
        i = 1
        same = (letters[0:-1] == letters[1:])
        while np.count_nonzero(same) > 0:
            (counts[i:])[same] *= self.numPower
            i += 1
            same = np.bitwise_and(same[0:-1], same[1:])
        
        #now I have letters and counts, time to fill in the array
        s = 0
        lInd = 0
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
    
    def getOccurrenceOfCharAtIndex(self, char, index):
        '''
        This functions gets the FM-index value of a character at the specified position
        @param char - the character to find the occurrence level
        @param index - the index we want to find the occurrence level at
        @return - the number of occurrences of char before the specified index
        '''
        return int(self.getFullFMAtIndex(index)[char])
    
    def getFullFMAtIndex(self, index):
        '''
        This function creates a complete FM-index for a specific position in the BWT.  Example using the above example:
        BWT    Full FM-index
                 $ A C G T
        C        0 0 1 0 0
        $        1 0 1 0 0
        C        1 0 2 0 0
        A        1 1 2 0 0
        @return - the above information in the form of an array
        '''
        if index == self.totalSize:
            return np.cumsum(self.totalCounts)
        
        #get the bin we start from
        binID = index >> self.bitPower
        bwtIndex = self.refFM[binID]
        
        #figure out how far in we really are
        ret = np.copy(self.partialFM[binID])
        trueIndex = np.sum(ret)-self.offsetSum
        dist = index-trueIndex
        if dist == 0:
            return ret
        
        #find the end of the region of interest
        if binID == self.refFM.shape[0]-1:
            endRange = self.bwt.shape[0]
        else:
            endRange = self.refFM[binID+1]+1
            while endRange < self.bwt.shape[0] and (self.bwt[endRange] & self.mask) == (self.bwt[endRange-1] & self.mask):
                endRange += 1
        
        #split the letters and numbers in the compressed bwt
        letters = np.bitwise_and(self.bwt[bwtIndex:endRange], self.mask)
        counts = np.right_shift(self.bwt[bwtIndex:endRange], self.letterBits, dtype='<u8')
        
        i = 1
        same = (letters[0:-1] == letters[1:])
        while np.count_nonzero(same) > 0:
            (counts[i:])[same] *= self.numPower
            i += 1
            same = np.bitwise_and(same[0:-1], same[1:])
        
        cs = np.subtract(np.cumsum(counts), counts)
        x = np.searchsorted(cs, dist, 'left')
        if x > 1:
            ret += np.bincount(letters[0:x-1], counts[0:x-1], minlength=self.vcLen)
        ret[letters[x-1]] += dist-cs[x-1]
        
        return ret
    
    '''
    def decompressBWT(self, outputFN):
        #TODO: make recovery smarter and move to MSBWTGen
        if self.fn == None:
            print 'Must load a BWT first'
        else:
            letterBits = 3
            numberBits = 8-letterBits
            numPower = 2**numberBits
            mask = 255 >> numberBits
            
            letters = np.bitwise_and(self.bwt, mask)
            counts = np.right_shift(self.bwt, letterBits)
            powers = np.zeros(dtype='<u1', shape=(self.bwt.shape[0],))
            
            i = 1
            same = (letters[0:-1] == letters[1:])
            while np.sum(same) > 0:
                (powers[i:])[same] += 1
                i += 1
                same = np.bitwise_and(same[0:-1], same[1:])
            
            size = np.sum(counts*(np.power(numPower, powers, dtype='<u4')))
            ret = np.lib.format.open_memmap(outputFN, 'w+', '<u1', (size,))
            
            offset = 0
            for i, c in enumerate(letters):
                tot = counts[i]*(numPower**powers[i])
                ret[offset:offset+tot] = c
                offset += tot
            
            return ret
    '''
    
def loadBWT(bwtDir, logger=None):
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
        
    #join into one massive string
    seqCopy = ''.join(seqCopy)
    
    #convert the sequences into uint8s and then save it
    seqCopy = np.fromstring(seqCopy, dtype='<u1')
    
    MSBWTGen.writeSeqsToFiles(seqCopy, seqFN, offsetFN, uniformLength)
    MSBWTGen.createFromSeqs(seqFN, offsetFN, mergedDir+'/msbwt.npy', numProcs, areUniform, logger)
        
def createMSBWTFromFastq(fastqFNs, outputDir, numProcs, areUniform, logger):
    '''
    This function takes a fasta filename and creates the BWT using the technique from Cox and Bauer
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
    MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)

def createMSBWTFromBam(bamFNs, outputDir, numProcs, areUniform, logger):
    '''
    This function takes a fasta filename and creates the BWT using the technique from Cox and Bauer
    @param bamFNs - a list of BAM filenames to extract sequences from
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
    MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)

def customiter(numpyArray):
    for x in numpyArray:
        yield tuple(x)

def preprocessFastqs(fastqFNs, seqFNPrefix, offsetFN, abtFN, areUniform, logger):
    '''
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
                #seqArray.append(line.strip('\n')+'$')
                if len(seqArray) == seqsPerFile:
                    if not areUniform or maxSeqLen == -1:
                        maxSeqLen = 0
                        for seq, fID, seqID in seqArray:
                        #for seq in seqArray:
                            if len(seq) > maxSeqLen:
                                maxSeqLen = len(seq)
                    
                    tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
                    subSortFNs.append(tempFN)
                    
                    tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
                    #tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen), (len(seqArray),))
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

'''
def getNextUniqueReads(bamFP):
    ret = []
    try:
        r = bamFP.next()
        qname = r.qname
    except:
        r = None
        qname = None
    while r != None and r.qname == qname:
'''

def preprocessBams(bamFNs, seqFNPrefix, offsetFN, abtFN, areUniform, logger):
    '''
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
        
        #TODO: open the bam file
        bamFile = pysam.Samfile(fn, 'rb')
        i = 0
        
        nr = bamFile.next()
        constantSize = len(nr.seq)
        
        #go through each line
        while nr != None:# and i < 100000:
            #collect all reads that are the same as nr
            aligns = []
            nqname = nr.qname
            while nr != None and nr.qname == nqname:
                aligns.append(nr)
                try:
                    nr = bamFile.next()
                except:
                    nr = None
            
            #print [a.seq for a in aligns]
            
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
                    #reads[1] = a.seq
                    if reads[1] == None:
                        reads[1] = seq
                    elif reads[1] != seq:
                        logger.warning('Two sequences with same flag and different seqs: '+reads[1]+'\n'+str(a))
                else:
                    #reads[0] = a.seq
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

def mergeSeqs(seqArray, mergedFN, numProcs, areUniform, logger):
    '''
    This function takes a series of sequences and creates a big BWT by merging the smaller ones 
    @param seqArray - a list of '$'-terminated strings to be placed into the array
    @param mergedFN - the final destination filename for the merged BWT
    @param numProcs - the number of processors the merge is allowed to create
    '''
    #first wipe away any traces of old information for the case of overwriting a BWT at mergedFN
    MSBWTGen.clearAuxiliaryData(mergedFN)
    
    #create two smaller ones
    midPoint = len(seqArray)/2
    mergedFN1 = mergedFN+'.0.npy'
    mergedFN2 = mergedFN+'.1.npy'
    createMSBWTFromSeqs(seqArray[0:midPoint], mergedFN1, numProcs, areUniform, logger)
    createMSBWTFromSeqs(seqArray[midPoint:], mergedFN2, numProcs, areUniform, logger)
    
    #now do the actual merging
    MSBWTGen.mergeMSBWT(mergedFN, mergedFN1, mergedFN2, numProcs, logger)

def mergeNewSeqs(seqArray, mergedDir, numProcs, areUniform, logger):
    '''
    This function takes a series of sequences and creates a big BWT by merging the smaller ones 
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

def interactiveTranscriptConstruction(bwtDir, seedKmer, endSeeds, threshold, numNodes, direction, logger):
    '''
    This function is intended to be an interactive technique for constructing transcripts
    @param bwtFN - the filename of the BWT to load
    @param seedKmer - the seed sequence to use for construction
    @param threshold - minimum number for a path to be considered a path
    @param direction - True is forward, False is backward
    @param logger - the logger
    @param 
    '''
    kmerLen = len(seedKmer)
    validChars = ['$', 'A', 'C', 'G', 'N', 'T']
    
    pileups = []
    
    logger.info('Loading '+bwtDir+'...')
    msbwt = loadBWT(bwtDir)
    if os.path.exists(bwtDir+'/origins.npy'):
        raise Exception("You haven\'t reimplemented the handling of origin files")
        origins = np.load(bwtDir+'/origins.npy', 'r')
    else:
        origins = None
    
    logger.info('Beginning with seed \''+seedKmer+'\', len='+str(kmerLen))
    
    kmer = seedKmer
    pos = kmerLen
    ret = ''+kmer
    
    #these variable are for counting the average pileup
    totalPileup = 0
    numCovered = 0
    #blockChoices = []
    
    discoveredBlocks = []
    discoveredEdges = []
    pathTups = []
    parentID = -1
    blockID = 0
    
    #mergeThreshold = 1.5
    
    #TODO: make it an input
    #we're stating that 5 reads indicates a path here
    pathThreshold = threshold
    
    foundKmers = {}
    movingAverage = 0
    
    for es in endSeeds:
        foundKmers[es] = 'END_SEED'
    
    terminate = False
    while not terminate and len(discoveredBlocks) < numNodes:
        
        if len(kmer) != kmerLen:
            print 'ERROR: DIFFERENT SIZED K-MER '+str(len(kmer))
            raise Exception('ERROR')
        
        #First, perform all the counts of paths going both forwards and backwards
        counts = {}
        revCounts = {}
        
        maxV = 0
        maxC = ''
        total = 0
        
        numPaths = 0
        numRevPaths = 0
        
        for c in validChars:
            #forward counts
            fr1 = msbwt.findIndicesOfStr(kmer+c)
            fr2 = msbwt.findIndicesOfStr(reverseComplement(kmer+c))
            
            #backward counts
            br1 = msbwt.findIndicesOfStr(c+kmer)
            br2 = msbwt.findIndicesOfStr(reverseComplement(c+kmer))
            
            counts[c] = (fr1[1]-fr1[0])+(fr2[1]-fr2[0])
            revCounts[c] = (br1[1]-br1[0])+(br2[1]-br2[0])
            
            if c != '$':
                total += counts[c]
                if counts[c] > maxV:
                    maxV = counts[c]
                    maxC = c
                
                if counts[c] > pathThreshold:
                    numPaths += 1
                    
                if revCounts[c] > pathThreshold:
                    numRevPaths += 1
            
            if origins != None:
                pass
        
        totalPileup += total
        numCovered += 1
        
        if numRevPaths > 1:
            discoveredBlocks.append((parentID, ret, pileups, 'MERGE_'+str(blockID+1)))
            discoveredEdges.append((blockID, blockID+1, revCounts))
            
            print 'INCOMING MERGE FOUND: '+str(discoveredBlocks[blockID])
            parentID = blockID
            blockID += 1
            
            ret = ''+kmer
            pileups = []
            
        if total == 0:
            print 'No strings found.'
            discoveredBlocks.append((parentID, ret, pileups, 'TERMINAL'))
            
            pileups = []
            
            print pathTups
            print discoveredBlocks
            
            if len(pathTups) == 0:
                terminate = True
            else:
                nextPathTup = pathTups.pop(0)
                print 'Handling1: '+str(nextPathTup)
                parentID = nextPathTup[1]
                direction = nextPathTup[2]
                kmer = nextPathTup[3]
                ret = ''+kmer
                    
                discoveredEdges.append((parentID, blockID+1, nextPathTup[0]))
            
            blockID += 1
            continue
        
        
        #now we identify this kmer as being part of the block
        foundKmers[kmer] = blockID
        r1 = msbwt.findIndicesOfStr(kmer)
        r2 = msbwt.findIndicesOfStr(reverseComplement(kmer))
        kmerCount = (r1[1]-r1[0])+(r2[1]-r2[0])
        pileups.append(kmerCount)
        
        if total == 0:
            perc = 0
        else:
            perc = float(maxV)/total
            
        if numPaths > 1:
            #TODO: reverse ret if direciton is reversed
            discoveredBlocks.append((parentID, ret, pileups, 'SPLIT'))
            
            for c in validChars[1:]:
                if counts[c] > pathThreshold:
                    #counts, parent block, direction, starting seed
                    if direction:
                        pathSeed = kmer[1:]+c
                    else:
                        pathSeed = c+kmer[0:-1]
                    
                    pathTup = (counts[c], blockID, direction, pathSeed)
                    pathTups.append(pathTup)
            
            print pathTups
            print discoveredBlocks
            
            if len(pathTups) == 0:
                terminate = True
            else:
                nextPathTup = pathTups.pop(0)
                print 'Handling2: '+str(nextPathTup)
                parentID = nextPathTup[1]
                direction = nextPathTup[2]
                kmer = nextPathTup[3]
                
                ret = ''+kmer    
                pileups = []
                                
                discoveredEdges.append((parentID, blockID+1, nextPathTup[0]))
            
            blockID += 1
            
        else:
            if direction:
                kmer = kmer[1:]+maxC
                ret += maxC
            else:
                kmer = maxC+kmer[0:-1]
                ret = maxC+ret
            pos += 1
            
            movingAverage = .9*movingAverage+.1*maxV
            print str(pos)+':\t'+kmer+'\t'+str(perc)+'\t'+str(maxV)+'/'+str(total)+'\t'+str(total-maxV)+'\t'+str(movingAverage)
            
        while foundKmers.has_key(kmer) and not terminate:
            #TODO: reverse ret if direciton is reversed
            discoveredBlocks.append((parentID, ret, pileups, 'MERGE_'+str(foundKmers[kmer])))
            discoveredEdges.append((blockID, foundKmers[kmer], ''))
            
            print pathTups
            print discoveredBlocks
            
            if len(pathTups) == 0:
                terminate = True
            else:
                nextPathTup = pathTups.pop(0)
                print 'Handling3: '+str(nextPathTup)
                #pileups.append(nextPathTup[0])
                parentID = nextPathTup[1]
                direction = nextPathTup[2]
                kmer = nextPathTup[3]
                
                ret = ''+kmer
                pileups = []
                    
                discoveredEdges.append((parentID, blockID+1, nextPathTup[0]))
            blockID += 1
        
        '''
        if (kmer in endSeeds):
            print 'Found end seed: '+kmer
            terminate = True
        '''
    
    return (discoveredBlocks, discoveredEdges)
    
def reverseComplement(seq):
    revComp = ''
    complement = {'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N', '$':'$'}
    for c in reversed(seq):
        revComp += complement[c]
    return revComp
    