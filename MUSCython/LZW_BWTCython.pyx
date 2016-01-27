#!python
#cython: boundscheck=False
#cython: wraparound=False

import copy
import math
import os
import time

import numpy as np
cimport numpy as np

cimport BasicBWT

cdef class LZW_BWT(BasicBWT.BasicBWT):
    '''
    This class is a BWT capable of hosting multiple strings inside one structure.  Basically, this would allow you to
    search for a given string across several strings simultaneously.  Note: this class is for the LZW-compressed version,
    for general purposes use the function loadBWT(...) which automatically detects whether this class is correct
    IMPORTANT: THIS CLASS IS NOT THREAD-SAFE DUE TO PRE-ALLOCATED SEARCH ARRAYS
    '''
    cdef unsigned long uncompLen
    cdef np.ndarray offsetArray
    cdef np.uint64_t [:] offsetArray_view
    
    cdef unsigned long searchPreAllocSize
    cdef unsigned long startingBits
    cdef unsigned long startingNumPatterns
    cdef unsigned long startingMask
    
    #all of these arrays are used for searching, we pre-allocate them now and re-use them
    #IMPORTANT: THIS CLASS IS NOT THREAD SAFE AS A RESULT
    cdef np.ndarray lookupList
    cdef np.uint16_t [:] lookupList_view
    cdef np.ndarray symbolList
    cdef np.uint8_t [:] symbolList_view
    cdef np.ndarray countsList
    cdef np.uint16_t [:] countsList_view
    cdef np.ndarray symbolCount
    cdef np.uint16_t [:] symbolCount_view
    cdef np.ndarray offsetList
    cdef np.uint16_t [:] offsetList_view
    cdef np.ndarray w0List
    cdef np.uint16_t [:] w0List_view
    
    cdef double totalTime
    def getTotalTime(LZW_BWT self):
        return self.totalTime
    
    def getCompSize(LZW_BWT self):
        return self.bwt.shape[0]
    
    def loadMsbwt(LZW_BWT self, char * dirName, bint useMemmap=True, logger=None):
        '''
        This functions loads a BWT file and constructs total counts, indexes start positions, and constructs an FM index
        on disk if it doesn't already exist
        @param dirName - the filename to load
        '''
        #open the file with our BWT in it
        self.dirName = dirName
        if useMemmap:
            self.bwt = np.memmap(self.dirName+'/comp_msbwt.dat', dtype='<u1', mode='r+')
        else:
            self.bwt = np.fromfile(self.dirName+'/comp_msbwt.dat', dtype='<u1')
        self.bwt_view = self.bwt
        self.offsetArray = np.load(self.dirName+'/comp_offsets.npy', 'r+')
        self.offsetArray_view = self.offsetArray
        
        #determine bin info
        self.bitPower = self.bwt_view[0]
        self.binSize = 2**self.bitPower
        self.totalTime = 0
        
        cdef unsigned long basesCovered = 0
        cdef unsigned long baseLen = 1
        cdef unsigned long y
        self.searchPreAllocSize = 1
        while basesCovered < self.binSize:
            for y in range(0, self.vcLen**(baseLen+1)):
                basesCovered += baseLen
                self.searchPreAllocSize += 1
                if basesCovered >= self.binSize:
                    break
            baseLen += 1
        #self.searchPreAllocSize *= 2
        
        #figure out starting stuff for all searches
        self.startingBits = 1
        self.startingNumPatterns = 2
        self.startingMask = 1
        while self.startingNumPatterns < self.vcLen:
            self.startingNumPatterns = (self.startingNumPatterns << 1)
            self.startingBits += 1
            self.startingMask = (self.startingMask << 1)+1
        
        if self.startingBits > 8:
            raise Exception('Cannot handle starting with more than 8 bits per symbol')
        
        #calculate the total number of symbols
        cdef unsigned long x
        self.uncompLen = 0
        for x in range(1, 9):
            self.uncompLen = self.uncompLen << 8
            self.uncompLen += self.bwt_view[x]
        
        #pre-allocate
        self.lookupList = np.empty(dtype='<u2', shape=(self.searchPreAllocSize, ))
        self.lookupList_view = self.lookupList
        self.symbolList = np.empty(dtype='<u1', shape=(self.searchPreAllocSize, ))
        self.symbolList_view = self.symbolList
        self.countsList = np.empty(dtype='<u2', shape=(self.searchPreAllocSize, ))
        self.countsList_view = self.countsList
        self.symbolCount = np.empty(dtype='<u2', shape=(self.searchPreAllocSize, ))
        self.symbolCount_view = self.symbolCount
        self.offsetList = np.empty(dtype='<u2', shape=(self.searchPreAllocSize, ))
        self.offsetList_view = self.offsetList
        self.w0List = np.empty(dtype='<u2', shape=(self.searchPreAllocSize, ))
        self.w0List_view = self.w0List
        
        #preset all our start values
        for x in range(0, self.vcLen):
            self.lookupList_view[x] = x
            self.symbolList_view[x] = x
            self.countsList_view[x] = 1
            self.symbolCount_view[x] = 0
            self.offsetList_view[x] = 0
            self.w0List_view[x] = x
            
        #we use a modified auxiliary construction that will do FM counts->totalCounts->startIndices->FM indices
        #it's faster b/c only one pass over full BWT which is important since we have to decomp it to calculate
        self.constructAuxiliary(logger)
        
        if os.path.exists(self.dirName+'/lcps.npy'):
            self.lcpsPresent = True
            self.lcps = np.load(self.dirName+'/lcps.npy', 'r+')
            self.lcps_view = self.lcps
        else:
            self.lcpsPresent = False
        
    def constructAuxiliary(LZW_BWT self, logger):
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
        fmIndexFN = self.dirName+'/lzw_fmIndex.npy'
        totalCountsFN = self.dirName+'/totalCounts.npy'
        
        cdef np.ndarray[np.uint64_t, ndim=1] counts
        cdef np.uint64_t [:] counts_view
        cdef unsigned long i, j
        cdef unsigned long numSamples
        
        cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] tmpBin
        cdef np.uint8_t [:] tmpBin_view
        
        if os.path.exists(fmIndexFN):
            self.partialFM = np.load(fmIndexFN, 'r+')
            self.partialFM_view = self.partialFM
            
            if os.path.exists(totalCountsFN):
                self.totalCounts = np.load(totalCountsFN, 'r+')
                self.totalCounts_view = self.totalCounts
            else:
                if logger != None:
                    logger.info('First time calculation of \'%s\'' % fmIndexFN)
                numSamples = int(math.ceil(1.0*self.uncompLen/self.binSize))
                self.totalCounts = np.zeros(dtype='<u8', shape=(self.vcLen, ))
                self.totalCounts_view = self.totalCounts
                
                #go through each bin
                for i in range(0, numSamples):
                    tmpBin = self.decompressBin(i)
                    tmpBin_view = tmpBin
                    
                    with nogil:
                        #go through each value in the decompressed bin
                        for j in range(0, tmpBin.shape[0]):
                            self.totalCounts_view[tmpBin_view[j]] += 1
                
                np.save(totalCountsFN, self.totalCounts)
            
            self.constructIndexing()
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % fmIndexFN)
            
            #pre-allocate space
            numSamples = int(math.ceil(1.0*self.uncompLen/self.binSize))
            self.partialFM = np.lib.format.open_memmap(fmIndexFN, 'w+', '<u8', (numSamples+1, self.vcLen))
            self.partialFM_view = self.partialFM
            
            #now perform each count and store it to disk
            counts = np.zeros(dtype='<u8', shape=(self.vcLen,))
            counts_view = counts
            
            for i in range(1, self.partialFM.shape[0]):
                #decompress each bin
                tmpBin = self.decompressBin(i-1)
                tmpBin_view = tmpBin
                
                with nogil:
                    #bincount the decompressed bin
                    for j in range(0, tmpBin.shape[0]):
                        counts_view[tmpBin_view[j]] += 1
                    
                    #write the counts
                    for j in range(0, self.vcLen):
                        self.partialFM_view[i][j] = counts_view[j]
            
            #now lets set up the totalCounts array
            if logger != None:
                logger.info('First time calculation of \'%s\'' % totalCountsFN)
            
            self.totalCounts = np.zeros(dtype='<u8', shape=(self.vcLen, ))
            self.totalCounts_view = self.totalCounts
            for i in range(0, self.vcLen):
                self.totalCounts_view[i] = self.partialFM_view[numSamples][i]
            
            np.save(totalCountsFN, self.totalCounts)
            
            #finally, correct the fm-index to be offsets, not counts
            self.constructIndexing()
            with nogil:
                for i in range(0, numSamples+1):
                    for j in range(0, self.vcLen):
                        self.partialFM_view[i][j] += self.startIndex_view[j]
        
        self.totalSize = np.sum(self.totalCounts)
    
    cpdef np.ndarray[np.uint8_t, ndim=1, mode='c'] decompressBin(LZW_BWT self, unsigned long binID):
        cdef unsigned long knownLength = min(self.binSize*(binID+1), self.uncompLen)-self.binSize*binID
        cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] ret = np.empty(dtype='<u1', shape=(knownLength, ))
        cdef np.uint8_t [:] ret_view = ret
        self.fillBin(ret_view, binID)
        return ret
    
    cdef void fillBin(LZW_BWT self, np.uint8_t [:] binToFill, unsigned long binID) nogil:
        #extract inputs
        cdef np.uint8_t [:] compressed_view = self.bwt_view[self.offsetArray_view[binID]:self.offsetArray_view[binID+1]]
        
        cdef unsigned long knownLength = min(self.binSize*(binID+1), self.uncompLen)-self.binSize*binID
        #cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] ret = np.empty(dtype='<u1', shape=(knownLength, ))
        #cdef np.uint8_t [:] ret_view = ret
        
        # Build the dictionary.
        cdef unsigned long dict_size = self.vcLen
        #cdef unsigned long numSymbols = dict_size
        
        cdef unsigned long x, z
        '''
        #this is handled in our constructor
        for x in range(0, dict_size):
            self.lookupList_view[x] = x
            self.symbolList_view[x] = x
            self.offsetList_view[x] = 0
        '''
        cdef unsigned long bitsPerSymbol = self.startingBits
        #cdef unsigned long numPatterns = self.startingNumPatterns
        cdef unsigned long mask = self.startingMask
        
        cdef unsigned long currByte = compressed_view[0]
        cdef unsigned long compressedIndex = 1
        cdef unsigned long currByteUse = 8-bitsPerSymbol
        
        #set up the first value
        #ret_view[0] = (currByte & (0xFF >> (8-bitsPerSymbol)))
        binToFill[0] = currByte & mask
        #currByteUse -= bitsPerSymbol
        currByte = currByte >> bitsPerSymbol
        
        #mark how many values we found, init to 1 obviously
        cdef unsigned long foundVals = 1
        cdef unsigned long prevK = binToFill[0]
        
        #we will set values in the next slot as if it were a special case
        self.symbolList_view[dict_size] = prevK
        self.lookupList_view[dict_size] = prevK
        self.offsetList_view[dict_size] = 1
        
        #with nogil:
        while foundVals < knownLength:
            #extract bytes until we have enough to make a k value
            while currByteUse < bitsPerSymbol:
                currByte |= ((<unsigned long>compressed_view[compressedIndex]) << currByteUse)
                compressedIndex += 1
                currByteUse += 8
            
            #get the 'k' value and remove bits from our buffer
            k = currByte & mask
            currByteUse -= bitsPerSymbol
            currByte = currByte >> bitsPerSymbol
            
            #iterate through filling in values reversed
            z = k
            while z >= self.vcLen:
                binToFill[foundVals+self.offsetList_view[z]] = self.symbolList_view[z]
                z = self.lookupList_view[z]
            binToFill[foundVals] = z
            foundVals += self.offsetList_view[k]+1
            
            #now build the next entry based on what we found
            self.symbolList_view[dict_size] = z
            self.lookupList_view[dict_size] = prevK
            self.offsetList_view[dict_size] = self.offsetList_view[prevK]+1
            dict_size += 1
            
            #also build an entry for the special case of a double up aka, w = entry+[w[0]]
            self.symbolList_view[dict_size] = z
            self.lookupList_view[dict_size] = k
            self.offsetList_view[dict_size] = self.offsetList_view[k]+1
            
            #check if we need to increase the number of bits per value, happens on powers of 2 obviously
            #if dict_size >= numPatterns:
            if (dict_size >> bitsPerSymbol):
                #numPatterns = (numPatterns << 1)
                bitsPerSymbol += 1
                mask = (mask << 1)+1
            
            #save which 'k' was this round
            prevK = k
        
        #return the numpy array
        #return ret
    
    cpdef unsigned long getCharAtIndex(LZW_BWT self, unsigned long index):# nogil:
        '''
        Used for searching, this function masks the complexity behind retrieving a specific character at a specific index
        in our compressed BWT.
        @param index - the index to retrieve the character from
        @param return - return the character in our BWT that's at a particular index (integer format)
        '''
        #get the bin we should start from
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long binPos = index - (binID << self.bitPower)
        
        #make a view for the region we care about
        cdef np.uint8_t [:] compressed_view = self.bwt_view[self.offsetArray_view[binID]:self.offsetArray_view[binID+1]]
        
        # Build the dictionary.
        cdef unsigned long dict_size = self.vcLen
        #cdef unsigned long numSymbols = dict_size
        
        cdef unsigned long x, z
        '''
        #this is now handled in the constructor
        for x in range(0, dict_size):
            self.lookupList_view[x] = x
            self.symbolList_view[x] = x
            self.countsList_view[x] = 1
            self.w0List_view[x] = x
        '''
        
        cdef unsigned long bitsPerSymbol = self.startingBits
        cdef unsigned long mask = self.startingMask
        
        cdef unsigned long currByte = compressed_view[0]
        cdef unsigned long compressedIndex = 1
        cdef unsigned long currByteUse = 8-bitsPerSymbol
        
        cdef unsigned long currIndex = 0
        cdef unsigned long prevK = currByte & mask
        
        currByte = currByte >> bitsPerSymbol
        
        #we will set values in the next slot as if it were a special case
        #self.symbolList_view[dict_size] = prevK
        #self.lookupList_view[dict_size] = prevK
        #self.countsList_view[dict_size] = 2
        self.w0List_view[dict_size] = prevK
        
        while currIndex < binPos:
            #extract bytes until we have enough to make a k value
            while currByteUse < bitsPerSymbol:
                currByte |= ((<unsigned long>compressed_view[compressedIndex]) << currByteUse)
                compressedIndex += 1
                currByteUse += 8
            
            #get the 'k' value and remove bits from our buffer
            k = currByte & mask
            currByteUse -= bitsPerSymbol
            currByte = currByte >> bitsPerSymbol
            
            #used to iterate through filling in values reversed, now just use lookup for 10x speedup
            z = self.w0List_view[k]
            
            #now build the next entry based on what we found
            self.symbolList_view[dict_size] = z
            self.lookupList_view[dict_size] = prevK
            self.countsList_view[dict_size] = self.countsList_view[prevK]+1
            self.w0List_view[dict_size] = self.w0List_view[prevK]
            
            #update how many symbols we've seen and the size of the dictionary
            currIndex += self.countsList_view[k]
            dict_size += 1
            
            #also build an entry for the special case of a double up aka, w = entry+[w[0]]
            #self.symbolList_view[dict_size] = z #doesn't get used before being set again
            #self.lookupList_view[dict_size] = k #doesn't get used before being set again
            #self.countsList_view[dict_size] = self.countsList_view[k]+1 #after moving the check for this, we don't need it set
            self.w0List_view[dict_size] = z
            
            #check if we need to increase the number of bits per value, happens on powers of 2 obviously
            if (dict_size >> bitsPerSymbol):
                bitsPerSymbol += 1
                mask = (mask << 1)+1
            
            #save which 'k' was this round
            prevK = k
    
        z = prevK
        #while currIndex > binPos:
        for x in range(currIndex, binPos, -1):
            z = self.lookupList_view[z]
            #currIndex -= 1
        
        #return the numpy array
        return <unsigned long> self.symbolList_view[z]
        
    cpdef unsigned long getOccurrenceOfCharAtIndex(LZW_BWT self, unsigned long sym, unsigned long index):# nogil:
        '''
        This functions gets the FM-index value of a character at the specified position
        @param sym - the character to find the occurrence level
        @param index - the index we want to find the occurrence level at
        @return - the number of occurrences of char before the specified index
        '''
        if index >= self.totalSize:
            return self.endIndex_view[sym]
        
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long binPos = index - (binID << self.bitPower)
        cdef unsigned long ret = self.partialFM_view[binID][sym]
        
        # Build the dictionary.
        cdef unsigned long dict_size = self.vcLen
        cdef unsigned long x, z
        '''
        #now handled in the constructor
        for x in range(0, dict_size):
            self.lookupList_view[x] = x
            self.symbolList_view[x] = x
            self.countsList_view[x] = 1
            self.symbolCount_view[x] = 0
            self.w0List_view[x] = x
        '''
        
        #this is search dependent, so we set it here and clear at the end of the call
        self.symbolCount_view[sym] = 1
            
        cdef unsigned long bitsPerSymbol = self.startingBits
        cdef unsigned long mask = self.startingMask
        
        #cdef unsigned long currByte = compressed_view[0]
        cdef unsigned long currByte = self.bwt_view[self.offsetArray_view[binID]]
        #cdef unsigned long compressedIndex = 1
        cdef unsigned long compressedIndex = self.offsetArray_view[binID]+1
        cdef unsigned long currByteUse = 8-bitsPerSymbol
        
        cdef unsigned long currIndex = 0
        cdef unsigned long prevK = currByte & mask
        
        currByte = currByte >> bitsPerSymbol
        
        #we will set values in the next slot as if it were a special case
        self.w0List_view[dict_size] = prevK
        
        while currIndex < binPos:
            #extract bytes until we have enough to make a k value
            while currByteUse < bitsPerSymbol:
                currByte |= ((<unsigned long>self.bwt_view[compressedIndex]) << currByteUse)
                compressedIndex += 1
                currByteUse += 8
                
            #get the 'k' value and remove bits from our buffer
            k = currByte & mask
            currByteUse -= bitsPerSymbol
            currByte = currByte >> bitsPerSymbol
            
            #used to iterate through filling in values reversed, now we just use a lookup and it's about 10x faster
            z = self.w0List_view[k]
            
            #now build the next entry based on what we found
            self.lookupList_view[dict_size] = prevK
            self.countsList_view[dict_size] = self.countsList_view[prevK]+1
            self.symbolCount_view[dict_size] = self.symbolCount_view[prevK]+(z == sym)
            self.w0List_view[dict_size] = self.w0List_view[prevK]
            
            #update how many symbols we've seen and the size of the dictionary
            currIndex += self.countsList_view[k]
            dict_size += 1
            
            #also build an entry for the special case of a double up aka, w = entry+[w[0]]
            self.w0List_view[dict_size] = z
            
            #check if we need to increase the number of bits per value, happens on powers of 2 obviously
            if (dict_size >> bitsPerSymbol):
                bitsPerSymbol += 1
                mask = (mask << 1)+1
            
            #save which 'k' was this round
            ret += self.symbolCount_view[prevK]
            prevK = k
        
        #unfortunately can't do a fast lookup here
        z = prevK
        for x in range(currIndex, binPos, -1):
            z = self.lookupList_view[z]
        
        if z >= self.vcLen:
            ret += self.symbolCount_view[self.lookupList_view[z]]
        
        #clear this for future searches
        self.symbolCount_view[sym] = 0
        return ret
    
    def getFullFMAtIndex(LZW_BWT self, unsigned long index):# nogil:
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.empty(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] ret_view = ret
        self.fillFmAtIndex(ret_view, index)
        return ret
        
    cdef void fillFmAtIndex(LZW_BWT self, np.uint64_t [:] ret_view, unsigned long index):
        '''
        This functions fills in the FM-index value of all characters at the specified position
        @param index - the index we want to find the occurrence level at
        @return - fills in ret_view with the index
        '''
        cdef unsigned long x, z
        if index >= self.totalSize:
            for x in range(0, self.vcLen):
                ret_view[x] = self.endIndex_view[x]
            return
        
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long binPos = index - (binID << self.bitPower)
        
        for x in range(0, self.vcLen):
            ret_view[x] = self.partialFM_view[binID][x]
        
        # Build the dictionary.
        cdef unsigned long dict_size = self.vcLen
        '''
        #now handled in the constructor
        for x in range(0, dict_size):
            self.lookupList_view[x] = x
            self.symbolList_view[x] = x
            self.countsList_view[x] = 1
            self.symbolCount_view[x] = 0
            self.w0List_view[x] = x
        '''
        cdef unsigned long bitsPerSymbol = self.startingBits
        cdef unsigned long mask = self.startingMask
        
        cdef unsigned long currByte = self.bwt_view[self.offsetArray_view[binID]]
        cdef unsigned long compressedIndex = self.offsetArray_view[binID]+1
        cdef unsigned long currByteUse = 8-bitsPerSymbol
        
        cdef unsigned long currIndex = 0
        cdef unsigned long prevK = currByte & mask
        
        currByte = currByte >> bitsPerSymbol
        
        #we will set values in the next slot as if it were a special case
        self.w0List_view[dict_size] = prevK
        
        while currIndex < binPos:
            #extract bytes until we have enough to make a k value
            while currByteUse < bitsPerSymbol:
                currByte |= ((<unsigned long>self.bwt_view[compressedIndex]) << currByteUse)
                compressedIndex += 1
                currByteUse += 8
                
            #get the 'k' value and remove bits from our buffer
            k = currByte & mask
            currByteUse -= bitsPerSymbol
            currByte = currByte >> bitsPerSymbol
            
            #used to iterate through filling in values reversed, now we just use a lookup and it's about 10x faster
            z = prevK
            while z >= self.vcLen:
                ret_view[self.symbolList_view[z]] += 1
                z = self.lookupList_view[z]
            ret_view[self.symbolList_view[z]] += 1
            
            z = self.w0List_view[k]
            
            #now build the next entry based on what we found
            self.symbolList_view[dict_size] = z
            self.lookupList_view[dict_size] = prevK
            self.countsList_view[dict_size] = self.countsList_view[prevK]+1
            self.w0List_view[dict_size] = self.w0List_view[prevK]
            
            #update how many symbols we've seen and the size of the dictionary
            currIndex += self.countsList_view[k]
            dict_size += 1
            
            #also build an entry for the special case of a double up aka, w = entry+[w[0]]
            self.w0List_view[dict_size] = z
            
            #check if we need to increase the number of bits per value, happens on powers of 2 obviously
            if (dict_size >> bitsPerSymbol):
                bitsPerSymbol += 1
                mask = (mask << 1)+1
            
            #save which 'k' was this round
            #ret += self.symbolCount_view[prevK]
            prevK = k
        
        z = prevK
        while z >= self.vcLen:
            if currIndex < binPos:
                ret_view[self.symbolList_view[z]] += 1
            z = self.lookupList_view[z]
            currIndex -= 1
        
        if currIndex < binPos:
            ret_view[self.symbolList_view[z]] += 1
          
    cpdef iterInit(LZW_BWT self):
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
    
    cpdef iterNext(LZW_BWT self):
        return self.iterNext_cython()
    
    cdef np.uint8_t iterNext_cython(LZW_BWT self) nogil:
        '''
        dummy function, override in all subclasses
        '''
        return 255
    
    