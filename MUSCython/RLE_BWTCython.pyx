#!python
#cython: boundscheck=False
#cython: wraparound=False

import math
import numpy as np
cimport numpy as np
import os

cimport BasicBWT
from cython.operator cimport preincrement as inc

cdef class RLE_BWT(BasicBWT.BasicBWT):
    '''
    This structure inherits from the BasicBWT and includes several functions with identical functionality to the ByteBWT
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
    
    def getCompSize(RLE_BWT self):
        return self.bwt.shape[0]
    
    def loadMsbwt(RLE_BWT self, char * dirName, bint useMemmap=True, logger=None):
        '''
        This functions loads a BWT file and constructs total counts, indexes start positions, and constructs an FM index in memory
        @param dirName - the directory to load, inside should be '<DIR>/comp_msbwt.npy' or it will fail
        '''
        #open the file with our BWT in it
        self.dirName = dirName
        if useMemmap:
            self.bwt = np.load(self.dirName+'/comp_msbwt.npy', 'r+')
        else:
            self.bwt = np.load(self.dirName+'/comp_msbwt.npy')
        self.bwt_view = self.bwt
        
        #build auxiliary structures
        self.constructTotalCounts(logger)
        self.constructIndexing()
        self.constructFMIndex(logger)
    
    def constructTotalCounts(RLE_BWT self, logger):
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
    
    def constructFMIndex(RLE_BWT self, logger):
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
        
    cpdef unsigned long getCharAtIndex(RLE_BWT self, unsigned long index):# nogil:
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
        
        cdef unsigned long prevChar = self.bwt_view[bwtIndex] & self.mask
        cdef unsigned long currentChar
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
    
    cdef void fillBin(RLE_BWT self, np.uint8_t [:] binToFill, unsigned long binID) nogil:
        '''
        This copies a slice of the BWT into an array the caller can manipulate, useful mostly for the compressed
        versions of this data structure.
        @param binToFill - the place we can copy the BWT into
        @param binID - the bin we're copying
        '''
        cdef unsigned long x
        cdef unsigned long startIndex = binID*self.binSize
        cdef unsigned long endIndex = min((binID+1)*self.binSize, self.totalSize)
        
        #get the bin we should start from
        cdef unsigned long bwtIndex = self.refFM_view[binID]
        
        #these are the values that indicate how far in we really are
        cdef unsigned long trueIndex = 0
        cdef unsigned long i
        for i in range(0, self.vcLen):
            trueIndex += self.partialFM_view[binID][i]
        trueIndex -= self.offsetSum
        
        cdef unsigned long prevChar = self.bwt_view[bwtIndex] & self.mask
        cdef unsigned long currentChar
        cdef unsigned long prevCount = self.bwt_view[bwtIndex] >> self.letterBits
        cdef unsigned powerMultiple = 1
        
        #first, we may need to skip ahead some
        while trueIndex + prevCount < startIndex:
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
        
        #now we actually do the same loop only with some writes
        cdef unsigned long realStart = 0
        while trueIndex + prevCount < endIndex:
            #first fill in any values from this bin
            for x in range(realStart, trueIndex+prevCount-startIndex):
                binToFill[x] = prevChar
            realStart = trueIndex+prevCount-startIndex
            
            #now do normal upkeep stuff
            trueIndex += prevCount
            inc(bwtIndex)
            
            #pull out the char and update powers/counts
            currentChar = self.bwt_view[bwtIndex] & self.mask
            if currentChar == prevChar:
                powerMultiple *= self.numPower
                prevCount = (self.bwt_view[bwtIndex] >> self.letterBits) * powerMultiple
            else:
                powerMultiple = 1
                prevCount = self.bwt_view[bwtIndex] >> self.letterBits
            prevChar = currentChar
        
        #finally fill in the remaining stuff
        for x in range(realStart, endIndex-startIndex):
            binToFill[x] = prevChar
        
    def getBWTRange(RLE_BWT self, unsigned long start, unsigned long end):
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
    
    def decompressBlocks(RLE_BWT self, unsigned long startBlock, unsigned long endBlock):
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
    
    cpdef unsigned long getOccurrenceOfCharAtIndex(RLE_BWT self, unsigned long sym, unsigned long index):# nogil:
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
    
    def getFullFMAtIndex(RLE_BWT self, np.uint64_t index):
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.empty(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] ret_view = ret
        self.fillFmAtIndex(ret_view, index)
        return ret
        
    cdef void fillFmAtIndex(RLE_BWT self, np.uint64_t [:] ret_view, unsigned long index):
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
        
        #cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.empty(dtype='<u8', shape=(self.vcLen, ))
        #cdef np.uint64_t [:] ret_view = ret
        
        for j in range(0, self.vcLen):
            bwtIndex += self.partialFM_view[binID][j]
            ret_view[j] = self.partialFM_view[binID][j]
        bwtIndex -= self.offsetSum
        
        '''
        cdef np.uint8_t prevChar = 255
        cdef np.uint8_t currentChar
        cdef unsigned long prevCount = 0
        cdef unsigned long powerMultiple = 1
        '''
        cdef np.uint8_t prevChar = self.bwt_view[compressedIndex] & self.mask
        cdef np.uint8_t currentChar
        cdef unsigned long prevCount = self.bwt_view[compressedIndex] >> self.letterBits
        cdef unsigned long powerMultiple = self.numPower
        compressedIndex += 1
        
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
        
        #return ret
    
    cdef np.uint8_t iterNext_cython(RLE_BWT self) nogil:
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