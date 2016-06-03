#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: profile=False

import math
import numpy as np
cimport numpy as np
import os

from libc.stdio cimport FILE, fopen, fwrite, fclose

cimport BasicBWT
import MSBWTGenCython as MSBWTGen
import AlignmentUtil
from cython.operator cimport preincrement as inc

cdef enum:
    letterBits = 3 #defined
    numberBits = 5 #8-letterBits
    numPower = 32  #2**numberBits
    mask = 7       #255 >> numberBits 
    
    bitPower = 11  #defined
    binSize = 2048 #2**self.bitPower
    
    vcLen = 6      #defined

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
    
    cdef bint useMemmapRLE
    
    def getCompSize(RLE_BWT self):
        return self.bwt.shape[0]
    
    def loadMsbwt(RLE_BWT self, char * dirName, bint useMemmap=True, logger=None):
        '''
        This functions loads a BWT file and constructs total counts, indexes start positions, and constructs an FM index in memory
        @param dirName - the directory to load, inside should be '<DIR>/comp_msbwt.npy' or it will fail
        '''
        #open the file with our BWT in it
        self.dirName = dirName
        self.useMemmapRLE = useMemmap
        if useMemmap:
            self.bwt = np.load(self.dirName+'/comp_msbwt.npy', 'r+')
        else:
            self.bwt = np.load(self.dirName+'/comp_msbwt.npy')
        self.bwt_view = self.bwt
        
        #build auxiliary structures
        self.constructTotalCounts(logger)
        self.constructIndexing()
        self.constructFMIndex(logger)
        
        if os.path.exists(self.dirName+'/lcps.npy'):
            self.lcpsPresent = True
            self.lcps = np.load(self.dirName+'/lcps.npy', 'r+')
            self.lcps_view = self.lcps
        else:
            self.lcpsPresent = False
        
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
        
        #self.letterBits = 3
        #self.numberBits = 8-self.letterBits
        #self.numPower = 2**self.numberBits
        #self.mask = 255 >> self.numberBits
        
        cdef str abtFN = self.dirName+'/totalCounts.npy'
        if os.path.exists(abtFN):
            if self.useMemmapRLE:
                self.totalCounts = np.load(abtFN, 'r+')
            else:
                self.totalCounts = np.load(abtFN)
            self.totalCounts_view = self.totalCounts
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % abtFN)
            
            self.totalCounts = np.zeros(dtype='<u8', shape=(vcLen, ))
            self.totalCounts_view = self.totalCounts
            numBytes = self.bwt.shape[0]
            prevChar = 255
            powerMultiple = 1
            
            with nogil:
                for i in range(0, numBytes):
                    currentChar = self.bwt_view[i] & mask
                    if currentChar == prevChar:
                        powerMultiple *= numPower
                    else:
                        powerMultiple = 1
                    prevChar = currentChar
                    
                    currentCount = (self.bwt_view[i] >> letterBits) * powerMultiple
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
        
        cdef str fmIndexFN = self.dirName+'/comp_fmIndex.npy'
        cdef str fmRefFN = self.dirName+'/comp_refIndex.npy'
        
        if os.path.exists(fmIndexFN) and os.path.exists(fmRefFN):
            #both exist, just memmap them
            if self.useMemmapRLE:
                self.partialFM = np.load(fmIndexFN, 'r+')
            else:
                self.partialFM = np.load(fmIndexFN)
            self.partialFM_view = self.partialFM
            
            if self.useMemmapRLE:
                self.refFM = np.load(fmRefFN, 'r+')
            else:
                self.refFM = np.load(fmRefFN)
            self.refFM_view = self.refFM
        else:
            if logger != None:
                logger.info('First time calculation of \'%s\'' % fmIndexFN)
            
            #pre-allocate space
            samplingSize = int(math.ceil(float(self.totalSize+1)/binSize))
            self.partialFM = np.lib.format.open_memmap(fmIndexFN, 'w+', '<u8', (samplingSize, vcLen))
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
                    currentChar = self.bwt_view[i] & mask
                    if currentChar == prevChar:
                        totalCharCount += (self.bwt_view[i] >> letterBits) * powerMultiple
                        powerMultiple *= numPower
                    else:
                        #first save this count
                        while bwtIndex + totalCharCount >= binEnd:
                            self.refFM_view[binID] = prevStart
                            for j in range(0, vcLen):
                                self.partialFM_view[binID,j] = countsSoFar_view[j]
                            binEnd += binSize
                            inc(binID)
                        
                        #add the previous
                        countsSoFar_view[prevChar] += totalCharCount
                        bwtIndex += totalCharCount
                        
                        prevChar = currentChar
                        prevStart = i
                        totalCharCount = (self.bwt_view[i] >> letterBits)
                        powerMultiple = numPower
                
                while bwtIndex + totalCharCount >= binEnd:
                    self.refFM_view[binID] = prevStart
                    for j in range(0, vcLen):
                        self.partialFM_view[binID,j] = countsSoFar_view[j]
                    binEnd += binSize
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
        cdef unsigned long binID = index >> bitPower
        cdef unsigned long bwtIndex = self.refFM_view[binID]
        
        #these are the values that indicate how far in we really are
        cdef unsigned long trueIndex = 0
        cdef unsigned long i
        for i in range(0, vcLen):
            trueIndex += self.partialFM_view[binID,i]
        trueIndex -= self.offsetSum
        
        cdef unsigned long prevChar = self.bwt_view[bwtIndex] & mask
        cdef unsigned long currentChar
        cdef unsigned long prevCount = self.bwt_view[bwtIndex] >> letterBits
        cdef unsigned long powerMultiple = 1
        
        while trueIndex + prevCount <= index:
            trueIndex += prevCount
            bwtIndex += 1
            
            currentChar = self.bwt_view[bwtIndex] & mask
            if currentChar == prevChar:
                powerMultiple *= numPower
                prevCount = (self.bwt_view[bwtIndex] >> letterBits) * powerMultiple
            else:
                powerMultiple = 1
                prevCount = self.bwt_view[bwtIndex] >> letterBits
            
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
        cdef unsigned long startIndex = binID*binSize
        cdef unsigned long endIndex = min((binID+1)*binSize, self.totalSize)
        
        #get the bin we should start from
        cdef unsigned long bwtIndex = self.refFM_view[binID]
        
        #these are the values that indicate how far in we really are
        cdef unsigned long trueIndex = 0
        cdef unsigned long i
        for i in range(0, vcLen):
            trueIndex += self.partialFM_view[binID,i]
        trueIndex -= self.offsetSum
        
        cdef unsigned long prevChar = self.bwt_view[bwtIndex] & mask
        cdef unsigned long currentChar
        cdef unsigned long prevCount = self.bwt_view[bwtIndex] >> letterBits
        cdef unsigned powerMultiple = 1
        
        #first, we may need to skip ahead some
        while trueIndex + prevCount < startIndex:
            trueIndex += prevCount
            #inc(bwtIndex)
            bwtIndex += 1
            
            currentChar = self.bwt_view[bwtIndex] & mask
            if currentChar == prevChar:
                powerMultiple *= numPower
                prevCount = (self.bwt_view[bwtIndex] >> letterBits) * powerMultiple
            else:
                powerMultiple = 1
                prevCount = self.bwt_view[bwtIndex] >> letterBits
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
            #inc(bwtIndex)
            bwtIndex += 1
            
            #pull out the char and update powers/counts
            currentChar = self.bwt_view[bwtIndex] & mask
            if currentChar == prevChar:
                powerMultiple *= numPower
                prevCount = (self.bwt_view[bwtIndex] >> letterBits) * powerMultiple
            else:
                powerMultiple = 1
                prevCount = self.bwt_view[bwtIndex] >> letterBits
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
        startBlockIndex = start >> bitPower
        endBlockIndex = int(math.floor(float(end)/binSize))
        trueStart = startBlockIndex*binSize
        
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
        expectedIndex = startBlock*binSize
        trueIndex = np.sum(self.partialFM[startBlock])-self.offsetSum
        dist = expectedIndex - trueIndex
        
        #find the end of the region of interest
        startRange = self.refFM[startBlock]
        if endBlock >= self.refFM.shape[0]-1:
            endRange = self.bwt.shape[0]
            returnSize = binSize*(endBlock-startBlock)+(self.totalSize % binSize)
        else:
            endRange = self.refFM[endBlock+1]+1
            returnSize = binSize*(endBlock-startBlock+1)
            while endRange < self.bwt.shape[0] and (self.bwt[endRange] & mask) == (self.bwt[endRange-1] & mask):
                endRange += 1
        
        ret = np.zeros(dtype='<u1', shape=(returnSize,))
        
        startRange = int(startRange)
        endRange = int(endRange)
        
        #split the letters and numbers in the compressed bwt
        letters = np.bitwise_and(self.bwt[startRange:endRange], mask)
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] counts = np.right_shift(self.bwt[startRange:endRange], letterBits, dtype='<u8')
        
        #multiply counts where needed
        i = 1
        #same = (letters[0:-1] == letters[1:])
        same = (letters[0:letters.shape[0]-1] == letters[1:])
        while np.count_nonzero(same) > 0:
            (counts[i:])[same] *= numPower
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
        cdef unsigned long binID = index >> bitPower
        cdef unsigned long compressedIndex = self.refFM_view[binID]
        cdef unsigned long bwtIndex = 0
        cdef unsigned long j
        for j in range(0, vcLen):
            bwtIndex += self.partialFM_view[binID,j]
        bwtIndex -= self.offsetSum
            
        cdef unsigned long ret = self.partialFM_view[binID,sym]
        
        cdef np.uint8_t prevChar = 255
        cdef np.uint8_t currentChar
        cdef unsigned long prevCount = 0
        cdef unsigned long powerMultiple = 1
        #cdef unsigned long powerMultiple = 0
        
        while bwtIndex + prevCount < index:
            currentChar = self.bwt_view[compressedIndex] & mask
            if currentChar == prevChar:
                prevCount += (self.bwt_view[compressedIndex] >> letterBits) * powerMultiple
                powerMultiple *= numPower
                #prevCount += <unsigned long>(self.bwt_view[compressedIndex] >> letterBits) << powerMultiple
                #powerMultiple += numberBits
            else:
                if prevChar == sym:
                    ret += prevCount
                
                bwtIndex += prevCount
                prevCount = (self.bwt_view[compressedIndex] >> letterBits)
                prevChar = currentChar
                powerMultiple = numPower
                #powerMultiple = numberBits
                
            compressedIndex += 1
        
        if prevChar == sym:
            ret += index-bwtIndex
        
        return ret
    
    cdef unsigned long getOccurrenceOfCharAtIndex_c(RLE_BWT self, unsigned long sym, unsigned long index):# nogil:
        '''
        This functions gets the FM-index value of a character at the specified position
        @param sym - the character to find the occurrence level
        @param index - the index we want to find the occurrence level at
        @return - the number of occurrences of char before the specified index
        '''
        cdef unsigned long binID = index >> bitPower
        cdef unsigned long compressedIndex = self.refFM_view[binID]
        cdef unsigned long bwtIndex = 0
        cdef unsigned long j
        for j in range(0, vcLen):
            bwtIndex += self.partialFM_view[binID,j]
        bwtIndex -= self.offsetSum
            
        cdef unsigned long ret = self.partialFM_view[binID,sym]
        
        cdef np.uint8_t prevChar = 255
        cdef np.uint8_t currentChar
        cdef unsigned long prevCount = 0
        cdef unsigned long powerMultiple = 1
        #cdef unsigned long powerMultiple = 0
        
        while bwtIndex + prevCount < index:
            currentChar = self.bwt_view[compressedIndex] & mask
            if currentChar == prevChar:
                prevCount += (self.bwt_view[compressedIndex] >> letterBits) * powerMultiple
                powerMultiple *= numPower
                #prevCount += <unsigned long>(self.bwt_view[compressedIndex] >> letterBits) << powerMultiple
                #powerMultiple += numberBits
            else:
                if prevChar == sym:
                    ret += prevCount
                
                bwtIndex += prevCount
                prevCount = (self.bwt_view[compressedIndex] >> letterBits)
                prevChar = currentChar
                powerMultiple = numPower
                #powerMultiple = numberBits
                
            compressedIndex += 1
        
        if prevChar == sym:
            ret += index-bwtIndex
        
        return ret
    
    cdef BasicBWT.bwtRange getOccurrenceOfCharAtRange(RLE_BWT self, unsigned long sym, BasicBWT.bwtRange inRange) nogil:
        '''
        This functions gets the FM-index value of a character at the specified position
        @param sym - the character to find the occurrence level
        @param index - the index we want to find the occurrence level at
        @return - the number of occurrences of char before the specified index
        '''
        cdef unsigned long binID = inRange.l >> self.bitPower
        cdef unsigned long compressedIndex = self.refFM_view[binID]
        cdef unsigned long bwtIndex = 0
        cdef unsigned long j
        for j in range(0, vcLen):
            bwtIndex += self.partialFM_view[binID,j]
        bwtIndex -= self.offsetSum
        
        cdef BasicBWT.bwtRange ret
        ret.l = self.partialFM_view[binID,sym]
        
        cdef np.uint8_t prevChar = 255
        cdef np.uint8_t currentChar
        cdef unsigned long prevCount = 0
        cdef unsigned long powerMultiple = 1
        
        while bwtIndex + prevCount < inRange.l:
            currentChar = self.bwt_view[compressedIndex] & mask
            if currentChar == prevChar:
                prevCount += (self.bwt_view[compressedIndex] >> letterBits) * powerMultiple
                powerMultiple *= numPower
            else:
                if prevChar == sym:
                    ret.l += prevCount
                
                bwtIndex += prevCount
                prevCount = (self.bwt_view[compressedIndex] >> letterBits)
                prevChar = currentChar
                powerMultiple = numPower
                
            compressedIndex += 1
        
        cdef unsigned long tempC = ret.l
        if prevChar == sym:
            ret.l += inRange.l-bwtIndex
        
        cdef unsigned long binID_h = inRange.h >> bitPower
        if binID == binID_h:
            #we can continue, just set this value
            ret.h = tempC
        else:
            #we need to load everything
            compressedIndex = self.refFM_view[binID_h]
            bwtIndex = 0
            for j in range(0, vcLen):
                bwtIndex += self.partialFM_view[binID_h,j]
            bwtIndex -= self.offsetSum
            
            ret.h = self.partialFM_view[binID_h,sym]
            
            prevChar = 255
            prevCount = 0
            powerMultiple = 1
        
        while bwtIndex + prevCount < inRange.h:
            currentChar = self.bwt_view[compressedIndex] & mask
            if currentChar == prevChar:
                prevCount += (self.bwt_view[compressedIndex] >> letterBits) * powerMultiple
                powerMultiple *= numPower
            else:
                if prevChar == sym:
                    ret.h += prevCount
                
                bwtIndex += prevCount
                prevCount = (self.bwt_view[compressedIndex] >> letterBits)
                prevChar = currentChar
                powerMultiple = numPower
                
            compressedIndex += 1
        
        if prevChar == sym:
            ret.h += inRange.h-bwtIndex
        
        return ret
    
    def getFullFMAtIndex(RLE_BWT self, np.uint64_t index):
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.empty(dtype='<u8', shape=(vcLen, ))
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
        cdef unsigned long binID = index >> bitPower
        cdef unsigned long compressedIndex = self.refFM_view[binID]
        cdef unsigned long bwtIndex = 0
        cdef unsigned long j
        
        for j in range(0, vcLen):
            bwtIndex += self.partialFM_view[binID,j]
            ret_view[j] = self.partialFM_view[binID,j]
        bwtIndex -= self.offsetSum
        
        cdef np.uint8_t prevChar = self.bwt_view[compressedIndex] & mask
        cdef np.uint8_t currentChar
        cdef unsigned long prevCount = self.bwt_view[compressedIndex] >> letterBits
        cdef unsigned long powerMultiple = numPower
        compressedIndex += 1
        
        while bwtIndex + prevCount < index:
            currentChar = self.bwt_view[compressedIndex] & mask
            if currentChar == prevChar:
                prevCount += (self.bwt_view[compressedIndex] >> letterBits) * powerMultiple
                powerMultiple *= numPower
            else:
                ret_view[prevChar] += prevCount
                
                bwtIndex += prevCount
                prevCount = (self.bwt_view[compressedIndex] >> letterBits)
                prevChar = currentChar
                powerMultiple = numPower
            
            compressedIndex += 1
            
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
            sym = self.bwt_view[self.iterIndex] & mask
            
            #increment our power if necessary
            if sym == self.iterCurrChar:
                inc(self.iterPower)
            else:
                self.iterCount = 0
                self.iterCurrCount = 0
                self.iterPower = 0
                self.iterCurrChar = sym
                
            #pull out the number of counts here and reset our counting
            self.iterCurrCount += (self.bwt_view[self.iterIndex] >> letterBits) * (numPower**self.iterPower)
            inc(self.iterCount) 
            ret = self.iterCurrChar
            inc(self.iterIndex)
        else:
            #we have run out of stuff
            ret = 255
        
        return ret
    
    def removeStrings(RLE_BWT self, set delSet, bint reloadBWT=True, logger=None):
        '''
        This function removes a set of readIDs from the BWT along with all their associated bases
        IMPORTANT: THE BWT MUST BE RELOADED AFTER CALLING THIS FUNCTION BECAUSE INDICES NEED TO BE REBUILT
        @param delSet - a set of readIDs to delete based on their position in the BWT
        @param reloadBWT - if true (default), it will reload the BWT in place
        @return None
        '''
        if len(delSet) == 0:
            return
        
        #make sure we are actually modifying the file
        if not self.useMemmapRLE:
            self.bwt = np.load(self.dirName+'/comp_msbwt.npy', 'r+')
            self.bwt_view = self.bwt
        
        #first, go through each id in the set and pull out the indices of all associated bases to add to a master deletion list
        #make sure that each index is a readID as well prior to adding it
        cdef unsigned long readID
        
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] tempIndexArray = np.zeros(dtype='<u8', shape=(1, ))
        cdef np.uint64_t [:] tempIndexArray_view = tempIndexArray
        
        cdef str deletionFN = self.dirName+'/deletion_indices.dat'
        cdef FILE * fp = fopen(deletionFN, 'w+')
        
        cdef unsigned long x, copyIndex
        cdef np.uint8_t indexByte
        
        if logger != None:
            logger.info('Identifying indices for deletion...')
        
        cdef unsigned long prevSym, currIndex
        
        for readID in delSet:
            #verify this is a valid read ID
            if readID >= self.totalCounts_view[0]:
                raise Exception(str(readID)+' is too large to be a string ID in BWT with '+str(self.totalCounts_view[0])+' strings.')
            
            #now pull out all the associated indices
            #figure out the first hop backwards
            prevSym = self.getCharAtIndex(readID)
            currIndex = self.getOccurrenceOfCharAtIndex(prevSym, readID)
            
            #while we haven't looped back to the start
            while currIndex != readID:
                #write the index
                #fwrite(&currIndex, 8, 1, fp)
                copyIndex = currIndex
                for x in range(0, 8):
                    indexByte = copyIndex & 0xFF
                    fwrite(&indexByte, 1, 1, fp)
                    copyIndex = copyIndex >> 8
                
                #figure out where to go from here
                prevSym = self.getCharAtIndex(currIndex)
                currIndex = self.getOccurrenceOfCharAtIndex(prevSym, currIndex)
            
            #write the read ID, which was the first index we found
            fwrite(&readID, 8, 1, fp)
            
        fclose(fp)
        
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] delIndices = np.memmap(deletionFN, dtype='<u8')
        
        if logger != None:
            logger.info('Sorting indices for deletion...')
            
        delIndices.sort()
        #second, sort the list and then go through in order modifying our file for our compressed BWT
        #deletionIndices.sort()
        #cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] delIndices = np.array(deletionIndices, dtype='<u8')
        cdef np.uint64_t [:] delIndices_view = delIndices
        cdef unsigned long currDel = 0
        cdef unsigned long totalDels = delIndices.shape[0]
        
        #go through each entry in the BWT
        cdef unsigned long numBytes = self.bwt.shape[0]
        
        #currentChar is the symbol we just read, prevChar is in the byte right before it
        cdef np.uint8_t currentChar
        cdef np.uint8_t prevChar = 255
        
        #the position we are currently writing
        cdef unsigned long writeByte = 0
        
        #the last run symbol and the last run count
        cdef np.uint8_t lastRunSym = 255
        cdef unsigned long lastRunCount = 0
        
        #these values are related to the current run we're looking at
        cdef unsigned long currentCount = 0
        cdef unsigned long powerMultiple = 1
        cdef unsigned long totalCounted = 0
        cdef np.uint8_t NUM_MASK = (0xFF >> letterBits)#0x1F
        
        if logger != None:
            logger.info('Deleting indices...')
        
        #i = the read byte index for this part of the code
        cdef unsigned long i, j
        
        #go through each byte in the BWT
        for i in range(0, numBytes):
            #figure out which character is in this run
            currentChar = self.bwt_view[i] & mask
            
            #compare the currentChar to the prevChar, tells us if a run is going on
            if currentChar == prevChar:
                #this is a continuation, so increment the power
                powerMultiple *= numPower
            else:
                #increase the number of counted symbols
                totalCounted += currentCount
                
                #while we still have deletions and those deletions are in this range, reduce the count and delete the string
                while currDel < totalDels and delIndices_view[currDel] < totalCounted:
                    currentCount -= 1
                    currDel += 1
                
                if currentCount > 0:
                    if prevChar == lastRunSym:
                        #we are basically extending a previously discovered run, so just add the counts on in
                        lastRunCount += currentCount
                    else:
                        #symbols are different, time to write out our previous run
                        while lastRunCount > 0:
                            self.bwt_view[writeByte] = ((lastRunCount & NUM_MASK) << letterBits) | lastRunSym
                            lastRunCount = lastRunCount >> numberBits
                            writeByte += 1
                        
                        #now store the next run
                        lastRunSym = prevChar
                        lastRunCount = currentCount
                else:
                    #no reason to change anything I think
                    pass
                
                #reset our read count stuff
                powerMultiple = 1
                currentCount = 0
                prevChar = currentChar
            
            #add the count based on the current multiple
            currentCount += (self.bwt_view[i] >> letterBits)*powerMultiple
        
        #do this one last time in case there's a late deletion
        totalCounted += currentCount
        while currDel < totalDels and delIndices_view[currDel] < totalCounted:
            currentCount -= 1
            currDel += 1
        
        #first, check if there's something in the run that *may* need to be sucked into a previous run
        if currentCount > 0:
            if prevChar == lastRunSym:
                #we are basically extending a previously discovered run, so just add the counts on in
                lastRunCount += currentCount
            else:
                #symbols are different, time to write out our previous run
                while lastRunCount > 0:
                    self.bwt_view[writeByte] = ((lastRunCount & NUM_MASK) << letterBits) | lastRunSym
                    lastRunCount = lastRunCount >> numberBits
                    writeByte += 1
                
                #now store the next run
                lastRunSym = prevChar
                lastRunCount = currentCount
        
        #now write out the final run which we know is there
        while lastRunCount > 0:
            self.bwt_view[writeByte] = ((lastRunCount & NUM_MASK) << letterBits) | lastRunSym
            lastRunCount = lastRunCount >> numberBits
            writeByte += 1
        
        #finally, clear out everything from the current write byte to the maximum array size
        for j in range(writeByte, numBytes):
            self.bwt_view[j] = 0x00
        
        #TODO: is there a way to clip the actual file so we don't have to clear?
        cdef unsigned long writeLcp
        cdef unsigned long readLcp
        if self.lcpsPresent:
            #initialize
            writeLcp = 0
            readLcp = 0
            currDel = 0
            
            #skip over deletions in the beginning
            while currDel < totalDels and delIndices_view[currDel] == readLcp:
                currDel += 1
                readLcp += 1
            
            #writeLcp will always be one past what we just wrote, readLcp is the same
            self.lcps_view[writeLcp] = self.lcps_view[readLcp]
            writeLcp += 1
            readLcp += 1
            
            #while we have more deletions
            while currDel < totalDels:
                #iterate straight to the next deletion
                for x in range(readLcp, delIndices_view[currDel]):
                    self.lcps_view[writeLcp] = self.lcps_view[readLcp]
                    writeLcp += 1
                    readLcp += 1
                
                #if we are deleting index 'x', then our write goes in index 'x-1' AND
                #is the minimum that 'x-1' shares with 'x' and 'x' shares with 'x+1'
                self.lcps_view[writeLcp-1] = min(self.lcps_view[writeLcp-1], self.lcps_view[readLcp])
                readLcp += 1
                currDel += 1
                
            #now clear out all that remains
            for x in range(writeLcp, self.lcps.shape[0]):
                self.lcps_view[x] = 0x00
            
        #clear Auxiliary data
        MSBWTGen.clearAuxiliaryData(self.dirName)
        try:
            os.remove(deletionFN)
        except:
            print 'Failed to remove '+deletionFN+' from the file system, manual removal necessary.'
        
        #reload if specified (by default, we will reload)
        if reloadBWT:
            #reload the bwt from the same directory, use whatever method was prescribed before
            self.loadMsbwt(self.dirName, self.useMemmapRLE, None)
        else:
            #USER IS RESPONSIBLE FOR RELOADING THE BWT
            pass
        
    cpdef set findReadsMatchingSeq(RLE_BWT self, bytes seq, unsigned long strLen):
        '''
        REQUIRES LCP 
        This function takes a sequence and finds all strings of length "stringLen" which exactly match the sequence
        @param seq - the sequence we want to match, no buffer 'N's are required here
        @param strLen - the length of the strings we are trying to extract
        @return - a set of dollar IDs corresponding to strings that exactly match the seq somewhere
        '''
        #currLen = the length of the l-h k-mer at all time
        cdef unsigned long currLen = 0
        cdef set readSet = set([])
        
        cdef unsigned long l = 0
        cdef unsigned long h = self.totalSize
        cdef unsigned long s = len(seq)
        cdef long x, y
        cdef unsigned long c
        
        cdef unsigned long newL
        cdef unsigned long newH
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        
        #with nogil:
        for x in range(s-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            newL = self.getOccurrenceOfCharAtIndex(c, l)
            newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            #currLen = the length of the l-h k-mer
            
            #while the new low and high are the same and we still have some length on the base k-mer
            while newL == newH and currLen > 0:
                #decrease currLen and loosen the k-mers to match it
                currLen -= 1
                while l > 0 and self.lcps_view[l-1] >= currLen:
                    l -= 1
                while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                    h += 1
                        
                #retry the search with the loosened parameters
                newL = self.getOccurrenceOfCharAtIndex(c, l)
                newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            if currLen == 0 and newL == newH:
                #if we have no current length and there's nothing in the search range, we basically skip this letter
                l = 0
                h = self.totalSize
            else:
                #else, we set our l/h to newL/newH and increment the length
                l = newL
                h = newH
                currLen += 1
                
                #also, we now check if we are at the read length
                if currLen == strLen:
                    #get all reads that match up
                    newL = self.getOccurrenceOfCharAtIndex(0, l)
                    newH = self.getOccurrenceOfCharAtIndex(0, h)
                    
                    #add each read on in
                    for y in xrange(newL, newH):
                        readSet.add(y)
                    
                    #now reduce the currLen and loosen
                    currLen -= 1
                    while l > 0 and self.lcps_view[l-1] >= currLen:
                        l -= 1
                    while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                        h += 1
                    
        #return the readSet
        return readSet
    
    cpdef set findReadsMatchingSeqWithError(RLE_BWT self, bytes seq, unsigned long strLen):
        '''
        REQUIRES LCP 
        This function takes a sequence and finds all strings of length "stringLen" which match the sequence
        while allowing for <= 1 errors.  This is basically a combination of two functions: findReadsMatchingSeq(...)
        and findStrWithError(...).
        @param seq - the sequence we want to match, assumed to be buffered on both ends with 'N' symbols
        @param strLen - the length of the strings we are trying to extract
        @return - a set of dollar IDs corresponding to strings that match with <= 1 base changes the seq somewhere
        '''
        #currLen = the length of the l-h k-mer at all time
        cdef unsigned long currLen = 0
        cdef set readSet = set([])
        
        cdef unsigned long l = 0
        cdef unsigned long h = self.totalSize
        cdef unsigned long lc, hc
        
        cdef unsigned long s = len(seq)
        cdef long x, y, z
        cdef unsigned long c, c2
        
        cdef unsigned long altC
        
        #altPos is only -1 initially, but since it CAN be negative, the altCurrLen must also be a long
        #I don't foresee any issues with this, but be aware future Matt; alternative is to do a shift of everything
        #which would be both programatically and mentally annoying
        cdef unsigned long altPos
        cdef unsigned long altCurrLen
        
        cdef unsigned long newL
        cdef unsigned long newH
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        
        #arrays for the fm-indices
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] lowArray = np.zeros(dtype='<u8', shape=(vcLen, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] highArray = np.zeros(dtype='<u8', shape=(vcLen, ))
        cdef np.uint64_t [:] lowArray_view = lowArray
        cdef np.uint64_t [:] highArray_view = highArray
        
        cdef unsigned long halfLen = strLen/2
        
        for x in range(s-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            self.fillFmAtIndex(lowArray_view, l)
            self.fillFmAtIndex(highArray_view, h)
            
            for c2 in xrange(1, vcLen):
                if c2 != c:# and (x < halfLen or currLen >= halfLen):#<---if you do this, need a different function below
                    #BEGIN SNP CHECK
                    #copy x into y, and set the altPos at -1 since it isn't actually added yet
                    y = x
                    #altPos = <long>-1
                    altPos = currLen+1
                    
                    #first copy into a temp area
                    lc = l
                    hc = h
                    altCurrLen = currLen
                    
                    #while we have seq left AND we haven't rotated our alternate symbol out
                    #while y >= 0 and altPos < <long>altCurrLen:
                    while y >= 0 and altPos > 0:
                        if y == x:
                            #first one is actually a different character 
                            altC = c2
                            newL = lowArray_view[altC]
                            newH = highArray_view[altC]
                        else:
                            #these are matching symbols now
                            altC = self.charToNum_view[seq_view[y]]
                            newL = self.getOccurrenceOfCharAtIndex(altC, lc)
                            newH = self.getOccurrenceOfCharAtIndex(altC, hc)
                        
                        #while our new range is length 0, and the altCurrLen > 0, and we haven't shifted our symbol out
                        #while newL == newH and altCurrLen > 0 and altPos < <long>altCurrLen:
                        while newL == newH and altCurrLen > 0 and altPos > 0:
                            #loosening time
                            altCurrLen -= 1
                            altPos -= 1
                            while lc > 0 and self.lcps_view[lc-1] >= altCurrLen:
                                lc -= 1
                            while hc < self.totalSize and self.lcps_view[hc-1] >= altCurrLen:
                                hc += 1
                            
                            #retry the search with the loosened parameters
                            newL = self.getOccurrenceOfCharAtIndex(altC, lc)
                            newH = self.getOccurrenceOfCharAtIndex(altC, hc)
                    
                        if altCurrLen == 0 and newL == newH:
                            #if we have no current length and there's nothing in the search range, we basically skip this letter
                            lc = 0
                            hc = self.totalSize
                            altPos = 0
                        else:
                            #else, we set our l/h to newL/newH and increment the length
                            lc = newL
                            hc = newH
                            altCurrLen += 1
                            #altPos += 1
                            
                            #also, we now check if we are at the read length
                            if altCurrLen == strLen:
                                #get all reads that match up
                                newL = self.getOccurrenceOfCharAtIndex(0, lc)
                                newH = self.getOccurrenceOfCharAtIndex(0, hc)
                                
                                #add each read on in
                                for z in xrange(newL, newH):
                                    readSet.add(z)
                                
                                #now reduce the currLen and loosen
                                altCurrLen -= 1
                                altPos -= 1
                                while lc > 0 and self.lcps_view[lc-1] >= altCurrLen:
                                    lc -= 1
                                while hc < self.totalSize and self.lcps_view[hc-1] >= altCurrLen:
                                    hc += 1
                                    
                        #now get the next symbol
                        y -= 1
                    
                    
                    #END SNP CHECK
            
            #set these to match the true symbol 
            newL = lowArray_view[c]
            newH = highArray_view[c]
            
            #while the new low and high are the same and we still have some length on the base k-mer
            while newL == newH and currLen > 0:
                #decrease currLen and loosen the k-mers to match it
                currLen -= 1
                while l > 0 and self.lcps_view[l-1] >= currLen:
                    l -= 1
                while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                    h += 1
                
                #retry the search with the loosened parameters
                newL = self.getOccurrenceOfCharAtIndex(c, l)
                newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            if currLen == 0 and newL == newH:
                #if we have no current length and there's nothing in the search range, we basically skip this letter
                l = 0
                h = self.totalSize
            else:
                #else, we set our l/h to newL/newH and increment the length
                l = newL
                h = newH
                currLen += 1
                
                #also, we now check if we are at the read length
                if currLen == strLen:
                    #get all reads that match up
                    newL = self.getOccurrenceOfCharAtIndex(0, l)
                    newH = self.getOccurrenceOfCharAtIndex(0, h)
                    
                    #add each read on in
                    for y in xrange(newL, newH):
                        readSet.add(y)
                    
                    #now reduce the currLen and loosen
                    currLen -= 1
                    while l > 0 and self.lcps_view[l-1] >= currLen:
                        l -= 1
                    while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                        h += 1
        
        #return the readSet
        return readSet
    
    cpdef set findReadsMatchingSeqWithError2(RLE_BWT self, bytes seq, unsigned long strLen):
        '''
        REQUIRES LCP 
        This function takes a sequence and finds all strings of length "stringLen" which match the sequence while
        allowing for one base error
        @param seq - the sequence we want to match, no buffer 'N's are required here
        @param strLen - the length of the strings we are trying to extract
        @return - a set of dollar IDs corresponding to strings that match with <= 1 base changes the seq somewhere
        '''
        #currLen = the length of the l-h k-mer at all time
        cdef unsigned long currLen = 0
        cdef set readSet = set([])
        
        cdef unsigned long l = 0
        cdef unsigned long h = self.totalSize
        cdef unsigned long s = len(seq)
        cdef long x, y, z, i
        cdef unsigned long c
        
        cdef unsigned long newL
        cdef unsigned long newH
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        
        #this will round down, which is fine
        cdef unsigned long halfLen = strLen/2
        cdef unsigned long hL = 0
        cdef unsigned long hH = self.totalSize
        cdef unsigned long hCurrLen = 0
        
        cdef bint isExactEntries = False
        cdef unsigned long exactMatchLow, exactMatchHigh
        
        #local align vars
        cdef str recoveredString
        cdef unsigned long alignScore
        cdef unsigned long c2, nextC
        cdef unsigned long dollarIndex
        cdef unsigned long symBefore, symAfter
        
        #arrays for the fm-indices
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] lowArray = np.zeros(dtype='<u8', shape=(vcLen, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] highArray = np.zeros(dtype='<u8', shape=(vcLen, ))
        cdef np.uint64_t [:] lowArray_view = lowArray
        cdef np.uint64_t [:] highArray_view = highArray
        
        cdef unsigned long altL, altH
        cdef unsigned long altCurrLen, altChangeOffset
        cdef unsigned long c3
        
        cdef unsigned long ind
        cdef unsigned long prevLZero, prevHZero
        cdef unsigned long currLZero, currHZero
        
        #with nogil:
        for x in range(s-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            newL = self.getOccurrenceOfCharAtIndex(c, l)
            newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            #while the new low and high are the same and we still have some length on the base k-mer
            while newL == newH and currLen > 0:
                #decrease currLen and loosen the k-mers to match it
                currLen -= 1
                while l > 0 and self.lcps_view[l-1] >= currLen:
                    l -= 1
                while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                    h += 1
                        
                #retry the search with the loosened parameters
                newL = self.getOccurrenceOfCharAtIndex(c, l)
                newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            if currLen == 0 and newL == newH:
                #if we have no current length and there's nothing in the search range, we basically skip this letter
                l = 0
                h = self.totalSize
                isExactEntries = False
            else:
                #else, we set our l/h to newL/newH and increment the length
                l = newL
                h = newH
                currLen += 1
                
                exactMatchLow = self.getOccurrenceOfCharAtIndex(0, l)
                exactMatchHigh = self.getOccurrenceOfCharAtIndex(0, h)
                
                #also, we now check if we are at the read length
                if currLen == strLen:
                    #get all reads that match up
                    #newL = self.getOccurrenceOfCharAtIndex(0, l)
                    #newH = self.getOccurrenceOfCharAtIndex(0, h)
                    newL = exactMatchLow
                    newH = exactMatchHigh
                    
                    #add each read on in
                    for y in xrange(newL, newH):
                        readSet.add(y)
                    
                    #now reduce the currLen and loosen
                    currLen -= 1
                    while l > 0 and self.lcps_view[l-1] >= currLen:
                        l -= 1
                    while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                        h += 1
                    
                    isExactEntries = True
                else:
                    isExactEntries = False
            
            #at this point, all exact matches have been found, now we do the off-by-one reads
            
            #Now we do everything again, but for the half length
            newL = self.getOccurrenceOfCharAtIndex(c, hL)
            newH = self.getOccurrenceOfCharAtIndex(c, hH)
            
            #while the new low and high are the same and we still have some length on the base k-mer
            while newL == newH and hCurrLen > 0:
                #decrease currLen and loosen the k-mers to match it
                hCurrLen -= 1
                while hL > 0 and self.lcps_view[hL-1] >= hCurrLen:
                    hL -= 1
                while hH < self.totalSize and self.lcps_view[hH-1] >= hCurrLen:
                    hH += 1
                        
                #retry the search with the loosened parameters
                newL = self.getOccurrenceOfCharAtIndex(c, hL)
                newH = self.getOccurrenceOfCharAtIndex(c, hH)
            
            if hCurrLen == 0 and newL == newH:
                #if we have no current length and there's nothing in the search range, we basically skip this letter
                hL = 0
                hH = self.totalSize
            else:
                #else, we set our l/h to newL/newH and increment the length
                hL = newL
                hH = newH
                hCurrLen += 1
                
                #also, we now check if we are at the half read length
                if hCurrLen == halfLen:
                    #so new plan using local alignments
                    #1 - take 50-mer range and find $+50-mer, then exclude those which exactly match, 
                    #    recover the string and local align the string
                    if x+strLen <= s:
                        #first, we need to find those with a '$' before them
                        newL = self.getOccurrenceOfCharAtIndex(0, hL)
                        newH = self.getOccurrenceOfCharAtIndex(0, hH)
                        
                        #TODO: is this complicated 'if' necessary when we are also checking for "y in readSet"?
                        if isExactEntries:
                            for y in range(newL, exactMatchLow):
                                if y in readSet:
                                    continue
                                else:
                                    recoveredString = self.recoverString(y)
                                    alignScore = AlignmentUtil.alignChanges(seq[x+halfLen:x+strLen], recoveredString[1+halfLen:])
                                    if alignScore <= 1:
                                        readSet.add(y)
                            for y in range(exactMatchHigh, newH):
                                if y in readSet:
                                    continue
                                else:
                                    recoveredString = self.recoverString(y)
                                    alignScore = AlignmentUtil.alignChanges(seq[x+halfLen:x+strLen], recoveredString[1+halfLen:])
                                    if alignScore <= 1:
                                        readSet.add(y)
                            
                        else:
                            #we have no exact matching entries, so we start with everything
                            for y in range(newL, newH):
                                if y in readSet:
                                    continue
                                if not isExactEntries or y < exactMatchLow or y >= exactMatchHigh:
                                    #not in the exact match range so handle it
                                    recoveredString = self.recoverString(y)
                                    alignScore = AlignmentUtil.alignChanges(seq[x+halfLen:x+strLen], recoveredString[1+halfLen:])
                                    if alignScore <= 1:
                                        readSet.add(y)
                        
                    #2 - take max level range, loosen until we have a c2+k-mer where c2 != nextC and k >= 50#
                    #    then crawl until we cycle the 'c2' out
                    if x > 0:
                        #get the next symbol and the f-index at this position
                        nextC = self.charToNum_view[seq_view[x-1]]
                        self.fillFmAtIndex(lowArray_view, hL)
                        self.fillFmAtIndex(highArray_view, hH)
                        
                        #for all other symbols
                        for c2 in range(1, vcLen):
                            if nextC != c2:
                                if highArray_view[c2] - lowArray_view[c2] <= 1:
                                    #THIS METHOD WORKS WELL WHEN THERE ARE SPARSE ERRORS
                                    #basically local align the one-of reads
                                    for y in range(lowArray_view[c2], highArray_view[c2]):
                                        recoveredString = self.recoverString(y)
                                        dollarIndex = recoveredString.find('$')
                                        #number of symbols before k-mer = 1+(strLen-dollarIndex)
                                        symBefore = 1+strLen-dollarIndex
                                        #number of symbols after k-mer = dollarIndex-(halfLen+1)
                                        symAfter = dollarIndex-(halfLen+1)
                                        
                                        if symBefore <= x and x+dollarIndex-1 < s:
                                            alignScore = (AlignmentUtil.alignChanges(seq[x-symBefore:x], recoveredString[dollarIndex+1:]+recoveredString[0:1])
                                                          + AlignmentUtil.alignChanges(seq[x+halfLen:x+halfLen+symAfter], recoveredString[1+halfLen:dollarIndex]))
                                            
                                            if alignScore <= 1:
                                                readSet.add(self.getSequenceDollarID(y))
                                else:
                                    #when we have a more than just one (i.e., a SNP we didn't know about)
                                    #we approach it by adding the "wrong" symbol and then forcing the rest to be correct
                                    #copy some values
                                    altL = l
                                    altH = h
                                    altCurrLen = currLen
                                    altChangeOffset = 0
                                    z = x - 2
                                    
                                    #try to tighten once
                                    newL = lowArray_view[c2]
                                    newH = highArray_view[c2]
                                    
                                    #loosen until we get a match we can tighten into
                                    while newL == newH and altCurrLen > halfLen:
                                        altCurrLen -= 1
                                        while altL > 0 and self.lcps_view[altL-1] >= altCurrLen:
                                            altL -= 1
                                        while altH < self.totalSize and self.lcps_view[altH-1] >= altCurrLen:
                                            altH += 1
                                        
                                        newL = self.getOccurrenceOfCharAtIndex(c2, altL)
                                        newH = self.getOccurrenceOfCharAtIndex(c2, altH)
                                    
                                    if altCurrLen == 0 and newL == newH:
                                        newL = 0
                                        newH = self.totalSize
                                    else:
                                        #tighten once
                                        altCurrLen += 1
                                        altL = newL
                                        altH = newH
                                        
                                        #if exact matches, copy them and loosen up
                                        if altCurrLen == strLen:
                                            for y in range(self.getOccurrenceOfCharAtIndex(0, altL), self.getOccurrenceOfCharAtIndex(0, altH)):
                                                readSet.add(y)
                                            
                                            altCurrLen -= 1
                                            while altL > 0 and self.lcps_view[altL-1] >= altCurrLen:
                                                altL -= 1
                                            while altH < self.totalSize and self.lcps_view[altH-1] >= altCurrLen:
                                                altH += 1
                                    
                                    #while our alternate symbol is in the front half of our k-mer range
                                    while altCurrLen - altChangeOffset > halfLen and z > 0:
                                        c3 = self.charToNum_view[seq_view[z]]
                                        newL = self.getOccurrenceOfCharAtIndex(c3, altL)
                                        newH = self.getOccurrenceOfCharAtIndex(c3, altH)
                                        
                                        #loosen until we get a match we can tighten into
                                        while newL == newH and altCurrLen - altChangeOffset > halfLen:
                                            altCurrLen -= 1
                                            while altL > 0 and self.lcps_view[altL-1] >= altCurrLen:
                                                altL -= 1
                                            while altH < self.totalSize and self.lcps_view[altH-1] >= altCurrLen:
                                                altH += 1
                                            
                                            newL = self.getOccurrenceOfCharAtIndex(c3, altL)
                                            newH = self.getOccurrenceOfCharAtIndex(c3, altH)
                                        
                                        if altCurrLen == 0 and newL == newH:
                                            newL = 0
                                            newH = self.totalSize
                                        else:
                                            #tighten once
                                            altCurrLen += 1
                                            altChangeOffset += 1
                                            altL = newL
                                            altH = newH
                                            
                                            #if exact matches, copy them and loosen up
                                            if altCurrLen == strLen:
                                                for y in range(self.getOccurrenceOfCharAtIndex(0, altL), self.getOccurrenceOfCharAtIndex(0, altH)):
                                                    readSet.add(y)
                                                
                                                altCurrLen -= 1
                                                while altL > 0 and self.lcps_view[altL-1] >= altCurrLen:
                                                    altL -= 1
                                                while altH < self.totalSize and self.lcps_view[altH-1] >= altCurrLen:
                                                    altH += 1
                                        
                                        z -= 1
                            
                    #now reduce the currLen and loosen
                    hCurrLen -= 1
                    while hL > 0 and self.lcps_view[hL-1] >= hCurrLen:
                        hL -= 1
                    while hH < self.totalSize and self.lcps_view[hH-1] >= hCurrLen:
                        hH += 1
            
        #return the readSet
        return readSet