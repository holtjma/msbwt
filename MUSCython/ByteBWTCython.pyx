#!python
#cython: boundscheck=False
#cython: wraparound=False

import math
import numpy as np
cimport numpy as np
import os

cimport BasicBWT
from cython.operator cimport preincrement as inc

cdef class ByteBWT(BasicBWT.BasicBWT):
    '''
    This class is a BWT capable of hosting multiple strings inside one structure.  Basically, this would allow you to
    search for a given string across several strings simultaneously.  Note: this class is for the non-compressed version,
    for general purposes use the function loadBWT(...) which automatically detects whether this class or CompressedMSBWT 
    is correct
    '''
    def loadMsbwt(ByteBWT self, char * dirName, logger):
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
    
    def constructTotalCounts(ByteBWT self, logger):
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
    
    def constructFMIndex(ByteBWT self, logger):
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
            
    cdef unsigned int getCharAtIndex(ByteBWT self, unsigned long index):# nogil:
        '''
        This function is only necessary for other functions which perform searches generically without knowing if the 
        underlying structure is compressed or not
        @param index - the index to retrieve the character from
        '''
        return self.bwt_view[index]
    
    def getBWTRange(ByteBWT self, unsigned long start, unsigned long end):
        '''
        This function is only necessary for other functions which perform searches generically without knowing if the 
        underlying structure is compressed or not
        @param start - the beginning of the range to retrieve
        @param end - the end of the range in normal python notation (bwt[end] is not part of the return)
        '''
        return self.bwt[start:end]
        
    cpdef unsigned long getOccurrenceOfCharAtIndex(ByteBWT self, unsigned int sym, unsigned long index):# nogil:
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
        
    def getFullFMAtIndex(ByteBWT self, unsigned long index):
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
        
    cdef np.uint8_t iterNext_cython(ByteBWT self) nogil:
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