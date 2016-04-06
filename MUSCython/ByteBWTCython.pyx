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
    is correct.  This particular BWT uses one byte per symbol and is the most intuitive of the BWT structures due to it's
    relative simplicity.  However, it's also the least compressed and not necessarily the fastest for searching.
    '''
    
    cdef bint useMemmap
    
    def loadMsbwt(ByteBWT self, char * dirName, bint useMemmap=True, logger=None):
        '''
        This functions loads a BWT file and constructs total counts, indexes start positions, and constructs an FM index
        on disk if it doesn't already exist
        @param dirName - the filename to load
        @param useMemmap - if True (default), the BWT is kept on disk and paged in as necessary
        @param logger - the logger to print output to (default: None)
        '''
        #open the file with our BWT in it
        self.dirName = dirName
        self.useMemmap = useMemmap
        if useMemmap:
            self.bwt = np.load(self.dirName+'/msbwt.npy', 'r+')
        else:
            self.bwt = np.load(self.dirName+'/msbwt.npy')
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
    
    def constructTotalCounts(ByteBWT self, logger):
        '''
        This function constructs the total count for each valid character in the array or loads them if they already exist.
        These will always be stored in '<DIR>/totalCounts.p', a pickled file
        @param logger - the logger to print output to
        '''
        cdef unsigned long i
        self.totalSize = self.bwt.shape[0]
        abtFN = self.dirName+'/totalCounts.npy'
        if os.path.exists(abtFN):
            if self.useMemmap:
                self.totalCounts = np.load(abtFN, 'r+')
            else:
                self.totalCounts = np.load(abtFN)
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
        This function iterates through the BWT and counts the letters as it goes to create the sampled FM index.  
        For example, the string 'ACC$' would have BWT 'C$CA'.  The FM index would iterate over this and count the 
        occurrence and offset of the letter it found and then sub-sample it at some power of two:
        BWT    FM-index
        C    0    1    2 <-sampled
        $    0    1    3 
        C    1    1    3 <-sampled
        A    1    1    4
             1    2    5 <-sampled
        This is necessary for finding the occurrence of a letter using the getOccurrenceOfCharAtIndex(...) function.
        In reality, this function creates a sampled FM-index so only one index every 2048 bases is filled in.
        This file is always stored in '<DIR>/fmIndex.npy'.
        @param logger - the logger to print output to
        '''
        #sampling method
        self.bitPower = 11
        self.binSize = 2**self.bitPower
        fmIndexFN = self.dirName+'/fmIndex.npy'
        
        cdef np.ndarray[np.uint64_t, ndim=1] counts
        cdef np.uint64_t [:] counts_view
        cdef unsigned long i, j
        cdef unsigned long samplingSize
        
        if os.path.exists(fmIndexFN):
            if self.useMemmap:
                self.partialFM = np.load(fmIndexFN, 'r+')
            else:
                self.partialFM = np.load(fmIndexFN)
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
            
    cpdef unsigned long getCharAtIndex(ByteBWT self, unsigned long index):# nogil:
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
    
    cdef void fillBin(ByteBWT self, np.uint8_t [:] binToFill, unsigned long binID) nogil:
        '''
        This copies a slice of the BWT into an array the caller can manipulate, useful mostly for the compressed
        versions of this data structure.
        @param binToFill - the place we can copy the BWT into
        @param binID - the bin we're copying
        '''
        cdef unsigned long x
        cdef unsigned long startIndex = binID*self.binSize
        cdef unsigned long endIndex = min((binID+1)*self.binSize, self.totalSize)
        
        for x in range(0, endIndex-startIndex):
            binToFill[x] = self.bwt_view[startIndex+x]
        
    cpdef unsigned long getOccurrenceOfCharAtIndex(ByteBWT self, unsigned long sym, unsigned long index):# nogil:
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
    
    cdef BasicBWT.bwtRange getOccurrenceOfCharAtRange(ByteBWT self, unsigned long sym, BasicBWT.bwtRange inRange) nogil:
        '''
        This functions gets the FM-index value of a character at the specified position
        @param sym - the character to find the occurrence level
        @param index - the index we want to find the occurrence level at
        @return - the number of occurrences of char before the specified index
        '''
        #sampling method
        cdef unsigned long binID = inRange.l >> self.bitPower
        cdef BasicBWT.bwtRange ret
        ret.l = self.partialFM_view[binID, sym]
        cdef unsigned long start = binID << self.bitPower
        cdef unsigned long i
        
        for i in range(start, inRange.l):
            if self.bwt_view[i] == sym:
                ret.l += 1
        
        cdef unsigned long binID_h = inRange.h >> self.bitPower
        if binID == binID_h:
            ret.h = ret.l
            start = inRange.l
        else:
            ret.h = self.partialFM_view[binID_h, sym]
            start = binID_h << self.bitPower
        
        for i in range(start, inRange.h):
            if self.bwt_view[i] == sym:
                ret.h += 1
        
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
    
    cdef void fillFmAtIndex(ByteBWT self, np.uint64_t [:] fill_view, unsigned long index):
        '''
        Same as getFmAtIndex, but with a pass in array to fill in
        @param fill_view - the view of the fmIndex we are going to fill in
        @param index - the index to extract the fm-index for
        '''
        cdef unsigned long binID = index >> self.bitPower
        cdef unsigned long bwtInd = binID * self.binSize
        cdef unsigned long i
        
        for i in range(0, self.vcLen):
            fill_view[i] = self.partialFM_view[binID][i]
        
        for i in range(bwtInd, index):
            inc(fill_view[self.bwt_view[i]])
    
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