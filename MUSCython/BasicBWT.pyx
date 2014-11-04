#!python
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

from cython.operator cimport preincrement as inc

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
    
    Override functions:
    loadMsbwt
    constructTotalCounts
    constructFMIndex
    getCharAtIndex
    getOccurrenceOfCharAtIndex
    getBWTRange
    getFullFMAtIndex
    iterInit
    iterNext
    iterNext_cython
    '''
    '''
    #declared in .pxd now
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
    '''
    
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
        
        #these are merely defaults, override if wanted but maintain the relationship of, binSize = 2**bitPower
        self.bitPower = 11
        self.binSize = 2**self.bitPower
    
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
    
    cpdef getTotalSize(BasicBWT self):
        '''
        @return - the total number of symbols in the BWT
        '''
        return self.totalSize
    
    cpdef unsigned long getSymbolCount(BasicBWT self, unsigned long symbol):
        '''
        @param symbol - this is an integer from [0, 6)
        @return - the total count for the passed in symbol
        '''
        #cdef unsigned long ret = self.totalCounts_view[symbol]
        #return ret
        return self.totalCounts_view[symbol]
    
    cpdef getBinBits(BasicBWT self):
        '''
        @return - the number of bits in a bin
        '''
        return self.bitPower
    
    cpdef countOccurrencesOfSeq(BasicBWT self, bytes seq, givenRange=None):
        '''
        This function counts the number of occurrences of the given sequence
        @param seq - the sequence to search for
        @param givenRange - the range to start from (if a partial search has already been run), default=whole range
        @return - an integer count of the number of times seq occurred in this BWT
        '''
        cdef unsigned long l, h
        cdef long x
        cdef unsigned long s
        cdef unsigned long c
        
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
    
    cpdef findIndicesOfStr(BasicBWT self, bytes seq, givenRange=None):
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
        cdef unsigned long c
        
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
    
    cpdef findIndicesOfRegex(BasicBWT self, bytes seq, givenRange=None):
        '''
        This function will search for a string and find the location of that string OR the last index less than it. It also
        will start its search within a given range instead of the whole structure.  Note that have a small tail string can 
        lead to fast exponential blowup of the solution space.
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
        cdef unsigned long c
        
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
    
    cpdef unsigned long getCharAtIndex(BasicBWT self, unsigned long index):# nogil:
        '''
        dummy function, shouldn't be called
        '''
        cdef unsigned long ret = 0
        return ret
    
    cdef void fillBin(BasicBWT self, np.uint8_t [:] binToFill, unsigned long binID) nogil:
        '''
        dummy funciton, shouldn't be called
        '''
        return
    
    cpdef unsigned long getOccurrenceOfCharAtIndex(BasicBWT self, unsigned long sym, unsigned long index):# nogil:
        '''
        dummy function, shouldn't be called
        '''
        cdef unsigned long ret = 0
        return ret
    
    cpdef iterInit(BasicBWT self):
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
    
    cpdef iterNext(BasicBWT self):
        return self.iterNext_cython()
    
    cdef np.uint8_t iterNext_cython(BasicBWT self) nogil:
        '''
        dummy function, override in all subclasses
        '''
        return 255
    
    cpdef getSequenceDollarID(BasicBWT self, unsigned long strIndex, bint returnOffset=False):
        '''
        This will take a given index and work backwards until it encounters a '$' indicating which dollar ID is
        associated with this read
        @param strIndex - the index of the character to start with
        @return - an integer indicating the dollar ID of the string the given character belongs to
        '''
        #figure out the first hop backwards
        cdef unsigned long currIndex = strIndex
        cdef unsigned long prevChar
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
    
    cpdef recoverString(BasicBWT self, unsigned long strIndex, bint withIndex=False):
        '''
        This will return the string that starts at the given index
        @param strIndex - the index of the string we want to recover
        @return - string that we found starting at the specified '$' index
        '''
        cdef list retNums = []
        cdef list indices = []
        
        #figure out the first hop backwards
        cdef unsigned long prevChar = self.getCharAtIndex(strIndex)
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
    
    cdef void fillFmAtIndex(BasicBWT self, np.uint64_t [:] fill_view, unsigned long index):
        '''
        dummy function, override in all subclasses
        '''
        pass