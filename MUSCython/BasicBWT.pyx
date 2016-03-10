#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np
cimport numpy as np

from cython.operator cimport preincrement as inc

import MultiStringBWTCython as MultiStringBWT

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
    
    cpdef unsigned long getTotalSize(BasicBWT self):
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
    
    cpdef unsigned long getBinBits(BasicBWT self):
        '''
        @return - the number of bits in a bin
        '''
        return self.bitPower
    
    cpdef unsigned long countOccurrencesOfSeq(BasicBWT self, bytes seq, tuple givenRange=None):
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
    
    cpdef tuple findIndicesOfStr(BasicBWT self, bytes seq, tuple givenRange=None):
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
        
        for x in range(s-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            l = self.getOccurrenceOfCharAtIndex(c, l)
            h = self.getOccurrenceOfCharAtIndex(c, h)
        
        #return the difference
        return (l, h)
    
    cpdef list findIndicesOfRegex(BasicBWT self, bytes seq, tuple givenRange=None):
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
    
    cpdef list findStrWithError(BasicBWT self, bytes seq, bytes bonusStr):
        '''
        This function will search the BWT for strings which match the given sequence allowing for one error.
        In this function, "seq" must be close to the length of the read or else the ends of the reads will be counted
        as long insertions leading to no matches in the data.
        @param seq - the sequence to search for with valid symbols [A, C, G, N, T], NOTE: we assume the string is implicity
                     flanked by '$' so do NOT pass the '$' in the string or no result will return
        @param bonusStr - in the case of a deletion in the search, this is an extra character that must match at the front
                          of seq, aka it must match (bonusStr+seq) with one symbol deleted
        @return - a python list of ranges representing the start and end of the sequence in the bwt, these ranges will be
                  in the '$' indices, so they will correspond to a specific read
                  NOTE: these results may overlap, user expected to check for overlaps if important
        '''
        cdef list ret = []
        
        cdef unsigned long l, h
        cdef unsigned long lc, hc
        cdef unsigned long s
        cdef long x, y
        cdef unsigned long c, altC
        
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] lowArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] highArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] lowArray_view = lowArray
        cdef np.uint64_t [:] highArray_view = highArray
        
        #initialize our search to the whole '$' range of the BWT since we want reads
        l = 0
        h = self.totalCounts_view[0]
        s = len(seq)
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        cdef unsigned char * bonusStr_view = bonusStr
        
        #start with the last symbol and work downwards as long as we have a range length > 0
        x = s-1
        while x >= 0 and l < h:
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            self.fillFmAtIndex(lowArray_view, l)
            self.fillFmAtIndex(highArray_view, h)
            
            for altC in xrange(1, self.vcLen):
                if altC != c:
                    lc = lowArray_view[altC]
                    hc = highArray_view[altC]
            
                    #this is the SNP version, start one symbol past and work down forcing exact matching now
                    y = x-1
                    while y >= 0 and lc < hc:
                        lc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], lc)
                        hc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], hc)
                        y -= 1
                    
                    lc = self.getOccurrenceOfCharAtIndex(0, lc)
                    hc = self.getOccurrenceOfCharAtIndex(0, hc)
                    if hc > lc:
                        ret.append((lc, hc))
                    
                    #this one is the insertion version, need to do fewer symbols, but starting at x now
                    lc = lowArray_view[altC]
                    hc = highArray_view[altC]
                    y = x
                    while y >= 1 and lc < hc:
                        lc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], lc)
                        hc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], hc)
                        y -= 1
                    
                    lc = self.getOccurrenceOfCharAtIndex(0, lc)
                    hc = self.getOccurrenceOfCharAtIndex(0, hc)
                    if hc > lc:
                        ret.append((lc, hc))
        
            #deletion is similar to SNP version but we start with the current range
            lc = l
            hc = h
            y = x-1
            while y >= 0 and lc < hc:
                lc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], lc)
                hc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], hc)
                y -= 1
            
            #go one further symbol using bonusStr
            lc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[bonusStr_view[0]], lc)
            hc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[bonusStr_view[0]], hc)
            lc = self.getOccurrenceOfCharAtIndex(0, lc)
            hc = self.getOccurrenceOfCharAtIndex(0, hc)
            if hc > lc:
                ret.append((lc, hc))
            
            #now we can update to our exact matching sequence
            #second handle the exact matches
            l = lowArray_view[c]
            h = highArray_view[c]
            x -= 1
        
        #finally add all strings that exactly terminate here
        lc = self.getOccurrenceOfCharAtIndex(0, l)
        hc = self.getOccurrenceOfCharAtIndex(0, h)
        if hc > lc:
            ret.append((lc, hc))
        
        return ret
    
    cpdef list findPatternWithError(BasicBWT self, bytes seq, bytes bonusStr):
        '''
        This function will search the BWT for strings which match the given sequence allowing for one error.
        In this function, "seq" must be close to the length of the read or else the ends of the reads will be counted
        as long insertions leading to no matches in the data.
        @param seq - the sequence to search for with valid symbols [A, C, G, N, T]
        @param bonusStr - in the case of a deletion in the search, this is an extra character that must match at the front
                          of seq, aka it must match (bonusStr+seq) with one symbol deleted
        @return - a python list of ranges representing the start and end of the sequence in the bwt, these ranges will be
                  in the '$' indices, so they will correspond to a specific read
                  NOTE: these results may overlap, user expected to check for overlaps if important
        '''
        cdef list ret = []
        
        cdef unsigned long l, h
        cdef unsigned long lc, hc
        cdef unsigned long s
        cdef long x, y
        cdef unsigned long c, altC
        
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] lowArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] highArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] lowArray_view = lowArray
        cdef np.uint64_t [:] highArray_view = highArray
        
        #initialize our search to the whole '$' range of the BWT since we want reads
        l = 0
        h = self.totalSize
        s = len(seq)
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        cdef unsigned char * bonusStr_view = bonusStr
        
        #start with the last symbol and work downwards as long as we have a range length > 0
        x = s-1
        while x >= 0 and l < h:
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            self.fillFmAtIndex(lowArray_view, l)
            self.fillFmAtIndex(highArray_view, h)
            
            for altC in xrange(1, self.vcLen):
                if altC != c:
                    lc = lowArray_view[altC]
                    hc = highArray_view[altC]
            
                    #this is the SNP version, start one symbol past and work down forcing exact matching now
                    y = x-1
                    while y >= 0 and lc < hc:
                        lc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], lc)
                        hc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], hc)
                        y -= 1
                    
                    if hc > lc:
                        ret.append((lc, hc))
                    
                    #this one is the insertion version, need to do fewer symbols, but starting at x now
                    lc = lowArray_view[altC]
                    hc = highArray_view[altC]
                    y = x
                    while y >= 1 and lc < hc:
                        lc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], lc)
                        hc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], hc)
                        y -= 1
                    
                    if hc > lc:
                        ret.append((lc, hc))
        
            #deletion is similar to SNP version but we start with the current range
            lc = l
            hc = h
            y = x-1
            while y >= 0 and lc < hc:
                lc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], lc)
                hc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[seq_view[y]], hc)
                y -= 1
            
            #go one further symbol using bonusStr
            lc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[bonusStr_view[0]], lc)
            hc = self.getOccurrenceOfCharAtIndex(self.charToNum_view[bonusStr_view[0]], hc)
            if hc > lc:
                ret.append((lc, hc))
            
            #now we can update to our exact matching sequence
            #second handle the exact matches
            l = lowArray_view[c]
            h = highArray_view[c]
            x -= 1
        
        #finally add all strings that exactly terminate here
        if h > l:
            ret.append((l, h))
        
        return ret
    
    cpdef set findReadsMatchingSeq(BasicBWT self, bytes seq, unsigned long strLen):
        '''
        REQUIRES LCP 
        This function takes a sequence and finds all strings of length "stringLen" which exactly match the sequence
        @param seq - the sequence we want to match, assumed to be buffered on both ends with 'N' symbols
        @param strLen - the length of the strings we are trying to extract
        @return - a list of dollar IDs corresponding to strings that exactly match the seq somewhere
        '''
        #TODO: this is temporarily all in RLE_BWT for testing purposes; if a release is done
        #it needs to be moved, to here and made generic for all BWTs (if not already generic)
        #if it can't be generalized, then specific impls/errors need to be written for subclasses
        return set([])
        
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
            i += 1
        
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
        cdef unsigned long i
        
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
        
        for i in range(0, self.vcLen):
            if strIndex < self.endIndex_view[i]:
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
    
    cpdef np.ndarray[np.uint64_t, ndim=1, mode='c'] countPileup(BasicBWT self, bytes seq, long kmerSize):
        '''
        This function takes an input sequence "seq" and counts the number of occurrences of all k-mers of size
        "kmerSize" in that sequence and return it in an array. Automatically includes reverse complement.
        @param seq - the seq to scan
        @param kmerSize - the size of the k-mer to count
        @return - a numpy array of size (len(seq)-kmerSize+1) containing the counts
        '''
        cdef long seqLen = len(seq)
        cdef long numCounts = max(0, seqLen-kmerSize+1)
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.zeros(dtype='<u8', shape=(numCounts, ))
        cdef np.uint64_t [:] ret_view = ret
        
        cdef bytes subseq, subseqRevComp
        cdef bytes revCompSeq = MultiStringBWT.reverseComplement(seq)
        
        cdef unsigned long x
        for x in range(0, numCounts):
            subseq = seq[x:x+kmerSize]
            subseqRevComp = revCompSeq[seqLen-kmerSize-x:seqLen-x]
            #ret_view[x] = self.countOccurrencesOfSeq(subseq)+self.countOccurrencesOfSeq(subseqRevComp)
            ret_view[x] = self.countOccurrencesOfSeq(subseq)+self.countOccurrencesOfSeq(subseqRevComp)
        
        return ret
        
    cpdef tuple countSeqMatches(BasicBWT self, bytes seq, unsigned long kmerSize):
        '''
        This function takes an input sequence "seq" and counts the number of occurrences of all k-mers of size
        "kmerSize" in that sequence and return it in an array.
        @param seq - the seq to scan
        @param kmerSize - the size of the k-mer to count
        @return - a numpy array of size (len(seq)-kmerSize+1) containing the counts
        '''
        cdef long numCounts = max(0, len(seq)-kmerSize+1)
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] otherChoices
        (ret, otherChoices) = self.countStrandedSeqMatches(seq, kmerSize)
        
        cdef np.uint64_t [:] ret_view = ret
        cdef np.int64_t [:] otherChoices_view = otherChoices
        
        cdef long x
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] retRevComp
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] otherChoicesRevComp
        (retRevComp, otherChoicesRevComp) = self.countStrandedSeqMatches(MultiStringBWT.reverseComplement(seq), kmerSize)
        
        cdef np.uint64_t [:] retRevComp_view = retRevComp
        cdef otherChoicesRevComp_view = otherChoicesRevComp
        
        for x in range(0, numCounts):
            ret_view[x] += retRevComp_view[numCounts-x-1]
            otherChoices_view[x] += otherChoicesRevComp_view[numCounts-x-1]
            
        return (ret, otherChoices)
        
    cpdef tuple countStrandedSeqMatches(BasicBWT self, bytes seq, unsigned long kmerSize):
        '''
        This function takes an input sequence "seq" and counts the number of occurrences of all k-mers of size
        "kmerSize" in that sequence and return it in an array.
        @param seq - the seq to scan
        @param kmerSize - the size of the k-mer to count
        @return - a numpy array of size (len(seq)-kmerSize+1) containing the counts, and the other choice also
        '''
        #get a view of the input
        cdef unsigned char * seq_view = seq
        cdef long s = len(seq)
        
        #array size stuff
        cdef long numCounts = s-kmerSize+1
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.zeros(dtype='<u8', shape=(max(0, numCounts), ))
        cdef np.uint64_t [:] ret_view = ret
        
        #array for OTHER symbols that aren't '$' or the matching symbol
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] otherChoices = np.zeros(dtype='<i8', shape=(max(0, numCounts), ))
        cdef np.int64_t [:] otherChoices_view = otherChoices
        
        #arrays for the fm-indices
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] lowArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] highArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] lowArray_view = lowArray
        cdef np.uint64_t [:] highArray_view = highArray
        
        #ranges for counting
        cdef unsigned long currLen = 0
        cdef unsigned long l = 0
        cdef unsigned long h = self.totalSize
        
        cdef unsigned long newL, newH
        
        #other vars
        cdef unsigned long c, altC
        cdef long x = len(seq)-1
        cdef long y = 0
        
        #now we start traversing
        for x in range(s-1, -1, -1):
            c = self.charToNum_view[seq_view[x]]
            newL = self.getOccurrenceOfCharAtIndex(c, l)
            newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            if currLen == kmerSize-1:
                self.fillFmAtIndex(lowArray_view, l)
                self.fillFmAtIndex(highArray_view, h)
                
                for altC in range(1, self.vcLen):
                    if altC != c:
                        otherChoices_view[x] += (highArray_view[altC] - lowArray_view[altC])
                
            while newL == newH and currLen > 0:
                #loosen up a bit
                currLen -= 1
                while l > 0 and self.lcps_view[l-1] >= currLen:
                    l -= 1
                while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                    h += 1
                
                #re-search
                newL = self.getOccurrenceOfCharAtIndex(c, l)
                newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            if newL == newH and currLen == 0:
                #this symbol just doesn't occur at all
                l = 0
                h = self.totalSize
            else:
                #else, we set our l/h to newL/newH and increment the length
                l = newL
                h = newH
                currLen += 1
                
                #check if we're ready to start counting
                if x < numCounts:
                    if currLen == kmerSize:
                        #store the count
                        ret_view[x] = h-l
                        
                        #now reduce the currLen and loosen
                        currLen -= 1
                        while l > 0 and self.lcps_view[l-1] >= currLen:
                            l -= 1
                        while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                            h += 1
                        
                    else:
                        #we're too small, this means the count is 0
                        ret_view[x] = 0
                
        return (ret, otherChoices)
    
    cdef bwtRange getOccurrenceOfCharAtRange(BasicBWT self, unsigned long sym, bwtRange inRange) nogil:
        cdef bwtRange ret
        ret.l = 0
        ret.h = 0
        return ret
    
    cdef bwtRange findRangeOfStr(BasicBWT self, bytes seq):
        '''
        This function will search for a string and find the location of that string OR the last index less than it. It also
        will start its search within a given range instead of the whole structure
        @param seq - the sequence to search for
        @param givenRange - the range to search for, whole range by default
        @return - a python range representing the start and end of the sequence in the bwt
        '''
        cdef bwtRange ret
        cdef unsigned long s
        cdef long x
        cdef unsigned long c
        
        #initialize our search to the whole BWT
        ret.l = 0
        ret.h = self.totalSize
        s = len(seq)
        
        #create a view of the sequence that can be used in a nogil region
        cdef unsigned char * seq_view = seq
        
        for x in range(s-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            ret = self.getOccurrenceOfCharAtRange(c, ret)
            
        #return the difference
        return ret
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] countPileup_c(BasicBWT self, bytes seq, long kmerSize):
        '''
        This function takes an input sequence "seq" and counts the number of occurrences of all k-mers of size
        "kmerSize" in that sequence and return it in an array. Automatically includes reverse complement.
        @param seq - the seq to scan
        @param kmerSize - the size of the k-mer to count
        @return - a numpy array of size (len(seq)-kmerSize+1) containing the counts
        '''
        cdef long seqLen = len(seq)
        cdef long numCounts = max(0, seqLen-kmerSize+1)
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.zeros(dtype='<u8', shape=(numCounts, ))
        cdef np.uint64_t [:] ret_view = ret
        
        #cdef bytes subseq, subseqRevComp
        cdef bytes revCompSeq = MultiStringBWT.reverseComplement(seq)
        
        cdef unsigned char * seq_view = seq
        cdef unsigned char * revCompSeq_view = revCompSeq
        
        cdef unsigned long x
        for x in range(0, numCounts):
            #subseq = seq[x:x+kmerSize]
            #subseqRevComp = revCompSeq[seqLen-kmerSize-x:seqLen-x]
            #ret_view[x] = (self.countOccurrencesOfSeq_c(subseq)+
            #               self.countOccurrencesOfSeq_c(subseqRevComp)
            ret_view[x] = (self.countOccurrencesOfSeq_c(&seq_view[x], kmerSize)+
                           self.countOccurrencesOfSeq_c(&revCompSeq_view[seqLen-kmerSize-x], kmerSize))
        
        return ret
    
    cdef unsigned long countOccurrencesOfSeq_c(BasicBWT self, unsigned char * seq_view, unsigned long seqLen, unsigned long mc=1):
        '''
        This function counts the number of occurrences of the given sequence
        @param seq - the sequence to search for
        @param givenRange - the range to start from (if a partial search has already been run), default=whole range
        @return - an integer count of the number of times seq occurred in this BWT
        '''
        #initialize our search to the whole BWT
        cdef bwtRange ret
        cdef unsigned long c
        
        #initialize our search to the whole BWT
        ret.l = 0
        ret.h = self.totalSize
        
        #create a view of the sequence that can be used in a nogil region
        cdef long x
        for x in range(seqLen-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            ret = self.getOccurrenceOfCharAtRange(c, ret)
            
            #early exit for counts
            #if ret.l == ret.h:
            #    return 0
            if ret.h-ret.l < mc:
                return ret.h-ret.l
        
        #return the difference
        return ret.h-ret.l
    
    cdef unsigned long getOccurrenceOfCharAtIndex_c(BasicBWT self, unsigned long sym, unsigned long index):# nogil:
        '''
        dummy function, shouldn't be called
        '''
        cdef unsigned long ret = 0
        return ret
    
    cdef bwtRange findRangeOfStr_c(BasicBWT self, unsigned char * seq_view, unsigned long seqLen):
        '''
        This function will search for a string and find the location of that string OR the last index less than it. It also
        will start its search within a given range instead of the whole structure
        @param seq - the sequence to search for
        @param givenRange - the range to search for, whole range by default
        @return - a python range representing the start and end of the sequence in the bwt
        '''
        cdef bwtRange ret
        cdef long x
        cdef unsigned long c
        
        #initialize our search to the whole BWT
        ret.l = 0
        ret.h = self.totalSize
        
        for x in range(seqLen-1, -1, -1):
            #get the character from the sequence, then search at both high and low
            c = self.charToNum_view[seq_view[x]]
            ret = self.getOccurrenceOfCharAtRange(c, ret)
            
        #return the difference
        return ret
    
    cpdef np.ndarray countStrandedSeqMatchesNoOther(BasicBWT self, bytes seq, unsigned long kmerSize):
        '''
        This function takes an input sequence "seq" and counts the number of occurrences of all k-mers of size
        "kmerSize" in that sequence and return it in an array.
        @param seq - the seq to scan
        @param kmerSize - the size of the k-mer to count
        @return - a numpy array of size (len(seq)-kmerSize+1) containing the counts
        '''
        #get a view of the input
        cdef unsigned char * seq_view = seq
        cdef long s = len(seq)
        
        #array size stuff
        cdef long numCounts = s-kmerSize+1
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.zeros(dtype='<u8', shape=(max(0, numCounts), ))
        cdef np.uint64_t [:] ret_view = ret
        
        #arrays for the fm-indices
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] lowArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] highArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] lowArray_view = lowArray
        cdef np.uint64_t [:] highArray_view = highArray
        
        #ranges for counting
        cdef unsigned long currLen = 0
        #cdef unsigned long l = 0
        #cdef unsigned long h = self.totalSize
        cdef bwtRange mainRange
        mainRange.l = 0
        mainRange.h = self.totalSize
        
        #cdef unsigned long newL, newH
        cdef bwtRange newRange
        
        #other vars
        cdef unsigned long c
        cdef long x = len(seq)-1
        cdef long y = 0
        
        #now we start traversing
        for x in range(s-1, -1, -1):
            c = self.charToNum_view[seq_view[x]]
            #newL = self.getOccurrenceOfCharAtIndex(c, l)
            #newH = self.getOccurrenceOfCharAtIndex(c, h)
            newRange = self.getOccurrenceOfCharAtRange(c, mainRange)
            
            if currLen == kmerSize-1:
                self.fillFmAtIndex(lowArray_view, mainRange.l)
                self.fillFmAtIndex(highArray_view, mainRange.h)
                
            #while newL == newH and currLen > 0:
            while newRange.l == newRange.h and currLen > 0:
                #loosen up a bit
                currLen -= 1
                while mainRange.l > 0 and self.lcps_view[mainRange.l-1] >= currLen:
                    mainRange.l -= 1
                while mainRange.h < self.totalSize and self.lcps_view[mainRange.h-1] >= currLen:
                    mainRange.h += 1
                
                #re-search
                #newL = self.getOccurrenceOfCharAtIndex(c, l)
                #newH = self.getOccurrenceOfCharAtIndex(c, h)
                newRange = self.getOccurrenceOfCharAtRange(c, mainRange)
            
            #if newL == newH and currLen == 0:
            if newRange.l == newRange.h and currLen == 0:
                #this symbol just doesn't occur at all
                mainRange.l = 0
                mainRange.h = self.totalSize
            else:
                #else, we set our l/h to newL/newH and increment the length
                #l = newL
                #h = newH
                mainRange.l = newRange.l
                mainRange.h = newRange.h
                currLen += 1
                
                #check if we're ready to start counting
                if x < numCounts:
                    if currLen == kmerSize:
                        #store the count
                        ret_view[x] = mainRange.h-mainRange.l
                        
                        #now reduce the currLen and loosen
                        currLen -= 1
                        while mainRange.l > 0 and self.lcps_view[mainRange.l-1] >= currLen:
                            mainRange.l -= 1
                        while mainRange.h < self.totalSize and self.lcps_view[mainRange.h-1] >= currLen:
                            mainRange.h += 1
                        
                    else:
                        #we're too small, this means the count is 0
                        ret_view[x] = 0
                
        return ret
    
    cpdef np.ndarray findKmerThreshold(BasicBWT self, bytes seq, unsigned long threshold):
        '''
        This function takes an input sequence "seq" and counts the number of occurrences of all k-mers of size
        "kmerSize" in that sequence and return it in an array.
        @param seq - the seq to scan
        @param kmerSize - the size of the k-mer to count
        @param isStranded - if True, it ONLY counts the forward strand (aka, exactly matches "seq")
                            if False, it counts forward strand and reverse-complement strand and adds them together
        @return - a numpy array of size (len(seq)-kmerSize+1) containing the counts
        '''
        cdef long numCounts = len(seq)
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret
        ret = self.findKmerThresholdStranded(seq, threshold)
        
        cdef np.uint64_t [:] ret_view = ret
        
        cdef long x
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] retRevComp
        retRevComp = self.findKmerThresholdStranded(MultiStringBWT.reverseComplement(seq), threshold)
        
        cdef np.uint64_t [:] retRevComp_view = retRevComp
        
        #we want the max from either side
        for x in range(0, numCounts):
            #ret_view[x] = max(ret_view[x], retRevComp_view[numCounts-x-1])
            ret_view[x] = ret_view[x] + retRevComp_view[numCounts-x-1]
        
        return ret
        
    cpdef np.ndarray findKmerThresholdStranded(BasicBWT self, bytes seq, unsigned long threshold):
        '''
        ??? need desc
        @param seq - the seq to scan
        @param kmerSize - the size of the k-mer to count
        @return - a numpy array of size (len(seq)-kmerSize+1) containing the counts
        '''
        #get a view of the input
        cdef unsigned char * seq_view = seq
        cdef unsigned long s = len(seq)
        
        #array size stuff
        cdef long numCounts = s
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] rightRet = np.zeros(dtype='<u8', shape=(numCounts, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] leftRet = np.zeros(dtype='<u8', shape=(numCounts, ))
        cdef np.uint64_t [:] rightRet_view = rightRet
        cdef np.uint64_t [:] leftRet_view = leftRet
        
        #arrays for the fm-indices
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] lowArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] highArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] lowArray_view = lowArray
        cdef np.uint64_t [:] highArray_view = highArray
        
        #ranges for counting
        cdef unsigned long currLen = 0
        cdef unsigned long l = 0
        cdef unsigned long h = self.totalSize
        
        cdef unsigned long newL, newH
        
        #other vars
        cdef unsigned long c, altC
        cdef long x = len(seq)-1
        cdef long y = 0
        
        #now we start traversing
        for x in range(s-1, -1, -1):
            c = self.charToNum_view[seq_view[x]]
            newL = self.getOccurrenceOfCharAtIndex(c, l)
            newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            #while the range we have is less than our threshold and the currLen is greater than 0
            while newH-newL < threshold and currLen > 0:
                #loosen up a bit
                currLen -= 1
                while l > 0 and self.lcps_view[l-1] >= currLen:
                    l -= 1
                while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                    h += 1
                
                #re-search
                newL = self.getOccurrenceOfCharAtIndex(c, l)
                newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            if newL == newH and currLen == 0:
                #this symbol just doesn't occur at all (this pretty much never actually happens, but included for completeness)
                l = 0
                h = self.totalSize
            else:
                #we have an extension that fits here
                l = newL
                h = newH
                currLen += 1
                
                #store the length of the k-mer we found
                rightRet_view[x] = currLen
        
        #TODO: currently returning (and calculating) just the right side, are other metrics better?
        return rightRet
                
    cpdef np.ndarray findKTOtherStranded(BasicBWT self, bytes seq, unsigned long threshold):
        '''
        This function takes an input sequence "seq" and counts the number of occurrences of all k-mers of size
        "kmerSize" in that sequence and return it in an array.
        @param seq - the seq to scan
        @param kmerSize - the size of the k-mer to count
        @param isStranded - if True, it ONLY counts the forward strand (aka, exactly matches "seq")
                            if False, it counts forward strand and reverse-complement strand and adds them together
        @return - a numpy array of size (len(seq)-kmerSize+1) containing the counts
        '''
        #get a view of the input
        cdef unsigned char * seq_view = seq
        cdef unsigned long s = len(seq)
        
        #array size stuff
        cdef long numCounts = s
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] ret = np.zeros(dtype='<u8', shape=(numCounts, ))
        cdef np.uint64_t [:] ret_view = ret
        
        #arrays for the fm-indices
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] lowArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] highArray = np.zeros(dtype='<u8', shape=(self.vcLen, ))
        cdef np.uint64_t [:] lowArray_view = lowArray
        cdef np.uint64_t [:] highArray_view = highArray
        
        #ranges for counting
        cdef unsigned long currLen = 0
        cdef unsigned long altCurrLen
        cdef unsigned long l = 0
        cdef unsigned long h = self.totalSize
        
        cdef unsigned long newL, newH
        cdef unsigned long altL, altH
        
        #other vars
        cdef unsigned long c, altC
        cdef long x = len(seq)-1
        cdef long y = 0
        
        cdef unsigned long maxAlt, altVal
        
        #now we start traversing
        for x in range(s-1, -1, -1):
            c = self.charToNum_view[seq_view[x]]
            self.fillFmAtIndex(lowArray_view, l)
            self.fillFmAtIndex(highArray_view, h)
            newL = lowArray_view[c]
            newH = highArray_view[c]
            
            if currLen > 20:
                maxAlt = 0
                for altC in range(1, self.vcLen):
                    if altC != c:
                        altVal = highArray_view[altC] - lowArray_view[altC]
                        if altVal > maxAlt:
                            maxAlt = altVal
                ret_view[x] = maxAlt
            
            #while the range we have is less than our threshold and the currLen is greater than 0
            while newH-newL < threshold and currLen > 0:
                #loosen up a bit
                currLen -= 1
                while l > 0 and self.lcps_view[l-1] >= currLen:
                    l -= 1
                while h < self.totalSize and self.lcps_view[h-1] >= currLen:
                    h += 1
                
                #re-search
                newL = self.getOccurrenceOfCharAtIndex(c, l)
                newH = self.getOccurrenceOfCharAtIndex(c, h)
            
            if newL == newH and currLen == 0:
                #this symbol just doesn't occur at all (this pretty much never actually happens, but included for completeness)
                l = 0
                h = self.totalSize
            else:
                #we have an extension that fits here
                l = newL
                h = newH
                currLen += 1
        
        return ret