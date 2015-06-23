
cimport numpy as np

cdef class BasicBWT(object):
    cdef np.ndarray numToChar
    cdef unsigned char [:] numToChar_view
    cdef np.ndarray charToNum
    cdef unsigned char [:] charToNum_view
    cdef unsigned long vcLen
    cdef unsigned long cacheDepth
    
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
    
    cdef bint lcpsPresent
    cdef np.ndarray lcps
    cdef np.uint8_t [:] lcps_view
    
    cdef void constructIndexing(BasicBWT self)
    cpdef unsigned long getTotalSize(BasicBWT self)
    cpdef unsigned long getSymbolCount(BasicBWT self, unsigned long symbol)
    cpdef unsigned long getBinBits(BasicBWT self)
    cpdef unsigned long countOccurrencesOfSeq(BasicBWT self, bytes seq, tuple givenRange=*)
    cpdef tuple findIndicesOfStr(BasicBWT self, bytes seq, tuple givenRange=*)
    cpdef list findIndicesOfRegex(BasicBWT self, bytes seq, tuple givenRange=*)
    cpdef list findStrWithError(BasicBWT self, bytes seq, bytes bonusStr)
    cpdef list findPatternWithError(BasicBWT self, bytes seq, bytes bonusStr)
    cpdef set findReadsMatchingSeq(BasicBWT self, bytes seq, unsigned long strLen)
    cpdef unsigned long getCharAtIndex(BasicBWT self, unsigned long index)
    cpdef unsigned long getOccurrenceOfCharAtIndex(BasicBWT self, unsigned long sym, unsigned long index)
    cpdef iterInit(BasicBWT self)
    cpdef iterNext(BasicBWT self)
    cdef np.uint8_t iterNext_cython(BasicBWT self) nogil
    cpdef getSequenceDollarID(BasicBWT self, unsigned long strIndex, bint returnOffset=*)
    cpdef recoverString(BasicBWT self, unsigned long strIndex, bint withIndex=*)
    cdef void fillBin(BasicBWT self, np.uint8_t [:] binToFill, unsigned long binID) nogil
    cdef void fillFmAtIndex(BasicBWT self, np.uint64_t [:] fill_view, unsigned long index)
    cpdef np.ndarray countPileup(BasicBWT self, bytes seq, long kmerSize)
    cpdef tuple countSeqMatches(BasicBWT self, bytes seq, unsigned long kmerSize)
    cpdef tuple countStrandedSeqMatches(BasicBWT self, bytes seq, unsigned long kmerSize)
    cpdef np.ndarray countStrandedSeqMatchesNoOther(BasicBWT self, bytes seq, unsigned long kmerSize)
    cpdef np.ndarray findKmerThreshold(BasicBWT self, bytes seq, unsigned long threshold)
    cpdef np.ndarray findKmerThresholdStranded(BasicBWT self, bytes seq, unsigned long threshold)
    cpdef np.ndarray findKTOtherStranded(BasicBWT self, bytes seq, unsigned long threshold)