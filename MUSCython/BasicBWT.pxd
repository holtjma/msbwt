
cimport numpy as np

cdef struct bwtRange:
    unsigned long l
    unsigned long h

cdef class BasicBWT(object):
    cdef np.ndarray numToChar
    cdef unsigned char [:] numToChar_view
    cdef np.ndarray charToNum
    cdef unsigned char [:] charToNum_view
    cdef unsigned long vcLen
    
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
    
    #called during initialization, no reason for a user to need this
    cdef void constructIndexing(BasicBWT self)
    
    #simple queries for basic BWT statistics
    cpdef unsigned long getTotalSize(BasicBWT self)
    cpdef unsigned long getSymbolCount(BasicBWT self, unsigned long symbol)
    cpdef unsigned long getBinBits(BasicBWT self)
    
    #common k-mer queries, both cython and python compatible
    cpdef unsigned long countOccurrencesOfSeq(BasicBWT self, bytes seq, tuple givenRange=*)
    cpdef tuple findIndicesOfStr(BasicBWT self, bytes seq, tuple givenRange=*)
    cpdef list findIndicesOfRegex(BasicBWT self, bytes seq, tuple givenRange=*)
    cpdef list findStrWithError(BasicBWT self, bytes seq, bytes bonusStr)
    cpdef list findPatternWithError(BasicBWT self, bytes seq, bytes bonusStr)
    cpdef set findReadsMatchingSeq(BasicBWT self, bytes seq, unsigned long strLen)
    cpdef list findKmerWithError(BasicBWT self, bytes seq, unsigned long minThresh=*)
    cpdef list findKmerWithErrors(BasicBWT self, bytes seq, unsigned long editDistance, unsigned long minThresh=*)
    
    #single character queries, both cython and python compatible
    cpdef unsigned long getCharAtIndex(BasicBWT self, unsigned long index)
    cpdef unsigned long getOccurrenceOfCharAtIndex(BasicBWT self, unsigned long sym, unsigned long index)
    
    #iterator functions
    cpdef iterInit(BasicBWT self)
    cpdef iterNext(BasicBWT self)
    cdef np.uint8_t iterNext_cython(BasicBWT self) nogil
    
    #full string functions
    cpdef getSequenceDollarID(BasicBWT self, unsigned long strIndex, bint returnOffset=*)
    cpdef recoverString(BasicBWT self, unsigned long strIndex, bint withIndex=*)
    
    #FM-index related queries
    cdef void fillBin(BasicBWT self, np.uint8_t [:] binToFill, unsigned long binID) nogil
    cdef void fillFmAtIndex(BasicBWT self, np.uint64_t [:] fill_view, unsigned long index)
    
    #Pileup counting without LCP array
    cpdef np.ndarray[np.uint64_t, ndim=1, mode='c'] countPileup(BasicBWT self, bytes seq, long kmerSize)
    
    #these functions are added specifically to reduce the Python overhead of some calls (input/output is a C struct)
    cdef bwtRange getOccurrenceOfCharAtRange(BasicBWT self, unsigned long sym, bwtRange inRange) nogil
    cdef bwtRange findRangeOfStr(BasicBWT self, bytes seq)
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] countPileup_c(BasicBWT self, bytes seq, long kmerSize)
    cdef unsigned long countOccurrencesOfSeq_c(BasicBWT self, unsigned char * seq_view, unsigned long seqLen, unsigned long mc=*)
    cdef unsigned long getOccurrenceOfCharAtIndex_c(BasicBWT self, unsigned long sym, unsigned long index)
    cdef bwtRange findRangeOfStr_c(BasicBWT self, unsigned char * seq_view, unsigned long seqLen)
    
    #the following functions require an LCP array
    cpdef tuple countSeqMatches(BasicBWT self, bytes seq, unsigned long kmerSize)
    cpdef tuple countStrandedSeqMatches(BasicBWT self, bytes seq, unsigned long kmerSize)
    cpdef np.ndarray countStrandedSeqMatchesNoOther(BasicBWT self, bytes seq, unsigned long kmerSize)
    cpdef np.ndarray findKmerThreshold(BasicBWT self, bytes seq, unsigned long threshold)
    cpdef np.ndarray findKmerThresholdStranded(BasicBWT self, bytes seq, unsigned long threshold)
    cpdef np.ndarray findKTOtherStranded(BasicBWT self, bytes seq, unsigned long threshold)