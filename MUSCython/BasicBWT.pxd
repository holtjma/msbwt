
cimport numpy as np

cdef class BasicBWT(object):
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
    
    cdef void constructIndexing(BasicBWT self)
    cpdef getTotalSize(BasicBWT self)
    cpdef getSymbolCount(BasicBWT self, unsigned int symbol)
    cpdef countOccurrencesOfSeq(BasicBWT self, bytes seq, givenRange=*)
    cpdef findIndicesOfStr(BasicBWT self, bytes seq, givenRange=*)
    cpdef findIndicesOfRegex(BasicBWT self, bytes seq, givenRange=*)
    cdef unsigned int getCharAtIndex(BasicBWT self, unsigned long index)
    cpdef unsigned long getOccurrenceOfCharAtIndex(BasicBWT self, unsigned int sym, unsigned long index)
    cpdef iterInit(BasicBWT self)
    cpdef iterNext(BasicBWT self)
    cdef np.uint8_t iterNext_cython(BasicBWT self) nogil
    cpdef getSequenceDollarID(BasicBWT self, unsigned long strIndex, bint returnOffset=*)
    cpdef recoverString(BasicBWT self, unsigned long strIndex, bint withIndex=*)