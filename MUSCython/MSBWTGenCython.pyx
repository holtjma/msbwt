#!python
#cython: boundscheck=False
#cython: wraparound=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

'''
Created on Mar 19, 2014

file for testing a Cython implementation of the very basic paper algorithm

@author: holtjma
'''

import time

import numpy as np
cimport numpy as np
from cython.operator cimport preincrement as inc
from multiprocessing.pool import ThreadPool

def mergeMsbwts(list inputDirs, char *outDir, unsigned int numProcs, logger):
    if numProcs <= 1:
        logger.info( 'Beginning single-threaded merge...')
        mergeCythonImpl_st(inputDirs, outDir, logger)
    else:
        logger.info( 'Beginning multi-threaded merge...')
        mergeCythonImpl_mt(inputDirs, outDir, numProcs, logger)

def mergeCythonImpl_st(list inputDirs, char *outDir, logger):
    #simple vars we will use throughout
    cdef int numValidChars = 6
    cdef unsigned int c, numInputs, b
    cdef unsigned long i, l
    numInputs = len(inputDirs)
    
    #tempBWT and bwt_view will be used in an initial pass as simple access vars, bwtAddress will hold pointers
    #in an array we can pass to subroutines
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] tempBWT
    cdef np.uint8_t [:] bwt_view
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtAddress = np.zeros(dtype='<u8', shape=(numInputs,))
    
    #prepare the total counts and bwtLen variable so we can fill it in the following loop
    cdef np.ndarray[np.uint64_t, ndim=1] totalCounts = np.zeros(dtype='<u8', shape=(numValidChars,))
    cdef np.uint64_t [:] totalCounts_view = totalCounts
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtLens = np.zeros(dtype='<u8', shape=(numInputs,))
    
    #ref holder for python
    bwts = [None]*numInputs
    
    #go through each input, grabbing lengths, pointers, and bincounts
    for i in range(numInputs):
        #load the file, also, we store it in bwts[i] to hold a pointer reference
        #IMPORTANT: for view level speed we require write permissions, 
        #BUT DO NOT WRITE ON THESE DATASETS
        #tempBWT = np.load(inputDirs[i]+'/msbwt.npy', 'r+')
        #bwts[i] = tempBWT
        bwts[i] = np.load(inputDirs[i]+'/msbwt.npy', 'r+')
        tempBWT = bwts[i]
        
        logger.info( 'Loading \''+inputDirs[i]+'\'')
        
        #transform into a view so we can use it in our addresses in the nogil
        bwt_view = tempBWT
        l = bwts[i].shape[0]
        
        #these two are stored long term
        bwtLens[i] = l
        bwtAddress[i] = <np.uint64_t>&bwt_view[0]
        
        #bincount basically for all files
        with nogil:
            for i in range(l):
                inc(totalCounts_view[bwt_view[i]])
    
    logger.info('Finished loading input.')
    
    #calculate the offsets for each letter into our merged BWT
    cdef np.ndarray[np.uint64_t, ndim=1] offsets = np.cumsum(totalCounts)-totalCounts
    cdef np.ndarray[np.uint64_t, ndim=1] offsetCopy
    cdef np.ndarray[np.uint64_t, ndim=1] ends = np.cumsum(totalCounts)
    
    #initial interleave
    print totalCounts
    print offsets
    print ends
    logger.info('Creating initial interleave of size '+str(ends[numValidChars-1])+'...')
    cdef np.ndarray[np.uint8_t, ndim=1] inter0 = np.lib.format.open_memmap(outDir+'/inter0.npy', 'w+', '<u1', (ends[numValidChars-1],))
    cdef np.uint8_t [:] inter0_view = inter0
    cdef unsigned long start = bwtLens[0]
    
    #0s followed by 1s, followed by 2s, followed by ...
    with nogil:
        for b in range(1, numInputs):
            inter0_view[start:start+bwtLens[b]] = b
            start += bwtLens[b]
            #for i in range(0, bwtLens[b]):
            #    inter0_view[start] = b
            #    start += 1
    
    logger.info('Allocating second interleave array...')
    #second interleave can be all zeroes for now, we will just fill it in
    cdef np.ndarray[np.uint8_t, ndim=1] inter1 = np.lib.format.open_memmap(outDir+'/inter1.npy', 'w+', '<u1', (ends[numValidChars-1],))
    cdef np.uint8_t [:] inter1_view = inter1
    
    #markers for the iteration number and timings
    cdef int iterID = 0
    cdef float st, et
    
    #iterate until convergence
    logger.info('Beginning iterations...')
    while not np.array_equal(inter0, inter1):
        offsetCopy = np.copy(offsets)
        
        #mergeIter(inArray, outArray)
        stw = time.time()
        st = time.clock()
        if iterID % 2 == 0:
            #mergeIter_st(inter0, inter1, bwtAddress, offsetCopy, ends[numValidChars-1])
            mergeIter_st(inter0_view, inter1_view, bwtAddress, offsetCopy, ends[numValidChars-1])
        else:
            #mergeIter_st(inter1, inter0, bwtAddress, offsetCopy, ends[numValidChars-1])
            mergeIter_st(inter1_view, inter0_view, bwtAddress, offsetCopy, ends[numValidChars-1])
        et = time.clock()
        etw = time.time()
        iterID += 1
        
        logger.info( 'Finished iter '+str(iterID)+' in '+str(et-st)+' CPU, '+str(etw-stw)+' secs')
        
    #create the output file and 0-offsets into each input
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] outBWT = np.lib.format.open_memmap(outDir+'/msbwt.npy', 'w+', '<u1', (ends[numValidChars-1],))
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtOffsets = np.zeros(dtype='<u8', shape=(l,))
    
    #build the views we need for nogil
    cdef np.uint64_t [:] bwtOffsets_view = bwtOffsets
    cdef np.uint64_t [:] bwtAddress_view = bwtAddress
    cdef np.uint8_t [:] outBWT_view = outBWT
    
    #helper var
    with nogil:
        #now iterate through the input interleave
        for i in range(ends[numValidChars-1]):
            #get the bit here and pull the appropriate symbol, c, increment the place we're looking
            b = inter0_view[i]
            c = (<np.uint8_t *>bwtAddress_view[b])[bwtOffsets_view[b]]
            inc(bwtOffsets_view[b])
            outBWT_view[i] = c
    
def mergeIter_st(#np.ndarray[np.uint8_t, ndim=1, mode='c'] inArray not None, 
              #np.ndarray[np.uint8_t, ndim=1, mode='c'] outArray not None, 
              np.uint8_t [:] inArray_view,
              np.uint8_t [:] outArray_view,
              np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtAddress not None,
              np.ndarray[np.uint64_t, ndim=1, mode='c'] offsets not None, 
              unsigned long end):
    #iterator, symbol (as uint), bwt ID from interleave, number of input BWTs
    cdef unsigned int sym, bwtID, numInputs
    cdef unsigned long i
    numInputs = bwtAddress.shape[0]
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtOffsets = np.zeros(dtype='<u8', shape=(numInputs,))
    
    #cdef np.uint8_t [:] outArray_view = outArray
    #cdef np.uint8_t [:] inArray_view = inArray
    cdef np.uint64_t [:] offsets_view = offsets
    cdef np.uint64_t [:] bwtOffsets_view = bwtOffsets
    cdef np.uint64_t [:] bwtAddress_view = bwtAddress
    
    with nogil:
        #TODO: remove hard-coded '5'
        #now iterate through the input interleave
        for i in range(end):
            #get the bit here and pull the appropriate symbol, c, increment the place we're looking
            bwtID = inArray_view[i]
            sym = (<np.uint8_t *>bwtAddress_view[bwtID])[bwtOffsets_view[bwtID]]
            inc(bwtOffsets_view[bwtID])
            
            #set the output and increment the offset for the symbol
            outArray_view[offsets_view[sym]] = bwtID
            inc(offsets_view[sym])

def mergeCythonImpl_mt(list inputDirs, char *outDir, unsigned int numProcs, logger):
    #simple vars we will use throughout
    cdef int numValidChars = 6
    cdef unsigned long i, j, k, l
    cdef unsigned int c, numInputs, b, begin
    numInputs = len(inputDirs)
    
    #tempBWT and bwt_view will be used in an initial pass as simple access vars, bwtAddress will hold pointers
    #in an array we can pass to subroutines
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] tempBWT
    cdef np.uint8_t [:] bwt_view
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtAddress = np.zeros(dtype='<u8', shape=(numInputs,))
    
    #prepare the total counts and bwtLen variable so we can fill it in the following loop
    cdef np.ndarray[np.uint64_t, ndim=1] totalCounts = np.zeros(dtype='<u8', shape=(numValidChars,))
    cdef np.ndarray[np.uint64_t, ndim=2] totalCountsByInput = np.zeros(dtype='<u8', shape=(numInputs, numValidChars))
    cdef np.ndarray[np.uint64_t, ndim=2] totalCountsByOnemer = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    
    cdef np.uint64_t [:] totalCounts_view = totalCounts
    cdef np.uint64_t [:, :] totalCountsByInput_view = totalCountsByInput
    cdef np.uint64_t [:, :] totalCountsByOnemer_view = totalCountsByOnemer
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtLens = np.zeros(dtype='<u8', shape=(numInputs,))
    
    #ref holder for python
    bwts = [None]*numInputs
    
    #go through each input, grabbing lengths, pointers, and bincounts
    for i in range(numInputs):
        #load the file, also, we store it in bwts[i] to hold a pointer reference
        #IMPORTANT: for view level speed we require write permissions, 
        #BUT DO NOT WRITE ON THESE DATASETS
        tempBWT = np.load(inputDirs[i]+'/msbwt.npy', 'r+')
        bwts[i] = tempBWT
        
        logger.info( 'Loading \''+inputDirs[i]+'\'')
        
        #transform into a view so we can use it in our addresses in the nogil
        bwt_view = tempBWT
        l = tempBWT.shape[0]
        
        #these two are stored long term
        bwtLens[i] = l
        bwtAddress[i] = <np.uint64_t>&bwt_view[0]
        
        #bincount basically for all files
        with nogil:
            #first construct the total counts for this input
            for j in range(l):
                #inc(totalCounts_view[bwt_view[j]])
                inc(totalCountsByInput_view[i][bwt_view[j]])
            
            #add it to the overall total counts
            for j in range(numValidChars):
                totalCounts_view[j] += totalCountsByInput_view[i][j]
            
            #go through again, calculating how many there are for each offset
            begin = 0
            for j in range(numValidChars):
                for k in range(begin, begin+totalCountsByInput_view[i][j]):
                    inc(totalCountsByOnemer_view[j][bwt_view[k]])
                begin += totalCountsByInput_view[i][j]
                
    #calculate the offsets for each letter into our merged BWT
    cdef np.ndarray[np.uint64_t, ndim=1] offsets = np.cumsum(totalCounts)-totalCounts
    cdef np.ndarray[np.uint64_t, ndim=1] offsetCopy = None
    cdef np.ndarray[np.uint64_t, ndim=1] ends = np.cumsum(totalCounts)
    
    #construct the offsets for each piece
    cdef np.ndarray[np.uint64_t, ndim=2] totalOffsetsByOnemer = np.cumsum(totalCountsByOnemer, axis=0)-totalCountsByOnemer
    totalOffsetsByOnemer += offsets
    cdef np.ndarray[np.uint64_t, ndim=2] totalOffsetsByOnemer_copy = None
    
    #initial interleave
    cdef np.ndarray[np.uint8_t, ndim=1] inter0 = np.lib.format.open_memmap(outDir+'/inter0.npy', 'w+', '<u1', (ends[numValidChars-1],))
    cdef np.uint8_t [:] inter0_view = inter0
    cdef unsigned long start = bwtLens[0]
    
    cdef np.ndarray[np.uint64_t, ndim=2] bwtOffsetsByOnemer = np.copy(np.transpose(np.cumsum(totalCountsByInput, axis=1) - totalCountsByInput))
    cdef np.ndarray[np.uint64_t, ndim=2] bwtOffsetsByOnemer_copy = None
    
    #0s followed by 1s, followed by 2s, followed by ...
    logger.info( 'Constructing initial interleave...')
    with nogil:
        begin = 0
        for i in range(numValidChars):
            for j in range(numInputs):
                for k in range(begin, begin+totalCountsByInput_view[j][i]):
                    inter0_view[k] = j
                begin += totalCountsByInput_view[j][i]
        '''
        for b in range(1, numInputs):
            for i in range(0, bwtLens[b]):
                inter0_view[start] = b
                start += 1
        '''
    
    #second interleave can be all zeroes for now, we will just fill it in
    cdef np.ndarray[np.uint8_t, ndim=1] inter1 = np.lib.format.open_memmap(outDir+'/inter1.npy', 'w+', '<u1', (ends[numValidChars-1],))
    
    #markers for the iteration number and timings
    cdef int iterID = 0
    cdef float st, et
    
    #cdef np.ndarray[np.uint8_t, ndim=1] offsetCopy = None
    
    #iterate until convergence
    logger.info( 'Beginning iterations...')
    while not np.array_equal(inter0, inter1):
        totalOffsetsByOnemer_copy = np.copy(totalOffsetsByOnemer)
        bwtOffsetsByOnemer_copy = np.copy(bwtOffsetsByOnemer)
        offsetCopy = np.copy(totalOffsetsByOnemer[0])
        #mergeIter(inArray, outArray)
        stw = time.time()
        st = time.clock()
        
        tups = []
        for i in range(numValidChars):
            if iterID % 2 == 0:
                tups.append((inter0, inter1, bwtAddress, totalOffsetsByOnemer_copy[i], bwtOffsetsByOnemer_copy[i], np.sum(bwtOffsetsByOnemer[i]), ends[i]))
            else:
                tups.append((inter1, inter0, bwtAddress, totalOffsetsByOnemer_copy[i], bwtOffsetsByOnemer_copy[i], np.sum(bwtOffsetsByOnemer[i]), ends[i]))
        
        tp = ThreadPool(numProcs)
        tp.map(mergeIterBySymbol_mt, tups)
        tp.terminate()
        tp.join()
        tp = None
        #'''
        et = time.clock()
        etw = time.time()
        iterID += 1
        
        logger.info( 'Finished iter '+str(iterID)+' in '+str(et-st)+' CPU, '+str(etw-stw)+' secs')
        
    #create the output file and 0-offsets into each input
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] outBWT = np.lib.format.open_memmap(outDir+'/msbwt.npy', 'w+', '<u1', (ends[numValidChars-1],))
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtOffsets = np.zeros(dtype='<u8', shape=(l,))
    
    #build the views we need for nogil
    cdef np.uint64_t [:] bwtOffsets_view = bwtOffsets
    cdef np.uint64_t [:] bwtAddress_view = bwtAddress
    cdef np.uint8_t [:] outBWT_view = outBWT
    
    #helper var
    with nogil:
        #now iterate through the input interleave
        for i in range(ends[numValidChars-1]):
            #get the bit here and pull the appropriate symbol, c, increment the place we're looking
            b = inter0_view[i]
            c = (<np.uint8_t *>bwtAddress_view[b])[bwtOffsets_view[b]]
            inc(bwtOffsets_view[b])
            outBWT_view[i] = c
            
def mergeIterBySymbol_mt(tup):
    '''
              np.ndarray[np.uint8_t, ndim=1, mode='c'] inArray not None, 
              np.ndarray[np.uint8_t, ndim=1, mode='c'] outArray not None, 
              np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtAddress not None,
              np.ndarray[np.uint64_t, ndim=1, mode='c'] offsets not None,
              np.ndarray[np.uint64_t, ndim=1, mode='c'] bwtOffsets not None,
              unsigned int start,
              unsigned int end):
    '''
    
    cdef unsigned long i
    cdef unsigned int sym, bwtID#, numInputs
    cdef np.uint8_t [:] inArray_view = tup[0]
    cdef np.uint8_t [:] outArray_view = tup[1]
    cdef np.uint64_t [:] bwtAddress_view = tup[2]
    cdef np.uint64_t [:] offsets_view = tup[3]
    cdef np.uint64_t [:] bwtOffsets_view = tup[4]
    cdef unsigned long start = tup[5]
    cdef unsigned long end = tup[6]
    
    with nogil:
        for i in range(start, end):
            bwtID = inArray_view[i]
            sym = (<np.uint8_t *>bwtAddress_view[bwtID])[bwtOffsets_view[bwtID]]
            inc(bwtOffsets_view[bwtID])
            
            #set the output and increment the offset for the symbol
            outArray_view[offsets_view[sym]] = bwtID
            inc(offsets_view[sym])
    
'''
def testThreading():
    print 'hello'
    tp = ThreadPool(4)
    cdef unsigned int l = 2000000000
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] data = np.ones(dtype='<u1', shape=(l, ))
    cdef np.uint8_t [:] data_view = data
    inputs = [(x, &data_view[0]) for x in range(1, 5)]
    ret = tp.map(runFunc, inputs)
    print ret
    
def runFunc(tup):
    cdef unsigned int x = tup[0]
    cdef unsigned int ret = 0
    cdef np.uint8_t * data_view = <np.uint8_t *>tup[1]
    with nogil:
        #garbage work to do
        for i in range(0, 2000000000):
            ret = (ret+x*data_view[i]) % 10
            ret += 1
            ret *= 2
            ret += 1
            ret *= 2
    return ret
'''