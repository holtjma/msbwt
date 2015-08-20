#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np
cimport numpy as np
import os
import time

import MultiStringBWTCython as MSBWT
cimport BasicBWT

def mergeTwoMSBWTs(char * inputMsbwtDir1, char * inputMsbwtDir2, char * mergedDir, unsigned long numProcs, logger):
    '''
    This function takes two BWTs as input and merges them into a single BWT in O(N*LCP_avg) time where N is the 
    total number of bases and LCP_avg is the average common prefix between adjacent entries in the merged result.
    @param inputMsbwtDir1 - the directory of the first MSBWT, must be in ByteBWT format (not compressed)
    @param inputMsbwtDir2 - the directory of the second MSBWT, must be in ByteBWT format (not compressed)
    @param mergedDir - the directory for output
    @param numProcs - number of processes we're allowed to use
    @param logger - use for logging outputs and progress
    '''
    logger.info('Beginning Merge:')
    logger.info('Input 1:\t'+inputMsbwtDir1)
    logger.info('Input 2:\t'+inputMsbwtDir2)
    logger.info('Output:\t'+mergedDir)
    logger.info('Num Procs:\t'+str(numProcs))
    
    if numProcs > 1:
        logger.info('Multi-processing not implemented, setting numProcs=1')
        numProcs = 1
    
    #hardcode this as we do everywhere else
    cdef unsigned long numValidChars = 6
    
    #map the seqs, note we map msbwt.npy because that's where all changes happen
    cdef BasicBWT.BasicBWT loadedBwt0 = MSBWT.loadBWT(inputMsbwtDir1, useMemmap=True, logger=logger)
    cdef BasicBWT.BasicBWT loadedBwt1 = MSBWT.loadBWT(inputMsbwtDir2, useMemmap=True, logger=logger)
    
    cdef unsigned long bwtLen1 = loadedBwt0.getTotalSize()
    cdef unsigned long bwtLen2 = loadedBwt1.getTotalSize()
    
    #prepare to construct total counts for the symbols
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] totalCounts = np.zeros(dtype='<u8', shape=(numValidChars, ))
    cdef np.uint64_t [:] totalCounts_view = totalCounts
    
    #first calculate the total counts for our region
    cdef unsigned long x, y
    for x in xrange(0, numValidChars):
        totalCounts_view[x] = loadedBwt0.getSymbolCount(x)+loadedBwt1.getSymbolCount(x)
    
    #now we should load the interleaves
    cdef unsigned long interleaveBytes
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] inter0
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] inter1
    
    #this needs 1 bit per base, so we allocate bases/8 bytes plus one due to integer division
    interleaveBytes = (bwtLen1+bwtLen2)/8+1
    
    #hardcoded as 1 GB right now
    cdef unsigned long interThresh = 1*10**9
    interleaveFN0 = mergedDir+'/inter0.npy'
    interleaveFN1 = mergedDir+'/inter1.npy'
    
    if interleaveBytes > interThresh:
        inter0 = np.lib.format.open_memmap(interleaveFN0, 'w+', '<u1', (interleaveBytes, ))
        inter1 = np.lib.format.open_memmap(interleaveFN1, 'w+', '<u1', (interleaveBytes, ))
    else:
        inter0 = np.zeros(dtype='<u1', shape=(interleaveBytes, ))
        inter1 = np.zeros(dtype='<u1', shape=(interleaveBytes, ))
    
    cdef np.uint8_t [:] inter0_view = inter0
    cdef np.uint8_t [:] inter1_view = inter1
    
    #initialize the first interleave based on the offsets
    cdef np.uint8_t * inter0_p
    
    #with two, we will initialize both arrays
    inter0_p = &inter0_view[0]
    
    #initialize the first half to all 0s, 0x00
    for y in xrange(0, bwtLen1/8):
        inter0_view[y] = 0x00
        inter1_view[y] = 0x00
    
    #one byte in the middle can be mixed zeros and ones
    inter0_view[bwtLen1/8] = 0xFF << (bwtLen1 % 8)
    inter1_view[bwtLen1/8] = 0xFF << (bwtLen1 % 8)
    
    #all remaining bytes are all ones, 0xFF
    for y in xrange(bwtLen1/8+1, (bwtLen1+bwtLen2)/8+1):
        inter0_view[y] = 0xFF
        inter1_view[y] = 0xFF
    
    #values tracking progress
    cdef unsigned int iterCount = 0
    cdef bint changesMade = True
    
    #make a copy of the offset that the subfunction can modify safely
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] ranges
    cdef double st, el
    
    #format is (position in bwt0, position in bwt1, total length)
    cdef unsigned long HARD_LIMIT = 4**10#at 1 million, we should force it to collapse down
    cdef unsigned long SOFT_LIMIT = 4**5 #at 1024 entries, we should be collapsing down to smaller ranges
    ranges = np.zeros(dtype='<u8', shape=(1, 3))
    ranges[0][2] = bwtLen1+bwtLen2
    cdef unsigned long totalLength = bwtLen1+bwtLen2
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fullCoverageRanges = np.copy(ranges)
    
    while changesMade:
        if iterCount % 2 == 0:
            st = time.time()
            ret = targetedIterationMerge2(loadedBwt0, loadedBwt1,
                                          &inter0_view[0], &inter1_view[0], bwtLen1+bwtLen2, 
                                          ranges, numValidChars, 
                                          iterCount, numProcs, ranges.shape[0] > SOFT_LIMIT)
            changesMade = ret[0]
            ranges = ret[1]
            el = time.time()-st
            logText = '\t'.join([str(val) for val in (0, iterCount, ranges.shape[0], el)])
        else:
            st = time.time()
            ret = targetedIterationMerge2(loadedBwt0, loadedBwt1, 
                                          &inter1_view[0], &inter0_view[0], bwtLen1+bwtLen2, 
                                          ranges, numValidChars, 
                                          iterCount, numProcs, ranges.shape[0] > SOFT_LIMIT)
            el = time.time()-st
            changesMade = ret[0]
            ranges = ret[1]
            logText = '\t'.join([str(val) for val in (0, iterCount, ranges.shape[0], el)])
        
        logger.info(logText)
        iterCount += 1
        
        if ranges.shape[0] > HARD_LIMIT:
            #we are too big, reset to a smaller set of buckets
            ranges = np.copy(fullCoverageRanges)
        elif ranges.shape[0] > SOFT_LIMIT:
            #we don't want to save bucket groups if they're bigger than the SOFT_LIMIT either
            pass
        elif np.sum(ranges[:, 2]) == totalLength:
            #check if this fully covers the whole bwt, if so it's a better breakdown of our ranges and still below the set limits
            fullCoverageRanges = np.copy(ranges)
    
    if interleaveBytes <= interThresh:
        np.save(interleaveFN0, inter0)
    
    interleaveTwoBwts(inputMsbwtDir1, inputMsbwtDir2, mergedDir, logger)
    
    if interleaveBytes > interThresh:
        os.remove(interleaveFN1)
        
    return iterCount
    
def interleaveTwoBwts(char * inputMsbwtDir1, char * inputMsbwtDir2, char * mergedDir, logger):
    #map the seqs, note we map msbwt.npy because that's where all changes happen
    cdef BasicBWT.BasicBWT loadedBwt0 = MSBWT.loadBWT(inputMsbwtDir1, useMemmap=True, logger=logger)
    cdef BasicBWT.BasicBWT loadedBwt1 = MSBWT.loadBWT(inputMsbwtDir2, useMemmap=True, logger=logger)
    
    cdef unsigned long bwtLen1 = loadedBwt0.getTotalSize()
    cdef unsigned long bwtLen2 = loadedBwt1.getTotalSize()
    
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] msbwt = np.lib.format.open_memmap(mergedDir+'/msbwt.npy', 'w+', '<u1', (bwtLen1+bwtLen2,))
    cdef np.uint8_t [:] msbwt_view = msbwt
    
    #hardcoded as 1 GB right now
    interleaveFN0 = mergedDir+'/inter0.npy'
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] inter0 = np.load(interleaveFN0, 'r+')
    cdef np.uint8_t [:] inter0_view = inter0
    cdef np.uint8_t * inter0_p
    
    #with two, we will initialize both arrays
    inter0_p = &inter0_view[0]
    
    cdef unsigned long readID
    cdef unsigned long pos1 = 0
    cdef unsigned long pos2 = 0
    
    cdef unsigned long binBits0 = loadedBwt0.getBinBits()
    cdef unsigned long binBits1 = loadedBwt1.getBinBits()
    cdef unsigned long binSize0 = 2**binBits0
    cdef unsigned long binSize1 = 2**binBits1
    
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currentBin0 = np.empty(dtype='<u1', shape=(binSize0, ))
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currentBin1 = np.empty(dtype='<u1', shape=(binSize1, ))
    cdef np.uint8_t [:] currentBin0_view = currentBin0
    cdef np.uint8_t [:] currentBin1_view = currentBin1
    cdef unsigned long currentBinID0 = 0
    cdef unsigned long currentBinID1 = 0
    cdef unsigned long currentBinUse0 = 0
    cdef unsigned long currentBinUse1 = 0
    
    #fill in the bin from the BasicBWT
    with nogil:
        loadedBwt0.fillBin(currentBin0_view, currentBinID0)
        loadedBwt1.fillBin(currentBin1_view, currentBinID1)
    
    cdef unsigned long x
    for x in range(0, bwtLen1+bwtLen2):
        #get the read, the symbol, and increment the position in that read
        if getBit_p(inter0_p, x):
            msbwt_view[x] = currentBin1_view[currentBinUse1]
            currentBinUse1 += 1
            if currentBinUse1 >= binSize1:
                currentBinID1 += 1
                loadedBwt1.fillBin(currentBin1_view, currentBinID1)
                currentBinUse1 = 0
        else:
            msbwt_view[x] = currentBin0_view[currentBinUse0]
            currentBinUse0 += 1
            if currentBinUse0 >= binSize0:
                currentBinID0 += 1
                loadedBwt0.fillBin(currentBin0_view, currentBinID0)
                currentBinUse0 = 0
                
cdef tuple targetedIterationMerge2(BasicBWT.BasicBWT bwt0, BasicBWT.BasicBWT bwt1, 
                                  np.uint8_t * inputInter_view, np.uint8_t * outputInter_view, 
                                  unsigned long bwtLen,
                                  np.ndarray[np.uint64_t, ndim=2, mode='c'] ranges, unsigned long nvc,
                                  unsigned long iterCount, unsigned long numThreads,
                                  bint collapseEntries):
    '''
    Performs a single pass over the data for a merge of at most 2 BWTs, allows for a bit array, instead of byte
    @param bwt0 - the first bwt, must be a subclass of BasicBWT
    @param bwt1 - the second bwt, must be a subclass of BasicBWT
    @param inputInter_view - a C pointer to the input interleave
    @param outputInter_view - a C pointer to where this function writes the output interleave
    @param bwtLen - the sum of all input string lengths
    @param ranges - the ranges we plan on iterating through
    @param nvc - Number of Valid Characters, aka 6
    @param iterCount - the iteration we're on
    @param numThreads - the number of threads we are allowing this sub-routine to create
    @param collapseEntries - if set, we will attempt to collapse adjacent entries at the end to reduce the 
        total number of entries that are returned (aka, reduces memory)
    '''
    #counters
    cdef unsigned long x, y, z
    
    #values associated with reading the input ranges
    cdef bint readID
    cdef np.uint8_t symbol
    cdef bint changesMade = False
    
    #3-d array indexed by [symbol, entry, entry-value], each entry is (start0, start1, distance)
    cdef np.ndarray[np.uint64_t, ndim=3, mode='c'] nextEntries = np.empty(dtype='<u8', shape=(nvc, ranges.shape[0], 3))
    cdef np.uint64_t [:, :, :] nextEntries_view = nextEntries
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] neIndex = np.zeros(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] neIndex_view = neIndex
    
    #these are things we need to process results of the sub-threads
    cdef unsigned long resultStart
    cdef unsigned long resultOffset
    cdef bint resultChangesMade
    
    #FM-index values at the start of a range
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmStart0 = np.empty(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] fmStart0_view = fmStart0
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmStart1 = np.empty(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] fmStart1_view = fmStart1
    
    #we update this index as we iterate through our range
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmCurrent0 = np.empty(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] fmCurrent0_view = fmCurrent0
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmCurrent1 = np.empty(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] fmCurrent1_view = fmCurrent1
    
    #boolean indicating if we made a change
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] symbolChange = np.zeros(dtype='<u1', shape=(nvc, ))
    cdef np.uint8_t [:] symbolChange_view = symbolChange
    
    #a view of the input ranges, it's a [# entries, 3] shape always
    cdef np.uint64_t [:, :] ranges_view = ranges
    
    #these values get pulled from each entry as needed
    cdef unsigned long startIndex0 = 0
    cdef unsigned long startIndex1 = 0
    cdef unsigned long dist
    
    #fill in the initial indices
    bwt0.fillFmAtIndex(fmCurrent0_view, 0)
    bwt1.fillFmAtIndex(fmCurrent1_view, 0)
    
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] byteMask = np.empty(dtype='<u1', shape=(nvc, ))
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] byteIDs = np.empty(dtype='<u8', shape=(nvc, ))
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currBytes = np.empty(dtype='<u1', shape=(nvc, ))
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] byteUses = np.empty(dtype='<u1', shape=(nvc, ))
    cdef np.uint8_t [:] byteMask_view = byteMask
    cdef np.uint64_t [:] byteIDs_view = byteIDs
    cdef np.uint8_t [:] currBytes_view = currBytes
    cdef np.uint8_t [:] byteUses_view = byteUses
    
    cdef unsigned long binBits0 = bwt0.getBinBits()
    cdef unsigned long binBits1 = bwt1.getBinBits()
    cdef unsigned long binSize0 = 2**binBits0
    cdef unsigned long binSize1 = 2**binBits1
    
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currentBin0 = np.empty(dtype='<u1', shape=(binSize0, ))
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currentBin1 = np.empty(dtype='<u1', shape=(binSize1, ))
    cdef np.uint8_t [:] currentBin0_view = currentBin0
    cdef np.uint8_t [:] currentBin1_view = currentBin1
    cdef unsigned long currentBinID0 = 0
    cdef unsigned long currentBinID1 = 0
    cdef unsigned long currentBinUse0 = 0
    cdef unsigned long currentBinUse1 = 0
    
    #fill in our initial bins
    bwt0.fillBin(currentBin0_view, currentBinID0)
    bwt1.fillBin(currentBin1_view, currentBinID1)
    
    if numThreads <= 1:
        #go through each input range
        for y in range(0, ranges.shape[0]):
            #need to calculate the FM-index for this range
            #check if we already have the right bin loaded
            if currentBinID0 == (ranges_view[y,0] >> binBits0):
                #right bin, just need to advance the indices
                for x in range(currentBinUse0, ranges_view[y,0] - (currentBinID0 << binBits0)):
                    fmCurrent0_view[currentBin0_view[x]] += 1
                startIndex0 = ranges_view[y,0]
                currentBinUse0 = ranges_view[y,0] - (currentBinID0 << binBits0)
            else:
                #wrong bin, need to get the right indexing and the right bin
                startIndex0 = ranges_view[y,0]
                bwt0.fillFmAtIndex(fmCurrent0_view, startIndex0)
                
                currentBinID0 = startIndex0 >> binBits0
                currentBinUse0 = startIndex0 - (currentBinID0 << binBits0)
                bwt0.fillBin(currentBin0_view, currentBinID0)
                
            if currentBinID1 == (ranges_view[y,1] >> binBits1):
                #right bin, just need to advance the indices
                for x in range(currentBinUse1, ranges_view[y,1] - (currentBinID1 << binBits1)):
                    fmCurrent1_view[currentBin1_view[x]] += 1
                startIndex1 = ranges_view[y,1]
                currentBinUse1 = ranges_view[y,1] - (currentBinID1 << binBits1)
            else:
                #wrong bin, need to get the right indexing and the right bin
                startIndex1 = ranges_view[y,1]
                bwt1.fillFmAtIndex(fmCurrent1_view, startIndex1)
                
                currentBinID1 = startIndex1 >> binBits1
                currentBinUse1 = startIndex1 - (currentBinID1 << binBits1)
                bwt1.fillBin(currentBin1_view, currentBinID1)
            
            #copy the current into the start
            for x in range(0, nvc):
                fmStart0_view[x] = fmCurrent0_view[x]
                fmStart1_view[x] = fmCurrent1_view[x]
            
            #this is data related to the merge bits
            for x in range(0, nvc):
                byteIDs_view[x] = (fmStart0_view[x]+fmStart1_view[x]) >> 3
                byteUses_view[x] = (fmStart0_view[x]+fmStart1_view[x]) & 0x7
                byteMask_view[x] = 0xFF >> (8-byteUses_view[x])
                currBytes_view[x] = 0
                
            #regardless, we pull out how far we need to go now
            dist = ranges_view[y,2]
            
            #clear our symbol change area
            for x in range(0, nvc):
                symbolChange_view[x] = 0
            
            #go through each bit in this interleave range
            for x in range(startIndex0+startIndex1, startIndex0+startIndex1+dist):
                readID = getBit_p(inputInter_view, x)
                if readID:
                    symbol = currentBin1_view[currentBinUse1]
                    currentBinUse1 += 1
                    
                    if currentBinUse1 >= binSize1:
                        currentBinID1 += 1
                        bwt1.fillBin(currentBin1_view, currentBinID1)
                        currentBinUse1 = 0
                    
                    currBytes_view[symbol] ^= (0x1 << byteUses_view[symbol])
                    fmCurrent1_view[symbol] += 1
                else:
                    symbol = currentBin0_view[currentBinUse0]
                    currentBinUse0 += 1
                    
                    if currentBinUse0 >= binSize0:
                        currentBinID0 += 1
                        bwt0.fillBin(currentBin0_view, currentBinID0)
                        currentBinUse0 = 0
                    
                    fmCurrent0_view[symbol] += 1
                
                byteUses_view[symbol] += 1
                if (byteUses_view[symbol] & 0x8):
                    currBytes_view[symbol] ^= byteMask_view[symbol] & outputInter_view[byteIDs_view[symbol]]
                    
                    #write it
                    if outputInter_view[byteIDs_view[symbol]] != currBytes_view[symbol]:
                        symbolChange_view[symbol] = True
                        outputInter_view[byteIDs_view[symbol]] = currBytes_view[symbol]
                    
                    #reset
                    byteIDs_view[symbol] += 1
                    currBytes_view[symbol] = 0
                    byteUses_view[symbol] = 0
                    byteMask_view[symbol] = 0x00
                    
            #write the changes to our nextEntries
            for x in range(0, nvc):
                if byteUses_view[x] > 0:
                    byteMask_view[x] ^= (0xFF << byteUses_view[x])
                    currBytes_view[x] ^= outputInter_view[byteIDs_view[x]] & byteMask_view[x]
                    if outputInter_view[byteIDs_view[x]] != currBytes_view[x]:
                        symbolChange_view[x] = True
                        outputInter_view[byteIDs_view[x]] = currBytes_view[x]
                
                if symbolChange_view[x]:
                    nextEntries_view[x,neIndex_view[x],0] = fmStart0_view[x]
                    nextEntries_view[x,neIndex_view[x],1] = fmStart1_view[x]
                    nextEntries_view[x,neIndex_view[x],2] = fmCurrent0_view[x]+fmCurrent1_view[x]-fmStart0_view[x]-fmStart1_view[x]
                    neIndex_view[x] += 1
                    changesMade = True
    
    #create our final output array
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] extendedEntries = np.empty(dtype='<u8', shape=(nvc*ranges.shape[0], 3))
    
    cdef np.uint64_t [:, :] extendedEntries_view = extendedEntries
    cdef unsigned long exIndex = 0
    
    #track these values so we can find ranges we missed
    cdef unsigned long start
    cdef unsigned long end, prevEnd
    cdef unsigned long input0c, input1c, output0c, output1c, prev0c, prev1c
    
    #these correspond to hidden ranges
    cdef unsigned long hiddenStart0, hiddenStart1, nextStart
    
    #init
    end = 0
    input0c = input1c = output0c = output1c = 0
    
    #go through each letter range in symbol order
    for z in range(0, nvc):
        #go through each entry for that symbol
        for x in range(0, neIndex_view[z]):
            #mark the previous end
            prevEnd = end
            
            #get the new start/end
            start = nextEntries_view[z,x,0]+nextEntries_view[z,x,1]
            end = start+nextEntries_view[z,x,2]
            
            if prevEnd != start:
                #if we're tackling a new region, clear out these counts
                input0c = input1c = output0c = output1c = 0
                prev0c = prev1c = 0
            else:
                #just update where the prev started, this basically means we're in a big change region
                prev0c = output0c
                prev1c = output1c
            
            #first handle this range
            for y in range(start, end):
                if getBit_p(inputInter_view, y):
                    input1c += 1
                else:
                    input0c += 1
                if getBit_p(outputInter_view, y):
                    output1c += 1
                else:
                    output0c += 1
            
            #append it
            extendedEntries_view[exIndex,0] = nextEntries_view[z,x,0]
            extendedEntries_view[exIndex,1] = nextEntries_view[z,x,1]
            extendedEntries_view[exIndex,2] = nextEntries_view[z,x,2]
            exIndex += 1
            
            #now check if there's a group after to add that we missed
            hiddenStart0 = nextEntries_view[z,x,0]+output0c-prev0c
            hiddenStart1 = nextEntries_view[z,x,1]+output1c-prev1c
            
            if x+1 < neIndex_view[z]:
                #the next one is in this range
                nextStart = nextEntries_view[z,x+1,0]+nextEntries_view[z,x+1,1]
            else:
                for y in range(z+1, nvc):
                    if neIndex_view[y] > 0:
                        nextStart = nextEntries_view[y,0,0]+nextEntries_view[y,0,1]
                        break
                else:
                    nextStart = bwtLen
            
            #we need to iterate for as long as the 0s and 1s haven't balanced back out
            while (input0c != output0c or input1c != output1c) and end < nextStart:
                #get bits and update counters
                if getBit_p(inputInter_view, end):
                    input1c += 1
                else:
                    input0c += 1
                if getBit_p(outputInter_view, end):
                    output1c += 1
                else:
                    output0c += 1
                end += 1
            
            #check if we found something, aka the dist > 0
            if end-hiddenStart0-hiddenStart1 > 0:
                #print 'hidden', extendedEntries.shape[0], exIndex
                extendedEntries_view[exIndex,0] = hiddenStart0
                extendedEntries_view[exIndex,1] = hiddenStart1
                extendedEntries_view[exIndex,2] = end-hiddenStart0-hiddenStart1
                exIndex += 1
    
    #shrink our entries so we don't just blow up in size
    extendedEntries = extendedEntries[0:exIndex]
    extendedEntries_view = extendedEntries
    
    #here's where we'll do the collapse
    cdef unsigned long shrinkIndex = 0, currIndex = 1
    #cdef bint collapseEntries = (iterCount <= 20)
    #cdef bint collapseEntries = True or (iterCount <= 20) or (exIndex >= 2**17)
    #cdef bint collapseEntries = (exIndex >= 2**10)
    #cdef bint collapseEntries = False
    if collapseEntries:
        for currIndex in xrange(1, exIndex):
            if (extendedEntries_view[shrinkIndex,0]+extendedEntries_view[shrinkIndex,1]+extendedEntries_view[shrinkIndex,2] ==
                extendedEntries_view[currIndex,0]+extendedEntries_view[currIndex,1]):
                #these are adjacent, extend the range 
                extendedEntries_view[shrinkIndex,2] += extendedEntries_view[currIndex,2]
            else:
                #not adjacent, start off the next range
                shrinkIndex += 1
                extendedEntries_view[shrinkIndex,0] = extendedEntries_view[currIndex,0]
                extendedEntries_view[shrinkIndex,1] = extendedEntries_view[currIndex,1]
                extendedEntries_view[shrinkIndex,2] = extendedEntries_view[currIndex,2]
        
        #collapse down to one past the shrink index
        extendedEntries = extendedEntries[0:shrinkIndex+1]
        extendedEntries_view = extendedEntries
    #'''
    #go through each of the next entries and correct our input
    cdef unsigned long totalStart
    
    #go through each symbol range we just changed and copy it back into our input
    for z in range(0, nvc):
        for y in range(0, neIndex_view[z]):
            #get the start and distance
            totalStart = nextEntries_view[z,y,0]+nextEntries_view[z,y,1]
            dist = nextEntries_view[z,y,2]
            
            #copy the first sub-byte
            while totalStart % 8 != 0 and dist > 0:
                if getBit_p(outputInter_view, totalStart):
                    setBit_p(inputInter_view, totalStart)
                else:
                    clearBit_p(inputInter_view, totalStart)
                totalStart += 1
                dist -= 1
            
            #copy all middle bytes
            for x in xrange(totalStart/8, (totalStart+dist)/8):
                inputInter_view[x] = outputInter_view[x]
            
            #copy the last sub-byte
            for x in range(((totalStart+dist)/8)*8, totalStart+dist):
                if getBit_p(outputInter_view, x):
                    setBit_p(inputInter_view, x)
                else:
                    clearBit_p(inputInter_view, x)
    
    #return a simple tuple that is (boolean, numpy array)
    cdef tuple ret = (changesMade, extendedEntries)
    return ret

cdef inline void setBit_p(np.uint8_t * bitArray, unsigned long index) nogil:
    #set a bit in an array
    bitArray[index >> 3] |= (0x1 << (index & 0x7))

cdef inline void clearBit_p(np.uint8_t * bitArray, unsigned long index) nogil:
    #clear a bit in an array
    bitArray[index >> 3] &= ~(0x1 << (index & 0x7))

cdef inline bint getBit_p(np.uint8_t * bitArray, unsigned long index) nogil:
    #get a bit from an array
    return (bitArray[index >> 3] >> (index & 0x7)) & 0x1
