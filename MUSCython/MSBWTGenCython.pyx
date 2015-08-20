#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False

'''
Created on Mar 19, 2014

Contains multiple BWT generation algorithms including a the column-wise approach of
Bauer et al., 2011.  A lot of this is re-implementation of previous work to get to 
the baseline of having MSBWTs to work with.

@author: holtjma
'''

import bisect
import copy
import gc
import math
import multiprocessing
import os
import pickle
import time

import MultiStringBWTCython as MultiStringBWT

import numpy as np
cimport numpy as np
#from cython.operator cimport preincrement as inc
from multiprocessing.pool import ThreadPool

def createMsbwtFromSeqs(bwtDir, unsigned int numProcs, logger):
    '''
    This function assumes the strings are all uniform length and that that uniform length is the
    values stored in offsetFN
    @param seqFNPrefix - the prefix of the sequence files
    @param offsetFN - a file containing offset information, should just be one value for this func
    @param numProcs - number of processes to use
    @param logger - a logger to dump output to
    '''
    #timing metrics, only for logging
    cdef double st, et, stc, etc, totalStartTime, totalEndTime
    st = time.time()
    totalStartTime = st
    stc = time.clock()
    
    #hardcoded values
    cdef unsigned int numValidChars = 6
    
    #clear anything that may already have been associated with our directory
    clearAuxiliaryData(bwtDir)
    
    #construct pre-arranged file names, God forbid we ever change these...
    bwtFN = bwtDir+'/msbwt.npy'
    seqFNPrefix = bwtDir+'/seqs.npy'
    offsetFN = bwtDir+'/offsets.npy'
    
    #offsets is really just storing the length of all string at this point
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] offsets = np.load(offsetFN, 'r+')
    cdef unsigned long seqLen = offsets[0]
    
    #finalSymbols stores the last real symbols in the strings (not the '$', the ones right before)
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] finalSymbols = np.load(seqFNPrefix+'.1.npy', 'r+')
    cdef np.uint8_t [:] finalSymbols_view = finalSymbols
    cdef unsigned long numSeqs = finalSymbols.shape[0]
    
    logger.warning('Beta version of Cython creation')
    logger.info('Preparing to merge '+str(numSeqs)+' sequences...')
    logger.info('Generating level 1 insertions...')
    
    #1 - load <DIR>/seqs.npy.<seqLen-2>.npy, these are the characters before the '$'s
    #    use them as the original inserts storing <insert position, insert character, sequence ID>
    #    note that inserts should be stored in a minimum of 6 files, one for each character
    initialInsertsFN = bwtDir+'/inserts.initial.npy'
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] initialInserts = np.lib.format.open_memmap(initialInsertsFN, 'w+', '<u8', (numSeqs, 3))
    cdef np.uint64_t [:, :] initialInserts_view = initialInserts
    
    #some basic counting variables
    cdef unsigned long i, j, c, columnID, c2
    
    #this will still the initial counts of symbols in the final column of our strings
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] initialFmDeltas = np.zeros(dtype='<u8', shape=(numValidChars, ))
    cdef np.uint64_t [:] initialFmDeltas_view = initialFmDeltas
    
    with nogil:
        for i in range(0, numSeqs):
            #index to insert, symbol, sequence
            initialInserts_view[i,0] = i
            initialInserts_view[i,1] = finalSymbols_view[i]
            initialInserts_view[i,2] = i
            initialFmDeltas_view[finalSymbols_view[i]] += 1
    
    #fmStarts is basically an fm-index offset, fmdeltas tells us how much fmStarts should change each iterations
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmStarts = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] fmStarts_view = fmStarts
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmDeltas = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] fmDeltas_view = fmDeltas
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmEnds = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] fmEnds_view = fmEnds
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] nextFmDeltas = np.empty(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] nextFmDeltas_view = nextFmDeltas
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] retFmDeltas
    cdef np.uint64_t [:, :] retFmDeltas_view
    
    #fmDeltas[0][:] = initialFmDeltas[:]
    for i in range(0, numValidChars):
        fmDeltas_view[0,i] = initialFmDeltas[i]
    
    #figure out the files we insert in each iteration
    cdef dict insertionFNDict = {}
    cdef dict nextInsertionFNDict = {}
    for c in range(0, numValidChars):
        #create an initial empty region for each symbol
        np.lib.format.open_memmap(bwtDir+'/state.'+str(c)+'.0.npy', 'w+', '<u1', (0, ))
        #np.memmap(bwtDir+'/state.'+str(c)+'.0.npy', '<u1', 'w+', shape=(0, ))
        
        #create an empty list of files for insertion
        insertionFNDict[c] = []
        nextInsertionFNDict[c] = []
        
    #add the single insertion file we generated earlier
    insertionFNDict[0].append(initialInsertsFN)
    
    etc = time.clock()
    et = time.time()
    logger.info('Finished init in '+str(et-st)+' seconds.')
    logger.info('Beginning iterations...')
    
    cdef unsigned long cumsum
    
    #2 - go back one column at a time, building new inserts
    for columnID in range(0, seqLen):
        st = time.time()
        stc = time.clock()
        
        #first take into accoun the new fmDeltas coming in from all insertions
        #fmStarts = fmStarts+(np.cumsum(fmDeltas, axis=0)-fmDeltas)
        #fmEnds = fmEnds+np.cumsum(fmDeltas, axis=0)
        for i in range(0, numValidChars):
            cumsum = 0
            for j in range(0, numValidChars-1):
                cumsum += fmDeltas_view[j,i]
                fmStarts_view[j+1,i] += cumsum
                fmEnds_view[j,i] += cumsum
                nextFmDeltas_view[j,i] = 0
            cumsum += fmDeltas_view[numValidChars-1,i]
            fmEnds_view[numValidChars-1,i] += cumsum
            nextFmDeltas_view[numValidChars-1,i] = 0
        
        #clear out the next fmDeltas
        #nextFmDeltas = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
        
        tups = []
        for c in range(0, numValidChars):
            currentSymbolFN = bwtDir+'/state.'+str(c)+'.'+str(columnID)+'.npy'
            nextSymbolFN = bwtDir+'/state.'+str(c)+'.'+str(columnID+1)+'.npy'
            nextSeqFN = seqFNPrefix+'.'+str((columnID+2) % seqLen)+'.npy'
            tup = (c, np.copy(fmStarts[c]), np.copy(fmDeltas[c]), np.copy(fmEnds[c]), insertionFNDict[c], currentSymbolFN, nextSymbolFN, bwtDir, columnID, nextSeqFN)
            tups.append(tup)
        '''
        rets = []
        for tup in tups:
            ret = iterateMsbwtCreate(tup)
            rets.append(ret)
        '''
        
        myPool = multiprocessing.Pool(numProcs)
        rets = myPool.imap(iterateMsbwtCreate, tups)
        
        for ret in rets:
            #update the fmDeltas
            #nextFmDeltas = nextFmDeltas + ret[0]
            retFmDeltas = ret[0]
            retFmDeltas_view = retFmDeltas
            for i in range(0, numValidChars):
                for j in range(0, numValidChars):
                    nextFmDeltas_view[i,j] += retFmDeltas_view[i,j]
            
            #update our insertions files
            for c2 in range(0, numValidChars):
                if ret[1][c2] == None:
                    #no new file
                    pass
                else:
                    nextInsertionFNDict[c2].append(ret[1][c2])
        
        myPool.close()
        
        #remove the old insertions and old state files
        for c in insertionFNDict:
            for fn in insertionFNDict[c]:
                try:
                    os.remove(fn)
                except:
                    logger.warning('Failed to remove \''+fn+'\' from file system.')
            
            if len(insertionFNDict[c]) > 0:
                try:
                    rmStateFN = bwtDir+'/state.'+str(c)+'.'+str(columnID)+'.npy'
                    os.remove(rmStateFN)
                except:
                    logger.warning('Failed to remove\''+rmStateFN+'\' from file system.')
        
        #copy the fmDeltas and insertion filenames
        #fmDeltas[:] = nextFmDeltas[:]
        for i in range(0, numValidChars):
            for j in range(0, numValidChars):
                fmDeltas_view[i,j] = nextFmDeltas_view[i,j]
        
        insertionFNDict = nextInsertionFNDict
        nextInsertionFNDict = {}
        for c in range(0, numValidChars):
            nextInsertionFNDict[c] = []
        
        etc = time.clock()
        et = time.time()
        logger.info('Finished iteration '+str(columnID+1)+' in '+str(et-st)+' seconds('+str(etc-stc)+' clock)...')
    
    logger.info('Creating final output...')
    
    #finally, join all the subcomponents
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] tempBWT
    cdef np.uint8_t [:] tempBWT_view
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] finalBWT
    cdef np.uint8_t [:] finalBWT_view
    
    cdef unsigned long totalLength = 0
    for c in range(0, numValidChars):
        tempBWT = np.load(bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.npy', 'r')
        totalLength += tempBWT.shape[0]
        
    #prepare the final structure for copying
    cdef unsigned long finalInd = 0
    cdef unsigned long tempLen
    finalBWT = np.lib.format.open_memmap(bwtFN, 'w+', '<u1', (totalLength, ))
    finalBWT_view = finalBWT
    
    for c in range(0, numValidChars):
        stateFN = bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.npy'
        if os.path.exists(stateFN):
            tempBWT = np.load(stateFN, 'r+')
        else:
            continue
        tempBWT_view = tempBWT
        tempLen = tempBWT.shape[0]
        
        with nogil:
            for i in range(0, tempLen):
                finalBWT_view[finalInd] = tempBWT_view[i]
                finalInd += 1
    
    #finally, clear the last state files
    tempBWT = None
    for c in range(0, numValidChars):
        for fn in insertionFNDict[c]:
            try:
                os.remove(fn)
            except:
                logger.warning('Failed to remove \''+fn+'\' from file system.')
        
        try:
            if os.path.exists(bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.npy'):
                os.remove(bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.npy')
        except:
            logger.warning('Failed to remove \''+bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.npy'+'\' from file system.')
    
    logger.info('Final output saved to \''+bwtFN+'\'.')
    totalEndTime = time.time()
    logger.info('Finished all iterations in '+str(totalEndTime-totalStartTime)+' seconds.')
            
def iterateMsbwtCreate(tuple tup):
    '''
    This function will perform the insertion iteration for a single symbol grouping
    '''
    #extract our inputs from the tuple
    cdef np.uint8_t idChar
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmIndex
    cdef np.uint64_t [:] fmIndex_view
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmEndex
    cdef np.uint64_t [:] fmEndex_view
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] prevFmDelta
    cdef np.uint64_t [:] prevFmDelta_view
    cdef unsigned long column
    (idChar, fmIndex, prevFmDelta, fmEndex, insertionFNs, currentSymbolFN, nextSymbolFN, bwtDir, column, nextSeqFN) = tup
    fmIndex_view = fmIndex
    fmEndex_view = fmEndex
    prevFmDelta_view = prevFmDelta
    
    #hardcoded
    cdef unsigned int numValidChars = 6
    
    #this stores the number of symbols we find in our range
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmDeltas = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] fmDeltas_view = fmDeltas
    
    #the input partial BWT for suffixes starting with 'idChar'
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currentBwt = np.load(currentSymbolFN, 'r+')
    #cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currentBwt = np.load(currentSymbolFN)
    cdef np.uint8_t [:] currentBwt_view = currentBwt
    
    #the output partial BWT for suffixes starting with 'idChar'
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] nextBwt
    cdef np.uint8_t [:] nextBwt_view
    
    #currentBWT + inserts = nextBWT
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] inserts
    cdef np.uint64_t [:, :] inserts_view
    
    #counting variables
    cdef unsigned long currIndex, insertLen, insertIndex, prevIndex, totalNewLen
    cdef unsigned long i, j
    cdef np.uint8_t c, symbol, nextSymbol
    
    #These values contain the numpy arrays for the new inserts, created below
    outputInserts = []
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] outputInsertPointers = np.zeros(dtype='<u8', shape=(numValidChars, ))
    cdef np.uint64_t [:] outputInsertPointers_view = outputInsertPointers
    cdef np.uint64_t * outputInsert_p
    
    #these indices will be used for knowing where we are in the outputs
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] outputInsertIndices = np.zeros(dtype='<u8', shape=(numValidChars, ))
    cdef np.uint64_t [:] outputInsertIndices_view = outputInsertIndices
    cdef np.uint64_t ind
    
    #these are temp array used to extract pointers
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] newInsertArray
    cdef np.uint64_t [:, :] newInsertArray_view
    
    #the symbols we are noting as being inserted next
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currSeqs = np.load(nextSeqFN, 'r+')
    cdef np.uint8_t [:] currSeqs_view = currSeqs
    
    retFNs = [None]*numValidChars
    
    if len(insertionFNs) == 0:
        #we don't need to do anything except rename our file
        #print str(idChar)+', No change'
        if os.path.exists(currentSymbolFN):
            os.rename(currentSymbolFN, nextSymbolFN)
    else:
        #first we need to count how big the new iteration is
        totalInsertionCounts = np.sum(prevFmDelta)
        totalNewLen = currentBwt.shape[0]+totalInsertionCounts
        nextBwt = np.lib.format.open_memmap(nextSymbolFN, 'w+', '<u1', (totalNewLen, ))
        nextBwt_view = nextBwt
        
        #allocate a numpy array for each new insert file
        for c in range(0, numValidChars):
            if prevFmDelta_view[c] == 0:
                #nothing to insert from c
                outputInserts.append(None)
            else:
                #add the insert to our return
                newInsertFN = bwtDir+'/inserts.'+str(c)+str(idChar)+'.'+str(column+1)+'.npy'
                retFNs[c] = newInsertFN
                
                #create the insert file, store pointers
                newInsertArray = np.lib.format.open_memmap(newInsertFN, 'w+', '<u8', (prevFmDelta_view[c], 3))
                newInsertArray_view = newInsertArray
                outputInserts.append(newInsertArray)
                outputInsertPointers_view[c] = <np.uint64_t> &(newInsertArray_view[0][0])
        
        #now build the iteration, we'll start at zero obviously
        prevIndex = 0
        currIndex = 0
        
        #go through each file, one at a time
        #for fn in insertionFNs:
        for fn in insertionFNs:
            #load the actual inserts
            #inserts = np.load(fn, 'r+')
            inserts = np.load(fn)
            inserts_view = inserts
            insertLen = inserts.shape[0]
            
            with nogil:
                #go through each insert
                for i in range(0, insertLen):
                    #get the position of the new insert
                    insertIndex = inserts_view[i,0]
                    
                    #just copy up to that position
                    for j in range(prevIndex, insertIndex):
                        symbol = currentBwt_view[currIndex]
                        nextBwt_view[j] = symbol
                        
                        fmIndex_view[symbol] += 1
                        currIndex += 1
                        
                    #now we actually write the value from the insert
                    symbol = inserts_view[i,1]
                    nextBwt_view[insertIndex] = symbol
                    
                    nextSymbol = currSeqs_view[inserts_view[i,2]]
                    fmDeltas_view[symbol, nextSymbol] += 1
                    prevIndex = insertIndex+1
                    
                    #now we need to add the information for our next insertion
                    outputInsert_p = <np.uint64_t *> outputInsertPointers_view[symbol]
                    ind = outputInsertIndices_view[symbol]
                    outputInsertIndices_view[symbol] += 3
                    
                    #finally, store the values
                    outputInsert_p[ind] = fmIndex_view[symbol]
                    fmIndex_view[symbol] += 1
                    outputInsert_p[ind+1] = nextSymbol
                    outputInsert_p[ind+2] = inserts_view[i,2]
                    
        with nogil:
            #at the end we need to copy all values that come after the final insert
            for j in range(prevIndex, totalNewLen):
                symbol = currentBwt_view[currIndex]
                nextBwt_view[j] = symbol
                
                fmIndex_view[symbol] += 1
                currIndex += 1
                
    ret = (np.copy(fmDeltas), retFNs)
    return ret
    
def compressBWT(char * inputFN, char * outputFN, unsigned long numProcs, logger):
    '''
    Current encoding scheme uses 3 LSB for the letter and 5 MSB for a count, note that consecutive ones of the same character
    combine to create one large count.  So to represent 34A, you would have 00010|001 followed by 00001|001 which can be though of
    as 2*1 + 32*1 = 34
    @param inputFN - the filename of the BWT to compress
    @param outputFN - the destination filename for the compressed BWT, .npy format
    @param numProcs - number of processes to use during compressing
    @param logger - logger from initLogger()
    '''
    #create bit spacings
    cdef unsigned long letterBits = 3
    cdef unsigned long numberBits = 8-letterBits
    cdef unsigned long numPower = 2**numberBits
    cdef np.uint8_t mask = 255 >> letterBits
    
    #load the thing to compress
    logger.info('Loading src file...')
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] bwt = np.load(inputFN, 'r+')
    logger.info('Original size:'+str(bwt.shape[0])+'B')
    numProcs = min(numProcs, bwt.shape[0])
    
    #first locate boundaries
    cdef list tups = []
    cdef unsigned long binSize = 2**20
    cdef unsigned long numBins = max(numProcs, bwt.shape[0]/binSize)
    
    cdef unsigned long i, startIndex, endIndex
    for i in range(0, numBins):
        startIndex = i*bwt.shape[0]/numBins
        endIndex = (i+1)*bwt.shape[0]/numBins
        tempFN = outputFN+'.temp.'+str(i)+'.npy'
        tups.append((inputFN, startIndex, endIndex, tempFN))
    
    logger.info('Compressing bwt...')
    
    #run our multi-processed builder
    cdef list rets
    if numProcs > 1:
        myPool = multiprocessing.Pool(numProcs)
        rets = myPool.map(compressBWTPoolProcess, tups)
    else:
        rets = []
        for tup in tups:
            rets.append(compressBWTPoolProcess(tup))
    
    #calculate how big it will be after combining the separate chunk
    cdef unsigned long totalSize = 0
    cdef long prevChar = -1
    cdef unsigned long prevTotal = 0
    for ret in rets:
        #start by just adding the raw size
        totalSize += ret[0]
        
        #check if we need to compensate for a break point like AAA|A
        if prevChar == ret[1]:
            totalSize -= int(math.floor(math.log(prevTotal, numPower)+1)+math.floor(math.log(ret[2], numPower)+1))
            prevTotal += ret[2]
            totalSize += int(math.floor(math.log(prevTotal, numPower)+1))
            if ret[0] == 1 or ret[6] == 1:
                #don't clear prev total
                pass
            else:
                prevTotal = ret[4]
        else:
            prevTotal = ret[4]
        prevChar = ret[3]
    
    #make the real output by joining all of the partial compressions into a single file
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] finalBWT = np.lib.format.open_memmap(outputFN, 'w+', '<u1', (totalSize,))
    cdef np.uint8_t [:] finalBWT_view = finalBWT
    logger.info('Calculated compressed size:'+str(totalSize)+'B')
    logger.info('Joining sub-compressions...')
    
    #iterate a second time, this time storing things
    prevChar = -1
    prevTotal = 0
    cdef unsigned long offset = 0
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] copyArr
    #cdef np.uint8_t [:] copyArr_view
    
    cdef unsigned long prevBytes, nextBytes, power
    for ret in rets:
        copyArr = np.load(ret[5], 'r+')
        #copyArr_view = copyArr
        if prevChar == ret[1]:
            #calculate byte usage of combining
            prevBytes = int(math.floor(math.log(prevTotal, numPower)+1))
            nextBytes = int(math.floor(math.log(ret[2], numPower)+1))
            prevTotal += ret[2]
            
            #actually combine them
            offset -= prevBytes
            power = 0
            while prevTotal >= numPower**power:
                finalBWT_view[offset] = (((prevTotal / (numPower**power)) & mask) << letterBits)+prevChar
                power += 1
                offset += 1
            
            #copy over the extra stuff and calculate the new offset
            finalBWT[offset:offset+(copyArr.shape[0]-nextBytes)] = copyArr[nextBytes:]
            offset += (copyArr.shape[0] - nextBytes)
            
            if ret[0] == 1 or ret[6] == 1:
                pass
            else:
                prevTotal = ret[4]
            
        else:
            #nothing shared, just copy the easy way
            finalBWT[offset:offset+copyArr.shape[0]] = copyArr
            offset += copyArr.shape[0]
            prevTotal = ret[4]
            
        prevChar = ret[3]
        
    #clear all intermediate files
    for ret in rets:
        os.remove(ret[5])
    
    logger.info('Compression finished.')
    
    #return this i guess
    return finalBWT
    
def compressBWTPoolProcess(tuple tup):
    '''
    During compression, each available process will calculate a subportion of the BWT independently using this 
    function.  This process takes the chunk and rewrites it into a given filename using the technique described
    in the compressBWT(...) function header
    '''
    #pull the tuple info
    cdef char * inputFN = tup[0]
    cdef unsigned long startIndex = tup[1]
    cdef unsigned long endIndex = tup[2]
    cdef char * tempFN = tup[3]
    
    #this shouldn't happen
    if startIndex == endIndex:
        print 'ERROR: EQUAL INDICES'
        return None
    
    #load the file
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] bwt = np.load(inputFN, 'r+')
    cdef np.uint8_t [:] bwt_view = bwt
    
    #create bit spacings
    cdef unsigned long letterBits = 3
    cdef unsigned long numberBits = 8-letterBits
    cdef unsigned long numPower = 2**numberBits
    cdef np.uint8_t mask = 255 >> letterBits
    
    #search for the places they're different
    cdef np.ndarray[np.int64_t, ndim=1, mode='c'] whereSol = np.add(startIndex+1, np.where(bwt[startIndex:endIndex-1] != bwt[startIndex+1:endIndex])[0])
    
    #this is the difference between two adjacent ones
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] deltas = np.zeros(dtype='<u4', shape=(whereSol.shape[0]+1,))
    cdef np.uint32_t [:] deltas_view = deltas
    if whereSol.shape[0] == 0:
        deltas_view[0] = endIndex-startIndex
    else:
        deltas_view[0] = whereSol[0]-startIndex
        deltas[1:deltas.shape[0]-1] = np.subtract(whereSol[1:], whereSol[0:whereSol.shape[0]-1])
        deltas_view[deltas.shape[0]-1] = endIndex - whereSol[whereSol.shape[0]-1]
    
    #calculate the number of bytes we need to store this information
    cdef unsigned long size = 0
    cdef unsigned long byteCount = 0
    cdef unsigned long lastCount = 1
    while lastCount > 0:
        lastCount = np.where(deltas >= 2**(numberBits*byteCount))[0].shape[0]
        size += lastCount
        byteCount += 1
    
    #create the file
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] ret = np.lib.format.open_memmap(tempFN, 'w+', '<u1', (size,))
    cdef np.uint8_t [:] ret_view = ret
    cdef unsigned long retIndex = 0
    cdef np.uint8_t c = bwt_view[startIndex]
    startChar = c
    delta = deltas_view[0]
    while delta > 0:
        ret_view[retIndex] = ((delta & mask) << letterBits)+c
        delta /= numPower
        retIndex += 1
    
    #fill in the values based on the bit functions
    cdef unsigned long i
    for i in range(0, whereSol.shape[0]):
        c = bwt_view[whereSol[i]]
        delta = deltas_view[i+1]
        while delta > 0:
            ret_view[retIndex] = ((delta & mask) << letterBits)+c
            delta /= numPower
            retIndex += 1
    endChar = c
    
    #return a lot of information so we can easily combine the results
    return (size, startChar, deltas[0], endChar, deltas[deltas.shape[0]-1], tempFN, deltas.shape[0])
    
##########################################################################################
#Everything below here has not been cythonized yet
##########################################################################################
def decompressBWT(inputDir, outputDir, numProcs, logger):
    '''
    This is called for taking a BWT and decompressing it back out to it's original form.  While unusual to do,
    it's included in this package for completion purposes.
    @param inputDir - the directory of the compressed BWT we plan on decompressing
    @param outputFN - the directory for the output decompressed BWT, it can be the same, we don't care
    @param numProcs - number of processes we're allowed to use
    @param logger - log all the things!
    '''
    #load it, force it to be a compressed bwt also
    #msbwt = MultiStringBWT.CompressedMSBWT()
    #msbwt.loadMsbwt(inputDir, logger)
    msbwt = MultiStringBWT.loadBWT(inputDir, logger)
    
    #make the output file
    outputFile = np.lib.format.open_memmap(outputDir+'/msbwt.npy', 'w+', '<u1', (msbwt.getTotalSize(),))
    del outputFile
    
    worksize = 2**20
    tups = [None]*(msbwt.getTotalSize()/worksize+1)
    x = 0
    
    if msbwt.getTotalSize() > worksize:
        for x in range(0, msbwt.getTotalSize()/worksize):
            tups[x] = (inputDir, outputDir, x*worksize, (x+1)*worksize)
        tups[len(tups)-1] = (inputDir, outputDir, (x+1)*worksize, msbwt.getTotalSize())
    else:
        tups[0] = (inputDir, outputDir, 0, msbwt.getTotalSize())
        
    if numProcs > 1:
        myPool = multiprocessing.Pool(numProcs)
        rets = myPool.map(decompressBWTPoolProcess, tups)
    else:
        rets = []
        for tup in tups:
            rets.append(decompressBWTPoolProcess(tup))
    
    #as of now, nothing to do we rets, so we're finished
    
def decompressBWTPoolProcess(tup):
    '''
    Individual process for decompression
    '''
    (inputDir, outputDir, startIndex, endIndex) = tup
    
    if startIndex == endIndex:
        return True
    
    #load the thing we'll be extracting from
    #msbwt = MultiStringBWT.CompressedMSBWT()
    #msbwt.loadMsbwt(inputDir, None)
    msbwt = MultiStringBWT.loadBWT(inputDir)
    
    #open our output
    outputBwt = np.load(outputDir+'/msbwt.npy', 'r+')
    outputBwt[startIndex:endIndex] = msbwt.getBWTRange(startIndex, endIndex)   
    
    return True 

def clearAuxiliaryData(dirName):
    '''
    This function removes auxiliary files associated with a given filename
    '''
    if dirName != None:
        if os.path.exists(dirName+'/auxiliary.npy'):
            os.remove(dirName+'/auxiliary.npy')
        
        if os.path.exists(dirName+'/totalCounts.p'):
            os.remove(dirName+'/totalCounts.p')
        
        if os.path.exists(dirName+'/totalCounts.npy'):
            os.remove(dirName+'/totalCounts.npy')
        
        if os.path.exists(dirName+'/fmIndex.npy'):
            os.remove(dirName+'/fmIndex.npy')
            
        if os.path.exists(dirName+'/comp_refIndex.npy'):
            os.remove(dirName+'/comp_refIndex.npy')
            
        if os.path.exists(dirName+'/comp_fmIndex.npy'):
            os.remove(dirName+'/comp_fmIndex.npy')
        
        if os.path.exists(dirName+'/backrefs.npy'):
            os.remove(dirName+'/backrefs.npy')
        
        if os.path.exists(dirName+'/lzw_fmIndex.npy'):
            os.remove(dirName+'/lzw_fmIndex.npy')

def writeSeqsToFiles(np.ndarray[np.uint8_t, ndim=1, mode='c'] seqArray, seqFNPrefix, offsetFN, uniformLength):    
    '''
    This function takes a seqArray and saves the values to a memmap file that can be accessed for multi-processing.
    Additionally, it saves some offset indices in a numpy file for quicker string access.
    @param seqArray - the list of '$'-terminated strings to be saved
    @param fnPrefix - the prefix for the temporary files, creates a prefix+'.seqs.npy' and prefix+'.offsets.npy' file
    '''
    
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] dArr
    cdef np.uint8_t [:] dArr_view
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] seqs
    cdef np.uint8_t [:] seq_view
    cdef np.uint8_t [:] seqArray_view = seqArray
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] seqOutArrayPointers
    cdef np.uint64_t [:] seqOutArrayPointers_view
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] offsets
    
    cdef unsigned long i, j
    cdef unsigned long k = 0
    cdef unsigned long seqLen
    cdef unsigned long numSeqs
    
    if uniformLength:
        #first, store the uniform size in our offsets file
        offsets = np.lib.format.open_memmap(offsetFN, 'w+', '<u8', (1,))
        offsets[0] = uniformLength
        seqLen = uniformLength
        numSeqs = seqArray.shape[0]/seqLen
        
        #define a constant character map for now
        d = {'$':0, 'A':1, 'C':2, 'G':3, 'N':4, 'T':5}
        dArr = np.add(np.zeros(dtype='<u1', shape=(256,)), len(d.keys()))
        
        dArr_view = dArr
        for c in d.keys():
            dArr[ord(c)] = d[c]
        
        seqOutArray = [None]*seqLen
        seqOutArrayPointers = np.zeros(dtype='<u8', shape=(seqLen, ))
        seqOutArrayPointers_view = seqOutArrayPointers
        for i in range(0, seqLen):
            seqOutArray[seqLen-1-i] = np.lib.format.open_memmap(seqFNPrefix+'.'+str(i)+'.npy', 'w+', '<u1', (numSeqs,))
            seq_view = seqOutArray[seqLen-1-i]
            seqOutArrayPointers_view[seqLen-1-i] = <np.uint64_t> &seq_view[0]
            
        with nogil:
            for i in range(0, numSeqs):
                for j in range(0, seqLen):
                    (<np.uint8_t *>seqOutArrayPointers_view[j])[i] = dArr_view[seqArray_view[k]]
                    k += 1
        
    else:
        #count how many terminal '$' exist, 36 = '$'
        lenSums = np.add(1, np.where(seqArray == 36)[0])
        numSeqs = lenSums.shape[0]
        totalLen = lenSums[numSeqs-1]
        
        #track the total length thus far and open the files we plan to fill in
        seqFN = seqFNPrefix
        seqs = np.lib.format.open_memmap(seqFN, 'w+', '<u1', (totalLen,))
        seq_view = seqs
        offsets = np.lib.format.open_memmap(offsetFN, 'w+', '<u8', (numSeqs+1,))
        offsets[1:] = lenSums
        
        #define a constant character map for now
        d = {'$':0, 'A':1, 'C':2, 'G':3, 'N':4, 'T':5}
        dArr = np.add(np.zeros(dtype='<u1', shape=(256,)), len(d.keys()))
        for c in d.keys():
            dArr[ord(c)] = d[c]
        dArr_view = dArr
        
        #copy the values
        '''
        chunkSize = 1000000
        i = 0
        while chunkSize*i < seqArray.shape[0]:
            seqs[chunkSize*i:chunkSize*(i+1)] = dArr[seqArray[chunkSize*i:chunkSize*(i+1)]]
            i += 1
        '''
        
        for i in xrange(0, seqArray.shape[0]):
            seq_view[i] = dArr_view[seqArray_view[i]]
        
        #clear memory
        del lenSums
        del offsets
    
    #return the two filenames
    return (seqFNPrefix, offsetFN)

def createFromSeqs(seqFNPrefix, offsetFN, mergedFN, numProcs, areUniform, logger):
    '''
    This function will take a preprocessed seqFN and offsetFN file and create the MSBWT from them, it 
    is the main function to call from MSBWTGen when attempting to build a BWT
    @param seqFNPrefix - the preprocessed seqFN file(s) prefix, '.npy' format of uint8s
    @param offsetFN - the offsets into seqFN indicating the start of a string
    @param mergedFN - the final destination output, '.npy' file
    @param numProcs - the maximum number of processes to use
    '''
    totalStartTime = time.time()
    numValidChars = 6
    
    #clear anything that may already have been associated with it
    clearAuxiliaryData(mergedFN)
    
    #start with column 2 for the iterations, (this is really l(s)-2 where s is a string to merge)
    startingColumn = 2
    
    #memmap the two string data files
    offsetData = np.load(offsetFN, 'r')
    if areUniform:
        uniformLength = int(offsetData[0])
        firstSeq = np.load(seqFNPrefix+'.0.npy', 'r')
        numSeqs = firstSeq.shape[0]
    else:
        numSeqs = offsetData.shape[0]-1
    
    logger.info('Preparing to merge '+str(numSeqs)+' sequences...')
    
    #depth should always be greater than or equal to 1
    depth = max(int(math.ceil(math.log(numProcs, numValidChars-1)))+1, int(math.ceil(math.log(numSeqs/1000000.0, numValidChars))), 1)
    
    logger.info('Setting depth to '+str(depth)+'...')        
    
    #initialize the total information
    if areUniform:
        totalSize = uniformLength*numSeqs
    else:
        totalSize = offsetData[offsetData.shape[0]-1]
    bwt = np.lib.format.open_memmap(mergedFN, 'w+', '<u1', (totalSize,))
    
    #initialize the count information
    fmStarts = {}
    allBwtCounts = [0]*numValidChars
    allFirstCounts = [0]*numValidChars
    cOffset = {}
    totalCounts = {}
    
    #prepare to make the first inserts
    logger.info('Generating level 1 insertions...')
    st = time.time()
    
    tups = []
    segLen = numSeqs / numProcs
    for i in xrange(0, numProcs-1):
        tup = (offsetFN, seqFNPrefix, mergedFN, depth, numValidChars, i*segLen, (i+1)*segLen, areUniform)
        tups.append(tup)
    
    tup = (offsetFN, seqFNPrefix, mergedFN, depth, numValidChars, segLen*(numProcs-1), numSeqs, areUniform)
    tups.append(tup)
    
    if numProcs > 1:
        #create a pool of processes based on the input
        myPool = multiprocessing.Pool(numProcs)
        initRets = myPool.imap(bwtInitialInsertionsPoolCall, tups)
    else:
        initRets = []
        for tup in tups:
            initRets.append(bwtInitialInsertionsPoolCall(tup))
    
    #create our initial information based on the return
    fmDeltas = {}
    insertFNs = {}
    for initRet in initRets:
        (retFmDeltas, retInsertFNs) = initRet
        for key in retFmDeltas.keys():
            if not fmDeltas.has_key(key):
                fmDeltas[key] = [0]*numValidChars
                np.lib.format.open_memmap(mergedFN+'.'+key+'.'+str(startingColumn-1)+'.npy', 'w+', '<u8,<u1,<u4', (0,))
            fmDeltas[key] = np.add(fmDeltas[key], retFmDeltas[key])
            
            if not insertFNs.has_key(key):
                insertFNs[key] = []
            insertFNs[key] += retInsertFNs[key]    
    
    if numProcs > 1:
        myPool.terminate()
        myPool.join()
        myPool = None
    
    for key in fmDeltas.keys():
        fmStarts[key] = [0]*numValidChars
        cOffset[key] = 0
        totalCounts[key] = 0
    
    et = time.time()
    
    #prepare to do deeper iterations
    i = 0
    
    #dummy, just for entering first loop
    logger.info('Finished init in '+str(et-st)+' seconds.')
    logger.info('Beginning iterations...')
    
    totalCounts = {}
    
    iterateCreateFromSeqs(startingColumn, fmStarts, fmDeltas, allFirstCounts, allBwtCounts, cOffset, totalCounts, numValidChars,
                          mergedFN, seqFNPrefix, offsetFN, insertFNs, numProcs, areUniform, depth, logger)
    
    totalEndTime = time.time()
    logger.info('Final output saved to \''+mergedFN+'\'.')
    logger.info('Finished all iterations in '+str(totalEndTime-totalStartTime)+' seconds.')

def iterateCreateFromSeqs(startingColumn, fmStarts, fmDeltas, allFirstCounts, allBwtCounts, cOffset, totalCounts, numValidChars,
                          mergedFN, seqFNPrefix, offsetFN, insertFNs, numProcs, areUniform, depth, logger):
    '''
    This function is the actual series of iterations that a BWT creation will perform.  It's separate so we can build a function
    for resuming construction if we fail for some reason.
    TODO: build a recovery function to start midway through BWT construction.
    TODO: @param values need explanation
    '''
    bwt = np.load(mergedFN, 'r+')
    
    #TODO REMOVE
    #numProcs = 1
    
    column = startingColumn
    newInserts = True
    while newInserts:
        st = time.time()
        
        #iterate through the sorted keys
        keySort = sorted(fmStarts.keys())
        for i, key in enumerate(keySort):
            #all deltas get copied
            for c2 in range(0, numValidChars):
                #copy only to the ones after key
                allFirstCounts[int(key[0])] += fmDeltas[key][c2]
                allBwtCounts[c2] += fmDeltas[key][c2]
                for key2 in keySort[i+1:]:
                    fmStarts[key2][c2] += fmDeltas[key][c2]
                    
                    if key[0] == key2[0]:
                        cOffset[key2] += fmDeltas[key][c2]
                    
                fmDeltas[key][c2] = 0
        
        #blank out the next insertions and make sure we set up the keys we already know about
        nextInsertFNs = {}
        for key in keySort:
            nextInsertFNs[key] = []
        
        #default to having no new insertions
        newInserts = False
        
        #generate tuples of data packets for processing
        tups = []
        for key in keySort:
            prevIterFN = mergedFN+'.'+key+'.'+str(column-1)+'.npy'
            cOff = cOffset[key]
            tup = (key, seqFNPrefix, offsetFN, mergedFN, prevIterFN, column, copy.deepcopy(fmStarts[key]), insertFNs[key], cOff, areUniform)
            tups.append(tup)
        
        if numProcs > 1:
            #TODO: chunksize?
            #create a pool of processes based on the input
            myPool = multiprocessing.Pool(numProcs)
            rets = myPool.imap(bwtPartialInsertPoolCall, tups, 1)
        else:
            rets = []
            for tup in tups:
                rets.append(bwtPartialInsertPoolCall(tup))
        
        for i, ret in enumerate(rets):
            (retFmDelta, retInsertFNs, retShape) = ret
            key = keySort[i]
            
            for c in range(1, numValidChars):
                if retInsertFNs[c] == None:
                    continue
                
                nextKey = (str(c)+key)[0:depth]
                if not fmStarts.has_key(nextKey):
                    fmDeltas[nextKey] = [0]*numValidChars
                        
                    keyInd = bisect.bisect(keySort, nextKey)
                    if keyInd < len(keySort):
                        fmStarts[nextKey] = copy.deepcopy(fmStarts[keySort[keyInd]])
                    else:
                        fmStarts[nextKey] = copy.deepcopy(allBwtCounts)
                    
                    if keyInd == len(keySort) or keySort[keyInd][0] != nextKey[0]:
                        cOffset[nextKey] = allFirstCounts[int(nextKey[0])]
                    else:
                        cOffset[nextKey] = cOffset[keySort[keyInd]]
                    
                    #np.lib.format.open_memmap(mergedFN+'.'+nextKey+'.'+str(column)+'.npy', 'w+', '<u8,<u1,<u4', (0,))
                    np.lib.format.open_memmap(mergedFN+'.'+nextKey+'.'+str(column)+'.npy', 'w+', '<u1', (0,))
                
                for c2 in range(0, numValidChars):
                    fmDeltas[nextKey][c2] += retFmDelta[c][c2]
                
                if not nextInsertFNs.has_key(nextKey):
                    nextInsertFNs[nextKey] = []
                
                if retInsertFNs[c] != None:
                    nextInsertFNs[nextKey].append(retInsertFNs[c])
                    newInserts = True
            
            #None means we didn't change anything
            if retShape != None:
                totalCounts[key] = retShape
        
        if numProcs > 1:
            myPool.terminate()
            myPool.join()
            myPool = None
        
        #at this point, we know everything is over, so we can clean up the previous step
        for key in keySort:
            prevIterFN = mergedFN+'.'+key+'.'+str(column-1)+'.npy'
            try:
                os.remove(prevIterFN)
            except:
                pass
            for fn in insertFNs[key]:
                try:
                    os.remove(fn)
                except:
                    pass
        
        insertFNs = nextInsertFNs
            
        #copy inserts and move to the next column
        column += 1
        
        et = time.time()
        logger.info('Finished iteration '+str(column-2)+' in '+str(et-st)+' seconds...')
            
    
    logger.info('Creating final output...')
    
    ei = 0
    sortedKeys = sorted(totalCounts.keys())
    for key in sortedKeys:
        copyArr = np.load(mergedFN+'.'+key+'.'+str(column-1)+'.npy', 'r')
        si = ei
        ei += totalCounts[key]
        bwt[si:ei] = copyArr[:]
        
    for key in sortedKeys:
        os.remove(mergedFN+'.'+key+'.'+str(column-1)+'.npy')

def bwtInitialInsertionsPoolCall(tup):
    (offsetFN, seqFNPrefix, mergedFN, depth, numValidChars, startingIndex, endingIndex, areUniform) = tup
    
    if areUniform:
        #figure out the length of uniformity
        offsets = np.load(offsetFN, 'r')
        uniformLength = int(offsets[0])
        
        #load the insertions
        mmapSeqs = np.load(seqFNPrefix+'.1.npy')
        numColumns = uniformLength
        
        #figure out the files to determine placement
        seqDepths = [None]*depth
        for i in xrange(0, depth):
            seqDepths[i] = np.load(seqFNPrefix+'.'+str((numColumns - i) % numColumns)+'.npy', 'r')
        
        fmDeltas = {}
        insertFNs = {}
        inserts = {}
        
        #calculate the key and npSeq
        i = startingIndex
        while i < endingIndex:
            npSeq = np.zeros(dtype='<u1', shape=(depth,))
            for j, arr in enumerate(seqDepths):
                npSeq[j] = arr[i]
            #key = str(npSeq)[1:-1].replace(' ', '')
            key = str(npSeq)
            key = key[1:len(key)-1].replace(' ', '')
            
            #initialize the relevant things for that sequence
            fmDeltas[key] = [0]*numValidChars
            
            #loop through matching all strings that fall into that bin
            j = i
            isSameBin = True
            while j < endingIndex and isSameBin:
                #build the key
                npSeq2 = np.zeros(dtype='<u1', shape=(depth,))
                for k, arr in enumerate(seqDepths):
                    npSeq2[k] = arr[j]
                
                #compare the key
                if not np.array_equal(npSeq, npSeq2):
                    #time to go to the next bin
                    isSameBin = False
                else:
                    #still matches, add it's info to the counts
                    fmDeltas[key][mmapSeqs[j]] += 1
                    j += 1
            
            #add the file for insertion
            insertFNs[key] = [mergedFN+'.'+str(startingIndex)+'.'+key+'.tempInserts.npy']
            inserts[key] = np.lib.format.open_memmap(insertFNs[key][0], 'w+', '<u8,<u1,<u4', (j-i,))
            
            #set the inserts
            for k in xrange(0, j-i):
                inserts[key][k] = (i+k, mmapSeqs[i+k], i+k)
                
            #set values for the next iteration
            inserts[key] = None
            i = j
        
    else:
        offsetData = np.load(offsetFN, 'r')
        mmapSeqs = np.load(seqFNPrefix, 'r')
    
        #loop through the strings time
        i = startingIndex
        ei = offsetData[i]
        
        fmDeltas = {}
        insertFNs = {}
        inserts = {}
        
        while i < endingIndex:
            #get the start and end of the string
            si = ei
            ei = offsetData[i+1]
            
            #TODO: naming probably needs to be changed
            #pull out the relevant sequence up to some depth
            npSeq = np.append([0], mmapSeqs[si:ei])
            while npSeq.shape[0] < depth:
                npSeq = np.append(npSeq, mmapSeqs[si:ei])
            npSeq = npSeq[0:depth]
            #key = str(npSeq)[1:-1].replace(' ', '')
            key = str(npSeq)
            key = key[1:len(key)-1].replace(' ', '')
            
            #initialize the relevant things for that sequence
            fmDeltas[key] = [0]*numValidChars
            
            #loop through matching all strings that fall into that bin
            j = i
            isSameBin = True
            ei2 = si
            while j < endingIndex and isSameBin:
                #get the start and end of the next seq
                si2 = ei2
                ei2 = offsetData[j+1]
                
                #build the key
                npSeq2 = np.append([0], mmapSeqs[si2:ei2])
                while npSeq2.shape[0] < depth:
                    npSeq2 = np.append(npSeq2, mmapSeqs[si2:ei2])
                npSeq2 = npSeq2[0:depth]
                
                #compare the key
                if not np.array_equal(npSeq, npSeq2):
                    #time to go to the next bin
                    isSameBin = False
                else:
                    #still matches, add it's info to the counts
                    fmDeltas[key][mmapSeqs[ei2-2]] += 1
                    j += 1
            
            #add the file for insertion
            insertFNs[key] = [mergedFN+'.'+str(startingIndex)+'.'+key+'.tempInserts.npy']
            inserts[key] = np.lib.format.open_memmap(insertFNs[key][0], 'w+', '<u8,<u1,<u4', (j-i,))
            
            #set the inserts
            for k, endOffset in enumerate(offsetData[i+1:j+1]):
                inserts[key][k] = (i+k, mmapSeqs[endOffset-2], i+k)
            
            #set values for the next iteration
            i = j
            ei = offsetData[i]
    
    return (fmDeltas, insertFNs)

def bwtPartialInsertPoolCall(tup):
    '''
    This is a function typically called by a multiprocess pool to do the partial insertions
    @param tup - the tuple of inputs, see below for breakdown
    '''
    #load these values from the tuple
    (procLabel, seqFNPrefix, offsetFN, finalOutFN, prevIterFN, columnStart, fmIndex, insertionFNs, cOffset, areUniform) = tup
    
    #mark the start time
    sTime = time.time()
    debug = False
    
    #TODO: param is hardcoded, bad?
    vcLen = 6
    
    #load the sequences and the offsets
    if areUniform:
        #figure out the length of uniformity
        offsets = np.load(offsetFN, 'r')
        uniformLength = int(offsets[0])
        
        mmapSeqs = np.load(seqFNPrefix+'.'+str(columnStart % uniformLength)+'.npy')
    else:
        offsetData = np.load(offsetFN, 'r')
        mmapSeqs = np.load(seqFNPrefix, 'r')
    fnToLoad = prevIterFN
    
    #if there's nothing to insert, we can just rename the file
    if len(insertionFNs) == 0:
        #TODO: is this screwing up our save points? note that we're renaming so we'd lose the old before guaranteed safe
        #nothing to do in this iteration
        oldFN = fnToLoad
        newFN = finalOutFN+'.'+procLabel+'.'+str(columnStart)+'.npy'
        os.rename(oldFN, newFN)
        
        #vcLen*vcLen matrix for fmDeltas
        fmDeltas = [[0]*vcLen]*vcLen
        nextInsertFNs = [None]*vcLen
        
        #return None as last to let it know not to update
        return (fmDeltas, nextInsertFNs, None)
        
    debugDump('Running.', procLabel, sTime, debug)
    
    #load the previous iteration and mark the column we're in
    column = columnStart
    prevIter = np.load(fnToLoad, 'r')
    
    #reset everything
    nextInserts = [None]*vcLen
    insertionArrays = []
    
    #this is a normal task
    #but first, clear the dump task
    newInsertSize = 0
    insertCounts = [0]*vcLen
    debugDump('Loading inserts...', procLabel, sTime, debug)
    for i, fn in enumerate(insertionFNs):
        #fully load these into memory for fastest results
        insertionArrays.append(np.load(fn))
        newInsertSize += insertionArrays[i].shape[0]
        insertCounts = np.add(insertCounts, np.bincount(insertionArrays[i]['f1'], minlength=vcLen))
        try:
            os.remove(fn)
        except:
            pass
        
    debugDump('Loaded inserts.', procLabel, sTime, debug)
    
    #create the file for the next iteration by pre-allocating for the new insertion size
    nextIter = np.zeros(shape=(prevIter.shape[0]+newInsertSize,), dtype='<u1')
    nextIter[:] = vcLen
    
    #mark the counts
    fmDeltas = [None]*vcLen
    nextInsertFNs = [None]*vcLen
    
    #insert everything that needs inserting
    for arr in insertionArrays:
        nextIter[arr['f0']-cOffset] = arr['f1'][:]
    
    #copy the other
    if prevIter.shape[0] > 0:
        nextIter[nextIter == vcLen] = prevIter[:]
    
    #save the next iteration
    debugDump('Saving...', procLabel, sTime, debug)
    np.save(finalOutFN+'.'+procLabel+'.'+str(column)+'.npy', nextIter)
    debugDump('Finished saving.', procLabel, sTime, debug)
    
    #no longer need this
    del prevIter
    try:
        os.remove(prevIterFN)
    except:
        pass
        
    for c in xrange(0, vcLen):
        #create this array
        nextInserts[c] = np.zeros(shape=(insertCounts[c],), dtype='<u8,<u1,<u4')
        
        #clear the fmDeltas
        fmDeltas[c] = [0]*vcLen
        
    bc = np.array(fmIndex)
    newInsertFilePos = [0]*vcLen
    p = 0
    for arr in insertionArrays:
        nz = np.nonzero(arr['f1'])[0]
        seqIds = arr['f2'][nz]
        
        if areUniform:
            nextC = mmapSeqs[seqIds]
        else:
            indices = np.subtract(offsetData[seqIds+1], (column+1))
            if seqIds.shape[0] > 0 and seqIds[0] == 0 and offsetData[1] < (column+1):
                indices[0] = offsetData[1]-1
            nextC = mmapSeqs[indices]
        
        pos = np.zeros(dtype='<u8', shape=(seqIds.shape[0], ))
        
        for i in xrange(0, seqIds.shape[0]):
            e = int(arr['f0'][nz[i]]-cOffset)
            if p == e:
                pass
            else:
                bc = np.add(bc, np.bincount(nextIter[p:e], minlength=vcLen))
            pos[i] = bc[arr['f1'][nz[i]]]
            p = e
        
        for c in xrange(1, vcLen):
            counters = np.where(arr['f1'][nz] == c)[0]
            nextInserts[c]['f0'][newInsertFilePos[c]:newInsertFilePos[c]+counters.shape[0]] = pos[counters]
            nextInserts[c]['f1'][newInsertFilePos[c]:newInsertFilePos[c]+counters.shape[0]] = nextC[counters]
            nextInserts[c]['f2'][newInsertFilePos[c]:newInsertFilePos[c]+counters.shape[0]] = seqIds[counters]
            newInsertFilePos[c] += counters.shape[0]
            
            if counters.shape[0] == 0:
                pass
            else:
                fmDeltas[c] = np.add(fmDeltas[c], np.bincount(nextC[counters], minlength=6))
    
    nextShape = nextIter.shape[0]
    
    nextInsertFNs = [None]*vcLen
    for c in xrange(1, vcLen):
        if insertCounts[c] > 0:
            nextInsertFN = finalOutFN+'.'+str(c)+procLabel+'.'+str(column)+'.temp.npy'
            nextInsertFNs[c] = nextInsertFN
            np.save(nextInsertFN, nextInserts[c])
            
    #cleanup 
    del nz
    del seqIds
    if not areUniform:
        del indices
    del nextC
    del counters
    del pos
    del nextIter
    del insertionArrays
    del nextInserts
    gc.collect()
    
    debugDump('Finished.', procLabel, sTime, debug)
    
    return (fmDeltas, nextInsertFNs, nextShape)
    
def debugDump(msg, procLabel, sTime, debug):
    if debug:
        print '[{0:.3f}] Process '.format(time.time()-sTime)+procLabel+': '+str(msg)

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