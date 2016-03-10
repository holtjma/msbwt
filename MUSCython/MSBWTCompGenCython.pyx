#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False

'''
Created on Mar 19, 2014

Contains a re-implementation of the MSBWTGenCython MSBWT creation algorithm only
it creates the MSBWT in a compressed format. The main idea is to reduce I/O volume
during creation while simultaneously building straight into the compressed version
of the MSBWT.  Presently, I only plan on implementing this for uniform strings, aka
Illumina like data, since that's where the computational bottleneck in our pipelines
currently resides.

@author: holtjma
'''

#C imports
from libc.stdio cimport FILE, fopen, fwrite, fclose

#python imports
import glob
import math
import multiprocessing
import numpy as np
cimport numpy as np
import os
import shutil
import time

#my imports
import MSBWTGenCython as MSBWTGen

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
    
    #clear anything that may already have been associated with our directory, re-use this function
    MSBWTGen.clearAuxiliaryData(bwtDir)
    
    #construct pre-arranged file names, God forbid we ever change these...
    bwtFN = bwtDir+'/comp_msbwt.npy'
    seqFNPrefix = bwtDir+'/seqs.npy'
    offsetFN = bwtDir+'/offsets.npy'
    
    #offsets is really just storing the length of all string at this point
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] offsets = np.load(offsetFN, 'r+')
    cdef unsigned long seqLen = offsets[0]
    
    #some basic counting variables
    cdef unsigned long i, j, c, columnID, c2
    cdef unsigned long initColumnID = 0
    for i in xrange(0, seqLen):
        for c in xrange(0, numValidChars):
            if not os.path.exists(bwtDir+'/state.'+str(c)+'.'+str(i)+'.dat'):
                break
        else:
            #do two more checks for information we need
            if (os.path.exists(bwtDir+'/fmStarts.'+str(i)+'.npy') and
                os.path.exists(bwtDir+'/fmDeltas.'+str(i)+'.npy')):
                initColumnID = i
                break
    
    #finalSymbols stores the last real symbols in the strings (not the '$', the ones right before)
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] finalSymbols = np.load(seqFNPrefix+'.1.npy', 'r+')
    cdef np.uint8_t [:] finalSymbols_view = finalSymbols
    cdef unsigned long numSeqs = finalSymbols.shape[0]
    
    logger.warning('Beta version of Cython compressed creation')
    logger.info('Preparing to merge '+str(numSeqs)+' sequences...')
    
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] initialInserts
    cdef np.uint64_t [:, :] initialInserts_view
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] initialFmDeltas
    cdef np.uint64_t [:] initialFmDeltas_view
    
    #fmStarts is basically an fm-index offset, fmdeltas tells us how much fmStarts should change each iterations
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmStarts = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] fmStarts_view = fmStarts
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmDeltas = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] fmDeltas_view = fmDeltas
    #cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmEnds = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    #cdef np.uint64_t [:, :] fmEnds_view = fmEnds
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] nextFmDeltas = np.empty(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] nextFmDeltas_view = nextFmDeltas
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] retFmDeltas
    cdef np.uint64_t [:, :] retFmDeltas_view
    
    cdef dict insertionFNDict = {}
    cdef dict nextInsertionFNDict = {}
    cdef FILE * tempFP
    
    if initColumnID == 0:
        logger.info('Generating level 1 insertions...')
        #1 - load <DIR>/seqs.npy.<seqLen-2>.npy, these are the characters before the '$'s
        #    use them as the original inserts storing <insert position, insert character, sequence ID>
        #    note that inserts should be stored in a minimum of 6 files, one for each character
        initialInsertsFN = bwtDir+'/inserts.initial.npy'
        initialInserts = np.lib.format.open_memmap(initialInsertsFN, 'w+', '<u8', (numSeqs, 3))
        initialInserts_view = initialInserts

        #this will store the initial counts of symbols in the final column of our strings
        initialFmDeltas = np.zeros(dtype='<u8', shape=(numValidChars, ))
        initialFmDeltas_view = initialFmDeltas
        
        for i in range(0, numSeqs):
            #index to insert, symbol, sequence
            initialInserts_view[i, 0] = i
            initialInserts_view[i, 1] = finalSymbols_view[i]
            initialInserts_view[i, 2] = i
            initialFmDeltas_view[finalSymbols_view[i]] += 1
        
        #fmDeltas[0][:] = initialFmDeltas[:]
        for i in range(0, numValidChars):
            fmDeltas_view[0, i] = initialFmDeltas[i]
        
        #figure out the files we insert in each iteration
        insertionFNDict = {}
        nextInsertionFNDict = {}
        for c in range(0, numValidChars):
            #create an initial empty region for each symbol
            tempFN = bwtDir+'/state.'+str(c)+'.0.dat'
            tempFP = fopen(tempFN, 'w+')
            fclose(tempFP)
            
            #create an empty list of files for insertion
            insertionFNDict[c] = []
            nextInsertionFNDict[c] = []
        
        #add the single insertion file we generated earlier
        insertionFNDict[0].append(initialInsertsFN)
        
        np.save(bwtDir+'/fmDeltas.0.npy', fmDeltas)
        np.save(bwtDir+'/fmStarts.0.npy', fmStarts)
        
        etc = time.clock()
        et = time.time()
        logger.info('Finished init in '+str(et-st)+' seconds.')
        
    else:
        #print 'FOUND PARTIAL RUN STARTING AT '+str(initColumnID)
        logger.info('Resuming previous run from '+str(initColumnID)+', loading partial BWT...')
        
        fmDeltas = np.load(bwtDir+'/fmDeltas.'+str(initColumnID)+'.npy')
        fmDeltas_view = fmDeltas
        fmStarts = np.load(bwtDir+'/fmStarts.'+str(initColumnID)+'.npy')
        fmStarts_view = fmStarts
        
        for c in range(0, numValidChars):
            #create an empty list of files for insertion
            insertionFNDict[c] = []
            nextInsertionFNDict[c] = []
            
        gl = sorted(glob.glob(bwtDir+'/inserts.*.'+str(initColumnID)+'.npy'))
        for fn in gl:
            firstSym = fn.split('/')
            firstSym = firstSym[len(firstSym)-1]
            firstSym = firstSym.split('.')
            firstSym = int(firstSym[1][0])
            
            insertionFNDict[firstSym].append(fn)
        
        #print insertionFNDict
        
        #raise Exception('')
    
    logger.info('Beginning iterations...')
    
    cdef unsigned long cumsum
    
    #2 - go back one column at a time, building new inserts
    for columnID in range(initColumnID, seqLen):
        st = time.time()
        stc = time.clock()
        
        #first take into account the new fmDeltas coming in from all insertions
        #fmStarts = fmStarts+(np.cumsum(fmDeltas, axis=0)-fmDeltas)
        #fmEnds = fmEnds+np.cumsum(fmDeltas, axis=0)
        for i in range(0, numValidChars):
            cumsum = 0
            for j in range(0, numValidChars-1):
                cumsum += fmDeltas_view[j,i]
                fmStarts_view[j+1, i] += cumsum
                #fmEnds_view[j, i] += cumsum
                nextFmDeltas_view[j, i] = 0
            cumsum += fmDeltas_view[numValidChars-1, i]
            #fmEnds_view[numValidChars-1, i] += cumsum
            nextFmDeltas_view[numValidChars-1, i] = 0
        
        #clear out the next fmDeltas
        tups = []
        for c in range(0, numValidChars):
            currentSymbolFN = bwtDir+'/state.'+str(c)+'.'+str(columnID)+'.dat'
            nextSymbolFN = bwtDir+'/state.'+str(c)+'.'+str(columnID+1)+'.dat'
            nextSeqFN = seqFNPrefix+'.'+str((columnID+2) % seqLen)+'.npy'
            #tup = (c, np.copy(fmStarts[c]), np.copy(fmDeltas[c]), np.copy(fmEnds[c]), insertionFNDict[c], currentSymbolFN, nextSymbolFN, bwtDir, columnID, nextSeqFN)
            tup = (c, np.copy(fmStarts[c]), np.copy(fmDeltas[c]), insertionFNDict[c], currentSymbolFN, nextSymbolFN, bwtDir, columnID, nextSeqFN)
            tups.append(tup)
        
        '''
        #this is just to force something to run in the same process
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
        
        #copy the fmDeltas and insertion filenames
        #fmDeltas[:] = nextFmDeltas[:]
        for i in range(0, numValidChars):
            for j in range(0, numValidChars):
                fmDeltas_view[i, j] = nextFmDeltas_view[i, j]
        
        np.save(bwtDir+'/fmDeltas.'+str(columnID+1)+'.npy', fmDeltas)
        np.save(bwtDir+'/fmStarts.'+str(columnID+1)+'.npy', fmStarts)
        
        #remove the old insertions and old state files
        for c in insertionFNDict:
            for fn in insertionFNDict[c]:
                try:
                    os.remove(fn)
                except:
                    logger.warning('Failed to remove \''+fn+'\' from file system.')
            
            #if len(insertionFNDict[c]) > 0:
            try:
                rmStateFN = bwtDir+'/state.'+str(c)+'.'+str(columnID)+'.dat'
                os.remove(rmStateFN)
            except:
                logger.warning('Failed to remove\''+rmStateFN+'\' from file system.')
        
        try:
            fn = bwtDir+'/fmDeltas.'+str(columnID)+'.npy'
            os.remove(fn)
        except:
            logger.warning('Failed to remove\''+fn+'\' from the file system.')
        
        try:
            fn = bwtDir+'/fmStarts.'+str(columnID)+'.npy'
            os.remove(fn)
        except:
            logger.warning('Failed to remove\''+fn+'\' from the file system.')
        
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
    
    #initialize our length to be the length of the '$' portion which must be there by our string definitions
    tempBWT = np.memmap(bwtDir+'/state.'+str(0)+'.'+str(seqLen)+'.dat', '<u1', 'r')
    cdef unsigned long totalLength = tempBWT.shape[0]
    
    #figure out what symbol is at the end and how big it is
    cdef np.uint8_t finalSymbol = tempBWT[totalLength-1] & 0x07
    cdef unsigned long finalSymbolCount = 0
    cdef unsigned long finalSymbolPos = totalLength
    cdef unsigned long finalSymbolBytes = 0
    while finalSymbolPos > 0 and (tempBWT[finalSymbolPos-1] & 0x07) == finalSymbol:
        finalSymbolCount = finalSymbolCount*32 + (tempBWT[finalSymbolPos-1] >> 3)
        finalSymbolBytes += 1
        finalSymbolPos -= 1
    
    cdef unsigned long modifier, extendBytes
    
    #now incorporate the other symbols
    for c in range(1, numValidChars):
        try:
            tempBWT = np.memmap(bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.dat', '<u1', 'r')
        except ValueError as e:
            if str(e) == 'cannot mmap an empty file':
                #this is normal behavior for my test cases with missing symbols
                tempBWT = np.zeros(dtype='<u1', shape=(0, ))
            else:
                raise e
        
        #figure out if they're the same
        finalSymbolPos = 0
        modifier = 0
        while finalSymbolPos < tempBWT.shape[0] and (tempBWT[finalSymbolPos] & 0x07) == finalSymbol:
            modifier += (tempBWT[finalSymbolPos] >> 3) * (32**finalSymbolPos)
            finalSymbolPos += 1
        
        #calculate if we added more bytes here
        extendBytes = max(1, math.floor(math.log(finalSymbolCount+modifier, 32.0))+1) - max(1, math.floor(math.log(finalSymbolCount, 32.0))+1)
        
        #add in the extended bytes followed by the total size subtracting the bytes used for the shared symbols
        totalLength += extendBytes + (tempBWT.shape[0] - finalSymbolPos)
        
        #now check if we need to update our final symbols
        if finalSymbolPos == tempBWT.shape[0]:
            #this is the atypical case where all 'C' are preceded by a single symbol for example, only matters in my test
            #cases really but the check is simple enough to implement that I should handle it
            #add the counts from this entire chunk to the count, but modify nothing else
            finalSymbolCount += modifier
        else:
            #this is the typical case, this chunk is not all one symbol, so we need to recalculate the final symbol range
            finalSymbol = tempBWT[tempBWT.shape[0]-1] & 0x07
            finalSymbolCount = 0
            finalSymbolPos = tempBWT.shape[0]
            finalSymbolBytes = 0
            while finalSymbolPos > 0 and (tempBWT[finalSymbolPos-1] & 0x07) == finalSymbol:
                finalSymbolCount = finalSymbolCount*32 + (tempBWT[finalSymbolPos-1] >> 3)
                finalSymbolBytes += 1
                finalSymbolPos -= 1
    
    #prepare the final structure for copying
    cdef unsigned long finalInd = 0
    cdef unsigned long tempLen
    finalBWT = np.lib.format.open_memmap(bwtFN, 'w+', '<u1', (totalLength, ))
    finalBWT_view = finalBWT
    
    cdef unsigned long startPoint
    
    #now the tricky part of actually combining them
    for c in range(0, numValidChars):
        stateFN = bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.dat'
        try:
            tempBWT = np.memmap(stateFN, '<u1', 'r+')
        except ValueError as e:
            if str(e) == 'cannot mmap an empty file':
                #this is normal behavior for my test cases with missing symbols
                tempBWT = np.zeros(dtype='<u1', shape=(0, ))
            else:
                raise e
        
        tempBWT_view = tempBWT
        tempLen = tempBWT.shape[0]
        
        if c == 0:
            startPoint = 0
        else:
            #special checks to see if the symbols are the same at the boundary
            startPoint = 0
            modifier = 0
            while startPoint < tempBWT.shape[0] and (tempBWT_view[startPoint] & 0x07) == finalSymbol:
                modifier += (tempBWT_view[startPoint] >> 3) * (32**startPoint)
                startPoint += 1
            
            #there is overlap, we need to re-write the boundary
            if modifier > 0:
                #reset finalInd to the start of this symbol boundary and add in the new counts
                finalInd = finalSymbolPos
                finalSymbolCount += modifier
                
                #now write it like normal
                while finalSymbolCount > 0:
                    #3 bits = symbol, 5 bits = piece of count in LSB form
                    finalBWT_view[finalInd] = finalSymbol + ((finalSymbolCount & 0x1F) << 3)
                    finalInd += 1
                    finalSymbolCount = finalSymbolCount >> 5
        
        #now write out the rest of this piece
        for i in range(startPoint, tempLen):
            finalBWT_view[finalInd] = tempBWT_view[i]
            finalInd += 1
        
        #finally, recalculate the final symbol stats for the next pass
        finalSymbol = finalBWT_view[finalInd-1] & 0x07
        finalSymbolCount = 0
        finalSymbolPos = finalInd
        finalSymbolBytes = 0
        while finalSymbolPos > 0 and (finalBWT_view[finalSymbolPos-1] & 0x07) == finalSymbol:
            finalSymbolCount = finalSymbolCount*32 + (finalBWT_view[finalSymbolPos-1] >> 3)
            finalSymbolBytes += 1
            finalSymbolPos -= 1
    
    #finally, clear the last state files
    tempBWT = None
    for c in range(0, numValidChars):
        for fn in insertionFNDict[c]:
            try:
                os.remove(fn)
            except:
                logger.warning('Failed to remove \''+fn+'\' from file system.')
        
        try:
            if os.path.exists(bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.dat'):
                os.remove(bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.dat')
        except:
            logger.warning('Failed to remove \''+bwtDir+'/state.'+str(c)+'.'+str(seqLen)+'.dat'+'\' from file system.')
    
    try:
        fn = bwtDir+'/fmDeltas.'+str(seqLen)+'.npy'
        os.remove(fn)
    except:
        logger.warning('Failed to remove\''+fn+'\' from the file system.')
    
    try:
        fn = bwtDir+'/fmStarts.'+str(seqLen)+'.npy'
        os.remove(fn)
    except:
        logger.warning('Failed to remove\''+fn+'\' from the file system.')
        
    
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
    #cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmEndex
    #cdef np.uint64_t [:] fmEndex_view
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] prevFmDelta
    cdef np.uint64_t [:] prevFmDelta_view
    cdef unsigned long column
    #(idChar, fmIndex, prevFmDelta, fmEndex, insertionFNs, currentSymbolFN, nextSymbolFN, bwtDir, column, nextSeqFN) = tup
    (idChar, fmIndex, prevFmDelta, insertionFNs, currentSymbolFN, nextSymbolFN, bwtDir, column, nextSeqFN) = tup
    fmIndex_view = fmIndex
    #fmEndex_view = fmEndex
    prevFmDelta_view = prevFmDelta
    
    #hardcoded
    cdef unsigned int numValidChars = 6
    
    #this stores the number of symbols we find in our range
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmDeltas = np.zeros(dtype='<u8', shape=(numValidChars, numValidChars))
    cdef np.uint64_t [:, :] fmDeltas_view = fmDeltas
    
    #the input partial BWT for suffixes starting with 'idChar'
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] currentBwt
    try:
        #TODO: make an "all-in-memory" version of this at some point, it's likely to actually fit on many machines
        currentBwt = np.memmap(currentSymbolFN, '<u1', 'r+')
        #currentBwt = np.fromfile(currentSymbolFN, '<u1')
    except ValueError as e:
        if str(e) == 'cannot mmap an empty file' or str(e) == 'mmap offset is greater than file size':
            #this is normal behavior during early iterations, make a dummy array just to get the code running
            currentBwt = np.zeros(dtype='<u1', shape=(0, ))
        else:
            raise e
        
    cdef np.uint8_t [:] currentBwt_view = currentBwt
    
    #the output partial BWT for suffixes starting with 'idChar'
    cdef FILE * nextBwtFP
    
    #currentBWT + inserts = nextBWT
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] inserts
    cdef np.uint64_t [:, :] inserts_view
    
    #counting variables
    cdef unsigned long currIndex, maxIndex, insertLen, insertIndex, prevIndex, totalNewLen
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
    
    #extra variables for counting the number of symbols of each type
    cdef unsigned long readBufferSymbol = 0
    cdef unsigned long readBufferCount = 0
    cdef unsigned long readBufferPower = 0
    
    cdef unsigned long writeBufferSymbol = 0
    cdef unsigned long writeBufferCount = 0
    
    cdef np.uint8_t writeValue = 0
    
    retFNs = [None]*numValidChars
    
    if len(insertionFNs) == 0:
        #we don't need to do anything except rename our file
        if os.path.exists(currentSymbolFN):
            #os.rename(currentSymbolFN, nextSymbolFN)
            shutil.copyfile(currentSymbolFN, nextSymbolFN)
    else:
        #first, open the file for output
        nextBwtFP = fopen(nextSymbolFN, 'w+')
        
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
                outputInsertPointers_view[c] = <np.uint64_t> &(newInsertArray_view[0,0])
        
        #now build the iteration, we'll start at zero obviously
        prevIndex = 0#tracks BWT index, aka the number of values written thus far, equivalently, the index of the start of the run next to be written
        currIndex = 0#tracks file index (this will be smaller because it's compressed)
        maxIndex = currentBwt.shape[0]
        
        #get the buffered symbol and count if they're present
        if currIndex < maxIndex:
            currIndex = getNextRLE(currentBwt_view, currIndex, maxIndex, &readBufferSymbol, &readBufferCount)
        
        #go through each file, one at a time
        for fn in insertionFNs:
            #load the actual inserts
            inserts = np.load(fn, 'r+')
            inserts_view = inserts
            insertLen = inserts.shape[0]
            
            #first check if there were adjacent writes across files, typically not except in the very early insertions
            #this loop is really just a copy of the second half of the main loop
            i = 0
            while i < insertLen and prevIndex+writeBufferCount == inserts_view[i, 0]:
                if inserts_view[i, 1] == writeBufferSymbol:
                    writeBufferCount += 1
                else:
                    fmIndex_view[writeBufferSymbol] += writeBufferCount
                    prevIndex += writeBufferCount
                    
                    setNextRLE(nextBwtFP, writeBufferSymbol, writeBufferCount)
                    writeBufferSymbol = inserts_view[i, 1]
                    writeBufferCount = 1
                
                symbol = writeBufferSymbol
                nextSymbol = currSeqs_view[inserts_view[i,2]]
                fmDeltas_view[symbol, nextSymbol] += 1
                
                #now we need to add the information for our next insertion
                outputInsert_p = <np.uint64_t *> outputInsertPointers_view[symbol]
                ind = outputInsertIndices_view[symbol]
                outputInsertIndices_view[symbol] += 3
                
                #finally, store the values
                outputInsert_p[ind] = fmIndex_view[symbol]+writeBufferCount-1
                outputInsert_p[ind+1] = nextSymbol
                outputInsert_p[ind+2] = inserts_view[i,2]
                i += 1
            
            #the real main loop
            while i < insertLen:
                insertIndex = inserts_view[i, 0]
                
                if writeBufferSymbol == readBufferSymbol:
                    #move the write buffer into the read buffer
                    readBufferCount += writeBufferCount
                    writeBufferCount = 0
                else:
                    #write it out
                    fmIndex_view[writeBufferSymbol] += writeBufferCount
                    prevIndex += writeBufferCount
                    
                    setNextRLE(nextBwtFP, writeBufferSymbol, writeBufferCount)
                    writeBufferCount = 0
                
                while prevIndex+readBufferCount < insertIndex:
                    fmIndex_view[readBufferSymbol] += readBufferCount
                    prevIndex += readBufferCount
                    
                    setNextRLE(nextBwtFP, readBufferSymbol, readBufferCount)
                    currIndex = getNextRLE(currentBwt_view, currIndex, maxIndex, &readBufferSymbol, &readBufferCount)
                
                writeBufferCount = insertIndex-prevIndex
                readBufferCount -= writeBufferCount
                writeBufferSymbol = readBufferSymbol
                    
                if readBufferCount == 0 and currIndex < maxIndex:
                    currIndex = getNextRLE(currentBwt_view, currIndex, maxIndex, &readBufferSymbol, &readBufferCount)
                    
                #now do the inserts, loop as long as there are consecutive insertions
                while i < insertLen and prevIndex+writeBufferCount == inserts_view[i, 0]:
                    if inserts_view[i, 1] == writeBufferSymbol:
                        writeBufferCount += 1
                    else:
                        fmIndex_view[writeBufferSymbol] += writeBufferCount
                        prevIndex += writeBufferCount
                        
                        setNextRLE(nextBwtFP, writeBufferSymbol, writeBufferCount)
                        writeBufferSymbol = inserts_view[i, 1]
                        writeBufferCount = 1
                    
                    symbol = writeBufferSymbol
                    nextSymbol = currSeqs_view[inserts_view[i,2]]
                    fmDeltas_view[symbol, nextSymbol] += 1
                    
                    #now we need to add the information for our next insertion
                    outputInsert_p = <np.uint64_t *> outputInsertPointers_view[symbol]
                    ind = outputInsertIndices_view[symbol]
                    outputInsertIndices_view[symbol] += 3
                    
                    #finally, store the values
                    outputInsert_p[ind] = fmIndex_view[symbol]+writeBufferCount-1
                    outputInsert_p[ind+1] = nextSymbol
                    outputInsert_p[ind+2] = inserts_view[i,2]
                    i += 1
        
        if writeBufferSymbol == readBufferSymbol:
            readBufferCount += writeBufferCount
        else:
            setNextRLE(nextBwtFP, writeBufferSymbol, writeBufferCount)
        
        if readBufferCount > 0:
            setNextRLE(nextBwtFP, readBufferSymbol, readBufferCount)
        
        while currIndex < maxIndex:
            currIndex = getNextRLE(currentBwt_view, currIndex, maxIndex, &readBufferSymbol, &readBufferCount)
            setNextRLE(nextBwtFP, readBufferSymbol, readBufferCount)
            
        #put this at the end when all writes are completed
        fclose(nextBwtFP)
      
    ret = (np.copy(fmDeltas), retFNs)
    return ret
    
cdef inline unsigned long getNextRLE(np.uint8_t [:] arr, unsigned long startingIndex, unsigned long maxIndex,
                                     unsigned long * sym, unsigned long * rl):
    cdef unsigned long pow32 = 0
    sym[0] = arr[startingIndex] & 0x7
    rl[0] = 0
    while startingIndex < maxIndex and sym[0] == (arr[startingIndex] & 0x7):
        rl[0] += (arr[startingIndex] >> 3) << (5*pow32)
        pow32 += 1
        startingIndex += 1
    
    return startingIndex
    
cdef inline void setNextRLE(FILE * fp, unsigned long sym, unsigned long rl):
    cdef np.uint8_t writeValue
    while rl > 0:
        writeValue = sym + ((rl & 0x1F) << 3)
        fwrite(&writeValue, 1, 1, fp)
        rl = (rl >> 5)
