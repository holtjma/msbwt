'''
Created on Aug 5, 2013

This file mostly contains BWT construction functions such as the initial construction, compression, decomp, etc.

@author: holtjma
'''

import bisect
import copy
import glob
import gc
import math
import multiprocessing
import numpy as np
import os
import shutil
import sys
import time

import MultiStringBWT

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
            key = str(npSeq)[1:-1].replace(' ', '')
        
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
        mmapSeqs = np.load(seqFNPrefix+'.npy', 'r')
    
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
            key = str(npSeq)[1:-1].replace(' ', '')
            
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
        mmapSeqs = np.load(seqFNPrefix+'.npy', 'r')
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
            e = arr['f0'][nz[i]]-cOffset
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
        totalSize = offsetData[-1]
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

#TODO: This is complicated to figure out initial state, review the entire process with redocumentation so you can 
#implement this correctly, MAKE SURE ITS TESTED
#def continueCreateFromSeqs(seqFNPrefix, offsetFN, mergedFN, numProcs, areUniform, logger):
    '''
    This function will take a preprocessed seqFN and offsetFN file and continue the creation of the MSBWT from them
    @param seqFNPrefix - the preprocessed seqFN file(s) prefix, '.npy' format of uint8s
    @param offsetFN - the offsets into seqFN indicating the start of a string
    @param mergedFN - the final destination output, '.npy' file
    @param numProcs - the maximum number of processes to use
    '''
'''
    #TODO: calculate starting column, hardcoded atm
    startingColumn = 54 #2 greater than last printed value
    
    numValidChars = 6
    
    #memmap the two string data files
    offsetData = np.load(offsetFN, 'r')
    if areUniform:
        #uniformLength = int(offsetData[0])
        firstSeq = np.load(seqFNPrefix+'.0.npy', 'r')
        numSeqs = firstSeq.shape[0]
    else:
        numSeqs = offsetData.shape[0]-1
    depth = max(int(math.ceil(math.log(numProcs, numValidChars-1)))+1, int(math.ceil(math.log(numSeqs/1000000.0, numValidChars))), 1)
    
    insertFNs = {}
    
    for fn in sorted(glob.glob(seqFNPrefix+'*.'+str(startingColumn)+'.*.npy')):
        tempFile = np.load(fn, 'r')
        fileIDNum = fn.split('.')[2]
        
        if len(fileIDNum) == depth:
            
        else:
            #first <depth> characters make the key for a file
            fileKey = fileIDNum[0:depth]
            
            if not insertFNs.has_key(fileKey):
                insertFNs[fileKey] = []
            insertFNs[fileKey].append(fn)
                
    iterateCreateFromSeqs(startingColumn, fmStarts, fmDeltas, allFirstCounts, allBwtCounts, cOffset, totalCounts, numValidChars,
                          mergedFN, seqFNPrefix, offsetFN, insertFNs, numProcs, areUniform, depth, logger)
'''
    
def iterateCreateFromSeqs(startingColumn, fmStarts, fmDeltas, allFirstCounts, allBwtCounts, cOffset, totalCounts, numValidChars,
                          mergedFN, seqFNPrefix, offsetFN, insertFNs, numProcs, areUniform, depth, logger):
    '''
    This function is the actual series of iterations that a BWT creation will perform.  It's separate so we can build a function
    for resuming construction if we fail for some reason.
    TODO: build a recovery function to start midway through BWT construction.
    TODO: @param values need explanation
    '''
    bwt = np.load(mergedFN, 'r+')
    
    column = startingColumn
    newInserts = True
    while newInserts:
        st = time.time()
        
        #iterate through the sorted keys
        keySort = sorted(fmStarts.keys())
        for i, key in enumerate(keySort):
            #all deltas get copied
            for c2 in xrange(0, numValidChars):
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
            
            for c in xrange(1, numValidChars):
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
                    
                    np.lib.format.open_memmap(mergedFN+'.'+nextKey+'.'+str(column)+'.npy', 'w+', '<u8,<u1,<u4', (0,))
                
                for c2 in xrange(0, numValidChars):
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

def writeSeqsToFiles(seqArray, seqFNPrefix, offsetFN, uniformLength):    
    '''
    This function takes a seqArray and saves the values to a memmap file that can be accessed for multi-processing.
    Additionally, it saves some offset indices in a numpy file for quicker string access.
    @param seqArray - the list of '$'-terminated strings to be saved
    @param fnPrefix - the prefix for the temporary files, creates a prefix+'.seqs.npy' and prefix+'.offsets.npy' file
    '''
    if uniformLength:
        #first, store the uniform size in our offsets file
        offsets = np.lib.format.open_memmap(offsetFN, 'w+', '<u8', (1,))
        offsets[0] = uniformLength
        
        #define a constant character map for now
        d = {'$':0, 'A':1, 'C':2, 'G':3, 'N':4, 'T':5}
        dArr = np.add(np.zeros(dtype='<u1', shape=(256,)), len(d.keys()))
        for c in d.keys():
            dArr[ord(c)] = d[c]
        
        seqLen = uniformLength
        b = np.reshape(seqArray, (-1, seqLen))
        numSeqs = b.shape[0]
        t = b.transpose()
        
        for i in xrange(0, seqLen):
            #create a file for this column
            seqs = np.lib.format.open_memmap(seqFNPrefix+'.'+str(i)+'.npy', 'w+', '<u1', (numSeqs,))
            chunkSize = 1000000
            j = 0
            while chunkSize*j < numSeqs:
                seqs[chunkSize*j:chunkSize*(j+1)] = dArr[t[-i-1][chunkSize*j:chunkSize*(j+1)]]
                j += 1
            del seqs
    else:
        #count how many terminal '$' exist, 36 = '$'
        lenSums = np.add(1, np.where(seqArray == 36)[0])
        numSeqs = lenSums.shape[0]
        totalLen = lenSums[-1]
        
        #track the total length thus far and open the files we plan to fill in
        seqFN = seqFNPrefix+'.npy'
        seqs = np.lib.format.open_memmap(seqFN, 'w+', '<u1', (totalLen,))
        offsets = np.lib.format.open_memmap(offsetFN, 'w+', '<u8', (numSeqs+1,))
        offsets[1:] = lenSums
        
        #define a constant character map for now
        d = {'$':0, 'A':1, 'C':2, 'G':3, 'N':4, 'T':5}
        dArr = np.add(np.zeros(dtype='<u1', shape=(256,)), len(d.keys()))
        for c in d.keys():
            dArr[ord(c)] = d[c]
        
        #copy the values
        chunkSize = 1000000
        i = 0
        while chunkSize*i < seqArray.shape[0]:
            seqs[chunkSize*i:chunkSize*(i+1)] = dArr[seqArray[chunkSize*i:chunkSize*(i+1)]]
            i += 1
        
        #clear memory
        del lenSums
        del seqs
        del offsets
    
    #return the two filenames
    return (seqFNPrefix, offsetFN)

def mergeNewMSBWTPoolCall(tup):
    '''
    This is a single process call of a chunk of the data to merge BWTs
    @param bID - the block ID this process is processing
    @param binSize - the size of a bin/block
    @param vcLen - 6, hardcoded upstream
    @param currOffsetCounts - the starting position in each input BWT for this process chunk, useful for FM extraction
    @param placeArrayFN - the filename for the input origin bits
    @param nextPlaceArrayFN - the filename for the output origin bits
    @param bwtDirs - the actual input BWT directories to merge
    '''
    (bID, binSize, vcLen, currOffsetCounts, placeArrayFN, nextPlaceArrayFN, bwtDirs) = tup
    
    #load things to run
    placeArray = np.load(placeArrayFN, 'r')
    nextPlaceArray = np.load(nextPlaceArrayFN, 'r+')
    
    numInputs = len(bwtDirs)
    msbwts = [None]*numInputs
    mergedLength = 0
    for i, bwtDir in enumerate(bwtDirs):
        msbwts[i] = MultiStringBWT.loadBWT(bwtDir)
        mergedLength += msbwts[i].totalSize
        
    #state info we need to pass back
    nextBinHasChanged = {}
    nextOffsetCounts = {}
    needsMoreIterations = False
    
    #get the region and count the number of 0s, 1s, 2s, etc.
    region = placeArray[bID*binSize:(bID+1)*binSize]
    srcCounts = np.bincount(region, minlength=numInputs)
    
    #first extract the two subregions from each bwt
    inputIndices = currOffsetCounts
    chunks = [None]*numInputs
    for x in xrange(0, numInputs):
        chunks[x] = msbwts[x].getBWTRange(int(inputIndices[x]), int(inputIndices[x]+srcCounts[x]))
        
    #count the number of characters of each
    bcs = [None]*numInputs
    for x in xrange(0, numInputs):
        bcs[x] = np.bincount(chunks[x], minlength=vcLen)
    
    #interleave these character based on the region
    cArray = np.zeros(dtype='<u1', shape=(region.shape[0],))
    for x in xrange(0, numInputs):
        cArray[region == x] = chunks[x]
    
    #calculate curr using the MSBWT searches
    curr = np.zeros(dtype='<u8', shape=(vcLen,))
    for x in xrange(0, numInputs):
        curr += msbwts[x].getFullFMAtIndex(int(inputIndices[x]))
    
    #this is the equivalent of bin sorting just this small chunk
    for c in xrange(0, vcLen):
        totalC = 0
        for x in xrange(0, numInputs):
            totalC += bcs[x][c]
        
        if totalC == 0:
            continue
        
        #extract the zeroes and ones for this character
        packed = region[cArray == c]
        
        #calculate which bin they are in and mark those bins are changed if different
        b1 = int(math.floor(curr[c]/binSize))
        b2 = int(math.floor((curr[c]+totalC)/binSize))
        if b1 == b2:
            if not np.array_equal(placeArray[curr[c]:curr[c]+packed.shape[0]], packed):
                nextBinHasChanged[b1] = True
                needsMoreIterations = True
            
            origins = np.bincount(packed, minlength=numInputs)
            nextOffsetCounts[b1] = origins+nextOffsetCounts.get(b1, (0,)*numInputs)
            
        else:
            #b1 and b2 are different bins
            delta = b2*binSize-curr[c]
            if not np.array_equal(placeArray[curr[c]:b2*binSize], packed[0:delta]):
                nextBinHasChanged[b1] = True
                needsMoreIterations = True
            
            if not np.array_equal(placeArray[b2*binSize:b2*binSize+packed.shape[0]-delta], packed[delta:]):
                nextBinHasChanged[b2] = True
                needsMoreIterations = True
            
            origins1 = np.bincount(packed[0:delta], minlength=numInputs)
            nextOffsetCounts[b1] = origins1+nextOffsetCounts.get(b1, (0,)*numInputs)
            
            origins2 = np.bincount(packed[delta:], minlength=numInputs)
            nextOffsetCounts[b2] = origins2+nextOffsetCounts.get(b2, (0,)*numInputs)
            
        #this is where we actually do the updating
        nextPlaceArray[curr[c]:curr[c]+totalC] = packed[:]
        
    #cleanup time
    del srcCounts
    del region
    del chunks
    del cArray
    gc.collect()
    
    return (bID, nextBinHasChanged, nextOffsetCounts, needsMoreIterations)

def mergeNewMSBWT(mergedDir, inputBwtDirs, numProcs, logger):
    '''
    This function will take a list of input BWTs (compressed or not) and merge them into a single BWT
    @param mergedFN - the destination for the final merged MSBWT
    @param inputBWTFN1 - the fn of the first BWT to merge
    @param inputBWTFN2 - the fn of the second BWT to merge
    @param numProcs - number of processes we're allowed to use
    @param logger - output goes here
    '''
    st = time.time()
    iterst = time.time()
    vcLen = 6
    
    #TODO: take advantage of these to skip an iteration or two perhaps
    numInputs = len(inputBwtDirs)
    msbwts = [None]*numInputs
    mergedLength = 0
    for i, dirName in enumerate(inputBwtDirs):
        '''
        NOTE: in practice, since we're allowing for multiprocessing, we construct the FM-index for each input BWT
        simply because in the long run, this allows us to figure out how to start processing chunks separately.
        Without this, we would need to track extra information that really just represent the FM-index.
        '''
        msbwts[i] = MultiStringBWT.loadBWT(dirName, logger)
        mergedLength += msbwts[i].totalSize
    
    #binSize = 2**1#small bin debugging
    #binSize = 2**15#this one is just for test purposes, makes easy to debug things
    #binSize = 2**25#diff in 22-23 is not that much, 23-24 was 8 seconds of difference, so REALLY no diff
    binSize = 2**28
    
    #allocate the mergedBWT space
    logger.info('Allocating space on disk...')
    mergedBWT = np.lib.format.open_memmap(mergedDir+'/msbwt.npy', 'w+', '<u1', (mergedLength,))
    
    #this one will create the array using bits
    logger.info('Initializing iterations...')
    placeArray = np.lib.format.open_memmap(mergedDir+'/temp.0.npy', 'w+', '<u1', (mergedBWT.shape[0],))
    copiedPlaceArray = np.lib.format.open_memmap(mergedDir+'/temp.1.npy', 'w+', '<u1', (mergedBWT.shape[0],))
    start = msbwts[0].totalSize
    end = 0
    
    #fill out the initial array with 0s, 1s, 2s, etc. as our initial condition
    for i, msbwt in enumerate(msbwts):
        end += msbwt.getTotalSize()
        placeArray[start:end].fill(i)
        copiedPlaceArray[start:end].fill(i)
        start = end
    
    #create something to track the offsets
    #TODO: x/binSize + 1 makes one too many bins if it's exactly divisible by binSize, ex: 4 length BWT with binSize 2
    nextBinHasChanged = np.ones(dtype='b', shape=(mergedBWT.shape[0]/binSize+1,))
    prevOffsetCounts = np.zeros(dtype='<u8', shape=(mergedBWT.shape[0]/binSize+1, numInputs))
    currOffsetCounts = np.zeros(dtype='<u8', shape=(mergedBWT.shape[0]/binSize+1, numInputs))
    nextOffsetCounts = np.zeros(dtype='<u8', shape=(mergedBWT.shape[0]/binSize+1, numInputs))
    binUpdates = [{}]*(mergedBWT.shape[0]/binSize+1)
    
    bwtInd = 0
    offsets = [0]*numInputs
    for x in xrange(0, currOffsetCounts.shape[0]):
        #set, then change for next iter
        nextOffsetCounts[x] = offsets
        remaining = binSize
        while remaining > 0 and bwtInd < numInputs:
            if remaining > msbwts[bwtInd].totalSize-offsets[bwtInd]:
                remaining -= msbwts[bwtInd].totalSize-offsets[bwtInd]
                offsets[bwtInd] = msbwts[bwtInd].totalSize
                bwtInd += 1
            else:
                offsets[bwtInd] += remaining
                remaining = 0
    
    ignored = 0
    
    #original
    sys.stdout.write('\rcp ')
    sys.stdout.flush()
        
    del copiedPlaceArray
    needsMoreIterations = True
    
    i = 0
    sameOffsetCount = 0
    while needsMoreIterations:
        prevOffsetCounts = currOffsetCounts
        currOffsetCounts = nextOffsetCounts
        nextOffsetCounts = np.zeros(dtype='<u8', shape=(mergedBWT.shape[0]/binSize+1, numInputs))
        needsMoreIterations = False
        sameOffsetCount = 0
        
        #this method uses a condensed byte and will ignore regions that are already finished
        sys.stdout.write('\rld ')
        sys.stdout.flush()
        ignored = 0
        
        iteret = time.time()
        sys.stdout.write('\r')
        logger.info('Finished iter '+str(i)+' in '+str(iteret-iterst)+'seconds')
        iterst = time.time()
        i += 1
        
        sys.stdout.write('\rld')
        sys.stdout.flush()
        
        #track which bins are actually different
        binHasChanged = nextBinHasChanged
        nextBinHasChanged = np.zeros(dtype='b', shape=(mergedBWT.shape[0]/binSize+1))
        
        tups = []
        
        for x in xrange(0, mergedBWT.shape[0]/binSize + 1):
            #check if the current offset matches the previous iteration offset
            sameOffset = np.array_equal(currOffsetCounts[x], prevOffsetCounts[x])
            
            if sameOffset:
                sameOffsetCount += 1
            
            '''
            TODO: the below False is there because this only works if you do a full file copy right now.  It's
            because unless we copy, then the appropriate parts of the nextPlaceArray isn't properly updated. It's
            unclear whether one of these is better than the other in terms of performance.  File copying is slow, but
            if only a couple sequences are similar then then skipping is good.  I think in general, we only skip at the
            beginning for real data though, so I'm going with the no-skip, no-copy form until I can resolve the
            problem (if there's a resolution).
            '''
            if False and not binHasChanged[x] and sameOffset:
                for key in binUpdates[x]:
                    nextOffsetCounts[key] += binUpdates[x][key]
                ignored += 1
            else:
                #note these are swapped depending on the iteration, saves time since there is no file copying
                if i % 2 == 0:
                    tup = (x, binSize, vcLen, currOffsetCounts[x], mergedDir+'/temp.0.npy', mergedDir+'/temp.1.npy', inputBwtDirs)
                else:
                    tup = (x, binSize, vcLen, currOffsetCounts[x], mergedDir+'/temp.1.npy', mergedDir+'/temp.0.npy', inputBwtDirs)
                tups.append(tup)
        
        if numProcs > 1:
            #TODO: tinker with chunksize, it might matter
            myPool = multiprocessing.Pool(numProcs)
            #myPool = multiprocessing.pool.ThreadPool(numProcs)
            rets = myPool.imap(mergeNewMSBWTPoolCall, tups, chunksize=10)
        else:
            rets = []
            for tup in tups:
                rets.append(mergeNewMSBWTPoolCall(tup))
        
        progressCounter = ignored
        sys.stdout.write('\r'+str(100*progressCounter*binSize/mergedBWT.shape[0])+'%')
        sys.stdout.flush()
            
        for ret in rets:
            #iterate through the returns so we can figure out information necessary for continuation
            (x, nBHC, nOC, nMI) = ret
            binUpdates[x] = nOC
            for k in nBHC:
                nextBinHasChanged[k] |= nBHC[k]
            for b in nOC:
                nextOffsetCounts[b] += nOC[b]
            needsMoreIterations |= nMI
            
            progressCounter += 1
            sys.stdout.write('\r'+str(min(100*progressCounter*binSize/mergedBWT.shape[0], 100))+'%')
            sys.stdout.flush()
        
        nextOffsetCounts = np.cumsum(nextOffsetCounts, axis=0)-nextOffsetCounts
        if numProcs > 1:
            myPool.terminate()
            myPool.join()
            myPool = None
        
    sys.stdout.write('\r')
    sys.stdout.flush()
    logger.info('Order solved, saving final array...')
    
    #TODO: make this better
    offsets = np.zeros(dtype='<u8', shape=(numInputs,))
    for i in xrange(0, mergedBWT.shape[0]/binSize+1):
        ind = placeArray[i*binSize:(i+1)*binSize]
        if i == mergedBWT.shape[0]/binSize:
            ind = ind[0:mergedBWT.shape[0]-i*binSize]
        
        bc = np.bincount(ind, minlength=numInputs)
        
        for x in xrange(0, numInputs):
            mergedBWT[np.add(i*binSize, np.where(ind == x))] = msbwts[x].getBWTRange(int(offsets[x]), int(offsets[x]+bc[x]))
        offsets += bc
        
    et = time.time()
    
    logger.info('Finished all merge iterations in '+str(et-st)+' seconds.')
    
def compressBWT(inputFN, outputFN, numProcs, logger):
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
    letterBits = 3
    numberBits = 8-letterBits
    numPower = 2**numberBits
    mask = 255 >> letterBits
    
    #load the thing to compress
    logger.info('Loading src file...')
    bwt = np.load(inputFN, 'r')
    logger.info('Original size:'+str(bwt.shape[0])+'B')
    numProcs = min(numProcs, bwt.shape[0])
    
    #first locate boundaries
    tups = []
    binSize = 1000000
    numBins = max(numProcs, bwt.shape[0]/binSize)
    
    for i in xrange(0, numBins):
        startIndex = i*bwt.shape[0]/numBins
        endIndex = (i+1)*bwt.shape[0]/numBins
        tempFN = outputFN+'.temp.'+str(i)+'.npy'
        tups.append((inputFN, startIndex, endIndex, tempFN))
    
    logger.info('Compressing bwt...')
    
    #run our multi-processed builder
    if numProcs > 1:
        myPool = multiprocessing.Pool(numProcs)
        rets = myPool.map(compressBWTPoolProcess, tups)
    else:
        rets = []
        for tup in tups:
            rets.append(compressBWTPoolProcess(tup))
    
    #calculate how big it will be after combining the separate chunk
    totalSize = 0
    prevChar = -1
    prevTotal = 0
    for ret in rets:
        #start by just adding the raw size
        totalSize += ret[0]
        
        #check if we need to compensate for a break point like AAA|A
        if prevChar == ret[1]:
            totalSize -= int(math.floor(math.log(prevTotal, numPower)+1)+math.floor(math.log(ret[2], numPower)+1))
            prevTotal += ret[2]
            totalSize += int(math.floor(math.log(prevTotal, numPower)+1))
            if ret[0] == 1:
                #don't clear prev total
                pass
            else:
                prevTotal = ret[4]
        else:
            prevTotal = ret[4]
        prevChar = ret[3]
    
    #make the real output by joining all of the partial compressions into a single file
    finalBWT = np.lib.format.open_memmap(outputFN, 'w+', '<u1', (totalSize,))
    logger.info('Calculated compressed size:'+str(totalSize)+'B')
    logger.info('Joining sub-compressions...')
    
    #iterate a second time, this time storing things
    prevChar = -1
    prevTotal = 0
    offset = 0
    for ret in rets:
        copyArr = np.load(ret[5], 'r')
        if prevChar == ret[1]:
            #calculate byte usage of combining
            prevBytes = int(math.floor(math.log(prevTotal, numPower)+1))
            nextBytes = int(math.floor(math.log(ret[2], numPower)+1))
            prevTotal += ret[2]
            
            #actually combine them
            offset -= prevBytes
            power = 0
            while prevTotal >= numPower**power:
                finalBWT[offset] = (((prevTotal / (numPower**power)) & mask) << letterBits)+prevChar
                power += 1
                offset += 1
            
            #copy over the extra stuff and calculate the new offset
            finalBWT[offset:offset+(copyArr.shape[0]-nextBytes)] = copyArr[nextBytes:]
            offset += (copyArr.shape[0] - nextBytes)
            
            if ret[0] == 1:
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
    
def compressBWTPoolProcess(tup):
    '''
    During compression, each available process will calculate a subportion of the BWT independently using this 
    function.  This process takes the chunk and rewrites it into a given filename using the technique described
    in the compressBWT(...) function header
    '''
    #pull the tuple info
    inputFN = tup[0]
    startIndex = tup[1]
    endIndex = tup[2]
    tempFN = tup[3]
    
    #this shouldn't happen
    if startIndex == endIndex:
        print 'ERROR: EQUAL INDICES'
        return None
    
    #load the file
    bwt = np.load(inputFN, 'r')
    
    #create bit spacings
    letterBits = 3
    numberBits = 8-letterBits
    numPower = 2**numberBits
    mask = 255 >> letterBits
    
    #search for the places they're different
    whereSol = np.add(startIndex+1, np.where(bwt[startIndex:endIndex-1] != bwt[startIndex+1:endIndex])[0])
    
    #this is the difference between two adjacent ones
    deltas = np.zeros(dtype='<u4', shape=(whereSol.shape[0]+1,))
    if whereSol.shape[0] == 0:
        deltas[0] = endIndex-startIndex
    else:
        deltas[0] = whereSol[0]-startIndex
        deltas[1:-1] = np.subtract(whereSol[1:], whereSol[0:-1])
        deltas[-1] = endIndex - whereSol[-1]
    
    #calculate the number of bytes we need to store this information
    size = 0
    byteCount = 0
    lastCount = 1
    while lastCount > 0:
        lastCount = np.where(deltas >= 2**(numberBits*byteCount))[0].shape[0]
        size += lastCount
        byteCount += 1
    
    #create the file
    ret = np.lib.format.open_memmap(tempFN, 'w+', '<u1', (size,))
    retIndex = 0
    c = bwt[startIndex]
    startChar = c
    delta = deltas[0]
    while delta > 0:
        ret[retIndex] = ((delta & mask) << letterBits)+c
        delta /= numPower
        retIndex += 1
    
    #fill in the values based on the bit functions
    for i in xrange(0, whereSol.shape[0]):
        c = bwt[whereSol[i]]
        delta = deltas[i+1]
        while delta > 0:
            ret[retIndex] = ((delta & mask) << letterBits)+c
            delta /= numPower
            retIndex += 1
    endChar = c
    
    #return a lot of information so we can easily combine the results
    return (size, startChar, deltas[0], endChar, deltas[-1], tempFN)
    
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
    msbwt = MultiStringBWT.CompressedMSBWT()
    msbwt.loadMsbwt(inputDir, logger)
    
    #make the output file
    outputFile = np.lib.format.open_memmap(outputDir+'/msbwt.npy', 'w+', '<u1', (msbwt.getTotalSize(),))
    del outputFile
    
    worksize = 1000000
    tups = [None]*(msbwt.getTotalSize()/worksize+1)
    x = 0
    
    if msbwt.getTotalSize() > worksize:
        for x in xrange(0, msbwt.getTotalSize()/worksize):
            tups[x] = (inputDir, outputDir, x*worksize, (x+1)*worksize)
        tups[-1] = (inputDir, outputDir, (x+1)*worksize, msbwt.getTotalSize())
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
    msbwt = MultiStringBWT.CompressedMSBWT()
    msbwt.loadMsbwt(inputDir, None)
    
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
        