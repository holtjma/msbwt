#!python
#cython: boundscheck=False
#cython: wraparound=False

import binascii
import gzip
import math
import multiprocessing
from multiprocessing.pool import ThreadPool
import numpy as np
cimport numpy as np
import os
import time

from cython.operator cimport preincrement as inc

import MSBWTGenCython as MSBWTGen

def createMSBWTFromSeqs(list seqArray, char * mergedDir, unsigned long numProcs, bint areUniform, logger):
    '''
    This function takes a list of input strings and builds the multi-string BWT for it using the merging
    method. Given R reads containing N bases (counting '$' as a base) this algorithm takes O(N*LCP*lg(R))
    time to complete and uses O(N) extra bits of memory in it's peak (aka, final) iteration.  Note that
    inputs and outputs are not counted in memory usage since they are read/written to disk.
    @param seqArray - the list of strings to be merged
    @param mergedDir - the directory we plan on saving the final result
    @param numProcs - the number of processors we're allowed to use
    @param areUniform - indicates if the reads are uniform length, use False if unknown
    @param logger - the logger to print all output
    '''
    #wipe the auxiliary data stored here
    MSBWTGen.clearAuxiliaryData(mergedDir)
    
    #create standard filenames
    seqFN = mergedDir+'/seqs.npy'
    offsetFN = mergedDir+'/offsets.npy'
    msbwtFN = mergedDir+'/msbwt.npy'
    
    #convert the sequences into BWTs in uint8 format and then save it
    formatSeqsForMerge(seqArray, seqFN, offsetFN, numProcs, areUniform, logger)
    
    #finally do the actual interleaving, hand it the output file
    #interleaveSeqs(mergedDir, areUniform, logger)
    interleaveLevelMerge(mergedDir, numProcs, areUniform, logger)
    
def createMSBWTFromFasta(list fastaFNs, char * outputDir, unsigned long numProcs, bint areUniform, logger):
    '''
    This function takes a list of input fasta filenames and builds the multi-string BWT for it using the merging
    method. Given R reads containing N bases (counting '$' as a base) this algorithm takes O(N*LCP*lg(R))
    time to complete and uses O(N) extra bits of memory in it's peak (aka, final) iteration.  Note that
    inputs and outputs are not counted in memory usage since they are read/written to disk.
    @param seqArray - the list of strings to be merged
    @param mergedDir - the directory we plan on saving the final result
    @param numProcs - the number of processors we're allowed to use
    @param areUniform - indicates if the reads are uniform length, use False if unknown
    @param logger - the logger to print all output
    '''
    #wipe the auxiliary data stored here
    MSBWTGen.clearAuxiliaryData(outputDir)
    
    #use standard filenames
    seqFN = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    bwtFN = outputDir+'/msbwt.npy'
    
    #build an iterator
    seqIter = fastaIterator(fastaFNs, logger)
    
    #convert the sequences into BWTs in uint8 format and then save it
    formatSeqsForMerge(seqIter, seqFN, offsetFN, numProcs, areUniform, logger)
    
    #finally do the actual interleaving, hand it the output file
    #interleaveSeqs(outputDir, areUniform, logger)
    interleaveLevelMerge(outputDir, numProcs, areUniform, logger)

def createMSBWTFromFastq(list fastqFNs, char * outputDir, unsigned long numProcs, bint areUniform, logger):
    '''
    This function takes a list of input fasta filenames and builds the multi-string BWT for it using the merging
    method. Given R reads containing N bases (counting '$' as a base) this algorithm takes O(N*LCP*lg(R))
    time to complete and uses O(N) extra bits of memory in it's peak (aka, final) iteration.  Note that
    inputs and outputs are not counted in memory usage since they are read/written to disk.
    @param seqArray - the list of strings to be merged
    @param mergedDir - the directory we plan on saving the final result
    @param numProcs - the number of processors we're allowed to use
    @param areUniform - indicates if the reads are uniform length, use False if unknown
    @param logger - the logger to print all output
    '''
    #wipe the auxiliary data stored here
    MSBWTGen.clearAuxiliaryData(outputDir)
    
    #use standard filenames
    seqFN = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    bwtFN = outputDir+'/msbwt.npy'
    
    #build an iterator
    seqIter = fastqIterator(fastqFNs, logger)
    
    #convert the sequences into BWTs in uint8 format and then save it
    formatSeqsForMerge(seqIter, seqFN, offsetFN, numProcs, areUniform, logger)
    
    #finally do the actual interleaving, hand it the output file
    #interleaveSeqs(outputDir, areUniform, logger)
    interleaveLevelMerge(outputDir, numProcs, areUniform, logger)

def mergeTwoMSBWTs(char * inputMsbwtDir1, char * inputMsbwtDir2, char * mergedDir, unsigned long numProcs, logger):
    '''
    @param inputMsbwtDir1 - the directory of the first MSBWT
    @param inputMsbwtDir2 - the directory of the second MSBWT
    @param mergedDir - the directory for output
    @param numProcs - number of processes we're allowed to use
    @param logger - use for logging outputs and progress
    '''
    logger.info('Beginning Merge:')
    logger.info('Input 1:\t'+inputMsbwtDir1)
    logger.info('Input 2:\t'+inputMsbwtDir2)
    logger.info('Output:\t'+mergedDir)
    
    #hardcode this as we do everywhere else
    cdef unsigned long numValidChars = 6
    
    #map the seqs, note we map msbwt.npy because that's where all changes happen
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] inputBwt1 = np.load(inputMsbwtDir1+'/msbwt.npy', 'r+')
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] inputBwt2 = np.load(inputMsbwtDir2+'/msbwt.npy', 'r+')
    cdef np.uint8_t [:] inputBwt1_view = inputBwt1
    cdef np.uint8_t [:] inputBwt2_view = inputBwt2
    cdef unsigned long bwtLen1 = inputBwt1.shape[0]
    cdef unsigned long bwtLen2 = inputBwt2.shape[0]
    
    #prepare to construct total counts for the symbols
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] totalCounts = np.zeros(dtype='<u8', shape=(numValidChars, ))
    cdef np.uint64_t [:] totalCounts_view = totalCounts
    
    #first calculate the total counts for our region
    cdef unsigned long x, y
    with nogil:
        for x in xrange(0, bwtLen1):
            totalCounts_view[inputBwt1_view[x]] += 1
        for x in xrange(0, bwtLen2):
            totalCounts_view[inputBwt2_view[x]] += 1
    
    #now find our offsets which we will keep FOREVER
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmStarts = np.cumsum(totalCounts)-totalCounts
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmCurrent
    cdef np.uint64_t [:] fmCurrent_view
    
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
    
    cdef unsigned long binBits = 11
    cdef unsigned long binSize = 2**binBits
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmIndex0
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmIndex1
    cdef np.uint64_t [:, :] fmIndex0_view
    cdef np.uint64_t [:, :] fmIndex1_view
    
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
    
    #fmindex stuff now
    logger.info('Initializing FM-indices...')
    fmIndex0 = np.zeros(dtype='<u8', shape=(bwtLen1/binSize + 2, numValidChars))
    fmIndex1 = np.zeros(dtype='<u8', shape=(bwtLen2/binSize + 2, numValidChars))
    fmIndex0_view = fmIndex0
    fmIndex1_view = fmIndex1
    #initializeFMIndex(&seqs_view[offsets_view[0]], &fmIndex0_view[0][0], offsets_view[1]-offsets_view[0], binSize, numValidChars)
    initializeFMIndex(&inputBwt1_view[0], &fmIndex0_view[0][0], bwtLen1, binSize, numValidChars)
    #initializeFMIndex(&seqs_view[offsets_view[1]], &fmIndex1_view[0][0], offsets_view[2]-offsets_view[1], binSize, numValidChars)
    initializeFMIndex(&inputBwt2_view[0], &fmIndex1_view[0][0], bwtLen2, binSize, numValidChars)
    
    #values tracking progress
    cdef unsigned int iterCount = 0
    cdef bint changesMade = True
    
    #make a copy of the offset that the subfunction can modify safely
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] ranges
    cdef double st, el
    
    #format is (position in bwt0, position in bwt1, total length)
    ranges = np.zeros(dtype='<u8', shape=(1, 3))
    ranges[0][2] = bwtLen1+bwtLen2
    
    while changesMade:
        #copy the fm info
        fmCurrent = np.copy(fmStarts)
        fmCurrent_view = fmCurrent
        
        if iterCount % 2 == 0:
            st = time.time()
            ret = targetedIterationMerge2(&inputBwt1_view[0], &inputBwt2_view[0], 
                                          &inter0_view[0], &inter1_view[0], bwtLen1+bwtLen2, 
                                          fmIndex0_view, fmIndex1_view, ranges, binBits, numValidChars, iterCount)
            changesMade = ret[0]
            ranges = ret[1]
            el = time.time()-st
            logText = '\t'.join([str(val) for val in (0, iterCount, ranges.shape[0], el, np.sum(ranges[:, 2]))])
        else:
            st = time.time()
            ret = targetedIterationMerge2(&inputBwt1_view[0], &inputBwt2_view[0], 
                                          &inter1_view[0], &inter0_view[0], bwtLen1+bwtLen2, 
                                          fmIndex0_view, fmIndex1_view, ranges, binBits, numValidChars, iterCount)
            el = time.time()-st
            changesMade = ret[0]
            ranges = ret[1]
            logText = '\t'.join([str(val) for val in (0, iterCount, ranges.shape[0], el, np.sum(ranges[:, 2]))])
        
        logger.info(logText)
        iterCount += 1
    
    #create access points to our result and a temporary result
    #cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] msbwt = np.load(mergedDir+'/msbwt.npy', 'r+')
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] msbwt = np.lib.format.open_memmap(mergedDir+'/msbwt.npy', 'w+', '<u1', (bwtLen1+bwtLen2,))
    cdef np.uint8_t [:] msbwt_view = msbwt
    
    cdef unsigned long readID
    cdef unsigned long pos1 = 0
    cdef unsigned long pos2 = 0

    for x in xrange(0, bwtLen1+bwtLen2):
        #get the read, the symbol, and increment the position in that read
        if getBit_p(inter0_p, x):
            msbwt_view[x] = inputBwt2_view[pos2]
            pos2 += 1
        else:
            msbwt_view[x] = inputBwt1_view[pos1]
            pos1 += 1
    '''
    print inputBwt1
    print fmIndex0
    print inputBwt2
    print fmIndex1
    print inter0
    print inter1
    print msbwt
    '''
    #remove this temp files also
    if interleaveBytes > interThresh:
        os.remove(interleaveFN1)
    else:
        np.save(interleaveFN0, inter0)
    
    #return the number of iterations it took us to converge
    return iterCount
    
def fastaIterator(list fastaFNs, logger):
    '''
    This function is a generator for parsing the a list of fasta files.  Each yield returns a string containing
    only the bases for the read (i.e., no quality or name).
    @param fastaFNs - a list of fasta filenames to read
    @param logger - logger for noting progress through the files
    '''
    currRead = ''
    for fn in fastaFNs:
        #load our file for processing
        logger.info('Loading \''+fn+'\'...')
        fp = open(fn, 'r')
        
        for line in fp:
            if line[0] == '>':
                #make sure we aren't trying to return an empty string
                if currRead != '':
                    yield currRead+'$'
                
                #reset to empty read now
                currRead = ''
            else:
                #add the line since a read spans multiple lines, so dumb, like so dumb for real
                currRead += line.strip('\n')
        
        #return the last one and close up the file
        if currRead != '':
            yield currRead+'$'
            currRead = ''
            
        #close this file and move on to the next
        fp.close()

def fastqIterator(list fastqFNs, logger):
    '''
    Iterate through the fastq input files which may be compressed
    @param fastqFNs - the fastq filenames
    @param logger - the logger
    '''
    cdef unsigned int i
    for fnID, fn in enumerate(fastqFNs):
        #open the file and read in starting form the second, every 4th line
        logger.info('Loading \''+fn+'\'...')
        if fn.endswith('.gz'):
            fp = gzip.open(fn, 'r')
        else:
            fp = open(fn, 'r')
        
        #go through each line
        i = 0
        for line in fp:
            if i % 4 == 1:
                #seqArray.append((line.strip('\n')+'$', fnID, i/4))
                yield line.strip('\n')+'$'
            i += 1
                
        fp.close()
    
def formatSeqsForMerge(seqIter, char * seqFN, char * offsetFN, unsigned long numProcs, bint areUniform, logger):
    '''
    This function takes an input iterable and reformats all of the input into a sequence file we can read for 
    merging.  Part of the reformat is to recode each input string as a BWT in this file.  Additionally, offset
    information is stored to indicate the start/end of each input.  Creates /seqs.npy and /offsets.npy.
    @param seqIter - the iterator that pulls strings
    @param seqFN - the sequence file we are writing to, typically '<BWT_DIR>/seqs.npy'
    @param offsetFN - the offset file we are writing to, typically '<BWT_DIR>/offsets.npy'
    @param numProcs - the number of processes we are allowed to use
    @parma areUniform - boolean indicating whether reads are of uniform length or not
    '''
    logger.info('Formatting sequences for merging...')
    
    #open the files for writing
    seqFP = open(seqFN, 'wb')
    offsetFP = open(offsetFN, 'wb')
    
    #everything is 64-bit offsets now, that should be more than enough for the dataset we plan on handling
    cdef unsigned long offsetNumBytes = 8
    cdef unsigned long offsetNumNibbles = 2*offsetNumBytes
    
    #define a constant character map for now
    cdef dict d = {'$':0, 'A':1, 'C':2, 'G':3, 'N':4, 'T':5}
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] dArr = np.add(np.zeros(dtype='<u1', shape=(256,)), len(d.keys()))
    cdef np.uint8_t [:] dArr_view = dArr
    
    #set the corresponding value for each symbol we care about
    for c in d.keys():
        dArr_view[<unsigned int>ord(c)] = d[c]
    
    #this values are for writing offsets to the memmap file
    #cdef np.uint64_t seqOffset = 0
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] seqOffsetArrayWrite = np.zeros(dtype='<u8', shape=(1, ))
    cdef unsigned long seqLen = 0
    
    cdef unsigned long numReads = 0
    
    #if the reads are not uniform, we stored each offset, so we need the all zero offset first
    if not areUniform:
        offsetFP.write(binascii.unhexlify('0'*offsetNumNibbles))
    
    #pool out the BWT creation, note we technically use numProcs+1 processes because this one handles results
    pool = multiprocessing.Pool(numProcs)
    res = pool.imap(memoryBWT, seqIter, chunksize=40)
    
    #values for determining when to log output to our logger
    cdef unsigned long bufferOutDist = 10**9
    cdef unsigned long nextOutput = bufferOutDist
    
    #we get back a string encoded numpy array and the length of the sequence in symbols
    for bwt, seqLen in res:
        #store the string with all our sequences
        seqFP.write(bwt)
        
        #update our offset
        #seqOffset += seqLen
        seqOffsetArrayWrite[0] += seqLen
        
        #we only write this offset if it's non-uniform
        if not areUniform:
            #create a 1-D array so we can use tostring
            #seqOffsetArrayWrite[0] = seqOffset
            offsetFP.write(seqOffsetArrayWrite.tostring())
        
        #check if we want to write anything to our logger
        if seqOffsetArrayWrite[0] > nextOutput:
            logger.info('Processed '+str(seqOffsetArrayWrite[0]/(1000000000))+' giga-bases...')
            nextOutput += bufferOutDist
            seqFP.flush()
            offsetFP.flush()
    
    #uniform seqs only need this written once
    if areUniform:
        #create a 1-D array so we can use tostring()
        seqOffsetArrayWrite[0] = seqLen
        offsetFP.write(seqOffsetArrayWrite.tostring())
    
    #cleanup of the pool
    pool.terminate()
    pool.join()
    pool = None
    
    #save it all
    seqFP.close()
    offsetFP.close()

def memoryBWT(seq):
    '''
    This function takes a single string as input and creates a BWT from it.
    @param seq - the sequence we want to turn into a BWT in memory
    @return - tuple containing
        (byte encoded BWT result (can be written straight to file),
         string length)
    '''
    #define a constant character map for now
    cdef dict d = {'$':0, 'A':1, 'C':2, 'G':3, 'N':4, 'T':5}
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] dArr = np.add(np.zeros(dtype='<u1', shape=(256,)), len(d.keys()))
    cdef np.uint8_t [:] dArr_view = dArr
    
    #create an easy reference array for converting from char to int
    for c in d.keys():
        dArr_view[<unsigned int>ord(c)] = d[c]
    
    #create the sequence permutations
    cdef unsigned long seqLen = len(seq)
    cdef char * seq_view = seq
    
    #interleave stuff
    cdef np.ndarray[np.uint16_t, ndim=1, mode='c'] arr0 = np.zeros(dtype='<u2', shape=(seqLen, ))
    cdef np.uint16_t [:] arr0_view = arr0
    cdef np.ndarray[np.uint16_t, ndim=1, mode='c'] arr1 = np.zeros(dtype='<u2', shape=(seqLen, ))
    cdef np.uint16_t [:] arr1_view = arr1
    cdef np.ndarray[np.uint16_t, ndim=1, mode='c'] tempArr
    
    #convert the string
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] converted = np.zeros(dtype='<u1', shape=(seqLen, ))
    cdef np.uint8_t [:] converted_view = converted
    
    #ret and totalcounts for each symbol
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] ret = np.zeros(dtype='<u1', shape=(seqLen, ))
    cdef np.uint8_t [:] ret_view = ret
    cdef np.ndarray[np.uint16_t, ndim=1, mode='c'] totalCounts = np.zeros(dtype='<u2', shape=(6, ))
    cdef np.uint16_t [:] totalCounts_view = totalCounts
    
    #iterate through the plain string doing both conversions and counts simultaneously
    cdef unsigned int x
    with nogil:
        for x in xrange(0, seqLen):
            converted_view[x] = dArr_view[seq_view[x]]
            totalCounts_view[converted_view[x]] += 1
            arr0_view[x] = x
    
    #fm index type stuff which helps build the thing
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmOffsets = np.cumsum(totalCounts)-totalCounts
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmOffsetsCopy
    cdef np.uint64_t [:] fmOffsetsCopy_view
    cdef unsigned long ind
    cdef np.uint8_t symbol
    
    #iterate until convergence, should be relatively quick for a single string
    while not np.array_equal(arr0, arr1):
        #swap input and output arrays, ping-pong it up
        tempArr = arr1
        arr1 = arr0
        arr1_view = arr1
        arr0 = tempArr
        arr0_view = arr0
        
        #copy the offsets into our working copy
        fmOffsetsCopy = np.copy(fmOffsets)
        fmOffsetsCopy_view = fmOffsetsCopy
        
        #now go through each symbol correcting the predecessor, aka radix sort it
        with nogil:
            for x in xrange(0, seqLen):
                #get the predecessor index
                if arr1_view[x] == 0:
                    ind = seqLen-1
                else:
                    ind = arr1_view[x]-1
                
                #retrieve the corresponding symbol and update the next array
                symbol = converted_view[ind]
                arr0_view[fmOffsetsCopy_view[symbol]] = ind
                fmOffsetsCopy_view[symbol] += 1
    
    #output the ord of the bwt sequence
    with nogil:
        for x in xrange(0, seqLen):
            if arr0_view[x] == 0:
                ret_view[x] = converted_view[seqLen-1]
            else:
                ret_view[x] = converted_view[arr0_view[x]-1]
    
    #finally return the tuple of (bwt, seqLen)
    return (ret.tostring(), seqLen)

def interleaveLevelMerge(char * mergedDir, unsigned long numProcs, bint areUniform, logger):
    '''
    This function is for use after all input reads have been processed into separate individual BWTs.  It merges
    the separate BWTs into a single MSBWT through several iterations.  At each iteration it merges 2 or more separate
    BWTs/MSBWTs into a new component.  At early iterations, it merges more inputs together before going to fewer to 
    stay in memory and maintain speed.
    @param mergedDir - the directory we are storing the final result in
    @param numProcs - the number of processes we're allowed to make
    @param areUniform - indicates whether reads are of equal length or not
    @param logger - logging logger used for output
    '''
    logger.info('Beginning MSBWT construction...')
    
    #hardcode this as we do everywhere else
    cdef unsigned long numValidChars = 6
    
    #load these from the input
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] seqs = np.memmap(mergedDir+'/seqs.npy', dtype='<u1')
    cdef np.uint8_t [:] seqs_view = seqs
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] offsets = np.memmap(mergedDir+'/offsets.npy', dtype='<u8')
    cdef np.uint64_t [:] offsets_view = offsets
    
    #we do things on a level basis in this method
    cdef unsigned long numSeqs
    if areUniform:
        numSeqs = seqs.shape[0]/offsets[0]
    else:
        numSeqs = offsets.shape[0]-1
    
    #currMultiplier indicates how many strings are in each group, at the start this is 1 since all are separate
    cdef unsigned long currMultiplier = 1
    
    #copy the input strings over to our final output, this is what we'll be manipulating
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] msbwt = np.lib.format.open_memmap(mergedDir+'/msbwt.npy', 'w+', '<u1', (seqs.shape[0], ))
    msbwt[:] = seqs[:]
    
    #only set to 2 or 256, eventually change allow 2, 4, 16, 256, 256^2
    cdef unsigned long splitDist
    
    x = 0
    while currMultiplier < numSeqs:
        #decide whether our level iterator is 2 or 256
        if x < 1:
            splitDist = 256
        else:
            splitDist = 2
        
        logger.info('Processing groups of size '+str(currMultiplier*splitDist)+'...')
        
        #this is an iterator which returns intervals for processing
        inputIter = levelIterator(offsets, areUniform, mergedDir, currMultiplier, splitDist)
        
        #get the results, even doing this with numProcs = 1 is okay
        if numProcs <= 1:
            results = []
            for tup in inputIter:
                results.append(buildViaMerge256(tup))
        else:
            pool = multiprocessing.Pool(numProcs)
            results = pool.imap(buildViaMerge256, inputIter)
        
        #make sure we wait for all inputs, the result is just the number of iterations needed
        totalIters = 0
        totalRets = 0
        for r in results:
            totalIters += r
            totalRets += 1
        
        logger.info('Average iterations: '+str(1.0*totalIters/totalRets))
        
        #clean up the pool
        if numProcs > 1:
            pool.terminate()
            pool.join()
            pool = None
        
        #recalculate where we stand in terms of merging
        currMultiplier *= splitDist
        x += 1
    
    logger.info('MSBWT construction finished.')

def levelIterator(offsets, bint areUniform, mergedDir, unsigned long currMultiplier, unsigned long splitDist):
    '''
    This function is a generator for creating offsets to merge.  It does so based on the current multiplier and 
    the splitDist.
    @param offsets - the offsets array for everything, it's memmaped earlier
    @param areUniform - indicates whether the reads are of uniform length or not
    @param mergedDir - the directory we are placing the final result into, must be python string since we yield it
    @param currMultiplier - the current size of groups, starts at one and increases as the program runs
    @param splitDist - how many groups we allocate to merge
    '''
    #initialize all these counters
    cdef unsigned long offsetLen = offsets.shape[0]
    cdef np.uint64_t [:] offsets_view = offsets
    cdef unsigned long x = 0
    cdef unsigned long y = 0
    
    #this is all we return - these values can't be made into np.ndarray for some odd reason
    seqs = np.memmap(mergedDir+'/seqs.npy', dtype='<u1')
    retOffsets = np.zeros(dtype='<u8', shape=(splitDist+1, ))
    
    cdef unsigned long numSeqs
    cdef unsigned long seqLen
    
    if areUniform:
        #seqlen and numSeqs are a function of the data input 
        seqLen = offsets_view[0]
        numSeqs = seqs.shape[0]/seqLen
        
        for x in xrange(0, numSeqs, currMultiplier):
            #first check if we filled the array
            if y == splitDist:
                #fill in the last one and yield the offset
                retOffsets[y] = x*seqLen
                yield (retOffsets, mergedDir)
                
                #reset for next
                retOffsets = np.zeros(dtype='<u8', shape=(splitDist+1, ))
                y = 0
                
            #now refill
            retOffsets[y] = x*seqLen
            y += 1
        
        #check if we need to yield one last entry which may be small
        if y > 0:
            retOffsets[y] = numSeqs*seqLen
            retOffsets = retOffsets[0:y+1]
            yield (retOffsets, mergedDir)
        
    else:
        #number of sequences must be retrieved
        numSeqs = offsetLen-1
        for x in xrange(0, numSeqs, currMultiplier):
            #first check if we filled the array
            if y == splitDist:
                #fill in the last one and yield the offset
                retOffsets[y] = offsets_view[x]
                yield (retOffsets, mergedDir)
                
                #reset for next
                retOffsets = np.zeros(dtype='<u8', shape=(splitDist+1, ))
                y = 0
                
            #now refill
            retOffsets[y] = offsets_view[x]
            y += 1
        
        #check if we need to yield one last entry which may be small
        if y > 0:
            retOffsets[y] = offsets_view[offsetLen-1]
            retOffsets = retOffsets[0:y+1]
            yield (retOffsets, mergedDir)
    
def buildViaMerge256(tup):
    '''
    This function takes an offset range and merges it into a single BWT
    @param tup[0], offsets - numpy array indicating where the strings in this merge start and end in our seqs.npy file
    @param tup[1], mergedDir - the directory for our final result
    '''
    #hardcode this as we do everywhere else
    cdef unsigned long numValidChars = 6
    
    #extract the input
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] offsets = tup[0]
    cdef char * mergedDir = tup[1]

    #offset stuff
    cdef unsigned long offsetLen = offsets.shape[0]
    cdef unsigned long numInputs = offsetLen-1
    cdef np.uint64_t [:] offsets_view = offsets
    
    #map the seqs, note we map msbwt.npy because that's where all changes happen
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] seqs = np.load(mergedDir+'/msbwt.npy', 'r+')
    cdef np.uint8_t [:] seqs_view = seqs
    
    #prepare to construct total counts for the symbols
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] totalCounts = np.zeros(dtype='<u8', shape=(numValidChars, ))
    cdef np.uint64_t [:] totalCounts_view = totalCounts
    
    #first calculate the total counts for our region
    cdef unsigned long x, y
    #with nogil:
    for x in xrange(offsets_view[0], offsets_view[offsetLen-1]):
        totalCounts_view[seqs_view[x]] += 1
    
    #now find our offsets which we will keep FOREVER
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmStarts = np.cumsum(totalCounts)-totalCounts
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmCurrent
    cdef np.uint64_t [:] fmCurrent_view
    
    #now we should load the interleaves
    #TODO: create methods to dynamically decide how to make the interleave array, size and/or on disk
    cdef unsigned long interleaveBytes
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] inter0
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] inter1
    if numInputs == 2:
        #this needs 1 bit per base, so we allocate bases/8 bytes plus one due to integer division
        interleaveBytes = (offsets_view[numInputs]-offsets_view[0])/8+1
    else:
        #this needs one bytes per base since we have 256 inputs
        interleaveBytes = offsets_view[numInputs]-offsets_view[0]
    
    #hardcoded as 1 GB right now
    cdef unsigned long interThresh = 1*10**9
    if interleaveBytes > interThresh:
        interleaveFN0 = mergedDir+'/inter0.'+str(offsets_view[0])+'.npy'
        interleaveFN1 = mergedDir+'/inter1.'+str(offsets_view[0])+'.npy'
        inter0 = np.lib.format.open_memmap(interleaveFN0, 'w+', '<u1', (interleaveBytes, ))
        inter1 = np.lib.format.open_memmap(interleaveFN1, 'w+', '<u1', (interleaveBytes, ))
    else:
        inter0 = np.zeros(dtype='<u1', shape=(interleaveBytes, ))
        inter1 = np.zeros(dtype='<u1', shape=(interleaveBytes, ))
    
    cdef np.uint8_t [:] inter0_view = inter0
    cdef np.uint8_t [:] inter1_view = inter1
    
    cdef unsigned long binBits = 11
    cdef unsigned long binSize = 2**binBits
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmIndex0
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] fmIndex1
    cdef np.uint64_t [:, :] fmIndex0_view
    cdef np.uint64_t [:, :] fmIndex1_view
    
    #initialize the first interleave based on the offsets
    cdef np.uint8_t * inter0_p
    if numInputs == 2:
        #with two, we will initialize both arrays
        inter0_p = &inter0_view[0]
        #with nogil:
        #initialize the first half to all 0s, 0x00
        for y in xrange(0, (offsets_view[1]-offsets_view[0])/8):
            inter0_view[y] = 0x00
            inter1_view[y] = 0x00
        
        #one byte in the middle can be mixed zeros and ones
        inter0_view[(offsets_view[1]-offsets_view[0])/8] = 0xFF << ((offsets_view[1]-offsets_view[0]) % 8)
        inter1_view[(offsets_view[1]-offsets_view[0])/8] = 0xFF << ((offsets_view[1]-offsets_view[0]) % 8)
        
        #all remaining bytes are all ones, 0xFF
        for y in xrange((offsets_view[1]-offsets_view[0])/8+1, (offsets_view[2]-offsets_view[0])/8+1):
            inter0_view[y] = 0xFF
            inter1_view[y] = 0xFF
        
        #fmindex stuff now
        fmIndex0 = np.zeros(dtype='<u8', shape=((offsets_view[1]-offsets_view[0])/binSize + 2, numValidChars))
        fmIndex1 = np.zeros(dtype='<u8', shape=((offsets_view[2]-offsets_view[1])/binSize + 2, numValidChars))
        fmIndex0_view = fmIndex0
        fmIndex1_view = fmIndex1
        initializeFMIndex(&seqs_view[offsets_view[0]], &fmIndex0_view[0][0], offsets_view[1]-offsets_view[0], binSize, numValidChars)
        initializeFMIndex(&seqs_view[offsets_view[1]], &fmIndex1_view[0][0], offsets_view[2]-offsets_view[1], binSize, numValidChars)
    else:
        #for each input, go through its range and set the byte to the input ID
        #with nogil:
        for x in xrange(0, numInputs):
            for y in xrange(offsets_view[x]-offsets_view[0], offsets_view[x+1]-offsets_view[0]):
                inter0_view[y] = x
    
    #values tracking progress
    cdef unsigned int iterCount = 0
    cdef bint changesMade = True
    
    #make a copy of the offset that the subfunction can modify safely
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] offsetsCopy = np.zeros(dtype='<u8', shape=(offsets.shape[0], ))
    cdef np.uint64_t [:] offsetsCopy_view = offsetsCopy
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] ranges
    cdef double st, el
    if numInputs == 2:
        #format is (position in bwt0, position in bwt1, total length)
        ranges = np.zeros(dtype='<u8', shape=(1, 3))
        ranges[0][2] = offsets[2]-offsets[0]
        
    while changesMade:
        #copy the fm info
        fmCurrent = np.copy(fmStarts)
        fmCurrent_view = fmCurrent
        
        #copy the offsets info
        #with nogil:
        for x in xrange(0, offsetsCopy.shape[0]):
            offsetsCopy_view[x] = offsets_view[x]-offsets_view[0]
        
        if iterCount % 2 == 0:
            if numInputs == 2:
                st = time.time()
                ret = targetedIterationMerge2(&seqs_view[offsets_view[0]], &seqs_view[offsets_view[1]],
                                              &inter0_view[0], &inter1_view[0], offsets[numInputs]-offsets[0], 
                                              fmIndex0_view, fmIndex1_view, ranges, binBits, numValidChars, iterCount)
                changesMade = ret[0]
                ranges = ret[1]
                el = time.time()-st
                print '\t'.join([str(val) for val in (offsets_view[0], iterCount, ranges.shape[0], el, np.sum(ranges[:, 2]))])
            else:
                changesMade = singleIterationMerge256(&seqs_view[offsets_view[0]], offsetsCopy_view, &inter0_view[0], &inter1_view[0], fmCurrent_view, offsets[numInputs]-offsets[0])
        else:
            if numInputs == 2:
                st = time.time()
                ret = targetedIterationMerge2(&seqs_view[offsets_view[0]], &seqs_view[offsets_view[1]],
                                              &inter1_view[0], &inter0_view[0], offsets[numInputs]-offsets[0], 
                                              fmIndex0_view, fmIndex1_view, ranges, binBits, numValidChars, iterCount)
                el = time.time()-st
                changesMade = ret[0]
                ranges = ret[1]
                print '\t'.join([str(val) for val in (offsets_view[0], iterCount, ranges.shape[0], el, np.sum(ranges[:, 2]))])
            else:
                changesMade = singleIterationMerge256(&seqs_view[offsets_view[0]], offsetsCopy_view, &inter1_view[0], &inter0_view[0], fmCurrent_view, offsets[numInputs]-offsets[0])
        
        iterCount += 1
    
    #create access points to our result and a temporary result
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] msbwt = np.load(mergedDir+'/msbwt.npy', 'r+')
    cdef np.uint8_t [:] msbwt_view = msbwt
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] msbwtOut
    cdef np.uint8_t [:] msbwtOut_view
    
    #TODO: currently set to half a gig, maybe make this a user option?
    #decide whether to hold the temp in memory or on disk
    cdef unsigned long inMemThreshold = 5*10**8
    if offsets_view[numInputs] - offsets_view[0] < inMemThreshold:
        msbwtOut = np.zeros(dtype='<u1', shape=(offsets[numInputs]-offsets[0], ))
    else:
        outFN = mergedDir+'/msbwt.temp.'+str(offsets[0])+'.npy'
        msbwtOut = np.lib.format.open_memmap(outFN, 'w+', '<u1', (offsets[numInputs]-offsets[0], ))
    msbwtOut_view = msbwtOut    
    
    cdef unsigned long readID
    
    #copy the offsets info
    #with nogil:
    for x in xrange(0, offsetsCopy.shape[0]):
        offsetsCopy_view[x] = offsets_view[x]

    if numInputs == 2:
        for x in xrange(0, offsets_view[numInputs]-offsets_view[0]):
            #get the read, the symbol, and increment the position in that read
            if getBit_p(inter0_p, x):
                msbwtOut_view[x] = seqs_view[offsetsCopy_view[1]]
                offsetsCopy_view[1] += 1
            else:
                msbwtOut_view[x] = seqs_view[offsetsCopy_view[0]]
                offsetsCopy_view[0] += 1
    else:
        for x in xrange(0, offsets_view[numInputs]-offsets_view[0]):
            #get the read, the symbol, and increment the position in that read
            readID = inter0_view[x]
            msbwtOut_view[x] = seqs_view[offsetsCopy_view[readID]]
            offsetsCopy_view[readID] += 1

    for x in xrange(offsets_view[0], offsets_view[numInputs]):
        msbwt_view[x] = msbwtOut_view[x-offsets[0]]
    
    #remove the temp file if we made it
    if offsets_view[numInputs] - offsets_view[0] >= inMemThreshold:
        os.remove(outFN)
    
    #remove these temp files also
    if interleaveBytes > interThresh:
        os.remove(interleaveFN0)
        os.remove(interleaveFN1)
    
    #return the number of iterations it took us to converge
    return iterCount

cdef void initializeFMIndex(np.uint8_t * seqs_view, np.uint64_t * fmIndex_view, unsigned long bwtLen, 
                      unsigned long binSize, unsigned long numValidChars):
    '''
    This function takes a BWT string and fills in the FM-index for it at given intervals
    @param seqs_view - the pointer to the string we care about here
    @param fmIndex_view - the pointer to the fmIndex we're filling in
    @param bwtLen - the length of the string
    @param binSize - distance between each entry
    @param numValidChars - 6 for now
    '''
    cdef unsigned long x
    cdef unsigned long y = 0
    cdef unsigned long z
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] counts = np.zeros(dtype='<u8', shape=(numValidChars, ))
    for x in xrange(0, bwtLen):
        if x % binSize == 0:
            for z in xrange(0, numValidChars):
                fmIndex_view[y] = counts[z]
                y += 1
        counts[seqs_view[x]] += 1
    
    #go one more past to catch the overflow values
    for z in xrange(0, numValidChars):
        fmIndex_view[y] = counts[z]
        y += 1
    
    counts = np.cumsum(counts)-counts
    y = 0
    for x in xrange(0, bwtLen/binSize+1):
        for z in xrange(0, numValidChars):
            fmIndex_view[y] += counts[z]
            y += 1

cdef void getFmAtIndex(np.uint8_t * seqs_view, np.uint64_t [:, :] fmIndex_view, 
                        unsigned long pos, unsigned long binBits, unsigned long nvc,
                        np.uint64_t [:] ret_view) nogil:
    
    cdef unsigned long startBin = pos >> binBits
    cdef unsigned long x
    for x in range(0, nvc):
        ret_view[x] = fmIndex_view[startBin][x]
    
    for x in range(startBin << binBits, pos):
        ret_view[seqs_view[x]] += 1
    
cdef bint singleIterationMerge256(np.uint8_t * seqs_view, np.uint64_t [:] offsets_view, np.uint8_t * inputInter_view, 
                               np.uint8_t * outputInter_view, np.uint64_t [:] fmCurrent_view, unsigned long bwtLen):
    '''
    Performs a single pass over the data for a merge of at most 256 BWTs
    @param seqs_view - a C pointer to the start of the strings specific to this merge
    @param offsets_view - a C pointer to the array indicating how long each string is
    @param inputInter_view - a C pointer to the input interleave
    @param outputInter_view - a C pointer to where this function writes the output interleave
    @param fmCurrent_view - an FM index this function can modify as it iterates
    @param bwtLen - the sum of all input string lengths
    '''
    cdef unsigned long x
    cdef unsigned long readID
    cdef np.uint8_t symbol
    cdef bint changesMade = False
    cdef unsigned long startIndex = offsets_view[0]
    
    with nogil:
        for x in xrange(0, bwtLen):
            #get the read, the symbol, and increment the position in that read
            readID = inputInter_view[x]
            
            symbol = seqs_view[offsets_view[readID]]
            offsets_view[readID] += 1
            
            #write the readID to the corresponding symbol position and increment
            if outputInter_view[fmCurrent_view[symbol]] != readID:
                changesMade = True
                outputInter_view[fmCurrent_view[symbol]] = readID
            fmCurrent_view[symbol] += 1
    
    return changesMade

cdef bint singleIterationMerge2(np.uint8_t * seqs0_view, np.uint8_t * seqs1_view, np.uint64_t [:] offsets_view, 
                                np.uint8_t * inputInter_view, np.uint8_t * outputInter_view, np.uint64_t [:] fmCurrent_view, 
                                unsigned long bwtLen):
    '''
    Performs a single pass over the data for a merge of at most 2 BWTs, allows for a bit array, instead of byte
    @param seqs0_view - a C pointer to the start of the first string specific to this merge
    @param seqs1_view - a C pointer to the start of the second string specific to this merge
    @param offsets_view - a C pointer to the array indicating how long each string is
    @param inputInter_view - a C pointer to the input interleave
    @param outputInter_view - a C pointer to where this function writes the output interleave
    @param fmCurrent_view - an FM index this function can modify as it iterates
    @param bwtLen - the sum of all input string lengths
    '''
    cdef unsigned long x
    cdef bint readID
    cdef np.uint8_t symbol
    cdef bint changesMade = False
    
    cdef unsigned long startIndex0 = 0
    cdef unsigned long startIndex1 = 0
    
    with nogil:
        for x in xrange(0, bwtLen):
            #get the read, the symbol, and increment the position in that read
            readID = getBit_p(inputInter_view, x)
            if readID:
                symbol = seqs1_view[startIndex1]
                startIndex1 += 1
            else:
                symbol = seqs0_view[startIndex0]
                startIndex0 += 1
            
            #write the readID to the corresponding symbol position and increment
            if getBit_p(outputInter_view, fmCurrent_view[symbol]):
                if not readID:
                    changesMade = True
                    clearBit_p(outputInter_view, fmCurrent_view[symbol])
            else:
                if readID:
                    changesMade = True
                    setBit_p(outputInter_view, fmCurrent_view[symbol])
            
            fmCurrent_view[symbol] += 1
    
    return changesMade

cdef tuple targetedIterationMerge2(np.uint8_t * seqs0_view, np.uint8_t * seqs1_view, 
                                  np.uint8_t * inputInter_view, np.uint8_t * outputInter_view, 
                                  unsigned long bwtLen, np.uint64_t [:, :] fmIndex0_view, np.uint64_t [:, :] fmIndex1_view, 
                                  np.ndarray[np.uint64_t, ndim=2, mode='c'] ranges, unsigned long binBits, unsigned long nvc,
                                  unsigned long iterCount):
    '''
    Performs a single pass over the data for a merge of at most 2 BWTs, allows for a bit array, instead of byte
    @param seqs0_view - a C pointer to the start of the first string specific to this merge
    @param seqs1_view - a C pointer to the start of the second string specific to this merge
    @param offsets_view - a C pointer to the array indicating how long each string is
    @param inputInter_view - a C pointer to the input interleave
    @param outputInter_view - a C pointer to where this function writes the output interleave
    @param bwtLen - the sum of all input string lengths
    '''
    #counters
    cdef unsigned long x, y, z
    
    #values associated with reading the input ranges
    cdef bint readID
    cdef np.uint8_t symbol
    cdef bint changesMade = False
    
    #3-d array indexed by [symbol, entry, entry-value], each entry is (start0, start1, distance)
    cdef np.ndarray[np.uint64_t, ndim=3, mode='c'] nextEntries = np.zeros(dtype='<u8', shape=(nvc, ranges.shape[0], 3))
    cdef np.uint64_t [:, :, :] nextEntries_view = nextEntries
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] neIndex = np.zeros(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] neIndex_view = neIndex
    
    #FM-index values at the start of a range
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmStart0 = np.zeros(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] fmStart0_view = fmStart0
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmStart1 = np.zeros(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] fmStart1_view = fmStart1
    
    #we update this index as we iterate through our range
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmCurrent0 = np.zeros(dtype='<u8', shape=(nvc, ))
    cdef np.uint64_t [:] fmCurrent0_view = fmCurrent0
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] fmCurrent1 = np.zeros(dtype='<u8', shape=(nvc, ))
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
    
    getFmAtIndex(seqs0_view, fmIndex0_view, 0, binBits, nvc, fmCurrent0_view)
    getFmAtIndex(seqs1_view, fmIndex1_view, 0, binBits, nvc, fmCurrent1_view)
    
    #go through each input range
    for y in xrange(0, ranges.shape[0]):
        #check if we need to recalculate the FM-index because we skipped some values
        if startIndex0 == ranges_view[y][0] and startIndex1 == ranges_view[y][1] and y != 0:
            #with nogil:
            #no recalc needed, just update our start to the current
            for x in range(0, nvc):
                fmStart0_view[x] = fmCurrent0_view[x]
                fmStart1_view[x] = fmCurrent1_view[x]
        else:
            #with nogil:
            if (startIndex0 >> binBits) == (ranges_view[y][0] >> binBits):
                for x in xrange(startIndex0, ranges_view[y][0]):
                    fmCurrent0_view[seqs0_view[x]] += 1
                startIndex0 = ranges_view[y][0]
            else:
                startIndex0 = ranges_view[y][0]
                getFmAtIndex(seqs0_view, fmIndex0_view, startIndex0, binBits, nvc, fmCurrent0_view)
                
            if (startIndex1 >> binBits) == (ranges_view[y][1] >> binBits):
                for x in xrange(startIndex1, ranges_view[y][1]):
                    fmCurrent1_view[seqs1_view[x]] += 1
                startIndex1 = ranges_view[y][1]
            else:
                startIndex1 = ranges_view[y][1]
                getFmAtIndex(seqs1_view, fmIndex1_view, startIndex1, binBits, nvc, fmCurrent1_view)
                
            #trying to rewrite this to remove unnecssarily large getFMAtIndex
            for x in range(0, nvc):
                fmStart0_view[x] = fmCurrent0_view[x]
                fmStart1_view[x] = fmCurrent1_view[x]
        
        #regardless, we pull out how far we need to go now
        dist = ranges_view[y][2]
        
        #with nogil:
        #clear our symbol change area
        for x in range(0, nvc):
            symbolChange_view[x] = 0
        
        #go through each bit in this interleave range
        for x in range(startIndex0+startIndex1, startIndex0+startIndex1+dist):
            readID = getBit_p(inputInter_view, x)
            if readID:
                symbol = seqs1_view[startIndex1]
                startIndex1 += 1
                
                #if not getBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol]):
                #if not getAndSetBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol]):
                #    changesMade = True
                #    symbolChange_view[symbol] = 1
                #    setBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol])
                
                symbolChange_view[symbol] |= setValAndRetToggle(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol], 1)
                fmCurrent1_view[symbol] += 1
            else:
                symbol = seqs0_view[startIndex0]
                startIndex0 += 1
                
                #if getBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol]):
                #if getAndClearBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol]):
                    #changesMade = True
                #    symbolChange_view[symbol] = 1
                    #clearBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol])
                
                symbolChange_view[symbol] |= setValAndRetToggle(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol], 0)
                fmCurrent0_view[symbol] += 1
            
            '''
            #get the read, the symbol, and increment the position in that read
            readID = getBit_p(inputInter_view, x)
            if readID:
                symbol = seqs1_view[startIndex1]
                startIndex1 += 1
            else:
                symbol = seqs0_view[startIndex0]
                startIndex0 += 1
            
            #write the readID to the corresponding symbol position and increment
            if getBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol]):
                if not readID:
                    changesMade = True
                    symbolChange_view[symbol] = 1
                    clearBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol])
            else:
                if readID:
                    changesMade = True
                    symbolChange_view[symbol] = 1
                    setBit_p(outputInter_view, fmCurrent0_view[symbol]+fmCurrent1_view[symbol])
            
            #finally, update the corresponding fm-current
            if readID:
                fmCurrent1_view[symbol] += 1
            else:
                fmCurrent0_view[symbol] += 1
            '''
        
        #write the changes to our nextEntries
        #with nogil:
        for x in range(0, nvc):
            if symbolChange_view[x]:
                nextEntries_view[x][neIndex_view[x]][0] = fmStart0_view[x]
                nextEntries_view[x][neIndex_view[x]][1] = fmStart1_view[x]
                nextEntries_view[x][neIndex_view[x]][2] = fmCurrent0_view[x]+fmCurrent1_view[x]-fmStart0_view[x]-fmStart1_view[x]
                neIndex_view[x] += 1
                changesMade = True
    
    #create our final output array
    cdef np.ndarray[np.uint64_t, ndim=2, mode='c'] extendedEntries = np.zeros(dtype='<u8', shape=(nvc*ranges.shape[0], 3))
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
    for z in xrange(0, nvc):
        
        #go through each entry for that symbol
        for x in xrange(0, neIndex_view[z]):
            #mark the previous end
            prevEnd = end
            
            #get the new start/end
            start = nextEntries_view[z][x][0]+nextEntries_view[z][x][1]
            end = start+nextEntries_view[z][x][2]
            
            if prevEnd != start:
                #if we're tackling a new region, clear out these counts
                input0c = input1c = output0c = output1c = 0
                prev0c = prev1c = 0
            else:
                #just update where the prev started, this basically means we're in a big change region
                prev0c = output0c
                prev1c = output1c
            
            #first handle this range
            for y in xrange(start, end):
                if getBit_p(inputInter_view, y):
                    input1c += 1
                else:
                    input0c += 1
                if getBit_p(outputInter_view, y):
                    output1c += 1
                else:
                    output0c += 1
            
            #append it
            extendedEntries_view[exIndex][0] = nextEntries_view[z][x][0]
            extendedEntries_view[exIndex][1] = nextEntries_view[z][x][1]
            extendedEntries_view[exIndex][2] = nextEntries_view[z][x][2]
            exIndex += 1
            
            #now check if there's a group after to add that we missed
            hiddenStart0 = nextEntries_view[z][x][0]+output0c-prev0c
            hiddenStart1 = nextEntries_view[z][x][1]+output1c-prev1c
            
            if x+1 < neIndex_view[z]:
                #the next one is in this range
                nextStart = nextEntries_view[z][x+1][0]+nextEntries_view[z][x+1][1]
            elif z+1 < nvc and np.sum(neIndex[z+1:]) > 0:
                #the next one starts with a different symbol after this symbol
                for y in xrange(z+1, nvc):
                    if neIndex_view[y] > 0:
                        nextStart = nextEntries_view[y][0][0]+nextEntries_view[y][0][1]
                        break
            else:
                #there is no next entry
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
                extendedEntries_view[exIndex][0] = hiddenStart0
                extendedEntries_view[exIndex][1] = hiddenStart1
                extendedEntries_view[exIndex][2] = end-hiddenStart0-hiddenStart1
                exIndex += 1
    
    #shrink our entries so we don't just blow up in size
    extendedEntries = extendedEntries[0:exIndex]
    extendedEntries_view = extendedEntries
    
    #here's where we'll do the collapse
    cdef unsigned long shrinkIndex = 0, currIndex = 1
    cdef bint collapseEntries = (iterCount <= 20)
    #cdef bint collapseEntries = (exIndex >= 1000)
    if collapseEntries:
        for currIndex in xrange(1, exIndex):
            if (extendedEntries_view[shrinkIndex][0]+extendedEntries_view[shrinkIndex][1]+extendedEntries_view[shrinkIndex][2] ==
                extendedEntries_view[currIndex][0]+extendedEntries_view[currIndex][1]):
                #these are adjacent, extend the range 
                extendedEntries_view[shrinkIndex][2] += extendedEntries_view[currIndex][2]
            else:
                #not adjacent, start off the next range
                shrinkIndex += 1
                extendedEntries_view[shrinkIndex][0] = extendedEntries_view[currIndex][0]
                extendedEntries_view[shrinkIndex][1] = extendedEntries_view[currIndex][1]
                extendedEntries_view[shrinkIndex][2] = extendedEntries_view[currIndex][2]
        
        #collapse down to one past the shrink index
        extendedEntries = extendedEntries[0:shrinkIndex+1]
        extendedEntries_view = extendedEntries
       
    #go through each of the next entries and correct our input
    cdef unsigned long totalStart
    
    #go through each symbol range we just changed and copy it back into our input
    for z in xrange(0, nvc):
        for y in xrange(0, neIndex_view[z]):
            #get the start and distance
            totalStart = nextEntries_view[z][y][0]+nextEntries_view[z][y][1]
            dist = nextEntries_view[z][y][2]
            
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
            for x in xrange(((totalStart+dist)/8)*8, totalStart+dist):
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
    
cdef inline bint getAndSetBit_p(np.uint8_t * bitArray, unsigned long index) nogil:
    cdef unsigned long arrIndex = index >> 3
    cdef np.uint8_t mask = (0x1 << (index & 0x7))
    cdef bint ret = bitArray[arrIndex] & mask
    bitArray[arrIndex] |= mask
    return ret

cdef inline bint getAndClearBit_p(np.uint8_t * bitArray, unsigned long index) nogil:
    cdef unsigned long arrIndex = index >> 3
    cdef np.uint8_t mask = (0x1 << (index & 0x7))
    cdef bint ret = bitArray[arrIndex] & mask
    bitArray[arrIndex] &= ~mask
    return ret

cdef inline bint setValAndRetToggle(np.uint8_t * bitArray, unsigned long index, bint val) nogil:
    cdef unsigned long arrIndex = index >> 3
    cdef np.uint8_t bitIndex = index & 0x7
    cdef np.uint8_t byteValue = bitArray[arrIndex]
    cdef bint changed = ((byteValue >> bitIndex) ^ val) & 0x1
    if changed:
        bitArray[arrIndex] = byteValue ^ (0x1 << bitIndex)
    #bitArray[arrIndex] = byteValue ^ (changed << bitIndex)
    return changed
    
