#!python
#cython: boundscheck=False
#cython: wraparound=False

'''
Created on Aug 5, 2013

This file mostly contains access utility for BWTs that are already created on disk.

@author: holtjma
'''

import bisect
import gc
import gzip
import heapq
import math
import numpy as np
cimport numpy as np
import os
import pickle
import pysam#@UnresolvedImport
import shutil
import sys
import time

import MSBWTGenCython as MSBWTGen
from ByteBWTCython import ByteBWT
from RLE_BWTCython import RLE_BWT
from LZW_BWTCython import LZW_BWT
cimport BasicBWT

from cython.operator cimport preincrement as inc

#flags for samtools
REVERSE_COMPLEMENTED_FLAG = 1 << 4#0x10
FIRST_SEGMENT_FLAG = 1 << 6#0x40
#SECOND_SEGMENT_FLAG = 1 << 7#0x80
    
def loadBWT(bwtDir, logger=None):
    '''
    Generic load function, this is recommended for anyone wishing to use this code as it will automatically detect compression
    and assign the appropriate class preferring the decompressed version if both exist.
    @return - a ByteBWT, RLE_BWT, or none if neither can be instantiated
    '''
    if os.path.exists(bwtDir+'/msbwt.npy'):
        msbwt = ByteBWT()
        msbwt.loadMsbwt(bwtDir, logger)
        return msbwt
    elif os.path.exists(bwtDir+'/comp_msbwt.npy'):
        msbwt = RLE_BWT()
        msbwt.loadMsbwt(bwtDir, logger)
        return msbwt
    elif os.path.exists(bwtDir+'/comp_msbwt.dat'):
        msbwt = LZW_BWT()
        msbwt.loadMsbwt(bwtDir, logger)
        return msbwt
    else:
        logger.error('Invalid BWT directory.')
        return None
    
def createMSBWTFromSeqs(seqArray, mergedDir, numProcs, areUniform, logger):
    '''
    This function takes a series of sequences and creates the BWT using the technique from Cox and Bauer
    @param seqArray - a list of '$'-terminated sequences to be in the MSBWT
    @param mergedFN - the final destination filename for the BWT
    @param numProcs - the number of processes it's allowed to use
    '''
    #wipe the auxiliary data stored here
    MSBWTGen.clearAuxiliaryData(mergedDir)
    
    #TODO: do we want a special case for N=1? there was one in early code, but we could just assume users aren't dumb
    seqFN = mergedDir+'/seqs.npy'
    offsetFN = mergedDir+'/offsets.npy'
    
    #sort the sequences
    seqCopy = sorted(seqArray)
    if areUniform:
        uniformLength = len(seqArray[0])
    else:
        uniformLength = 0
        #raise Exception('non-uniform is not handled anymore due to speed, consider building uniform and merging')
        
    #join into one massive string
    seqCopy = ''.join(seqCopy)
    
    #convert the sequences into uint8s and then save it
    seqCopy = np.fromstring(seqCopy, dtype='<u1')
    
    MSBWTGen.writeSeqsToFiles(seqCopy, seqFN, offsetFN, uniformLength)
    if areUniform:
        MSBWTGen.createMsbwtFromSeqs(mergedDir, numProcs, logger)
    else:
        MSBWTGen.createFromSeqs(seqFN, offsetFN, mergedDir+'/msbwt.npy', numProcs, areUniform, logger)
        
def createMSBWTFromFastq(fastqFNs, outputDir, numProcs, areUniform, logger):
    '''
    This function takes fasta filenames and creates the BWT using the technique from Cox and Bauer by simply loading
    all string prior to computation
    @param fastqFNs - a list of fastq filenames to extract sequences from
    @param outputDir - the directory for all of the bwt related data
    @param numProcs - the number of processes it's allowed to use
    @areUniform - true if all the sequences passed into the function are of equal length
    '''
    #generate the files we will reference and clear out the in memory array before making the BWT
    logger.info('Saving sorted sequences...')
    seqFN = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    bwtFN = outputDir+'/msbwt.npy'
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessFastqs(fastqFNs, seqFN, offsetFN, abtFN, areUniform, logger)
    #MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
    if areUniform:
        MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    else:
        MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
    
def createMSBWTFromBam(bamFNs, outputDir, numProcs, areUniform, logger):
    '''
    This function takes a fasta filename and creates the BWT using the technique from Cox and Bauer
    @param bamFNs - a list of BAM filenames to extract sequences from, READS MUST BE SORTED BY NAME
    @param outputDir - the directory for all of the bwt related data
    @param numProcs - the number of processes it's allowed to use
    @areUniform - true if all the sequences passed into the function are of equal length
    '''
    #generate the files we will reference and clear out the in memory array before making the BWT
    logger.info('Saving sorted sequences...')
    seqFN = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    bwtFN = outputDir+'/msbwt.npy'
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessBams(bamFNs, seqFN, offsetFN, abtFN, areUniform, logger)
    #MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
    if areUniform:
        MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    else:
        MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
        
def createMSBWTFromFasta(fastaFNs, outputDir, numProcs, areUniform, logger):
    logger.info('Saving sorted sequences...')
    seqFN = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    bwtFN = outputDir+'/msbwt.npy'
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessFastas(fastaFNs, seqFN, offsetFN, abtFN, areUniform, logger)
    if areUniform:
        MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    else:
        MSBWTGen.createFromSeqs(seqFN, offsetFN, bwtFN, numProcs, areUniform, logger)
    
def customiter(numpyArray):
    '''
    dummy iterator, for some reason numpy doesn't like to act like one by default
    '''
    for x in numpyArray:
        yield tuple(x)

def preprocessFastqs(fastqFNs, seqFNPrefix, offsetFN, abtFN, areUniform, logger):
    '''
    This function does the grunt work behind string extraction for fastq files
    @param fastqFNs - a list of .fq filenames for parsing
    @param seqFNPrefix - this is always of the form '<DIR>/seqs.npy'
    @param offsetFN - this is always of the form '<DIR>/offsets.npy'
    @param abtFN - this is always of the form '<DIR>/about.npy'
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    #create a seqArray
    seqArray = []
    
    tempFileId = 0
    seqsPerFile = 1000000
    maxSeqLen = -1
    numSeqs = 0
    
    subSortFNs = []
    
    for fnID, fn in enumerate(fastqFNs):
        #open the file and read in starting form the second, every 4th line
        logger.info('Loading \''+fn+'\'...')
        if fn.endswith('.gz'):
            fp = gzip.open(fn, 'r')
        else:
            fp = open(fn, 'r')
        i = -1
        
        #go through each line
        for line in fp:
            if i % 4 == 0:
                seqArray.append((line.strip('\n')+'$', fnID, i/4))
                if len(seqArray) == seqsPerFile:
                    if not areUniform or maxSeqLen == -1:
                        maxSeqLen = 0
                        for seq, fID, seqID in seqArray:
                            if len(seq) > maxSeqLen:
                                maxSeqLen = len(seq)
                    
                    tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
                    subSortFNs.append(tempFN)
                    
                    tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
                    tempArray[:] = sorted(seqArray)
                    numSeqs += len(seqArray)
                    del tempArray
                    tempFileId += 1
                    seqArray = []
            i += 1
                
        fp.close()
    
    if len(seqArray) > 0:
        if not areUniform or maxSeqLen == -1:
            maxSeqLen = 0
            for seq, fID, seqID in seqArray:
                if len(seq) > maxSeqLen:
                    maxSeqLen = len(seq)
        
        tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
        subSortFNs.append(tempFN)
        
        tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
        tempArray[:] = sorted(seqArray)
        numSeqs += len(seqArray)
        del tempArray
        tempFileId += 1
        seqArray = []
    
    logger.info('Pre-sorting '+str(numSeqs)+' sequences...')
    iters = []
    for fn in subSortFNs:
        iters.append(customiter(np.load(fn, 'r')))
    
    #save it
    tempFN = seqFNPrefix+'.temp.npy'
    fp = open(tempFN, 'w+')
    
    aboutFile = np.lib.format.open_memmap(abtFN, 'w+', '<u1,<u8', (numSeqs,))
    ind = 0
    
    for tup in heapq.merge(*iters):
        (seq, fID, seqID) = tup
        aboutFile[ind] = (fID, seqID)
        fp.write(seq)
        ind += 1
        
    fp.close()
    
    #clean up disk space
    for fn in subSortFNs:
        os.remove(fn)
    
    #convert the sequences into uint8s and then save it
    del seqArray
    seqArray = np.memmap(tempFN)
    
    if areUniform:
        uniformLength = maxSeqLen
    else:
        uniformLength = 0
    
    logger.info('Saving sorted sequences for BWT construction...')
    MSBWTGen.writeSeqsToFiles(seqArray, seqFNPrefix, offsetFN, uniformLength)
    
    #wipe this
    del seqArray
    os.remove(tempFN)

def preprocessBams(bamFNs, seqFNPrefix, offsetFN, abtFN, areUniform, logger):
    '''
    This does the grunt work behind read extraction from a name-sorted BAM file.  If it isn't sorted, this will not work
    as intended.
    @param bamFNs - a list of '.bam' filenames for parsing
    @param seqFNPrefix - this is always of the form '<DIR>/seqs.npy'
    @param offsetFN - this is always of the form '<DIR>/offsets.npy'
    @param abtFN - this is always of the form '<DIR>/about.npy'
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    #create a seqArray
    seqArray = []
    
    #prep to break this into several smaller sorted sequences
    tempFileId = 0
    seqsPerFile = 10000000
    maxSeqLen = -1
    numSeqs = 0
    
    subSortFNs = []
    
    for fnID, fn in enumerate(bamFNs):
        #open the file and read in starting form the second, every 4th line
        logger.info('Loading \''+fn+'\'...')
        
        bamFile = pysam.Samfile(fn, 'rb')
        i = 0
        
        nr = bamFile.next()
        constantSize = len(nr.seq)
        
        #go through each line
        while nr != None:
            #collect all reads that are the same as nr
            aligns = []
            nqname = nr.qname
            while nr != None and nr.qname == nqname:
                aligns.append(nr)
                try:
                    nr = bamFile.next()
                except:
                    nr = None
            
            #reduce this to a simpler set
            reads = [None, None]
            for a in aligns:
                if len(a.seq) != constantSize:
                    print 'DIFF SIZE='+str(len(a.seq))
                
                if a.flag & REVERSE_COMPLEMENTED_FLAG == 0:
                    #not reverse complemented
                    seq = a.seq
                else:
                    seq = reverseComplement(a.seq)
                
                if a.flag & FIRST_SEGMENT_FLAG == 0:
                    #second segment
                    if reads[1] == None:
                        reads[1] = seq
                    elif reads[1] != seq:
                        logger.warning('Two sequences with same flag and different seqs: '+reads[1]+'\n'+str(a))
                else:
                    if reads[0] == None:
                        reads[0] = seq
                    elif reads[0] != seq:
                        logger.warning('Two sequences with same flag and different seqs: '+reads[0]+'\n'+str(a))
            
            for j, r in enumerate(reads):
                if r == None:
                    continue
                
                if r[1-j] == None:
                    seqArray.append((r+'$', fnID, i, 0xFF, 0xFFFFFFFFFFFFFFFF))
                else:
                    seqArray.append((r+'$', fnID, i, fnID, i+1-2*j))
                
                if len(seqArray) == seqsPerFile:
                    if not areUniform or maxSeqLen == -1:
                        maxSeqLen = 0
                        for seq, fID, seqID, pfID, pseqID in seqArray:
                            if len(seq) > maxSeqLen:
                                maxSeqLen = len(seq)
                    
                    tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
                    subSortFNs.append(tempFN)
                    
                    sys.stdout.write('\rWriting file '+str(tempFileId))
                    sys.stdout.flush()
                    tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8,<u1,<u8', (len(seqArray),))
                    tempArray[:] = sorted(seqArray)
                    numSeqs += len(seqArray)
                    del tempArray
                    tempFileId += 1
                    seqArray = []
                i += 1
                
    sys.stdout.write('\n')
    if len(seqArray) > 0:
        if not areUniform or maxSeqLen == -1:
            maxSeqLen = 0
            for seq, fID, seqID, pfID, pseqID in seqArray:
                if len(seq) > maxSeqLen:
                    maxSeqLen = len(seq)
        
        tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
        subSortFNs.append(tempFN)
        
        tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8,<u1,<u8', (len(seqArray),))
        tempArray[:] = sorted(seqArray)
        numSeqs += len(seqArray)
        del tempArray
        tempFileId += 1
        seqArray = []
    
    logger.info('Pre-sorting '+str(numSeqs)+' sequences...')
    iters = []
    for fn in subSortFNs:
        iters.append(customiter(np.load(fn, 'r')))
    
    #save it
    tempFN = seqFNPrefix+'.temp.npy'
    fp = open(tempFN, 'w+')
    
    aboutFile = np.lib.format.open_memmap(abtFN, 'w+', '<u1,<u8,<u1,<u8', (numSeqs,))
    ind = 0
    
    for tup in heapq.merge(*iters):
        (seq, fID, seqID, pfID, pseqID) = tup
        aboutFile[ind] = (fID, seqID, pfID, pseqID)
        fp.write(seq)
        ind += 1
        
    fp.close()
    
    for fn in subSortFNs:
        os.remove(fn)
    
    #convert the sequences into uint8s and then save it
    del seqArray
    seqArray = np.memmap(tempFN)
    
    if areUniform:
        uniformLength = maxSeqLen
    else:
        uniformLength = 0
    
    logger.info('Saving sorted sequences for BWT construction...')
    MSBWTGen.writeSeqsToFiles(seqArray, seqFNPrefix, offsetFN, uniformLength)
    
    #wipe this
    del seqArray
    os.remove(tempFN)

def preprocessFastas(fastaFNs, seqFNPrefix, offsetFN, abtFN, areUniform, logger):
    '''
    This function does the grunt work behind string extraction for fasta files
    @param fastaFNs - a list of .fq filenames for parsing
    @param seqFNPrefix - this is always of the form '<DIR>/seqs.npy'
    @param offsetFN - this is always of the form '<DIR>/offsets.npy'
    @param abtFN - this is always of the form '<DIR>/about.npy'
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    #create a seqArray
    seqArray = []
    
    #TODO: make the seqPerFile work better for when they aren't uniform
    tempFileId = 0
    seqsPerFile = 1000000
    maxSeqLen = -1
    numSeqs = 0
    
    subSortFNs = []
    
    for fnID, fn in enumerate(fastaFNs):
        #open the file and read in starting form the second, every 4th line
        logger.info('Loading \''+fn+'\'...')
        fp = open(fn, 'r')
        i = 0
        
        currRead = ''
        
        #go through each line
        for line in fp:
            if line[0] == '>':
                if currRead == '':
                    continue
                
                #end of a read
                seqArray.append((currRead+'$', fnID, i))
                if len(seqArray) == seqsPerFile:
                    if not areUniform or maxSeqLen == -1:
                        maxSeqLen = 0
                        for seq, fID, seqID in seqArray:
                            if len(seq) > maxSeqLen:
                                maxSeqLen = len(seq)
                    
                    tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
                    subSortFNs.append(tempFN)
                    
                    tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
                    tempArray[:] = sorted(seqArray)
                    numSeqs += len(seqArray)
                    del tempArray
                    tempFileId += 1
                    seqArray = []
                
                #reset to empty read now
                currRead = ''
                i += 1
                
            else:
                currRead += line.strip('\n')
            
        fp.close()
    
    if len(seqArray) > 0 or currRead != '':
        seqArray.append((currRead+'$', fnID, i))
        
        if not areUniform or maxSeqLen == -1:
            maxSeqLen = 0
            for seq, fID, seqID in seqArray:
                if len(seq) > maxSeqLen:
                    maxSeqLen = len(seq)
        
        tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
        subSortFNs.append(tempFN)
        
        tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
        tempArray[:] = sorted(seqArray)
        numSeqs += len(seqArray)
        del tempArray
        tempFileId += 1
        seqArray = []
    
    logger.info('Pre-sorting '+str(numSeqs)+' sequences...')
    iters = []
    for fn in subSortFNs:
        iters.append(customiter(np.load(fn, 'r')))
    
    #save it
    tempFN = seqFNPrefix+'.temp.npy'
    fp = open(tempFN, 'w+')
    
    aboutFile = np.lib.format.open_memmap(abtFN, 'w+', '<u1,<u8', (numSeqs,))
    ind = 0
    
    for tup in heapq.merge(*iters):
        (seq, fID, seqID) = tup
        aboutFile[ind] = (fID, seqID)
        fp.write(seq)
        ind += 1
        
    fp.close()
    
    #clean up disk space
    for fn in subSortFNs:
        os.remove(fn)
    
    #convert the sequences into uint8s and then save it
    del seqArray
    seqArray = np.memmap(tempFN)
    
    if areUniform:
        uniformLength = maxSeqLen
    else:
        uniformLength = 0
    
    logger.info('Saving sorted sequences for BWT construction...')
    MSBWTGen.writeSeqsToFiles(seqArray, seqFNPrefix, offsetFN, uniformLength)
    
    #wipe this
    del seqArray
    os.remove(tempFN)

def mergeNewSeqs(seqArray, mergedDir, numProcs, areUniform, logger):
    '''
    This function takes a series of sequences and creates a big BWT by merging the smaller ones 
    Mostly a test function, no real purpose to the tool as of now
    @param seqArray - a list of '$'-terminated strings to be placed into the array
    @param mergedFN - the final destination filename for the merged BWT
    @param numProcs - the number of processors the merge is allowed to create
    '''
    #first wipe away any traces of old information for the case of overwriting a BWT at mergedFN
    MSBWTGen.clearAuxiliaryData(mergedDir)
    
    #create two smaller ones
    midPoint = len(seqArray)/3
    mergedDir1 = mergedDir+'0'
    mergedDir2 = mergedDir+'1'
    mergedDir3 = mergedDir+'2'
    
    try:
        shutil.rmtree(mergedDir1)
    except:
        pass
    try:
        shutil.rmtree(mergedDir2)
    except:
        pass
    try:
        shutil.rmtree(mergedDir3)
    except:
        pass
    os.makedirs(mergedDir1)
    os.makedirs(mergedDir2)
    os.makedirs(mergedDir3)
    
    createMSBWTFromSeqs(seqArray[0:midPoint], mergedDir1, numProcs, areUniform, logger)
    createMSBWTFromSeqs(seqArray[midPoint:2*midPoint], mergedDir2, numProcs, areUniform, logger)
    createMSBWTFromSeqs(seqArray[2*midPoint:], mergedDir3, numProcs, areUniform, logger)
    
    #now do the actual merging
    MSBWTGen.mergeNewMSBWT(mergedDir, [mergedDir1, mergedDir2, mergedDir3], numProcs, logger)

def compareKmerProfiles(profileFN1, profileFN2):
    '''
    This function takes two kmer profiles and compare them for similarity.
    @param profileFN1 - the first kmer-profile to compare to
    @param profileFN2 - the second kmer-profile to compare to
    @return - a tuple of the form (1-norm, 2-norm, sum of differences, normalized Dot product)
    '''
    fp1 = open(profileFN1, 'r')
    fp2 = open(profileFN2, 'r')
    
    oneNorm = 0
    twoNorm = 0
    sumDeltas = 0
    dotProduct = 0
    
    tot1 = float(fp1.readline().strip('\n').split(',')[1])
    tot2 = float(fp2.readline().strip('\n').split(',')[1])
    
    (seq1, count1) = parseProfileLine(fp1)
    (seq2, count2) = parseProfileLine(fp2)
    
    while seq1 != None or seq2 != None:
        if seq1 == seq2:
            delta = abs(count1/tot1-count2/tot2)
            dotProduct += (count1/tot1)*(count2/tot2)
            (seq1, count1) = parseProfileLine(fp1)
            (seq2, count2) = parseProfileLine(fp2)
        elif seq2 == None or (seq1 != None and seq1 < seq2):
            delta = count1/tot1
            (seq1, count1) = parseProfileLine(fp1)
        else:
            delta = count2/tot2
            (seq2, count2) = parseProfileLine(fp2)
        
        if delta > oneNorm:
            oneNorm = delta
        
        twoNorm += delta*delta
        sumDeltas += delta
    
    fp1.close()
    fp2.close()
    
    twoNorm = math.sqrt(twoNorm)
    #print '1-norm:\t\t'+str(oneNorm)
    #print '2-norm:\t\t'+str(twoNorm)
    #print 'Delta sum:\t'+str(sumDeltas)
    return (oneNorm, twoNorm, sumDeltas, dotProduct)
    
def parseProfileLine(fp):
    '''
    Helper function for profile parsing
    @param fp - the file pointer to get the next line from
    @return - (kmer, kmerCount) as (string, int)
    '''
    nextLine = fp.readline()
    if nextLine == None or nextLine == '':
        return (None, None)
    else:
        pieces = nextLine.strip('\n').split(',')
        return (pieces[0], int(pieces[1]))
    
def reverseComplement(seq):
    '''
    Helper function for generating reverse-complements
    '''
    revComp = ''
    complement = {'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N', '$':'$', '?':'?', '*':'*'}
    for c in reversed(seq):
        revComp += complement[c]
    return revComp
    