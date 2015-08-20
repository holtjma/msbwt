#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False

'''
Created on Aug 5, 2013

This file mostly contains access utility for BWTs that are already created on disk.

@author: holtjma
'''

import bisect
import gc
import glob
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
import MSBWTCompGenCython as MSBWTCompGen
from ByteBWTCython import ByteBWT
from RLE_BWTCython import RLE_BWT
from LZW_BWTCython import LZW_BWT
cimport BasicBWT

from cython.operator cimport preincrement as inc

#flags for samtools
REVERSE_COMPLEMENTED_FLAG = 1 << 4#0x10
FIRST_SEGMENT_FLAG = 1 << 6#0x40
#SECOND_SEGMENT_FLAG = 1 << 7#0x80
    
def loadBWT(bwtDir, useMemmap=True, logger=None):
    '''
    Generic load function, this is recommended for anyone wishing to use this code as it will automatically detect compression
    and assign the appropriate class preferring the decompressed version if both exist.
    @return - a ByteBWT, RLE_BWT, or none if neither can be instantiated
    '''
    if os.path.exists(bwtDir+'/msbwt.npy'):
        msbwt = ByteBWT()
        msbwt.loadMsbwt(bwtDir, useMemmap, logger)
        return msbwt
    elif os.path.exists(bwtDir+'/comp_msbwt.npy'):
        msbwt = RLE_BWT()
        msbwt.loadMsbwt(bwtDir, useMemmap, logger)
        return msbwt
    elif os.path.exists(bwtDir+'/comp_msbwt.dat'):
        msbwt = LZW_BWT()
        msbwt.loadMsbwt(bwtDir, useMemmap, logger)
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
    
    seqFN = mergedDir+'/seqs.npy'
    offsetFN = mergedDir+'/offsets.npy'
    
    #sort the sequences
    if areUniform:
        uniformLength = len(seqArray[0])
    else:
        raise Exception('Must be uniform length for this function')
    
    seqCopy = sorted(seqArray)
        
    #join into one massive string
    seqCopy = ''.join(seqCopy)
    
    #convert the sequences into uint8s and then save it
    seqCopy = np.fromstring(seqCopy, dtype='<u1')
    
    MSBWTGen.writeSeqsToFiles(seqCopy, seqFN, offsetFN, uniformLength)
    MSBWTGen.createMsbwtFromSeqs(mergedDir, numProcs, logger)
    cleanupTemporaryFiles(mergedDir)
    
def createMSBWTCompFromSeqs(seqArray, mergedDir, numProcs, areUniform, logger):
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
    if areUniform:
        uniformLength = len(seqArray[0])
    else:
        raise Exception('Must be uniform length for this function')
    
    seqCopy = sorted(seqArray)
        
    #join into one massive string
    seqCopy = ''.join(seqCopy)
    
    #convert the sequences into uint8s and then save it
    seqCopy = np.fromstring(seqCopy, dtype='<u1')
    
    MSBWTGen.writeSeqsToFiles(seqCopy, seqFN, offsetFN, uniformLength)
    MSBWTCompGen.createMsbwtFromSeqs(mergedDir, numProcs, logger)
    cleanupTemporaryFiles(mergedDir)
        
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
    
    if not areUniform:
        raise Exception('Must be uniform length for this function')
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessFastqs(fastqFNs, outputDir, areUniform, logger)
    MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    cleanupTemporaryFiles(outputDir)
    
def createMSBWTCompFromFastq(fastqFNs, outputDir, numProcs, areUniform, logger):
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
    
    if not areUniform:
        raise Exception('Must be uniform length for this function')
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessFastqs(fastqFNs, outputDir, areUniform, logger)
    MSBWTCompGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    cleanupTemporaryFiles(outputDir)
    
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
    
    if not areUniform:
        raise Exception('Must be uniform length for this function')
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessBams(bamFNs, outputDir, areUniform, logger)
    MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    cleanupTemporaryFiles(outputDir)
        
def createMSBWTFromFasta(fastaFNs, outputDir, numProcs, areUniform, logger):
    logger.info('Saving sorted sequences...')
    
    if not areUniform:
        raise Exception('Must be uniform length for this function')
    
    MSBWTGen.clearAuxiliaryData(outputDir)
    preprocessFastas(fastaFNs, outputDir, areUniform, logger)
    MSBWTGen.createMsbwtFromSeqs(outputDir, numProcs, logger)
    cleanupTemporaryFiles(outputDir)
    
def customiter(numpyArray):
    '''
    dummy iterator, for some reason numpy doesn't like to act like one by default
    '''
    for x in numpyArray:
        yield tuple(x)

def preprocessFastqs(list fastqFNs, outputDir, bint areUniform, logger):
    '''
    This function does the grunt work behind string extraction for fastq files
    @param fastqFNs - a list of .fq filenames for parsing
    @param outputDir - the directory we are doing all our building in
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    if not areUniform:
        raise Exception('preprocessFastqs() does not work for non-uniform sequence.')
    
    #pre-defined filenames based on the directory
    seqFNPrefix = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    
    #create a seqArray
    cdef list seqArray = []
    
    cdef unsigned long tempFileId = 0
    cdef unsigned long seqsPerFile = 10000000
    cdef long maxSeqLen = -1
    cdef unsigned long numSeqs = 0
    
    cdef list subSortFNs = []
    cdef unsigned long fnID
    cdef unsigned long i
    
    cdef long uniformSeqLen = -1
    cdef np.ndarray tempArray
    
    for fnID, fn in enumerate(fastqFNs):
        #open the file and read in starting form the second, every 4th line
        logger.info('Loading \''+fn+'\'...')
        if fn.endswith('.gz'):
            fp = gzip.open(fn, 'r')
        else:
            fp = open(fn, 'r')
        i = 0
        
        #go through each line
        for line in fp:
            if (i & 0x3) == 1:
                seqArray.append((line.strip('\n')+'$', fnID, i >> 2))
                if len(seqArray) == seqsPerFile:
                    for seq, fID, seqID in seqArray:
                        if len(seq) != uniformSeqLen:
                            if uniformSeqLen == -1:
                                uniformSeqLen = len(seq)
                                maxSeqLen = len(seq)
                            else:
                                raise ValueError('Strings are not of uniform length')
                    
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
        for seq, fID, seqID in seqArray:
            if len(seq) != uniformSeqLen:
                if uniformSeqLen == -1:
                    uniformSeqLen = len(seq)
                    maxSeqLen = len(seq)
                else:
                    raise ValueError('Strings are not of uniform length')
        
        tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
        subSortFNs.append(tempFN)
        
        tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
        tempArray[:] = sorted(seqArray)
        numSeqs += len(seqArray)
        del tempArray
        tempFileId += 1
        seqArray = []
    
    mergeSubSorts(subSortFNs, numSeqs, areUniform, maxSeqLen, outputDir, logger)

def preprocessBams(bamFNs, outputDir, areUniform, logger):
    '''
    This does the grunt work behind read extraction from a name-sorted BAM file.  If it isn't sorted, this will not work
    as intended.
    @param bamFNs - a list of '.bam' filenames for parsing
    @param outputDir - the directory to do all our building inside of
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    if not areUniform:
        raise Exception('preprocessBams() does not work for non-uniform sequence.')
    
    #pre-defined filenames based on the directory
    seqFNPrefix = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    
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
    
    mergeSubSorts(subSortFNs, numSeqs, areUniform, maxSeqLen, outputDir, logger)

def preprocessFastas(fastaFNs, outputDir, areUniform, logger):
    '''
    This function does the grunt work behind string extraction for fasta files
    @param fastaFNs - a list of .fq filenames for parsing
    @param outputDir - the directory to build our bwt in
    @param areUniform - True if all sequences are of uniform length
    @param logger - logger object for output 
    '''
    if not areUniform:
        raise Exception('preprocessFastas() does not work for non-uniform sequence.')
    
    #pre-defined filenames based on the directory
    seqFNPrefix = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    
    #create a seqArray
    cdef list seqArray = []
    
    #TODO: make the seqPerFile work better for when they aren't uniform
    cdef unsigned long tempFileId = 0
    cdef unsigned long seqsPerFile = 10000000
    cdef long maxSeqLen = -1
    cdef unsigned long numSeqs = 0
    cdef long uniformSeqLen = -1
    
    cdef list subSortFNs = []
    cdef unsigned long fnID
    cdef unsigned long i
    
    cdef np.ndarray tempArray
    
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
                    for seq, fID, seqID in seqArray:
                        if len(seq) != uniformSeqLen:
                            if uniformSeqLen == -1:
                                uniformSeqLen = len(seq)
                                maxSeqLen = len(seq)
                            else:
                                raise ValueError('Strings are not of uniform length')
                            
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
        
        for seq, fID, seqID in seqArray:
            if len(seq) != uniformSeqLen:
                if uniformSeqLen == -1:
                    uniformSeqLen = len(seq)
                    maxSeqLen = len(seq)
                else:
                    raise ValueError('Strings are not of uniform length')
        
        tempFN = seqFNPrefix+'.sortTemp.'+str(tempFileId)+'.npy'
        subSortFNs.append(tempFN)
        
        tempArray = np.lib.format.open_memmap(tempFN, 'w+', 'a'+str(maxSeqLen)+',<u1,<u8', (len(seqArray),))
        tempArray[:] = sorted(seqArray)
        numSeqs += len(seqArray)
        del tempArray
        tempFileId += 1
        seqArray = []
    
    mergeSubSorts(subSortFNs, numSeqs, areUniform, maxSeqLen, outputDir, logger)

def mergeSubSorts(list subSortFNs, unsigned long numSeqs, bint areUniform, maxSeqLen, outputDir, logger):
    '''
    A shared function for use by the preprocess related functions
    @param subSortFNs - the list of partially sorted string filenames
    @param numSeqs - the total number of strings to sort
    @param areUniform - indicated whether the strings are of uniform length, this should have been checked earlier though
    @param maxSeqLen - really the uniform length of the strings
    @param outputDir - the output directory
    @param logger - our standard logging object
    '''
    #pre-defined filenames based on the directory
    seqFNPrefix = outputDir+'/seqs.npy'
    offsetFN = outputDir+'/offsets.npy'
    abtFN = outputDir+'/about.npy'
    
    logger.info('Pre-sorting '+str(numSeqs)+' sequences...')
    cdef list iters = []
    for fn in subSortFNs:
        iters.append(customiter(np.load(fn, 'r')))
    
    #save it
    tempFN = seqFNPrefix+'.temp.npy'
    fp = open(tempFN, 'w+')
    
    cdef np.ndarray aboutFile = np.lib.format.open_memmap(abtFN, 'w+', '<u1,<u8', (numSeqs,))
    cdef unsigned long ind = 0
    
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
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] seqArrayMmap = np.memmap(tempFN)
    
    if areUniform:
        uniformLength = maxSeqLen
    else:
        uniformLength = 0
    
    logger.info('Saving sorted sequences for BWT construction...')
    MSBWTGen.writeSeqsToFiles(seqArrayMmap, seqFNPrefix, offsetFN, uniformLength)
    
    #wipe this
    os.remove(tempFN)
    
def cleanupTemporaryFiles(outputDir):
    '''
    This function is for cleaning up all the excess seq.npy files after construction and whatever else needs removal
    '''
    #remove seq files
    for fn in glob.glob(outputDir+'/seqs.npy*'):
        os.remove(fn)
    
    #remove the offset file too
    os.remove(outputDir+'/offsets.npy')

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
    complement = {'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N', '$':'$', '?':'?', '*':'*'}
    revComp = ''.join([complement[c] for c in reversed(seq)])
    #for c in reversed(seq):
    #    revComp += complement[c]
    return revComp
    