#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: profile=False

import numpy as np
cimport numpy as np
import os

from libc.stdio cimport FILE, fopen, fread, fwrite, fclose, stdin

def compressInput(str fn, str bwtDir):
    '''
    This function takes an input file or STDIN stream and converted it to a numpy file containing the Run-Length 
    Encoded (RLE) BWT.
    @param fn - the filename containing the uncompressed BWT string, all symbols must be '$ACGNT\n' or else this
        code will raise an exception; if fn == None, then the code reads from STDIN, allowing for piping
    @param bwtDir - the directory to save our compressed output to; used for loading the BWT later
    @return - None
    '''
    cdef FILE * inputStream
    if fn == None:
        inputStream = stdin
    else:
        inputStream = fopen(fn, 'r')
    
    if not os.path.exists(bwtDir):
        os.makedirs(bwtDir)
    
    cdef str outputFN = bwtDir+'/comp_msbwt.npy'
    cdef FILE * outputStream = fopen(outputFN, 'w+')
    
    cdef unsigned long BUFFER_SIZE = 1024
    cdef bytes strBuffer = <bytes>('\x00'*BUFFER_SIZE)
    cdef unsigned char * buffer = strBuffer
    
    #most of the files I've seen are 80 and '\x46', I'm increasing it just in case
    cdef unsigned long headerSize = 96
    cdef str headerHex = '\x56'
    
    cdef unsigned long x
    
    for x in xrange(0, headerSize-1):
        buffer[x] = 32 #hex value 20 = ' '
    buffer[headerSize-1] = 10 #hex value 0a = '\n'
    fwrite(buffer, 1, headerSize, outputStream)
    
    #set up the translation
    cdef list validSymbols = ['$', 'A', 'C', 'G', 'N', 'T']
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] translator = np.array([255]*256, dtype='<u1')
    cdef np.uint8_t [:] translator_view = translator
    
    x = 0
    cdef str c
    for c in validSymbols:
        translator_view[ord(c)] = x
        x += 1
    
    cdef unsigned long readBytes = fread(buffer, 1, BUFFER_SIZE, inputStream)
    
    cdef unsigned char currSym = buffer[0]
    cdef unsigned long currCount = 0
    cdef unsigned char writeByte
    cdef unsigned long bytesWritten = 0
    
    while readBytes > 0:
        for x in range(0, readBytes):
            if currSym == buffer[x]:
                currCount += 1
            else:
                #if it's the new line symbol, we will ignore it
                if translator_view[currSym] == 255:
                    if currSym == 10:
                        pass
                    else:
                        raise Exception('UNEXPECTED SYMBOL DETECTED: '+currSym)
                else:
                    #we are at the end of the run so handle it
                    #print translator_view[currSym], currCount
                    #writeByte = translator_view[currSym]
                    while currCount > 0:
                        writeByte = translator_view[currSym] | ((currCount & 0x1F) << 3)
                        fwrite(&writeByte, 1, 1, outputStream)
                        currCount = currCount >> 5
                        bytesWritten += 1
                        
                    #the symbol is expected
                    currSym = buffer[x]
                    currCount = 1
        
        readBytes = fread(buffer, 1, BUFFER_SIZE, inputStream)
    
    #handle the last run
    #if it's the new line symbol, we will ignore it
    if translator_view[currSym] == 255:
        if currSym == 10:
            pass
        else:
            raise Exception('UNEXPECTED SYMBOL DETECTED: '+currSym)
    else:
        #we are at the end of the run so handle it
        while currCount > 0:
            writeByte = translator_view[currSym] | ((currCount & 0x1F) << 3)
            fwrite(&writeByte, 1, 1, outputStream)
            currCount = currCount >> 5
            bytesWritten += 1
            
        #the symbol is expected
        currSym = 0
        currCount = 0
    
    #we have finished the compression
    fclose(inputStream)
    fclose(outputStream)
    
    #now that we know the total length, fill in the bytes for our header
    cdef bytes initialWrite = '\x93NUMPY\x01\x00'+headerHex+'\x00{\'descr\': \'|u1\', \'fortran_order\': False, \'shape\': ('+str(bytesWritten)+',), }'
    buffer = initialWrite
    
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] mmapTemp = np.memmap(bwtDir+'/comp_msbwt.npy', '<u1', 'r+')
    cdef np.uint8_t [:] mmapTemp_view = mmapTemp
    for x in range(0, len(initialWrite)):
        mmapTemp_view[x] = buffer[x]
    