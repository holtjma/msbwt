#!python
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
import os

from libc.stdio cimport FILE, fopen, fwrite, fclose

cimport BasicBWT
import MultiStringBWTCython as MultiStringBWT

def lcpGenerator(bytes bwtDir, np.uint8_t maxLCP, logger):
    cdef BasicBWT.BasicBWT msbwt = MultiStringBWT.loadBWT(bwtDir)
    
    cdef unsigned long ts = msbwt.getTotalSize()
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] lcps = np.zeros(dtype='<u1', shape=(ts, ))
    lcps[:] = maxLCP
    cdef np.uint8_t [:] lcps_view = lcps
    
    #currList contains the known end of ranges
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] currList = msbwt.getFullFMAtIndex(ts)
    cdef np.uint64_t [:] currList_view = currList
    
    cdef unsigned long vcLen = 6
    cdef unsigned long lcpValue
    cdef unsigned long i, z
    cdef unsigned long x
    cdef unsigned long currListLen = vcLen
    cdef unsigned long nextListLen = 0
    cdef unsigned long ind
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] currFMIndex = np.zeros(dtype='<u8', shape=(vcLen, ))
    cdef np.uint64_t [:] currFMIndex_view = currFMIndex
    
    cdef str currListFN = ''
    cdef str nextListFN = bwtDir+'/lcpList.0.dat'
    
    cdef FILE * nextListFP = fopen(nextListFN, 'w+')
    cdef np.uint8_t writeByte
    cdef np.uint64_t writeValue
    
    #write all of the initial values to our initial file
    for i in range(0, currList.shape[0]):
        writeValue = currList_view[i]
        for x in range(0, 8):
            writeByte = writeValue & 0xFF
            fwrite(&writeByte, 1, 1, nextListFP)
            writeValue = writeValue >> 8
    fclose(nextListFP)
    
    for lcpValue in range(0, maxLCP):
        logger.info('Generating LCPs of '+str(lcpValue)+'...')
        
        #get all the file names in order
        if currListFN != '':
            try:
                os.remove(currListFN)
            except:
                logger.warning('Failed to remove '+currListFN+' from the file system.')
        currListFN = nextListFN
        nextListFN = bwtDir+'/lcpList.'+str(lcpValue+1)+'.dat'
        
        #memmap it in
        currList = np.memmap(currListFN, '<u8', 'r+')
        currList_view = currList
        currListLen = currList.shape[0]
        
        #condition to break out early
        if currListLen == 0:
            break
        
        #prep the next list
        nextListFP = fopen(nextListFN, 'w+')
        nextListLen = 0
        
        #for ind in currList:
        for i in range(0, currListLen):
            ind = currList_view[i]
            if ind > 0 and lcps_view[ind-1] > lcpValue:
                #first, set the LCP
                lcps_view[ind-1] = lcpValue
                
                #now get it's values cause they'll be set next
                msbwt.fillFmAtIndex(currFMIndex_view, ind)
                for z in range(0, vcLen):
                    writeValue = currFMIndex_view[z]
                    nextListLen += 1
                    for x in range(0, 8):
                        writeByte = writeValue & 0xFF
                        fwrite(&writeByte, 1, 1, nextListFP)
                        writeValue = writeValue >> 8
        
        #close up and loop back around
        fclose(nextListFP)
        
        if nextListLen == 0:
            break
    
    if currListFN != '':
        try:
            os.remove(currListFN)
        except:
            logger.warning('Failed to remove '+currListFN+' from the file system.')
    
    return lcps