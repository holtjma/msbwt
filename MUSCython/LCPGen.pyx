#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np
cimport numpy as np
import os

from libc.stdio cimport FILE, fopen, fwrite, fclose

cimport BasicBWT
import MultiStringBWTCython as MultiStringBWT

def lcpGenerator(bytes bwtDir, np.uint8_t maxLCP, logger):
    '''
    This is basically a breadth-first, branch-and-bound, LCP generator.  Unfortunately, if any particular
    branch is too "wide", it's disk usage becomes very, very high.  Recommended use of this generator is only
    for small BWTs where "small" is defined by the "width" of the branching.
    @param bwtDir - the directory of the BWT
    @param maxLCP - the maximum LCP you want to store, typically this is read length + 1 (the +1 is the '$')
    @param logger - for output
    @return - a numpy uint8 array containing the LCPs
    '''
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

def linearLcpGenerator(bytes bwtDir, np.uint8_t maxLCP, logger):
    '''
    This is basically a scan-based, LCP generator. Each pass is a linear scan of the BWT and LCP array so far.  Since
    this is a linear scan, there is far less memory consumption (only the memory to actual hold the LCP array) and 
    not really any other data structure.  Recommended for any large BWT due to minimal memory overhead.  It also works
    well for small BWTs, but may be slightly slower compared to the regular "lcpGenerator(...)" function.
    @param bwtDir - the directory of the BWT
    @param maxLCP - the maximum LCP you want to store, typically this is read length + 1 (the +1 is the '$')
    @param logger - for output
    @return - a numpy uint8 array containing the LCPs
    '''
    #load the BWT
    cdef BasicBWT.BasicBWT msbwt = MultiStringBWT.loadBWT(bwtDir)
    
    #create an array and fill it with the max LCP
    cdef unsigned long ts = msbwt.getTotalSize()
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] lcps = np.empty(dtype='<u1', shape=(ts, ))
    lcps[:] = maxLCP
    cdef np.uint8_t [:] lcps_view = lcps
    
    #currList contains the known end of ranges
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] currList = msbwt.getFullFMAtIndex(ts)
    cdef np.uint64_t [:] currList_view = currList
    
    cdef unsigned long vcLen = 6
    cdef unsigned long lcpValue, nextLcpValue
    cdef unsigned long i, z
    cdef unsigned long ind
    
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] currFMIndex = np.zeros(dtype='<u8', shape=(vcLen, ))
    cdef np.uint64_t [:] currFMIndex_view = currFMIndex
    
    #set all the 0 LCP values
    logger.info('Generating LCPs of 0...')
    for i in range(0, currList.shape[0]):
        lcps_view[currList_view[i]-1] = 0
    cdef unsigned long changedLCP = vcLen
    
    for lcpValue in range(0, maxLCP-1):
        logger.info('Found '+str(changedLCP)+' suffixes with LCP='+str(lcpValue))
        if changedLCP == 0:
            logger.info('None found, terminating early.')
            break
        
        changedLCP = 0
        logger.info('Generating LCPs of '+str(lcpValue+1)+'...')
        nextLcpValue = lcpValue+1
        
        #scan through the whole BWT identifying anything with LCP=lcpValue
        for ind in range(0, ts):
            if lcps_view[ind] == lcpValue:
                msbwt.fillFmAtIndex(currFMIndex_view, ind+1)
                for z in range(0, vcLen):
                    if currFMIndex_view[z] > 0 and lcps_view[currFMIndex_view[z]-1] > nextLcpValue:
                        lcps_view[currFMIndex_view[z]-1] = nextLcpValue
                        changedLCP += 1
        
    return lcps