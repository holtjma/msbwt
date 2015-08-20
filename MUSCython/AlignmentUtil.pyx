#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np
cimport numpy as np

def fullAlign(bytes original, bytes modified):
    '''
    This version checks for matches, mismatches, and indels with a GAP_OPEN cost; as a result, it's an n^3 algorithm
    due to the way gap open/extend works
    '''
    #default scoring for Bowtie2 for end-to-end alignment
    cdef unsigned long MATCH = 0
    cdef unsigned long MISMATCH = 6
    cdef unsigned long GAP_OPEN = 5
    cdef unsigned long GAP_EXTEND = 3
    
    #get the string sizes
    cdef unsigned long oLen = len(original)
    cdef unsigned long mLen = len(modified)
    
    #initialize the scores
    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] scoreArray = np.empty(dtype='<u4', shape=(oLen+1, mLen+1))
    cdef np.uint32_t [:, :] scoreArray_view = scoreArray
    scoreArray[:] = 0xFFFFFFFF
    scoreArray_view[0, 0] = 0
    
    cdef unsigned long x, y, z
    for x in range(1, oLen+1):
        scoreArray_view[x, 0] = GAP_OPEN+x*GAP_EXTEND
    for x in range(1, mLen+1):
        scoreArray_view[0, x] = GAP_OPEN+x*GAP_EXTEND
    
    #initialize the jumpers
    cdef np.ndarray[np.uint32_t, ndim=3, mode='c'] previousPos = np.zeros(dtype='<u4', shape=(oLen+1, mLen+1, 2))
    cdef np.uint32_t [:, :, :] previousPos_view = previousPos
    
    cdef char * original_view = original
    cdef char * modified_view = modified
    cdef unsigned long diagScore, jumpScore
    
    for x in range(0, oLen+1):
        for y in range(0, mLen+1):
            #make sure there is a diagonal before trying to handle it
            if x < oLen and y < mLen:
                #first, handle the diagonal
                if original_view[x] == modified_view[y]:
                    diagScore = scoreArray_view[x, y]+MATCH
                else:
                    diagScore = scoreArray_view[x, y]+MISMATCH
                if scoreArray_view[x+1, y+1] > diagScore:
                    scoreArray_view[x+1, y+1] = diagScore
                    previousPos_view[x+1, y+1, 0] = x
                    previousPos_view[x+1, y+1, 1] = y
            
            #now handle deletions to the original
            jumpScore = scoreArray_view[x, y]+GAP_OPEN
            for z in xrange(x+1, oLen+1):
                jumpScore += GAP_EXTEND
                if scoreArray_view[z, y] > jumpScore:
                    scoreArray_view[z, y] = jumpScore
                    previousPos_view[z, y, 0] = x
                    previousPos_view[z, y, 1] = y
            
            #now handle insertions to the original
            jumpScore = scoreArray_view[x, y]+GAP_OPEN
            for z in xrange(y+1, mLen+1):
                jumpScore += GAP_EXTEND
                if scoreArray_view[x, z] > jumpScore:
                    scoreArray_view[x, z] = jumpScore
                    previousPos_view[x, z, 0] = x
                    previousPos_view[x, z, 1] = y
    
    cdef unsigned long MATCH_T = 0
    cdef unsigned long MISMATCH_T = 1
    cdef unsigned long INSERTION_T = 2
    cdef unsigned long DELETION_T = 3
    
    cdef list typeToCig = ['=', 'X', 'I', 'D']
    
    cdef unsigned long numMatches = 0
    cdef list cig = []
    cdef unsigned long nextX, nextY
    cdef unsigned long instructionType = MATCH_T
    cdef unsigned long instructionCount = 0
    cdef unsigned long currType
    cdef unsigned long currCount
    
    x = oLen
    y = mLen
    
    while x != 0 and y != 0:
        nextX = previousPos_view[x, y, 0]
        nextY = previousPos_view[x, y, 1]
        
        if nextX == x-1 and nextY == y-1:
            #diagonal
            if scoreArray_view[nextX, nextY] == scoreArray_view[x, y]+MATCH:
                #match
                currType = MATCH_T
                currCount = 1
            else:
                #mismatch
                currType = MISMATCH_T
                currCount = 1
        elif nextY == y:
            #deletion to original
            currType = DELETION_T
            currCount = x-nextX
        else:
            #insertion to the original
            currType = INSERTION_T
            currCount = y-nextY
        
        if currType == instructionType:
            instructionCount += currCount
        else:
            cig.append((instructionCount, typeToCig[instructionType]))
            instructionCount = currCount
            instructionType = currType
            
        x = nextX
        y = nextY
    
    cig.append((instructionCount, typeToCig[instructionType]))
    cig.reverse()
    
    return cig

def fullAlign_noGO(bytes original, bytes modified):
    '''
    This version checks for matches, mismatches, and indels with a constant cost for a single base indel; 
    as a result, it's an n^2 algorithm since there are no GAP_OPEN penalties; additionally, it finds edit distance
    because cost of mismatch or indel is identical
    '''
    #default scoring for Bowtie2 for end-to-end alignment
    cdef unsigned long MATCH = 0
    cdef unsigned long MISMATCH = 1
    cdef unsigned long GAP_EXTEND = 1
    
    #get the string sizes
    cdef unsigned long oLen = len(original)
    cdef unsigned long mLen = len(modified)
    
    #initialize the scores
    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] scoreArray = np.empty(dtype='<u4', shape=(oLen+1, mLen+1))
    cdef np.uint32_t [:, :] scoreArray_view = scoreArray
    scoreArray[:] = 0xFFFFFFFF
    scoreArray_view[0, 0] = 0
    
    cdef unsigned long x, y, z
    for x in range(1, oLen+1):
        scoreArray_view[x, 0] = x*GAP_EXTEND
    for x in range(1, mLen+1):
        scoreArray_view[0, x] = x*GAP_EXTEND
    
    #initialize the jumpers
    cdef np.ndarray[np.uint32_t, ndim=3, mode='c'] previousPos = np.zeros(dtype='<u4', shape=(oLen+1, mLen+1, 2))
    cdef np.uint32_t [:, :, :] previousPos_view = previousPos
    
    cdef char * original_view = original
    cdef char * modified_view = modified
    cdef unsigned long diagScore, jumpScore
    
    for x in range(0, oLen+1):
        for y in range(0, mLen+1):
            #make sure there is a diagonal before trying to handle it
            if x < oLen and y < mLen:
                #first, handle the diagonal
                if original_view[x] == modified_view[y]:
                    diagScore = scoreArray_view[x, y]+MATCH
                else:
                    diagScore = scoreArray_view[x, y]+MISMATCH
                if scoreArray_view[x+1, y+1] > diagScore:
                    scoreArray_view[x+1, y+1] = diagScore
                    previousPos_view[x+1, y+1, 0] = x
                    previousPos_view[x+1, y+1, 1] = y
            
            #now handle deletions to the original
            '''
            jumpScore = scoreArray_view[x, y]+GAP_OPEN
            for z in xrange(x+1, oLen+1):
                jumpScore += GAP_EXTEND
                if scoreArray_view[z, y] > jumpScore:
                    scoreArray_view[z, y] = jumpScore
                    previousPos_view[z, y, 0] = x
                    previousPos_view[z, y, 1] = y
            '''
            if x < oLen:
                jumpScore = scoreArray_view[x, y]+GAP_EXTEND
                if scoreArray_view[x+1, y] > jumpScore:
                    scoreArray_view[x+1, y] = jumpScore
                    previousPos_view[x+1, y, 0] = x
                    previousPos_view[x+1, y, 1] = y
                
            
            #now handle insertions to the original
            '''
            jumpScore = scoreArray_view[x, y]+GAP_OPEN
            for z in xrange(y+1, mLen+1):
                jumpScore += GAP_EXTEND
                if scoreArray_view[x, z] > jumpScore:
                    scoreArray_view[x, z] = jumpScore
                    previousPos_view[x, z, 0] = x
                    previousPos_view[x, z, 1] = y
            '''
            if y < mLen:
                jumpScore = scoreArray_view[x, y]+GAP_EXTEND
                if scoreArray_view[x, y+1] > jumpScore:
                    scoreArray_view[x, y+1] = jumpScore
                    previousPos_view[x, y+1, 0] = x
                    previousPos_view[x, y+1, 1] = y
    
    cdef unsigned long MATCH_T = 0
    cdef unsigned long MISMATCH_T = 1
    cdef unsigned long INSERTION_T = 2
    cdef unsigned long DELETION_T = 3
    
    cdef list typeToCig = ['=', 'X', 'I', 'D']
    
    cdef unsigned long numMatches = 0
    cdef list cig = []
    cdef unsigned long nextX, nextY
    cdef unsigned long instructionType = MATCH_T
    cdef unsigned long instructionCount = 0
    cdef unsigned long currType
    cdef unsigned long currCount
    
    x = oLen
    y = mLen
    
    while x != 0 and y != 0:
        nextX = previousPos_view[x, y, 0]
        nextY = previousPos_view[x, y, 1]
        
        if nextX == x-1 and nextY == y-1:
            #diagonal
            if scoreArray_view[nextX, nextY] == scoreArray_view[x, y]+MATCH:
                #match
                currType = MATCH_T
                currCount = 1
            else:
                #mismatch
                currType = MISMATCH_T
                currCount = 1
        elif nextY == y:
            #deletion to original
            currType = DELETION_T
            currCount = x-nextX
        else:
            #insertion to the original
            currType = INSERTION_T
            currCount = y-nextY
        
        if currType == instructionType:
            instructionCount += currCount
        else:
            cig.append((instructionCount, typeToCig[instructionType]))
            instructionCount = currCount
            instructionType = currType
            
        x = nextX
        y = nextY
    
    cig.append((instructionCount, typeToCig[instructionType]))
    cig.reverse()
    
    return cig

cpdef unsigned long alignChanges(bytes original, bytes modified):
    '''
    This function takes two strings and does a local alignment returning the number of base changes required for the 
    two strings to match up
    TODO: add a buffer parameter to handle indels
    @param original - really just the first string; can be thought of as "reference" string
    @param modified - really just the second string; can be thought of as the "change" to the "reference"
    @return - the number of bases that need to be changed to match them match
    #TODO: change it so it includes indels
    '''
    #scoring when we only care about the number of bases that don't match
    cdef unsigned long MATCH = 0
    cdef unsigned long MISMATCH = 1
    #cdef unsigned long GAP_OPEN = 0
    cdef unsigned long GAP_EXTEND = 1
    
    #get the string sizes
    cdef unsigned long oLen = len(original)
    cdef unsigned long mLen = len(modified)
    
    if oLen != mLen:
        raise Exception("Indels not handled right now")
    
    #initialize the scores
    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] scoreArray = np.empty(dtype='<u4', shape=(oLen+1, mLen+1))
    cdef np.uint32_t [:, :] scoreArray_view = scoreArray
    scoreArray[:] = 0xFFFFFFFF
    scoreArray_view[0, 0] = 0
    
    cdef unsigned long x, y, z
    for x in range(1, oLen+1):
        scoreArray_view[x, 0] = x*GAP_EXTEND
    for x in range(1, mLen+1):
        scoreArray_view[0, x] = x*GAP_EXTEND
    
    cdef char * original_view = original
    cdef char * modified_view = modified
    cdef unsigned long diagScore, jumpScore
    
    for x in range(0, oLen):
        for y in range(0, mLen):
            #first, handle the diagonal
            if original_view[x] == modified_view[y]:
                diagScore = scoreArray_view[x, y]+MATCH
            else:
                diagScore = scoreArray_view[x, y]+MISMATCH
            
            #now see if that's better
            if scoreArray_view[x+1, y+1] > diagScore:
                scoreArray_view[x+1, y+1] = diagScore
            
            #deletions to the original
            if scoreArray_view[x+1, y] > scoreArray_view[x, y]+GAP_EXTEND:
                scoreArray_view[x+1, y] = scoreArray_view[x, y]+GAP_EXTEND
            
            #insertions to the original
            if scoreArray_view[x, y+1] > scoreArray_view[x, y]+GAP_EXTEND:
                scoreArray_view[x, y+1] = scoreArray_view[x, y]+GAP_EXTEND
            
            '''
            #now handle deletions to the original
            jumpScore = scoreArray_view[x, y]+GAP_OPEN
            for z in xrange(x+1, oLen+1):
                jumpScore += GAP_EXTEND
                if scoreArray_view[z, y] > jumpScore:
                    scoreArray_view[z, y] = jumpScore
            
            #now handle insertions to the original
            jumpScore = scoreArray_view[x, y]+GAP_OPEN
            for z in xrange(y+1, mLen+1):
                jumpScore += GAP_EXTEND
                if scoreArray_view[x, z] > jumpScore:
                    scoreArray_view[x, z] = jumpScore
            '''
    
    #just return the final number
    return scoreArray_view[oLen, mLen]