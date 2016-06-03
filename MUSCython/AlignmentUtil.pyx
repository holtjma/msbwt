#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np
cimport numpy as np

def fullAlign(bytes original, bytes modified):
    '''
    This version checks for matches, mismatches, and indels with a GAP_OPEN cost
    '''
    #default scoring for Bowtie2 for end-to-end alignment
    cdef unsigned long MATCH = 0
    cdef unsigned long MISMATCH = 6
    cdef unsigned long GAP_OPEN = 5
    cdef unsigned long GAP_EXTEND = 3
    
    #get the string sizes
    cdef unsigned long oLen = len(original)
    cdef unsigned long mLen = len(modified)
    
    '''
    initialize the scores
    [x, y, 0] corresponds to match/mismatch original
    [x, y, 1] corresponds to deletion from original
    [x, y, 2] corresponds to insertion to original
    '''
    cdef np.ndarray[np.uint32_t, ndim=3, mode='c'] scoreArray = np.empty(dtype='<u4', shape=(oLen+1, mLen+1, 3))
    cdef np.uint32_t [:, :, :] scoreArray_view = scoreArray
    scoreArray_view[:, :, :] = 0x7FFFFFFF
    scoreArray_view[0, 0, 0] = 0
    scoreArray_view[0, 0, 1] = 0
    scoreArray_view[0, 0, 2] = 0
    
    #initialize the jumpers: 0 coming from M, 1 coming from X, 2 coming from Y
    cdef np.ndarray[np.uint8_t, ndim=3, mode='c'] previousPos = np.zeros(dtype='<u1', shape=(oLen+1, mLen+1, 3))
    cdef np.uint8_t [:, :, :] previousPos_view = previousPos
    
    cdef unsigned long x, y, z
    for x in range(1, oLen+1):
        scoreArray_view[x, 0, 1] = GAP_OPEN+x*GAP_EXTEND
        previousPos_view[x, 0, 1] = 1
    for x in range(1, mLen+1):
        scoreArray_view[0, x, 2] = GAP_OPEN+x*GAP_EXTEND
        previousPos_view[0, x, 2] = 2
    
    cdef char * original_view = original
    cdef char * modified_view = modified
    cdef unsigned long diagScore, jumpScore
    
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] scores = np.zeros(dtype='<u4', shape=(3, ))
    cdef np.uint32_t [:] scores_view = scores
    cdef unsigned long choice, nc
    
    for x in range(1, oLen+1):
        for y in range(1, mLen+1):
            #the "M" matrix
            if original_view[x-1] == modified_view[y-1]:
                diagScore = MATCH
            else:
                diagScore = MISMATCH
            #scores_view[:] = scoreArray_view[x-1, y-1, :]
            scores_view[0] = scoreArray_view[x-1, y-1, 0]
            scores_view[1] = scoreArray_view[x-1, y-1, 1]
            scores_view[2] = scoreArray_view[x-1, y-1, 2]
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            
            scoreArray_view[x, y, 0] = diagScore+scores_view[choice]
            previousPos_view[x, y, 0] = choice
            
            #the "X" matrix
            scores_view[0] = scoreArray_view[x-1, y, 0]+GAP_OPEN+GAP_EXTEND
            scores_view[1] = scoreArray_view[x-1, y, 1]+GAP_EXTEND
            scores_view[2] = scoreArray_view[x-1, y, 2]+GAP_OPEN+GAP_EXTEND
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            scoreArray_view[x, y, 1] = scores_view[choice]
            previousPos_view[x, y, 1] = choice
            
            #the "Y" matrix
            scores_view[0] = scoreArray_view[x, y-1, 0]+GAP_OPEN+GAP_EXTEND
            scores_view[1] = scoreArray_view[x, y-1, 1]+GAP_OPEN+GAP_EXTEND
            scores_view[2] = scoreArray_view[x, y-1, 2]+GAP_EXTEND
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            scoreArray_view[x, y, 2] = scores_view[choice]
            previousPos_view[x, y, 2] = choice
    
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
    
    if scoreArray_view[oLen, mLen, 0] < scoreArray_view[oLen, mLen, 1]:
        choice = 0
    else:
        choice = 1
    if scoreArray_view[oLen, mLen, 2] < scoreArray_view[oLen, mLen, choice]:
        choice = 2
    
    x = oLen
    y = mLen
    while x != 0 or y != 0:
        nc = previousPos_view[x, y, choice]
        if choice == 0:
            x -= 1
            y -= 1
            
            if original_view[x] == modified_view[y]:
                currType = MATCH_T
            else:
                currType = MISMATCH_T
            
        elif choice == 1:
            x -= 1
            currType = DELETION_T
        else:
            y -= 1
            currType = INSERTION_T
        
        if currType == instructionType:
            instructionCount += 1
        else:
            if instructionCount > 0:
                cig.append((instructionCount, typeToCig[instructionType]))
            instructionType = currType
            instructionCount = 1
        choice = nc
    
    if instructionCount > 0:
        cig.append((instructionCount, typeToCig[instructionType]))
    cig.reverse()
    return cig

cpdef unsigned long fullED_score(bytes original, bytes modified):
    '''
    This function calculates the edit distance from an original sequence to a modified sequence and 
    returns just that numerical value
    @param original - the original string
    @param modifed - the modified string
    @return - an integer equal to the the edit distance between the two strings
    '''
    #since this is edit distance, there are no gap open penalties
    cdef unsigned long MATCH = 0
    cdef unsigned long MISMATCH = 1
    cdef unsigned long GAP_OPEN = 0
    cdef unsigned long GAP_EXTEND = 1
    
    #get the string sizes
    cdef unsigned long oLen = len(original)
    cdef unsigned long mLen = len(modified)
    
    cdef np.ndarray[np.uint32_t, ndim=3, mode='c'] scoreArray = np.empty(dtype='<u4', shape=(oLen+1, mLen+1, 3))
    cdef np.uint32_t [:, :, :] scoreArray_view = scoreArray
    scoreArray_view[:, :, :] = 0x7FFFFFFF
    scoreArray_view[0, 0, 0] = 0
    scoreArray_view[0, 0, 1] = 0
    scoreArray_view[0, 0, 2] = 0
    
    #initialize the jumpers: 0 coming from M, 1 coming from X, 2 coming from Y
    cdef np.ndarray[np.uint8_t, ndim=3, mode='c'] previousPos = np.zeros(dtype='<u1', shape=(oLen+1, mLen+1, 3))
    cdef np.uint8_t [:, :, :] previousPos_view = previousPos
    
    cdef unsigned long x, y, z
    for x in range(1, oLen+1):
        scoreArray_view[x, 0, 1] = GAP_OPEN+x*GAP_EXTEND
        previousPos_view[x, 0, 1] = 1
    for x in range(1, mLen+1):
        scoreArray_view[0, x, 2] = GAP_OPEN+x*GAP_EXTEND
        previousPos_view[0, x, 2] = 2
    
    cdef char * original_view = original
    cdef char * modified_view = modified
    cdef unsigned long diagScore, jumpScore
    
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] scores = np.zeros(dtype='<u4', shape=(3, ))
    cdef np.uint32_t [:] scores_view = scores
    cdef unsigned long choice, nc
    
    for x in range(1, oLen+1):
        for y in range(1, mLen+1):
            #the "M" matrix
            if original_view[x-1] == modified_view[y-1]:
                diagScore = MATCH
            else:
                diagScore = MISMATCH
            #scores_view[:] = scoreArray_view[x-1, y-1, :]
            scores_view[0] = scoreArray_view[x-1, y-1, 0]
            scores_view[1] = scoreArray_view[x-1, y-1, 1]
            scores_view[2] = scoreArray_view[x-1, y-1, 2]
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            
            scoreArray_view[x, y, 0] = diagScore+scores_view[choice]
            previousPos_view[x, y, 0] = choice
            
            #the "X" matrix
            scores_view[0] = scoreArray_view[x-1, y, 0]+GAP_OPEN+GAP_EXTEND
            scores_view[1] = scoreArray_view[x-1, y, 1]+GAP_EXTEND
            scores_view[2] = scoreArray_view[x-1, y, 2]+GAP_OPEN+GAP_EXTEND
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            scoreArray_view[x, y, 1] = scores_view[choice]
            previousPos_view[x, y, 1] = choice
            
            #the "Y" matrix
            scores_view[0] = scoreArray_view[x, y-1, 0]+GAP_OPEN+GAP_EXTEND
            scores_view[1] = scoreArray_view[x, y-1, 1]+GAP_OPEN+GAP_EXTEND
            scores_view[2] = scoreArray_view[x, y-1, 2]+GAP_EXTEND
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            scoreArray_view[x, y, 2] = scores_view[choice]
            previousPos_view[x, y, 2] = choice
    
    if scoreArray_view[oLen, mLen, 0] < scoreArray_view[oLen, mLen, 1]:
        choice = 0
    else:
        choice = 1
    if scoreArray_view[oLen, mLen, 2] < scoreArray_view[oLen, mLen, choice]:
        choice = 2
    
    return scoreArray_view[oLen, mLen, choice]

cdef struct scoreLen:
    unsigned long ed
    unsigned long length

cpdef scoreLen fullED_minimize(bytes original, bytes modified):
    '''
    This function calculates the edit distance from an original sequence to a modified sequence.  It allows
    the modified string to be clipped at the end to minimize the edit distance.  Both the score and the clipped
    length are returned
    @param original - the original string
    @param modifed - the modified string
    @return - two integer values; 'ed' is the edit distance, 'length' is the number of symbols of 'modified'
        that are used to get the edit distance 'ed'
    '''
    #since this is edit distance, there are no gap open penalties
    cdef unsigned long MATCH = 0
    cdef unsigned long MISMATCH = 1
    cdef unsigned long GAP_OPEN = 0
    cdef unsigned long GAP_EXTEND = 1
    
    #get the string sizes
    cdef unsigned long oLen = len(original)
    cdef unsigned long mLen = len(modified)
    
    cdef np.ndarray[np.uint32_t, ndim=3, mode='c'] scoreArray = np.empty(dtype='<u4', shape=(oLen+1, mLen+1, 3))
    cdef np.uint32_t [:, :, :] scoreArray_view = scoreArray
    scoreArray_view[:, :, :] = 0x7FFFFFFF
    scoreArray_view[0, 0, 0] = 0
    scoreArray_view[0, 0, 1] = 0
    scoreArray_view[0, 0, 2] = 0
    
    #initialize the jumpers: 0 coming from M, 1 coming from X, 2 coming from Y
    cdef np.ndarray[np.uint8_t, ndim=3, mode='c'] previousPos = np.zeros(dtype='<u1', shape=(oLen+1, mLen+1, 3))
    cdef np.uint8_t [:, :, :] previousPos_view = previousPos
    
    cdef unsigned long x, y, z
    for x in range(1, oLen+1):
        scoreArray_view[x, 0, 1] = GAP_OPEN+x*GAP_EXTEND
        previousPos_view[x, 0, 1] = 1
    for x in range(1, mLen+1):
        scoreArray_view[0, x, 2] = GAP_OPEN+x*GAP_EXTEND
        previousPos_view[0, x, 2] = 2
    
    cdef char * original_view = original
    cdef char * modified_view = modified
    cdef unsigned long diagScore, jumpScore
    
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] scores = np.zeros(dtype='<u4', shape=(3, ))
    cdef np.uint32_t [:] scores_view = scores
    cdef unsigned long choice, nc
    
    for x in range(1, oLen+1):
        for y in range(1, mLen+1):
            #the "M" matrix
            if original_view[x-1] == modified_view[y-1]:
                diagScore = MATCH
            else:
                diagScore = MISMATCH
            #scores_view[:] = scoreArray_view[x-1, y-1, :]
            scores_view[0] = scoreArray_view[x-1, y-1, 0]
            scores_view[1] = scoreArray_view[x-1, y-1, 1]
            scores_view[2] = scoreArray_view[x-1, y-1, 2]
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            
            scoreArray_view[x, y, 0] = diagScore+scores_view[choice]
            previousPos_view[x, y, 0] = choice
            
            #the "X" matrix
            scores_view[0] = scoreArray_view[x-1, y, 0]+GAP_OPEN+GAP_EXTEND
            scores_view[1] = scoreArray_view[x-1, y, 1]+GAP_EXTEND
            scores_view[2] = scoreArray_view[x-1, y, 2]+GAP_OPEN+GAP_EXTEND
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            scoreArray_view[x, y, 1] = scores_view[choice]
            previousPos_view[x, y, 1] = choice
            
            #the "Y" matrix
            scores_view[0] = scoreArray_view[x, y-1, 0]+GAP_OPEN+GAP_EXTEND
            scores_view[1] = scoreArray_view[x, y-1, 1]+GAP_OPEN+GAP_EXTEND
            scores_view[2] = scoreArray_view[x, y-1, 2]+GAP_EXTEND
            #choice = np.argmin(scores)
            if scores_view[0] < scores_view[1]:
                choice = 0
            else:
                choice = 1
            if scores_view[2] < scores_view[choice]:
                choice = 2
            scoreArray_view[x, y, 2] = scores_view[choice]
            previousPos_view[x, y, 2] = choice
    
    #we want the last occurrence of the minimum, hence the [::-1]
    cdef unsigned long argmin0 = mLen-np.argmin(scoreArray[oLen, :, 0][::-1])
    cdef unsigned long argmin1 = mLen-np.argmin(scoreArray[oLen, :, 1][::-1])
    cdef unsigned long argmin2 = mLen-np.argmin(scoreArray[oLen, :, 2][::-1])
    
    cdef scoreLen ret
    if scoreArray_view[oLen, argmin0, 0] < scoreArray_view[oLen, argmin1, 1]:
        ret.ed = scoreArray_view[oLen, argmin0, 0]
        ret.length = argmin0+1
    else:
        ret.ed = scoreArray_view[oLen, argmin1, 1]
        ret.length = argmin1+1
    if scoreArray_view[oLen, argmin2, 2] < ret.ed:
        ret.ed = scoreArray_view[oLen, argmin2, 2]
        ret.length = argmin2+1
    
    return ret

def fullAlign_noGO(bytes original, bytes modified):
    '''
    DEPRECATED
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
    
    #initialize the jumpers
    cdef np.ndarray[np.uint32_t, ndim=3, mode='c'] previousPos = np.zeros(dtype='<u4', shape=(oLen+1, mLen+1, 2))
    cdef np.uint32_t [:, :, :] previousPos_view = previousPos
    
    cdef unsigned long x, y, z
    for x in range(1, oLen+1):
        scoreArray_view[x, 0] = x*GAP_EXTEND
        previousPos_view[x, 0, 0] = x-1
    for x in range(1, mLen+1):
        scoreArray_view[0, x] = x*GAP_EXTEND
        previousPos_view[0, x, 1] = x-1
    
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
            if x < oLen:
                jumpScore = scoreArray_view[x, y]+GAP_EXTEND
                if scoreArray_view[x+1, y] > jumpScore:
                    scoreArray_view[x+1, y] = jumpScore
                    previousPos_view[x+1, y, 0] = x
                    previousPos_view[x+1, y, 1] = y
                
            
            #now handle insertions to the original
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
    
    while x != 0 or y != 0:
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
    
    #if oLen != mLen:
    #    raise Exception("Indels not handled right now")
    
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
    
    #just return the final number
    return scoreArray_view[oLen, mLen]