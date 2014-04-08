import os

import numpy as np

import pyximport
pyximport.install(setup_args=dict(include_dirs=np.get_include()))
from MUSCython import MultiStringBWTCython as MultiStringBWT

class PathNode(object):
    def __init__(self, nodeID, startingKmer, countK, pathK, threshold, overloadThreshold, msbwt, minDistToSeed):
        self.nodeID = nodeID
        self.execOrder = -1
        self.seq = startingKmer
        self.countK = countK
        self.pathK = pathK
        self.pathThreshold = threshold
        self.overloadThreshold = overloadThreshold
        self.msbwt = msbwt
        self.validChars = ['$', 'A', 'C', 'G', 'N', 'T']
        self.termCondition = None
        self.pileups = []
        self.minDistToSeed = minDistToSeed
        self.inversionSet = set([])
    
    def firstTimeExtension(self, foundKmers, unexploredPaths, nodes, edges):
        '''
        @param foundKmers - Each kmer we find will be checked against this and added if not present
        @param unexploredPaths - if we find a new path split, we add the things here, also merges are important to add here
        @param nodes - the list of nodes if we find a new one
        '''
        pc = ''
        kmer = self.seq
        terminate = False
        while not terminate:
            if len(kmer) != self.pathK:
                print 'ERROR: DIFFERENT SIZED K-MER '+str(len(kmer))
                raise Exception('ERROR')
        
            #First, perform all the counts of paths going both forwards and backwards
            counts = {}
            revCounts = {}
            
            #maxV - the count of the (k+1)-mer with maxC on it, total is the total counts of valid chars
            maxV = 0
            maxC = ''
            total = 0
            
            #count the number of forward and reversed paths
            numPaths = 0
            numRevPaths = 0
            
            for c in self.validChars:
                counts[c] = self.msbwt.countOccurrencesOfSeq(kmer+c)+self.msbwt.countOccurrencesOfSeq(MultiStringBWT.reverseComplement(kmer+c))
                revCounts[c] = self.msbwt.countOccurrencesOfSeq(c+kmer)+self.msbwt.countOccurrencesOfSeq(MultiStringBWT.reverseComplement(c+kmer))
                
                if c != '$':
                    total += counts[c]
                    if counts[c] > maxV:
                        maxV = counts[c]
                        maxC = c
                    
                    if counts[c] >= self.pathThreshold:
                        numPaths += 1
                        
                    #if we have evidence from the counts OR if the previous character was known to be that character
                    if revCounts[c] >= self.pathThreshold or c == pc:
                        numRevPaths += 1
                
            #check if we have incoming edges, in which case we need to end this block
            if numRevPaths > 1 and kmer != self.seq:
                #this will lead to repeating the same counts later, but that's okay
                newID = len(nodes)
                newHistMers = set([])
                nodes.append(PathNode(newID, kmer, self.countK, self.pathK, self.pathThreshold, self.overloadThreshold, self.msbwt, self.minDistToSeed+len(self.pileups)))
                edges.append(PathEdge(len(edges), self.nodeID, newID, pc+', '+str(revCounts)))
                self.termCondition = 'MERGE_'+str(newID)
                foundKmers[kmer] = newID
                
                unexploredPaths.append(nodes[newID])
                
                print 'Ending block for merge'
                terminate = True
                
            elif total == 0:
                print 'No strings found.'
                self.termCondition = 'TERMINAL'
                terminate = True
            else:
                #the kmer was found in this block and it may have multiple extensions
                foundKmers[kmer] = self.nodeID
                revMer = MultiStringBWT.reverseComplement(kmer)
                if foundKmers.has_key(revMer):
                    otherID = foundKmers[revMer]
                    self.inversionSet.add(otherID)
                    nodes[otherID].inversionSet.add(self.nodeID)
                
                r1 = self.msbwt.findIndicesOfStr(kmer[-self.countK:])
                r2 = self.msbwt.findIndicesOfStr(MultiStringBWT.reverseComplement(kmer[-self.countK:]))
                kmerCount = (r1[1]-r1[0])+(r2[1]-r2[0])
                self.pileups.append(kmerCount)
                perc = float(maxV)/total
                
                #if kmerCount > self.overloadThreshold:
                if self.pileups[0] > self.overloadThreshold:
                    #this path is too heavy, we probably won't figure out what's going on downstream
                    self.termCondition = 'OVERLOAD'
                    terminate = True
                    
                elif numPaths > 1:
                    self.termCondition = 'SPLIT'
                    for c in self.validChars[1:]:
                        if counts[c] >= self.pathThreshold:
                            newKmer = kmer[1:]+c
                            if foundKmers.has_key(newKmer):
                                otherNID = foundKmers[newKmer]
                                nodes[otherNID].minDistToSeed = min(nodes[otherNID].minDistToSeed, self.minDistToSeed+len(self.pileups))
                                edges.append(PathEdge(len(edges), self.nodeID, otherNID, c+': '+str(counts[c])))
                                
                            else:
                                newID = len(nodes)
                                newHistMers = set([])
                                nodes.append(PathNode(newID, newKmer, self.countK, self.pathK, self.pathThreshold, self.overloadThreshold, self.msbwt, self.minDistToSeed+len(self.pileups)))
                                edges.append(PathEdge(len(edges), self.nodeID, newID, c+': '+str(counts[c])))
                                foundKmers[newKmer] = newID
                                
                                unexploredPaths.append(nodes[newID])
                                
                    terminate = True
                else:
                    #this is data pertaining to this k-mer
                    #print ':\t'+kmer+maxC+'\t'+str(perc)+'\t'+str(maxV)+'/'+str(total)+'\t'+str(total-maxV)+'\t'
                    pc = kmer[0]
                    kmer = kmer[1:]+maxC
                    #check if we've found the new k-mer before
                    if foundKmers.has_key(kmer):
                        otherNID = foundKmers[kmer]
                        nodes[otherNID].minDistToSeed = min(nodes[otherNID].minDistToSeed, self.minDistToSeed+len(self.pileups))
                        if counts[maxC] >= self.pathThreshold:
                            edges.append(PathEdge(len(edges), self.nodeID, otherNID, pc+': '+str(counts[maxC])))
                            self.termCondition = 'MERGE_'+str(otherNID)
                        else:
                            edges.append(PathEdge(len(edges), self.nodeID, otherNID, pc+': '+str(counts[maxC]), 'dashed'))
                            self.termCondition = 'MERGE_'+str(otherNID)+', THRESHOLD'
                            
                        terminate = True
                    else:
                        self.seq += maxC
                    
        print 'END EXPLORE RESULTS:'
        print self.seq[-100:]
        print kmer
        print
    
    def followNewHistory(self, newHistMer):
        print 'UNHANDLED '

class PathEdge(object):
    def __init__(self, edgeID, fromNodeID, toNodeID, label, style='solid'):
        self.edgeID = edgeID
        self.fromID = fromNodeID
        self.toID = toNodeID
        self.label = str(label)
        self.edgeStyle = style

def interactiveTranscriptConstruction(bwtDir, seedKmer, endSeeds, pathK, countK, pathThreshold, overloadThreshold, numNodes, direction, logger):
    '''
    This function is intended to be an interactive technique for constructing transcripts, probably to be released
    in a future version of msbwt
    @param bwtFN - the filename of the BWT to load
    @param seedKmer - the seed sequence to use for construction
    @param threshold - minimum number for a path to be considered a path
    @param direction - True is forward, False is backward
    @param logger - the logger
    @param 
    '''
    
    #kmerLen = len(seedKmer)
    validChars = ['$', 'A', 'C', 'G', 'N', 'T']
    
    logger.info('Loading '+bwtDir+'...')
    msbwt = MultiStringBWT.loadBWT(bwtDir, logger)
    if os.path.exists(bwtDir+'/origins.npy'):
        raise Exception("You haven\'t reimplemented the handling of origin files")
        origins = np.load(bwtDir+'/origins.npy', 'r')
    else:
        origins = None
    
    kmer = seedKmer
    
    foundKmers = {kmer:0}
    nodes = []
    nodes.append(PathNode(len(nodes), kmer, countK, pathK, pathThreshold, overloadThreshold, msbwt, 0))
    edges = []
    
    for i, endSeed in enumerate(endSeeds):
        if len(endSeed) != pathK:
            raise Exception(endSeed+': NOT CORRECT LENGTH')
        else:
            endID = len(nodes)
            nodes.append(PathNode(endID, endSeed, countK, pathK, pathThreshold, overloadThreshold, msbwt, 0))
            nodes[endID].termCondition = 'END_SEED_'+str(i)
            foundKmers[endSeed] = endID
    
    logger.info('Beginning with seed \''+seedKmer+'\', pathK='+str(pathK)+', countK='+str(countK))
    
    #unexploredPaths = [(0, NEW_NODE)]
    unexploredPaths = [nodes[0]]
    
    #init the kmer dictionary
    execID = 0
    
    while len(unexploredPaths) > 0:
        #uncomment to make this smallest first
        unexploredPaths.sort(key = lambda node: node.minDistToSeed)
        print 'UP: '+'['+','.join([str((node.minDistToSeed, node.nodeID)) for node in unexploredPaths])+']'
        nextNode = unexploredPaths.pop(0)
        #print path
        
        if nextNode.nodeID < numNodes:
            nextNode.execOrder = execID
            execID += 1
            
            logger.info('Exploring new node')
            nextNode.firstTimeExtension(foundKmers, unexploredPaths, nodes, edges)
        else:
            nextNode.termCondition = 'UNEXPLORED'

    retData = []
    retEdges = []
    for node in nodes:
        retData.append((node.execOrder, node.seq, node.pileups, node.termCondition, node.minDistToSeed, node.inversionSet))
    for edge in edges:
        retEdges.append((edge.fromID, edge.toID, edge.label, edge.edgeStyle))
    
    return (retData, retEdges)
        