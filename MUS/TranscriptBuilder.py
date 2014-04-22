import os

import numpy as np

import pyximport
pyximport.install(setup_args=dict(include_dirs=np.get_include()))
from MUSCython import MultiStringBWTCython as MultiStringBWT

class PathNode(object):
    '''
    def __init__(self, nodeID, startingKmer, countK, pathK, threshold, overloadThreshold, msbwt, minDistToSeed, 
                 drawDollarTerminals, isMerged):
    '''
    def __init__(self, nodeID, startingKmer, msbwt, minDistToSeed, settingsDict):
        #passed in node specifics
        self.nodeID = nodeID
        self.execOrder = -1
        self.seq = startingKmer
        self.msbwt = msbwt
        self.minDistToSeed = minDistToSeed
        
        #default values only
        self.termCondition = None
        self.pileups = []
        self.validChars = ['$', 'A', 'C', 'G', 'N', 'T']
        self.inversionSet = set([])
        self.readSet = set([])
        self.pairedNodes = {}
        self.sourceCounts = {}
        
        #settings for path following
        self.settingsDict = settingsDict
        self.pathK = settingsDict.get('kmerSize', len(startingKmer))
        self.countK = settingsDict.get('countK', self.pathK)
        self.pathThreshold = settingsDict.get('pathThreshold', 10)
        self.overloadThreshold = settingsDict.get('overloadThreshold', 1000)
        self.drawDollarTerminals = settingsDict.get('drawDollarTerminals', False)
        self.isMerged = settingsDict.get('isMerged', False)
        if self.isMerged:
            self.interleave = np.load(settingsDict['interleaveFN'], 'r')
        else:
            self.interleave = None
    
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
                
                if self.drawDollarTerminals or c != '$':
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
                
                #remove the last kmer, because it's actually in the new node we merge into
                self.seq = self.seq[0:-1]
                
                #this will lead to repeating the same counts later, but that's okay
                newID = len(nodes)
                newHistMers = set([])
                nodes.append(PathNode(newID, kmer, self.msbwt, self.minDistToSeed+len(self.pileups), self.settingsDict))
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
                
                for i in xrange(r1[0], r1[1]):
                    self.readSet.add((int(self.msbwt.getSequenceDollarID(i)), 0))
                for i in xrange(r2[0], r2[1]):
                    self.readSet.add((int(self.msbwt.getSequenceDollarID(i)), 1))
                
                #if kmerCount > self.overloadThreshold:
                if self.pileups[0] > self.overloadThreshold:
                    #this path is too heavy, we probably won't figure out what's going on downstream
                    self.termCondition = 'OVERLOAD'
                    terminate = True
                    
                elif numPaths > 1:
                    self.termCondition = 'SPLIT'
                    for c in self.validChars:
                        if counts[c] >= self.pathThreshold:
                            newKmer = kmer[1:]+c
                            if foundKmers.has_key(newKmer):
                                otherNID = foundKmers[newKmer]
                                nodes[otherNID].minDistToSeed = min(nodes[otherNID].minDistToSeed, self.minDistToSeed+len(self.pileups))
                                edges.append(PathEdge(len(edges), self.nodeID, otherNID, c+': '+str(counts[c])))
                            
                            else:
                                if self.drawDollarTerminals or c != '$':
                                    newID = len(nodes)
                                    newHistMers = set([])
                                    nodes.append(PathNode(newID, newKmer, self.msbwt, self.minDistToSeed+len(self.pileups), self.settingsDict))
                                    edges.append(PathEdge(len(edges), self.nodeID, newID, c+': '+str(counts[c])))
                                    foundKmers[newKmer] = newID
                                    
                                    if c != '$':
                                        unexploredPaths.append(nodes[newID])
                                    else:
                                        nodes[newID].termCondition = '$ Ext'
                                
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
                        if maxC == '$':
                            self.termCondition = '$ Max'
                            terminate = True
                    
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

'''
def interactiveTranscriptConstruction(bwtDir, seedKmer, endSeeds, pathK, countK, pathThreshold, overloadThreshold, 
                                      numNodes, direction, drawDollarTerminals, trackPairs, isMerged, logger):
'''
def interactiveTranscriptConstruction(bwtDir, seedKmer, endSeeds, settingsDict, logger):
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
    
    pathK = settingsDict.get('kmerSize', len(seedKmer))
    countK = settingsDict.get('countK', pathK)
    isMerged = settingsDict.get('isMerged', False)
    trackPairs = settingsDict.get('trackPairs', False)
    
    numNodes = settingsDict['numNodes']
    validChars = ['$', 'A', 'C', 'G', 'N', 'T']
    
    logger.info('Loading '+bwtDir+'...')
    msbwt = MultiStringBWT.loadBWT(bwtDir, logger)
    if os.path.exists(bwtDir+'/origins.npy'):
        raise Exception("You haven\'t reimplemented the handling of origin files")
        origins = np.load(bwtDir+'/origins.npy', 'r')
    else:
        origins = None
    
    '''
    if isMerged:
        interleave = np.load(bwtDir+'/inter0.npy', 'r')
    else:
        interleave = None
    '''
    settingsDict['interleaveFN'] = bwtDir+'/inter0.npy'
    
    kmer = seedKmer
    
    foundKmers = {kmer:0}
    nodes = []
    nodes.append(PathNode(len(nodes), kmer, msbwt, 0, settingsDict))
    edges = []
    
    for i, endSeed in enumerate(endSeeds):
        if len(endSeed) != pathK:
            raise Exception(endSeed+': NOT CORRECT LENGTH')
        else:
            endID = len(nodes)
            nodes.append(PathNode(endID, endSeed, msbwt, 0, settingsDict))
            nodes[endID].termCondition = 'END_SEED_'+str(i)
            foundKmers[endSeed] = endID
    
    logger.info('Beginning with seed \''+seedKmer+'\', pathK='+str(pathK)+', countK='+str(countK))
    
    unexploredPaths = [nodes[0]]
    
    #init the kmer dictionary
    execID = 0
    
    while len(unexploredPaths) > 0:
        #uncomment to make this smallest first
        unexploredPaths.sort(key = lambda node: node.minDistToSeed)
        print 'UP: '+'['+','.join([str((node.minDistToSeed, node.nodeID)) for node in unexploredPaths])+']'
        nextNode = unexploredPaths.pop(0)
        
        if nextNode.nodeID < numNodes:
            nextNode.execOrder = execID
            execID += 1
            
            logger.info('Exploring new node')
            nextNode.firstTimeExtension(foundKmers, unexploredPaths, nodes, edges)
        else:
            nextNode.termCondition = 'UNEXPLORED'
        
    if isMerged:
        interleaveFN = bwtDir+'/inter0.npy'
        interleave = np.load(interleaveFN, 'r')
        
        for node in nodes:
            dIDs = set([])
            for s in xrange(0, len(node.seq)-pathK+1):
                kmer = node.seq[s:s+pathK]
                r = msbwt.findIndicesOfStr(kmer)
                for ind in xrange(r[0], r[1]):
                    dID = msbwt.getSequenceDollarID(ind)
                    dIDs.add(dID)
                
                r2 = msbwt.findIndicesOfStr(MultiStringBWT.reverseComplement(kmer))
                for ind in xrange(r2[0], r2[1]):
                    dID = msbwt.getSequenceDollarID(ind)
                    dIDs.add(dID)
            
            for dID in dIDs:
                sourceID = interleave[dID]
                node.sourceCounts[sourceID] = node.sourceCounts.get(sourceID, 0)+1
        
    if trackPairs:
        abtFN = bwtDir+'/abt.npy'
        abt = np.load(abtFN, 'r')
                
        abtDict = {}
        
        for node in nodes:
            '''
            dIDs = set([])
            for s in xrange(0, len(node.seq)-pathK+1):
                kmer = node.seq[s:s+pathK]
                r = msbwt.findIndicesOfStr(kmer)
                for ind in xrange(r[0], r[1]):
                    dID = msbwt.getSequenceDollarID(ind)
                    dIDs.add((dID, 0))
                
                r2 = msbwt.findIndicesOfStr(MultiStringBWT.reverseComplement(kmer))
                for ind in xrange(r2[0], r2[1]):
                    dID = msbwt.getSequenceDollarID(ind)
                    dIDs.add((dID, 1))
            '''
            dIDs = node.readSet
            
            for dID, direction in dIDs:
                (fID, rID) = abt[dID]
                if fID % 2 == 0:
                    oFID = fID+1
                else:
                    oFID = fID-1
                
                if abtDict.has_key((oFID, rID, 1-direction)):
                    otherNIDs = abtDict[(oFID, rID, 1-direction)][1]
                    for n in otherNIDs:
                        nodes[n].pairedNodes[node.nodeID] = nodes[n].pairedNodes.get(node.nodeID, 0)+1
                        node.pairedNodes[n] = node.pairedNodes.get(n, 0)+1
                
                if not abtDict.has_key((fID, rID, direction)):
                    abtDict[(fID, rID, direction)] = (dID, set([]))
                abtDict[(fID, rID, direction)][1].add(node.nodeID)
    
    retData = []
    retEdges = []
    for node in nodes:
        retData.append((node.nodeID, node.execOrder, node.seq, node.pileups, node.termCondition, node.minDistToSeed,
                        node.inversionSet, node.readSet, node.pairedNodes, node.sourceCounts))
    for edge in edges:
        retEdges.append((edge.fromID, edge.toID, edge.label, edge.edgeStyle))
    
    return (retData, retEdges)
        