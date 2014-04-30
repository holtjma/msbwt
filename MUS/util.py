'''
Created on Nov 1, 2013
@summary: this file mostly contains some auxiliary checks for the command line interface to make sure it's
handed correct file types
@author: holtjma
'''

import os
import argparse as ap

#I see no need for the versions to be different as of now
DESC = "A multi-string BWT package for DNA and RNA."
VERSION = '0.1.7b'
PKG_VERSION = VERSION

validCharacters = set(['$', 'A', 'C', 'G', 'N', 'T'])

def readableFastqFile(fileName): 
    '''
    @param filename - must be both an existing and readable fastq file, supported under '.txt' and '.gz' as of now
    '''
    if os.path.isfile(fileName) and os.access(fileName, os.R_OK):
        if fileName.endswith('.txt') or fileName.endswith('.gz'):
            return fileName
        else:
            raise ap.ArgumentTypeError("Wrong file format ('.txt' or '.gz' required): '%s'" % fileName)
    else:
        raise ap.ArgumentTypeError("Cannot read file '%s'." % fileName)

'''
TODO: REMOVE UNUSED FUNCTION
'''
def readableNpyFile(fileName):
    if os.path.isfile(fileName) and os.access(fileName, os.R_OK):
        if fileName.endswith('.npy'):
            return fileName
        else:
            raise ap.ArgumentTypeError("Wrong file format ('.npy' required): '%s'" % fileName)
    else:
        raise ap.ArgumentTypeError("Cannot read file '%s'." % fileName)

'''
TODO: REMOVE UNUSED FUNCTION
'''
def writableNpyFile(fileName):
    if os.access(os.path.dirname(fileName), os.W_OK):
        if fileName.endswith('.npy'):
            return fileName
        else:
            raise ap.ArgumentTypeError("Wrong file format ('.npy' required): '%s'." % fileName)
    else:        
        raise ap.ArgumentTypeError("Cannot write file '%s'." % fileName)


def newDirectory(dirName):
    '''
    @param dirName - will make a directory with this name, aka, this must be a new directory
    '''
    #strip any tail '/'
    if dirName[-1] == '/':
        dirName = dirName[0:-1]
    
    #this will raise it's own exception
    os.makedirs(dirName)
    return dirName

def existingDirectory(dirName):
    '''
    @param dirName - checks to make sure this directory already exists
    TODO: add checks for the bwt files?
    '''
    #strip any tail '/'
    if dirName[-1] == '/':
        dirName = dirName[0:-1]
    
    if os.path.isdir(dirName):
        return dirName
    else:
        raise ap.ArgumentTypeError("Directory does not exist: '%s'" % dirName)

def newOrExistingDirectory(dirName):
    '''
    @param dirName - the directory could be pre-existing, if not it's created
    '''
    if dirName[-1] == '/':
        dirName = dirName[0:-1]
    
    if os.path.isdir(dirName):
        return dirName
    else:
        os.makedirs(dirName)
        return dirName

def validKmer(kmer):
    '''
    @param kmer - must be contained in the characters used for our sequencing
    '''
    for c in kmer:
        if not (c in validCharacters):
            raise ap.ArgumentTypeError("Invalid k-mer: All characters must be in ($, A, C, G, N, T)")
    return kmer
