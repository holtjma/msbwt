'''
Created on Nov 1, 2013

@author: holtjma
'''

import os
import argparse as ap

validCharacters = set(['$', 'A', 'C', 'G', 'N', 'T'])

def readableFastaFile(fileName):
    if os.path.isfile(fileName) and os.access(fileName, os.R_OK):
        if fileName.endswith('.txt') or fileName.endswith('.gz'):
            return fileName
        else:
            raise ap.ArgumentTypeError("Wrong file format ('.txt' or '.gz' required): '%s'" % fileName)
    else:
        raise ap.ArgumentTypeError("Cannot read file '%s'." % fileName)

def readableNpyFile(fileName):
    if os.path.isfile(fileName) and os.access(fileName, os.R_OK):
        if fileName.endswith('.npy'):
            return fileName
        else:
            raise ap.ArgumentTypeError("Wrong file format ('.npy' required): '%s'" % fileName)
    else:
        raise ap.ArgumentTypeError("Cannot read file '%s'." % fileName)

def writableNpyFile(fileName):
    if os.access(os.path.dirname(fileName), os.W_OK):
        if fileName.endswith('.npy'):
            return fileName
        else:
            raise ap.ArgumentTypeError("Wrong file format ('.npy' required): '%s'." % fileName)
    else:        
        raise ap.ArgumentTypeError("Cannot write file '%s'." % fileName)

def newDirectory(dirName):
    #strip any tail '/'
    if dirName[-1] == '/':
        dirName = dirName[0:-1]
    
    #this will raise it's own exception
    os.makedirs(dirName)
    return dirName

def existingDirectory(dirName):
    #strip any tail '/'
    if dirName[-1] == '/':
        dirName = dirName[0:-1]
    
    if os.path.isdir(dirName):
        return dirName
    else:
        raise ap.ArgumentTypeError("Directory does not exist: '%s'" % dirName)
    
def newOrExistingDirectory(dirName):
    if dirName[-1] == '/':
        dirName = dirName[0:-1]
    
    if os.path.isdir(dirName):
        return dirName
    else:
        os.makedirs(dirName)
        return dirName
    
def validKmer(kmer):
    for c in kmer:
        if not (c in validCharacters):
            raise ap.ArgumentTypeError("Invalid k-mer: All characters must be in ($, A, C, G, N, T)")
    
    return kmer
