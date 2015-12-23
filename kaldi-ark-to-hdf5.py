
import sys
import os
import re
import h5py
import random
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Generate HDF5 file based on .ark and .pdf.txt.')

parser.add_argument('feat', type=argparse.FileType('r'),
                   help='input feat ark file in text mode')
parser.add_argument('pdf', type=argparse.FileType('r'),
                   help='train.pdf.txt')
parser.add_argument('hdf5_dir', metavar='hdf5_dir', action='store', 
                   help='directory name of the hdf5 files')
parser.add_argument('--input-dim', dest='input_dim', action='store', type=int,
                   default=621,
                   help='dimension of the input feature default 621')
parser.add_argument('--data-key', dest='data', action='store', 
                   default='data',
                   help='the key field of features in HDF5 database')
parser.add_argument('--label-key', dest='label', action='store', 
                   default='label',
                   help='the key field of labels in HDF5 database')
parser.add_argument('--map', dest='mapping', type=argparse.FileType('r'),
                   help='mapping file')
args = parser.parse_args()
pdf_map = {}
mapping = {}
if args.mapping:
    for line in args.mapping:
        tokens = line.rstrip('\n').split()
        mapping[ int(tokens[0]) ] = int(tokens[1]) 

for line in args.pdf:
    line = line.rstrip('\n')
    tokens = line.split()
    utt_id = tokens[0]
    if args.mapping:
        pdf_map[ utt_id ] = [ mapping[ int(f)] for f in tokens[1:] ]
    else:
        pdf_map[ utt_id ] = [ int(f) for f in tokens[1:] ]

args.pdf.close()

# ensure dir
if not os.path.exists(args.hdf5_dir):
    os.makedirs( args.hdf5_dir )


input_dim = args.input_dim
   
utt_id = ''
h5_fn = ''
feats = []

for line in args.feat:
    line = line.rstrip('\n') 
    if not line:
        break;
    tokens = line.split()
    if tokens[1] == "[":
        utt_id = tokens[0]
        if utt_id not in pdf_map:
            sys.stderr.write( "Warning: " + utt_id + " not in the alignments.\n")
        else:
            sys.stderr.write( "Processing: " + utt_id +".\n")
        feats = []
        h5_fn = args.hdf5_dir + "/" + utt_id + ".h5" 
        continue
    
    if utt_id not in pdf_map:
        continue
    if len( tokens ) != input_dim and len( tokens ) != input_dim + 1:
        raise Exception( "Error: incorrect input dimension: " + str(input_dim) )

    feats.append(  [ float(f)  for f in tokens[0:input_dim] ] )
    
    if len( tokens ) == input_dim + 1 and tokens[-1] == "]":
        with h5py.File( h5_fn , 'w' ) as f:
	        data = f.create_dataset( args.data , ( len(feats) ,  input_dim , 1, 1 ) , maxshape=(None,input_dim,1,1) , chunks=True, dtype='float32' )
	        label = f.create_dataset( args.label , ( len(feats) , 1 ) , maxshape=(None, 1) , dtype='float32' , chunks=True)
	        for i in range( len(feats) ):
		        data[i , :, 0 , 0 ] = feats[i]
                if len(pdf_map[utt_id]) != len(feats):
                    raise Exception( "Mismatch frame counter: " + utt_id )
                label[: , 0] = pdf_map[ utt_id]


args.feat.close()
