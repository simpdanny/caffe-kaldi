import numpy as np 
import os
import sys
import argparse
import glob
import time
import caffe
import math
parser = argparse.ArgumentParser()

parser.add_argument("prototxt" , 
                    help="the deploy nnet description file." )
parser.add_argument("caffemodel" ,
                    help="the trained model weights file" )
parser.add_argument("input_blob",
                    help="the name of input blobs" )
parser.add_argument("output_blob" ,
                    help="the name of output blobs")
parser.add_argument('--class-frame-count', dest='count' , type=argparse.FileType('r'),
                   help='counts of individual classes')
parser.add_argument('--epsilon' , dest='epsilon', type=float, default=1e-32,
                    help='minimum prob to avoid log(0)' )
parser.add_argument('--no-log' , dest='no_log' , action='store_true', 
                    help='minimum prob to avoid log(0)' )
args = parser.parse_args()

pdf_prior_log = None
if args.count:
    pdf_prior_line = args.count.readline().rstrip('\n').split()
    args.count.close()
    pdf_prior = [  float(f)  for f in pdf_prior_line[1:-1] ] 
    pdf_prior = [  max(1 , f) for f in pdf_prior ]
    pdf_count_sum = sum(pdf_prior)
    pdf_prior = [  f / float(pdf_count_sum)  for f in pdf_prior ] 
    pdf_prior_log = [ math.log( float(f)  ) for f in pdf_prior ]
        
 
net = caffe.Net(args.prototxt, args.caffemodel , caffe.TEST )
utt_id = None
input_feats = []
for line in sys.stdin:
    tokens = line.rstrip('\n').split()
    if tokens[1] == "[":
        utt_id = tokens[0]
        sys.stdout.write(line)
        input_feats = []
        continue
    
    to_end = False
    if tokens[-1] == "]":
        to_end = True
        tokens = tokens[:-1] 

    input_feats.append( [ float(f)  for f in tokens] ) 

    if to_end:
        input_batch = np.ndarray( dtype='float32' , shape=(len(input_feats) , len(input_feats[0] ) , 1 ,1)  )
        for i in range( len(input_feats) ):
            input_batch[ i , : , 0 , 0 ] = input_feats[i][:]

        net.blobs[ args.input_blob ].reshape( input_batch.shape[0] , input_batch.shape[1] , 1 , 1  )
        blobs = { args.input_blob: input_batch }
        output = net.forward( **blobs )
        output_blob = output[args.output_blob]
        if output_blob.shape[0] != len(input_feats):
            raise Exception( "Mismatch number of instances")
        for i in range( output_blob.shape[0] ):
            output_vec = output_blob[i , :  ]
            if not args.no_log:
                output_vec = np.log( output_vec +args.epsilon )
                if args.count:
                    output_vec = np.subtract( output_vec , pdf_prior_log )
            else:
                if args.count:
                    output_vec = np.divide( output_vec , pdf_prior )
            output_str = str( output_vec[0] )
            for v in output_vec[1:]:
                output_str += " " + str( v ) 
            if i == output_blob.shape[0] -1:
                output_str += " ]"
            print output_str
                
        
