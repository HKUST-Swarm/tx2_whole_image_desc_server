#!/usr/bin/env python
# Tensorflow
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.platform import gfile
from sensor_msgs.msg import Image
# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Misc Python packages
import numpy as np
import os
import time

# ROS
import rospy
import math

# Pkg msg definations
from tx2_whole_image_desc_server.srv import WholeImageDescriptorComputeTS, WholeImageDescriptorComputeTSResponse

from TerminalColors import bcolors
tcol = bcolors()

QUEUE_SIZE = 10


def imgmsg_to_cv2( msg ):
    assert msg.encoding == "8UC3" or msg.encoding == "8UC1" or msg.encoding == "bgr8" or msg.encoding == "mono8", \
        "Expecting the msg to have encoding as 8UC3 or 8UC1, received"+ str( msg.encoding )
    if msg.encoding == "8UC3" or msg.encoding=='bgr8':
        X = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        return X

    if msg.encoding == "8UC1" or msg.encoding=='mono8':
        X = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return X


class ProtoBufferModelImageDescriptor:
    """
    This class loads the net structure from the .h5 file. This file contains
    the model weights as well as architecture details.
    In the argument `frozen_protobuf_file`
    you need to specify the full path (keras model file).
    """

    def __init__(self, frozen_protobuf_file, im_rows=600, im_cols=960, im_chnls=3):
        start_const = time.time()
        # return
        ## Build net
        # from keras.backend.tensorflow_backend import set_session
        # tf.set_random_seed(42)
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.visible_device_list = "0"
        config.intra_op_parallelism_threads=1
        tf.keras.backend.set_session(tf.Session(config=config))
        tf.keras.backend.set_learning_phase(0)

        self.sess = tf.keras.backend.get_session()
        self.queue = []

        self.im_rows = int(im_rows)
        self.im_cols = int(im_cols)
        self.im_chnls = int(im_chnls)

        LOG_DIR = '/'.join( frozen_protobuf_file.split('/')[0:-1] )
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
        print(  '++++++++++ (HDF5ModelImageDescriptor) LOG_DIR=', LOG_DIR )
        print( '++++++++++ im_rows=', im_rows, ' im_cols=', im_cols, ' im_chnls=', im_chnls )
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
        model_type = LOG_DIR.split('/')[-1]
        self.model_type = model_type



        assert os.path.isdir( LOG_DIR ), "The LOG_DIR doesnot exist, or there is a permission issue. LOG_DIR="+LOG_DIR


        assert os.path.isfile( frozen_protobuf_file ), 'The model weights file doesnot exists or there is a permission issue.'+"frozen_protobuf_file="+frozen_protobuf_file


        #---
        # Load .pb (protobuf file)
        print( tcol.OKGREEN , 'READ: ', frozen_protobuf_file, tcol.ENDC )
        # f = gfile.FastGFile(frozen_protobuf_file, 'rb')
        f = tf.gfile.GFile( frozen_protobuf_file, 'rb')
        graph_def = tf.GraphDef()
        # Parses a serialized binary message into the current message.
        graph_def.ParseFromString(f.read())
        f.close()


        #---
        # Setup computation graph
        print( 'Setup computational graph')
        start_t = time.time()
        sess = K.get_session()
        sess.graph.as_default()
        # Import a serialized TensorFlow `GraphDef` protocol buffer
        # and place into the current default `Graph`.
        tf.import_graph_def(graph_def)
        print( 'Setup computational graph done in %4.2f sec ' %(time.time() - start_t ) )


        #--
        # Output Tensor.
        # Note: the :0. Without :0 it will mean the operator, whgich is not what you want
        # Note: import/
        self.output_tensor = sess.graph.get_tensor_by_name('import/net_vlad_layer_1/l2_normalize_1:0')
        self.sess = K.get_session()


        # Doing this is a hack to force keras to allocate GPU memory. Don't comment this,
        print ('Allocating GPU Memory...')
        # Sample Prediction
        tmp_zer = np.zeros( (1,self.im_rows,self.im_cols,self.im_chnls), dtype='float32' )
        tmp_zer_out = self.sess.run(self.output_tensor, {'import/input_1:0': tmp_zer})

        print( 'model input.shape=', tmp_zer.shape, '\toutput.shape=', tmp_zer_out.shape )
        print( 'model_type=', self.model_type )

        print( '-----' )
        print( '\tinput_image.shape=', tmp_zer.shape )
        print( '\toutput.shape=', tmp_zer_out.shape )
        print( '\tminmax(tmp_zer_out)=', np.min( tmp_zer_out ), np.max( tmp_zer_out ) )
        print( '\tnorm=', np.linalg.norm( tmp_zer_out ) )
        print( '\tdtype=', tmp_zer_out.dtype )
        print( '-----' )

        print ( 'tmp_zer_out=', tmp_zer_out )
        self.n_request_processed = 0
        print( 'Constructor done in %4.2f sec ' %(time.time() - start_const ) )
        print( tcol.OKGREEN , 'READ: ', frozen_protobuf_file, tcol.ENDC )
        # quit()

    def on_image_recv(self, msg):
        self.queue.append(msg)
        # print("Adding msg to queue", len(self.queue))
        if len(self.queue) > QUEUE_SIZE:
            del self.queue[0]

    def pop_image_by_timestamp(self, stamp):
        print("Find...", stamp, "queue_size", len(self.queue))
        index = -1
        for i in range(len(self.queue)):
            if math.fabs(self.queue[i].header.stamp.to_sec() - stamp.to_sec()) < 0.001:
                index = i
                break
        if index >= 0:
            cv_image = imgmsg_to_cv2( self.queue[index] )
            del self.queue[0:index+1]
            if cv_image.shape[0] != 480 or cv_image.shape[1] != 640:
                cv_image = cv2.resize(cv_image, (640, 480))
            return cv_image
        return None

    def handle_req( self, req ):
        """ The received image from CV bridge has to be [0,255]. In function makes it to
        intensity range [-1 to 1]
        """
        start_time_handle = time.time()
        stamp = req.stamp.data
        cv_image = self.pop_image_by_timestamp(stamp)
        if cv_image is None:
            rospy.logerr("Unable to find such image")
            result = WholeImageDescriptorComputeTSResponse()
            return result

        print( '[ProtoBufferModelImageDescriptor Handle Request#%5d] cv_image.shape' %(self.n_request_processed), cv_image.shape, '\ta=', req.a, '\tt=', stamp )
        if len(cv_image.shape)==2:
            # print 'Input dimensions are NxM but I am expecting it to be NxMxC, so np.expand_dims'
            cv_image = np.expand_dims( cv_image, -1 )
        elif len( cv_image.shape )==3:
            pass
        else:
            assert False


        assert (cv_image.shape[0] == self.im_rows and cv_image.shape[1] == self.im_cols and cv_image.shape[2] == self.im_chnls) , \
                "\n[whole_image_descriptor_compute_server] Input shape of the image \
                does not match with the allocated GPU memory. Expecting an input image of \
                size %dx%dx%d, but received : %s" %(self.im_rows, self.im_cols, self.im_chnls, str(cv_image.shape) )

        ## Compute Descriptor
        start_time = time.time()
        i__image = (np.expand_dims( cv_image.astype('float32'), 0 ) - 128.)*2.0/255. #[-1,1]
        print( 'Prepare in %4.4fms' %( 1000. *(time.time() - start_time_handle) ) )

        # u = self.model.predict( i__image )
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # u = self.model.predict( i__image )
                u = self.sess.run(self.output_tensor, {'import/input_1:0': i__image})

        print( tcol.HEADER, 'Descriptor Computed in %4.4fms' %( 1000. *(time.time() - start_time) ), tcol.ENDC )
        print( '\tinput_image.shape=', cv_image.shape, )
        print( '\tinput_image dtype=', cv_image.dtype )
        print( tcol.OKBLUE, '\tinput image (to neuralnet) minmax=', np.min( i__image ), np.max( i__image ), tcol.ENDC )
        print( '\tdesc.shape=', u.shape, )
        print( '\tdesc minmax=', np.min( u ), np.max( u ), )
        print( '\tnorm=', np.linalg.norm(u[0]) )
        print( '\tmodel_type=', self.model_type )



        ## Populate output message
        result = WholeImageDescriptorComputeTSResponse()
        # result.desc = [ cv_image.shape[0], cv_image.shape[1] ]
        result.desc = u[0,:]
        result.model_type = self.model_type
        print( '[ProtoBufferModelImageDescriptor Handle Request] Callback returned in %4.4fms' %( 1000. *(time.time() - start_time_handle) ) )
        return result

if __name__ == '__main__':
    rospy.init_node( 'whole_image_descriptor_compute_server' )

    ##
    ## Load the config file and read image row, col
    ##
    fs_image_width = -1
    fs_image_height = -1
    fs_image_chnls = 1

    if True: # read from param `config_file`
        if not rospy.has_param( '~config_file'):
            print( 'FATAL...cannot find param ~config_file. This is needed to determine size of the input image to allocate GPU memory. If you do not specify the config_file, you need to atleast specify the nrows, ncols, nchnls' )
            rospy.logerr( '[whole_image_descriptor_compute_server]FATAL...cannot find param ~config_file. This is needed to determine size of the input image to allocate GPU memory. If you do not specify the config_file, you need to atleast specify the nrows, ncols, nchnls' )

            if ( rospy.has_param( '~nrows') and rospy.has_param( '~ncols') ):
                print( tcol.OKGREEN, 'However, you seem to have set the parameters nrows and ncols, so will read those.', tcol.ENDC )
            else:
                quit()
            # quit only if you cannot see nrows and ncols

        else:
            config_file = rospy.get_param('~config_file')
            print( '++++++++\n++++++++ config_file: ', config_file )
            if not os.path.isfile(config_file):
                print( 'FATAL...cannot find config_file: ', config_file )
                rospy.logerr( '[whole_image_descriptor_compute_server]FATAL...cannot find config_file: '+ config_file )
                quit()


            print( '++++++++ READ opencv-yaml file: ', config_file )
            fs = cv2.FileStorage(config_file, cv2.FILE_STORAGE_READ)
            fs_image_width = int(  fs.getNode( "image_width" ).real() )
            fs_image_height = int( fs.getNode( "image_height" ).real() )
            print( '++++++++ opencv-yaml:: image_width=', fs_image_width, '   image_height=', fs_image_height )
            print( '++++++++' )


    ##
    ## Load nrows and ncols directly as config
    ##
    if True:  # read from param `nrows` and `ncols`
        if fs_image_width < 0 :
            if ( not rospy.has_param( '~nrows') or not rospy.has_param( '~ncols') or not rospy.has_param( '~nchnls') ):
                print( 'FATAL...cannot find param either of ~nrows, ~ncols, ~nchnls. This is needed to determine size of the input image to allocate GPU memory' )
                rospy.logerr( '[whole_image_descriptor_compute_server] FATAL...cannot find param either of ~nrows, ~ncols, nchnls. This is needed to determine size of the input image to allocate GPU memory' )
                quit()
            else:
                fs_image_height = rospy.get_param('~nrows')
                fs_image_width = rospy.get_param('~ncols')
                fs_image_chnls = rospy.get_param('~nchnls')

                print ( '~~~~~~~~~~~~~~~~' )
                print ( '~nrows = ', fs_image_height, '\t~ncols = ', fs_image_width, '\t~nchnls = ', fs_image_chnls )
                print ( '~~~~~~~~~~~~~~~~' )


    ##
    ## Load Channels
    ##
    if True:
        if not rospy.has_param( '~nchnls' ):
            rospy.logerr( "[whole_image_descriptor_compute_server] FATAL....cannot file cmd param nchnls.")
            quit()
        else:
            fs_image_chnls = rospy.get_param('~nchnls')


    print( '~~~@@@@ OK...' )
    ##
    ## Start Server
    ##
    # frozen_protobuf_file = '/models.keras/Apr2019/gray_conv6_K16Ghost1__centeredinput/core_model.%d.keras' %(500)
    if rospy.has_param( '~frozen_protobuf_file'):
        frozen_protobuf_file = rospy.get_param('~frozen_protobuf_file')
    else:
        print( tcol.FAIL, 'FATAL...missing specification of model file. You need to specify ~frozen_protobuf_file', tcol.ENDC )
        quit()

    gpu_netvlad = ProtoBufferModelImageDescriptor( frozen_protobuf_file=frozen_protobuf_file, im_rows=fs_image_height, im_cols=fs_image_width, im_chnls=fs_image_chnls )

    s = rospy.Service( 'whole_image_descriptor_compute_ts', WholeImageDescriptorComputeTS, gpu_netvlad.handle_req)
    sub = rospy.Subscriber("left_camera", Image, gpu_netvlad.on_image_recv, queue_size=5, tcp_nodelay=True)
    print (tcol.OKGREEN )
    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print( '+++  whole_image_descriptor_compute_server is running +++' )
    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print( tcol.ENDC )
    rospy.spin()
