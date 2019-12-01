#!/usr/bin/env python3

# Sample Client to call the `whole_image_descriptor_compute_server`

from tx2_whole_image_desc_server.srv import WholeImageDescriptorCompute

import rospy
import numpy as np
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import rospkg
THIS_PKG_BASE_PATH = rospkg.RosPack().get_path('tx2_whole_image_desc_server')

print ('Attempt connecting to server')
rospy.wait_for_service( 'whole_image_descriptor_compute' )
print ('Connection successful!!')
try:
    res = rospy.ServiceProxy( 'whole_image_descriptor_compute', WholeImageDescriptorCompute )

    # X = np.zeros( (100, 100), dtype=np.uint8 )
    X = cv2.resize( cv2.imread( THIS_PKG_BASE_PATH+'/resources/lena_color.jpg' ), (640,480) )
    X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    # X = cv2.resize( cv2.imread( THIS_PKG_BASE_PATH+'/resources/lena_color.jpg' ), (7,5) )
    print( 'X.shape=', X.shape )
    i = CvBridge().cv2_to_imgmsg( X )
    u = res( i, 23  )
    print ('received: ', u )
except rospy.ServiceException as e:
    print( 'failed', e )
