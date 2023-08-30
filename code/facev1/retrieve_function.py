# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import json

import numpy as np
import cv2 as cv
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from facev1.yunet import YuNet
from facev1.sface import SFace



class faceRetrieve:

    def __init__(self,
                 base_path, 
                 yunet_model, 
                 sface_model, 
                 dis_type=0, 
                 backend=cv.dnn.DNN_BACKEND_OPENCV, 
                 target=cv.dnn.DNN_TARGET_CPU):
        '''
        base_path:      Path to the database (with .npy, .json).
        yunet_model:    Path to the yunet model.
        sface_model:    Path to the sface model.
        dis_type:       Distance type. 0: cosine, 1: norm_l1.
        backend:        cv.dnn.DNN_BACKEND_OPENCV or cv.dnn.DNN_BACKEND_CUDA or cv.dnn.DNN_BACKEND_TIMVX
        target:         cv.dnn.DNN_TARGET_CPU or cv.dnn.DNN_TARGET_CUDA or cv.dnn.DNN_TARGET_CUDA_FP16 or cv.dnn.DNN_TARGET_NPU
        '''

        # Instantiate YuNet for face detection
        self.detector = YuNet(modelPath=yunet_model,
                              inputSize=[320, 320],
                              confThreshold=0.9,
                              nmsThreshold=0.3,
                              topK=5000,
                              backendId=backend,
                              targetId=target)
        # Instantiate SFace for face recognition
        self.recognizer = SFace(modelPath=sface_model, 
                                disType=dis_type, 
                                backendId=backend, 
                                targetId=target)
        
        # Database features and filenames
        self.f_base = np.load(base_path + '.npy')
        with open(base_path + '.json') as f:
            self.path_map = json.load(f)['filenames']
    

    def imread(self, path):
        img = cv.imread(path)
        H, W, C = img.shape
        _H, _W = 500, int(500 / H * W)
        img = cv.resize(img, (_W, _H))

        return img


    def match(self, query, database, disType=0, threshold_cosine=0.363, threshold_norml2=1.128):
        assert disType in [0, 1], "0: Cosine similarity, 1: norm-L2 distance, others: invalid"

        if len(query.shape) == 1:
            query = np.expand_dims(query, axis=0)

        if disType == 0: # COSINE
            cosine_scores = cosine_similarity(query, database)
            max_scores = np.max(cosine_scores, axis=1)
            max_indexs = np.argmax(cosine_scores, axis=1)
            failure = -np.ones_like(max_indexs)
            prediction = np.where(max_scores >= threshold_cosine, max_indexs, failure)
        else: # NORM_L2
            norml2_distance = euclidean_distances(query, database)
            min_distances = np.min(norml2_distance, axis=1)
            min_indexs = np.argmin(norml2_distance, axis=1)
            failure = -np.ones_like(max_indexs)
            prediction = np.where(min_distances <= threshold_norml2, min_indexs, failure)
        
        return prediction
    

    def retrieve(self, query_path, topk=3):
        query = self.imread(query_path)

        # Detect faces
        self.detector.setInputSize([query.shape[1], query.shape[0]])
        faces = self.detector.infer(query)
        assert faces is not None, 'Cannot find a face in {}'.format(query_path)
        # assert faces.shape[0] > 0, 'Cannot find a face in {}'.format(query_path)

        # Feature
        f_query = []
        for i in range(len(faces)):
            f_query.append(self.recognizer.infer(query, faces[i]))
        f_query = np.concatenate(f_query)

        actor_ids = self.match(f_query, self.f_base)
        prediction = []
        for actor_id in actor_ids:
            if actor_id >= 0:
                prediction.append(self.path_map[actor_id])
            if len(prediction) == topk:
                break
            
        return prediction
    

# model = faceRetrieve(base_path='database', 
#                      yunet_model='models/face_detection_yunet_2022mar.onnx',
#                      sface_model='models/face_recognition_sface_2021dec.onnx')

# prediction = model.retrieve(query_path='image2.jpg', topk=2)
# print(prediction)




        


