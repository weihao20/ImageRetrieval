from retrieval_image import ClipRetrieve
from facev1 import faceRetrieve

print('loading ClipRetrieve model...')
fr_model = ClipRetrieve()
print('loading faceRetrieve model...')
face_model = faceRetrieve(base_path='facev1/database', 
                          yunet_model='facev1/models/face_detection_yunet_2022mar.onnx',
                          sface_model='facev1/models/face_recognition_sface_2021dec.onnx')

fr_list = fr_model.i2i_match(query_path='frame.jpg')
face_list = face_model.retrieve(query_path='frame.jpg', topk=2)
