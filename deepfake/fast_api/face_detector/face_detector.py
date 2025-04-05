from mtcnn import MTCNN
import cv2
from os.path import join
from numpy import array
from shutil import copyfileobj
from fastapi import HTTPException
from configs import configs


def video_save(file):

    '''It will take take the file object from the API and check if it is video.\n If it is video then save it in disk and return a path else 0'''
    
    if file.content_type.split(sep="/")[0] == 'video':

        path = join(configs.VIDEO_DB_PATH,file.filename)

        with open(path, "wb") as buffer:

            copyfileobj(file.file, buffer)

        return path
    else:
        raise HTTPException(status_code=404, detail=f'Please upload a video instead of {file.content_type}')


def video_to_numpy(path):

    detector = MTCNN()
    array_list = []
    cap = cv2.VideoCapture(path)
    frame_count = 0
    if cap.isOpened() == False:
        raise HTTPException(status_code=404, detail= "Video is not readable")

    while cap.isOpened():
        ret, frame = cap.read()
    
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detector.detect_faces(frame)
            if not face :
                continue
            col_1, row_1, col_2, row_2 = face[0]['box']
            col_1_, row_1_, col_2_, row_2_ = col_1, row_1, col_2 + col_1, row_2 + row_1
            frame = frame[row_1_ : row_2_, col_1_ : col_2_]
            
            if frame is None or frame.size == 0:
                continue
                
            frame = cv2.resize(frame, dsize = (224,224))
            array_list.append(frame)
            frame_count = frame_count + 1

            if frame_count == configs.N_FRAMES:
                break
            
        else: 
            break

    cap.release()
            
    return array(array_list)


