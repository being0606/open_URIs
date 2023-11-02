import cv2, os
from os import listdir
from os.path import isfile, join
import numpy as np
import csv
import copy

lxd = 1

face_classifier = None

custom_model = None

def file_select():
    from tkinter import filedialog
    file = filedialog.askopenfilenames()
    file = ''.join(file)
    return file

def folder_select():
    from tkinter import filedialog
    folder = filedialog.askdirectory()
    folder = ''.join(folder)
    return folder
    
def camera_a():
    global lxd
    global camera
    if lxd == 1:
        camera = cv2.VideoCapture(0)
        lxd = 2
        return camera
    else:
        return camera
    

def get_frame(camera):
    ret, frame = camera.read()
    #frame = frame[60:480,60:580]
    return frame

def save_face(cap, save_path, count):
    face = None
    while face is None:
        #sleep(0.1)
        face = face_extractor(cap)
    #얼굴 이미지 크기를 200x200으로 조정 
    face = cv2.resize(face,(200,200))
    #조정된 이미지를 흑백으로 변환 
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    os.makedirs(save_path, exist_ok=True)

    #faces폴더에 jpg파일로 저장 
    file_name_path = save_path + '/user'+str(count)+'.jpg'
    cv2.imwrite(file_name_path,face)
    print(file_name_path + " is saved.")

def train_face(save_path):
    data_path = save_path + '/'
    #faces폴더에 있는 파일 리스트 얻기

    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    #데이터와 매칭될 라벨 변수

    Training_Data, Labels = [], []

    #파일 개수 만큼 루프 
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        #이미지 불러오기 
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #이미지 파일이 아니거나 못 읽어 왔다면 무시
        if images is None:
            continue    
        #Training_Data 리스트에 이미지를 바이트 배열로 추가 
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        #Labels 리스트엔 카운트 번호 추가 
        Labels.append(i)
        

    #훈련할 데이터가 없다면 종료.
    if len(Labels) == 0:
        return False
    
    #Labels를 32비트 정수로 변환
    Labels = np.asarray(Labels, dtype=np.int32)

    #모델 생성 
    model = cv2.face.LBPHFaceRecognizer_create()

    #학습 시작
    model.train(np.asarray(Training_Data), np.asarray(Labels))

    return model

def train_face(save_path):
    data_path = save_path + '/'
    #faces폴더에 있는 파일 리스트 얻기

    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    #데이터와 매칭될 라벨 변수

    Training_Data, Labels = [], []

    #파일 개수 만큼 루프 
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        #이미지 불러오기 
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #이미지 파일이 아니거나 못 읽어 왔다면 무시
        if images is None:
            continue    
        #Training_Data 리스트에 이미지를 바이트 배열로 추가 
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        #Labels 리스트엔 카운트 번호 추가 
        Labels.append(i)
        

    #훈련할 데이터가 없다면 종료.
    if len(Labels) == 0:
        return False
    
    #Labels를 32비트 정수로 변환
    Labels = np.asarray(Labels, dtype=np.int32)

    #모델 생성 
    model = cv2.face.LBPHFaceRecognizer_create()

    #학습 시작
    model.train(np.asarray(Training_Data), np.asarray(Labels))

    return model

def face_extractor(cap):
    global face_classifier

    img = get_frame(cap)
    
    cv2.imshow('FaceCheck', img)
    cv2.waitKey(1)
    
    #흑백처리 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #얼굴 찾기
    if face_classifier is None:
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #찾은 얼굴이 없으면 None으로 리턴 
    if faces is():
        return None
    #얼굴이 있으면 
    for(x,y,w,h) in faces:
        #해당 얼굴 크기만큼 cropped_face에 넣기 
        cropped_face = img[y:y+h, x:x+w]
        #roi = cv2.resize(cropped_face, (200,200))
    #cropped_face 반환 
    return cropped_face #, roi

def compare_face(face, model):
    try:
        face = cv2.resize(face, (200,200))
        #검출된 사진을 흑백으로 변환 
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #학습한 모델로 예측시도
        result = model.predict(face)

        #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))

        #75 보다 크면 동일 인물
        if confidence > 75:
            return True
        #75 이하면 타인
        else:
            return False
    except:
        return False


def test(facenet, model, i):
    ret, img = camera_a().read()

    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(405, 405), mean=(104., 177., 123.))         #블럭2
    facenet.setInput(blob)
    dets = facenet.forward()


    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2]
        face = face/256

        if (x2 >= w or y2 >= h):
            continue
        if (x1<=0 or y1<=0):
            continue

        face_input = cv2.resize(face,(200, 200))
        face_input = np.expand_dims(face_input, axis=0)
        face_input = np.array(face_input)

        modelpredict = model.predict(face_input)
        mask=modelpredict[0][0]
        nomask=modelpredict[0][1]

        if mask > nomask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)
            #frequency = 2500  # Set Frequency To 2500 Hertz
            #duration = 1000  # Set Duration To 1000 ms == 1 second
            #winsound.Beep(frequency, duration)

        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)
    return img

def readCSV(file):
    landmarks = {}
    ids = []
    coordinates = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count != 0):
                landmarks[line_count] = {'id': int(row[0]),
                                         'x' : int(row[1]),
                                         'y' : int(row[2])}
                ids.append(int(row[0]))
                coordinates.append([int(row[1]),int(row[2])])

            line_count += 1
    
    return landmarks,ids,coordinates

def warpImage(image,landmarks_coord,mask_file,mask_coord,selected,test):
	im_src = cv2.imread(mask_file,cv2.IMREAD_UNCHANGED)
	pts_src = np.array(mask_coord, dtype=float)

	pts_dst = np.array(landmarks_coord, dtype=float)
	h, status = cv2.findHomography(pts_src, pts_dst)
	im_out = cv2.warpPerspective(im_src, h, (image.shape[1],image.shape[0]))

	src = im_out.astype(float)
	src = src / 255.0
	alpha_foreground = src[:,:,3]

	dst = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
	dst = dst.astype(float)
	dst = dst / 255.0

	# Additional code required for blending alpha parameter from mask
	#	image to the live feed, hence the need of transparent backgrounds
	if(selected != test):
		for color in range(0, test):
		    dst[:,:,color] = alpha_foreground*src[:,:,color] + (1-alpha_foreground)*dst[:,:,color]
	
	dst[:,:,:] = cv2.erode(dst[:,:,:],(5,5),0)
	dst[:,:,:] = cv2.GaussianBlur(dst[:,:,:],(3,3),0)

	return dst

def displayImage(output,face_land_img,masks_files,selected,hover):
    face_land_img = cv2.cvtColor(face_land_img, cv2.COLOR_BGR2BGRA)
    face_land_img = face_land_img / 255.0
    height, width = face_land_img.shape[:2]

    positions = []

    # Place landmark image on the top left corner
    #if(face_land_img is not None):
    #    output[:height,:width] = face_land_img
    #    output = cv2.rectangle(output, (0,0), (width,height), (0,0,250), 5)

    # Place mask images on the right
    # Depending on the mask selected or hovered over, it shifts it
    #   to the left
    mask_height,mask_width = masks_files[0].shape[:2]
    for i,mask in enumerate(masks_files):
        if(selected == i or hover == i):
            shift = 15
        else:
            shift = 0

        pos_y = [10+i*15+i*mask_height,10+i*15+(i+1)*mask_height]
        pos_x = [output.shape[1]-mask_width-10-shift,output.shape[1]-10-shift]
        positions.append(pos_y+pos_x)
        output[pos_y[0]:pos_y[1],pos_x[0]:pos_x[1]] = mask

        if(selected == i):
            output = cv2.rectangle(output,(pos_x[0],pos_y[0]),(pos_x[1],pos_y[1]), (0,200,0), 3)
    
    return output,positions

def find_object_from_frame(o_frame, data_path):
    global custom_model
    from keras.models import load_model
    categories = np.load("./" + str(data_path) + "_labels.npy")
    categories = categories.tolist()
    
    frame = copy.deepcopy(o_frame)
    if custom_model is None:
        custom_model = load_model(data_path + '.h5')

    try:
        frame_conv = cv2.resize(frame, (32, 32))
        frame_conv = np.array(frame_conv)
        frame_conv = frame_conv.astype("float") / 255.0
        #frame = img_to_array(frame)
        frame_conv = np.expand_dims(frame_conv, axis=0)
        #predict = custom_model.predict_classes(frame_conv)
        predict = np.argmax(custom_model.predict(frame_conv), axis=-1)
        predicted = "N/A"
        percent = custom_model.predict(frame_conv)
    except:
        return frame, "ERROR"
    
    for i in range(len(predict)):
        # write label and confidence above face rectangle
        for j in range(len(categories)):
            if(float(percent[0][j])==max(list(percent)[0])):
                predicted = str(categories[predict[i]])
                cv2.putText(frame, str(categories[predict[i]]), (00, 40),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
                #cv2.putText(frame, str(float(percent[0][j])), (00, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, predicted