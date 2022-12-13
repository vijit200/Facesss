import sys
from src.insightface.deploy import face_model

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from imutils import paths
import numpy as np
# import face_model
import pickle
import cv2
import os


class GeneratingFaceEmbedding:

    def __init__(self,args):
        self.args = args
        self.image_size = '112,112'
        self.model = "./insightface/models/model-y1-test2/model,0"
        self.threshold = 1.24
        self.det = 0
        

    def genFaceEmbedding(self):

        imagePaths = list(paths.list_images(self.args.dataset))
        embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)
        knownEmbeddings = []
        knownNames = []
        total = 0
        for (i, imagePath) in enumerate(imagePaths):
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1))
            face_embedding = embedding_model.get_feature(nimg)
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            total += 1

        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(self.args.embeddings, "wb")
        f.write(pickle.dumps(data))
        f.close()
        #self.logger.info("{} EMBEDDING CREATED {}".format('*'*20,'*'*20))
