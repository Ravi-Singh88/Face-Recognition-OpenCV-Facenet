# Face-Recognition-OpenCV-Facenet
This is an implementation Facial Recognition using OpenCV implementation of Facenet and SVM


Setup before running the code:
1. Create dataset folder
2. Create sub-folders with names of the faces you want to recognize
3. In each sub-folder add the image of the person. There is no need to crop images. However, make sure that each image has only the pic of one person
4. Create a video folder
5. Add the video on which you want to run the algorithm on - preferably an mp4 video
6. Create an output folder. This will save the pickle files and the final output video.
7. Make sure you have the relevant libraries of Opencv, imutils, pickle etc..



To run the code:
1. Extract the 128-D embeddings of the images in the dataset. This uses the Openface implementation of Facenet model. The embeddings are saved in the output folder "embeddings.pickle" file

==> python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

2. Train the embeddings on a Support Vector Machine (SVM) classifier. This is used to detect different faces in the video. The SVM encodings are saved in "recognizer.pickle". The name encodings of the different faces you want to detect are stored in "le.pickle" file

==> python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

3. Run the model on the video. The output video file can be mp4 or avi. 

===> python recognize_video_file.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --input video/<input video file> --output output/<outut video file>
  
  
 Special thanks to Adrian Rosebrock for teaching me OpenCV. 
 Check out his blog: https://www.pyimagesearch.com/ to learn more about Computer Vision.
 
 Code from pyimage search: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition
 
 Linkedin Video: https://www.linkedin.com/feed/update/urn:li:ugcPost:6485447434771238912/
 
