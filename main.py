#pip install opencv-python (Instalimi i librarise OpenCV)
import os
import cv2

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Left_index_finger_Obl.BMP") #lexojme imazhin dhe e ruajme ne variablen SAMPLE(Moster)
# sample = cv2.resize(sample, None, fx=2.5, fy=2.5) #ndryshojme permasat e imazhit

#Printojme imazhit qe kemi marre nga dataseti yne
# cv2.imshow("Sample", sample)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Variablat per rezultatet finale
best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None #keypoints (pikat kryesore) , matchpoint (pika perputhese)

#trajnimi i programit
for file in os.listdir("SOCOFing/Real")[:1000]: #marrim 1000 fotot e para
    
    fingerprint_image = cv2.imread("SOCOFing/Real/" + file)
    sift = cv2.SIFT_create() #Scale Invariant Feature Transform (SIFT) 
    #na lejon te nxjerrim keypoints dhe pershkrimet per imazhet individuale

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    #FlannBasedMatcher libraria me e shpejte per gjetjen e fqinjint me te afert (pikat qe jane me te peraferta me piken aktuale)
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2) #kerkojme per dy pikat me te peraferta

    match_points = [] #relevant matches (ndeshjet perkatese)

    #deklarojm distancen ne mes pikave
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    #kontrollojme gjatesine me te vogel per dy pikat
    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

    #kalkulojme rezultatin e perputhjeve
    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

print("BEST MATCH: " + filename)
print("SCORE: " + str(best_score))

#shikojme perputhjen ne mes dy imazheve
result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()