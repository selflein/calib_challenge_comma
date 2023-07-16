import numpy as np
import cv2 as cv
import argparse
import matplotlib.pyplot as plt

import constants


import torchvision
import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.utils import draw_segmentation_masks


MODEL_WEIGHTS = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
MODEL = fcn_resnet50(weights=MODEL_WEIGHTS)
MODEL.eval()
CLASS_TO_IDX = {cls: idx for (idx, cls) in enumerate(MODEL_WEIGHTS.meta["categories"])}
IGNORE_CLASSES = ["motorbike", "car", "bus"]


def slam(p0, p1):
    E, mask = cv.findEssentialMat(
        points1=p0,
        points2=p1,
        cameraMatrix=constants.CAMERA_MAT,
        method=cv.LMEDS,
        prob=0.99,
        mask=None,
    )

    _, R, t, mask, point_4d_hom = cv.recoverPose(
        E=E,
        points1=p0,
        points2=p1,
        cameraMatrix=constants.CAMERA_MAT,
        distanceThresh=1e5,
        mask=mask
    )
    point_4d = point_4d_hom / point_4d_hom[-1, :]
    point_3d = point_4d[:3, :].T
    return R, t, point_3d[mask[:, 0] != 0]


def get_center_of_motion(t):
    angles = np.arccos(t)
    yaw = -(angles[2, 0] - np.pi)
    pitch = -(angles[1, 0] - np.pi / 2)
    return np.clip(np.array([yaw, pitch]), a_min=(-constants.MAX_YAW, -constants.MAX_PITCH), a_max=(constants.MAX_YAW, constants.MAX_PITCH))



def get_matches_orb(train_img, query_img):
    # Convert it to grayscale
    query_img_bw = cv.cvtColor(query_img,cv.COLOR_BGR2GRAY)
    train_img_bw = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv.ORB_create(nfeatures=1_000)
    
    # Now detect the keypoints and compute the descriptors for the query image and train image
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    # Initialize the Matcher for matching the keypoints and then match the keypoints
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    matches = matcher.match(queryDescriptors, trainDescriptors)

    p0 = np.array([trainKeypoints[match.trainIdx].pt for match in matches])
    p1 = np.array([queryKeypoints[match.queryIdx].pt for match in matches])
    return p0, p1


def get_matches_orb_flann(img1, img2):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2,k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return np.array(pts1), np.array(pts2)


def get_matches_lk(old_frame, frame):

    preprocess = MODEL_WEIGHTS.transforms()

    img = torch.from_numpy(cv.cvtColor(old_frame, cv.COLOR_BGR2RGB)).permute((2, 0, 1))

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and visualize the prediction
    prediction = torch.softmax(MODEL(batch)["out"], dim=1)
    prediction = torchvision.transforms.Resize((constants.HEIGHT, constants.WIDTH))(prediction)

    ignore_class_probs = torch.index_select(prediction, dim=1, index=torch.LongTensor([CLASS_TO_IDX[cl] for cl in IGNORE_CLASSES])).sum(1)[0]
    mask = (~(ignore_class_probs > 0.1)).numpy().astype(np.uint8)

    p0 = cv.goodFeaturesToTrack(
        cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY),
        maxCorners = 1000,
        qualityLevel = 0.01,
        minDistance = 20,
        blockSize = 5,
        mask=mask
    )
    
    # Mask points on the hood of the car
    p0 = p0[p0[:, 0, 1] < 600]

    # calculate optical flow
    p1, status, _ = cv.calcOpticalFlowPyrLK(
        prevImg=old_frame,
        nextImg=frame,
        prevPts=p0,
        nextPts=None, 
        winSize=(30, 30),
        maxLevel=10,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    return p0[status == 1], p1[status == 1]


def angles_to_com(yaw_pitch):
    return (constants.PP + np.tan(yaw_pitch) * np.array([1., -1.]) * constants.FOCAL_LENGTH).astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=str, help='path to image file')
    parser.add_argument('--show', help='path to image file', action="store_true")
    args = parser.parse_args()

    cap = cv.VideoCapture(f'labeled/{args.scene}.hevc')

    # Rotation around z axis (pitch) and y axis (yaw) 
    yaw_pitch_labels = np.flip(np.loadtxt(f'labeled/{args.scene}.txt'), axis=-1)

    ret, old_frame = cap.read()

    preds = []
    smoothed = None
    n_frame = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try: 
            frame = cv.circle(frame, angles_to_com(yaw_pitch_labels[n_frame]), radius=5, color=(255, 255, 255), thickness=-1)

            p0, p1 = get_matches_lk(old_frame, frame)
            # p0, p1 = get_matches_orb(old_frame, frame)
            # p0, p1 = get_matches_orb_flann(old_frame, frame)

            R, t, points_3 = slam(p0, p1)
            yaw_pitch = get_center_of_motion(t)


            if smoothed is None:
                smoothed = yaw_pitch
            else:
                diff = np.abs(smoothed - yaw_pitch)
                smoothed = (0.9 * smoothed) + (0.1 * yaw_pitch)

            preds.append(smoothed)
            frame = cv.circle(frame, angles_to_com(smoothed), radius=5, color=(255, 255, 0), thickness=-1)

        except Exception as e:
            preds.append(np.array([np.nan, np.nan]))
            print(e)

        if args.show:
            frame = cv.circle(frame, constants.PP.astype(int), radius=2, color=(0, 0, 255), thickness=-1)
            cv.imshow('frame', frame)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

        # Now update the previous frame
        old_frame = frame
        n_frame += 1
    
    np.savetxt(f"labeled/pred_labeled/{args.scene}.txt", np.flip(np.stack(preds[:1] + preds), axis=-1))
    cv.destroyAllWindows()