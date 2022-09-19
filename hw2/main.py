from random import sample, shuffle
from time import time
import cv2 as cv
import os

dataset_path = "./rgb-dataset/"


def main():
    obj_types = os.listdir(dataset_path)
    dataset_train = {}
    dataset_test = []

    for type in obj_types:
        dataset_train[type] = os.listdir(os.path.join(dataset_path, type))

    for type, set in dataset_train.items():
        samp = sample(set, len(set)//10)
        for e in samp:
            set.remove(e)
        samp = [os.path.join(type, x) for x in samp]
        dataset_test += samp

    shuffle(dataset_test)

    time_classify(classify_SIFT,(dataset_train,dataset_test))
    time_classify(classify_ORB,(dataset_train,dataset_test,12))
    time_classify(classify_ORB,(dataset_train,dataset_test,10))
    time_classify(classify_ORB_KNN,(dataset_train,dataset_test,12))
    time_classify(classify_ORB_KNN,(dataset_train,dataset_test,10))
    time_classify(classify_FAST_SIFT,(dataset_train,dataset_test))

CEND = '\33[0m'
CRED = '\33[31m'
CGREEN = '\33[32m'

# COMMENT OUT IF THE OUTPUT IS BROKEN FOR SOME REASON
def colored_print(color, text):
    print(color + text + CEND)

# COMMENT IN IF THE OUTPUT IS BROKEN FOR SOME REASON
# def colored_print(color,text):
#     print(text)

def print_conf_matrix(conf_matrix):
    print("Confusion Matrix(Columns are Predicted Class, Rows are True Class):")
    print(f"{'':15}", end="")
    for type in conf_matrix:
        print(f"{type:^17}", end="")
    print()
    for type in conf_matrix:
        print(f"{type:15} ", end=" ")
        for innertype in conf_matrix:
            print(f"{conf_matrix[type].get(innertype,0):^15} ", end=" ")
        print()
    print()

def time_classify(classify, args):
    start_time = time()
    classify(*args)
    exec_time = time() - start_time
    print(f"Time of execution: {exec_time//60:.0f}m{exec_time%60:.3f}s")

def classify_SIFT(dataset_train, dataset_test):
    sift = cv.SIFT_create()
    sift_desc = {}
    conf_matrix = {}

    print("SIFT Training Started!")
    i = 0
    print(len(dataset_train))
    for type in dataset_train:
        desList = []
        k = 0
        j = 0
        print(f"{100*i/len(dataset_train)}% is completed.")
        for obj in dataset_train[type]:
            img = cv.imread(os.path.join(dataset_path, type, obj),
                            cv.IMREAD_GRAYSCALE)
            kp, des = sift.detectAndCompute(img, None)
            desList.append(des)
        sift_desc[type] = desList
        conf_matrix[type] = {}
        i += 1
    print("SIFT Training Completed!")

    bf = cv.BFMatcher()
    correct = 0
    i = 0
    for test_obj in dataset_test:
        color = CRED
        i += 1
        img = cv.imread(os.path.join(dataset_path, test_obj))
        tkp, tdest = sift.detectAndCompute(img, None)
        matchList = {}
        for type, desList in sift_desc.items():
            for des in desList:
                matches = bf.knnMatch(des, tdest, k=2)
                good = []
                try:
                    for num, pair in enumerate(matches):
                        m,n = pair
                        if m.distance < 0.40*n.distance:
                            good.append([m])
                except ValueError:
                    print("hey")
                    pass
                matchList[type] = max(matchList.get(type, 0), len(good))
        max_match = -1
        matched_type = None
        for type in matchList:
            if(matchList[type] > max_match):
                max_match = matchList[type]
                matched_type = type
        correct_type = test_obj.split("/", 1)[0]
        conf_matrix[correct_type][matched_type] = conf_matrix[correct_type].get(matched_type,0) + 1
        if(matched_type == correct_type):
            correct += 1
            color = CGREEN
        colored_print(color,
                      f"{i}/{len(dataset_test)} File: {test_obj}, Match: {matched_type} w/{max_match} matches. Correct match: {correct_type} w/{matchList[correct_type]} matches")
    print(
        f"SIFT Accuracy is: {correct/len(dataset_test)} with {correct}/{len(dataset_test)} matchings")
    print_conf_matrix(conf_matrix)

def classify_ORB(dataset_train, dataset_test,threshold=12):
    orb = cv.ORB_create(nfeatures=100, edgeThreshold=threshold)
    orb_desc = {}
    conf_matrix = {}

    print("ORB Training Started!")
    i = 0
    print(len(dataset_train))
    for type in dataset_train:
        desList = []
        print(f"{100*i/len(dataset_train)}% is completed.")
        for obj in dataset_train[type]:
            img = cv.imread(os.path.join(dataset_path, type, obj),
                            cv.IMREAD_GRAYSCALE)
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                desList.append(des)
        orb_desc[type] = desList
        conf_matrix[type] = {}
        i += 1
    print("ORB Training Completed!")

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    correct = 0
    i = 0
    for test_obj in dataset_test:
        color = CRED
        i += 1
        img = cv.imread(os.path.join(dataset_path, test_obj))
        tkp, tdest = orb.detectAndCompute(img, None)
        if(tdest is None):
            colored_print(CRED, f"{i}/{len(dataset_test)} File: {test_obj}. Couldn't find any descriptors")
            continue
        matchList = {}
        for type, desList in orb_desc.items():
            for des in desList:
                matches = bf.match(des, tdest)
                matchList[type] = max(matchList.get(type, 0), len(matches))
        max_match = -1
        matched_type = None
        for type in matchList:
            if(matchList[type] > max_match):
                max_match = matchList[type]
                matched_type = type
        correct_type = test_obj.split("/", 1)[0]
        conf_matrix[correct_type][matched_type] = conf_matrix[correct_type].get(matched_type,0) + 1
        if(matched_type == correct_type):
            correct += 1
            color = CGREEN
        colored_print(color,
                      f"{i}/{len(dataset_test)} File: {test_obj}, Match: {matched_type} w/{max_match} matches. Correct match: {correct_type} w/{matchList[correct_type]} matches")
    print(
        f"ORB Accuracy is: {correct/len(dataset_test)} with {correct}/{len(dataset_test)} matchings")
    print_conf_matrix(conf_matrix)


def classify_ORB_KNN(dataset_train, dataset_test,threshold=12):
    orb = cv.ORB_create(nfeatures=100, edgeThreshold=threshold)
    orb_desc = {}
    conf_matrix = {}

    print("ORB_KNN Training Started!")
    i = 0
    print(len(dataset_train))
    for type in dataset_train:
        desList = []
        print(f"{100*i/len(dataset_train)}% is completed.")
        for obj in dataset_train[type]:
            img = cv.imread(os.path.join(dataset_path, type, obj),
                            cv.IMREAD_GRAYSCALE)
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                desList.append(des)
        orb_desc[type] = desList
        conf_matrix[type] = {}
        i += 1
    print("ORB_KNN Training Completed!")

    bf = cv.BFMatcher()
    correct = 0
    i = 0
    for test_obj in dataset_test:
        color = CRED
        i += 1
        img = cv.imread(os.path.join(dataset_path, test_obj))
        tkp, tdest = orb.detectAndCompute(img, None)
        if(tdest is None):
            colored_print(CRED, f"{i}/{len(dataset_test)} File: {test_obj}. Couldn't find any descriptors")
            continue
        matchList = {}
        for type, desList in orb_desc.items():
            for des in desList:
                matches = bf.knnMatch(des, tdest, k=2)
                good = []
                try:
                    for num, pair in enumerate(matches):
                        m,n = pair
                        if m.distance < 0.40*n.distance:
                            good.append([m])
                except ValueError:
                    pass
                matchList[type] = max(matchList.get(type, 0), len(good))
        max_match = -1
        matched_type = None
        for type in matchList:
            if(matchList[type] > max_match):
                max_match = matchList[type]
                matched_type = type
        correct_type = test_obj.split("/", 1)[0]
        conf_matrix[correct_type][matched_type] = conf_matrix[correct_type].get(matched_type,0) + 1
        if(matched_type == correct_type):
            correct += 1
            color = CGREEN
        colored_print(color,
                      f"{i}/{len(dataset_test)} File: {test_obj}, Match: {matched_type} w/{max_match} matches. Correct match: {correct_type} w/{matchList[correct_type]} matches")
    print(
        f"ORB_KNN Accuracy is: {correct/len(dataset_test)} with {correct}/{len(dataset_test)} matchings")
    print_conf_matrix(conf_matrix)


def classify_FAST_SIFT(dataset_train, dataset_test):
    fast = cv.FastFeatureDetector_create()
    sift = cv.SIFT_create()
    fast_desc = {}
    conf_matrix = {}

    print("FAST Training Started!")
    i = 0
    print(len(dataset_train))
    for type in dataset_train:
        desList = []
        print(f"{100*i/len(dataset_train)}% is completed.")
        for obj in dataset_train[type]:
            img = cv.imread(os.path.join(dataset_path, type, obj),
                            cv.IMREAD_GRAYSCALE)
            kp = fast.detect(img, None)
            kp,des = sift.compute(img,kp)
            if des is not None:
                desList.append(des)
        fast_desc[type] = desList
        conf_matrix[type] = {}
        i += 1
    print("FAST Training Completed!")

    bf = cv.BFMatcher()
    correct = 0
    i = 0
    for test_obj in dataset_test:
        color = CRED
        i += 1
        img = cv.imread(os.path.join(dataset_path, test_obj))
        tkp = fast.detect(img, None)
        tkp,tdest = sift.compute(img,tkp)
        matchList = {}
        for type, desList in fast_desc.items():
            for des in desList:
                matches = bf.knnMatch(des, tdest, k=2)
                good = []
                try:
                    for num, pair in enumerate(matches):
                        m,n = pair
                        if m.distance < 0.40*n.distance:
                            good.append([m])
                except ValueError:
                    pass
                matchList[type] = max(matchList.get(type, 0), len(good))
        max_match = -1
        matched_type = None
        for type in matchList:
            if(matchList[type] > max_match):
                max_match = matchList[type]
                matched_type = type
        correct_type = test_obj.split("/", 1)[0]
        conf_matrix[correct_type][matched_type] = conf_matrix[correct_type].get(matched_type,0) + 1
        if(matched_type == correct_type):
            correct += 1
            color = CGREEN
        colored_print(color,
                      f"{i}/{len(dataset_test)} File: {test_obj}, Match: {matched_type} w/{max_match} matches. Correct match: {correct_type} w/{matchList[correct_type]} matches")
    print(
        f"FAST Accuracy is: {correct/len(dataset_test)} with {correct}/{len(dataset_test)} matchings")
    print_conf_matrix(conf_matrix)

main()
