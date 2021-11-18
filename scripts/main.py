from utils import *
import pandas as pd
import pickle
import os
from haar_filter import HaarFilter
from weak_classifier import WeakClassifier
import matplotlib
from copy import deepcopy
import matplotlib.pyplot as plt
from data_collector import capture_data


def select_best_classifier(clfs, weights, X, Y):
    best_clf, best_accuracy, accuracy_map = None, 0, None
    best_idx = None
    for idx, clf in enumerate(clfs):
        prediction = clf.predict(X).astype(np.int)
        accuracy = np.sum(np.multiply(prediction == Y, weights))
        if accuracy > best_accuracy:
            best_clf = clf
            best_idx = idx
            best_accuracy = accuracy
            accuracy_map = prediction == Y
    return best_clf, best_accuracy, best_idx, accuracy_map


def train_weak_classifiers(X, Y, W, filters, verbose=True):
    trained_clfs = []
    for idx, haar_filter in enumerate(filters):
        if not idx % 100 and verbose:
            print(f"... Trained {idx}/{len(filters)} weak classifiers")
        wc = WeakClassifier(haar_filter)
        theta, polarity, best_error, face_scores, none_face_scores = wc.train(X=X, Y=Y, W=W)
        trained_clfs.append(wc)
    return trained_clfs


def train(filters, X, Y, N=1, verbose=False, output_path="strong_clf.pkl"):
    W = np.ones(len(Y)) * 0.5
    W[Y == 1] = W[Y == 1] * 1 / np.sum(W[Y == 1])
    W[Y == -1] = W[Y == -1] * 1 / np.sum(W[Y == -1])
    W = W / W.sum()
    weak_classifiers = train_weak_classifiers(X, Y, W, filters, verbose=verbose)
    best_classifiers = []
    alphas = []
    for n in range(N):
        W = W / W.sum()
        best_clf, best_acc, best_idx, accuracy_map = select_best_classifier(weak_classifiers, W, X, Y)
        best_acc = np.clip(best_acc, 1e-10, 0.99)
        error = 1 - best_acc
        beta = (1 - best_acc) / best_acc
        alpha = 1 / error
        alphas.append(alpha)
        # remove best clf from weak_classifiers
        weak_classifiers = weak_classifiers[:best_idx] + weak_classifiers[best_idx+1:]
        W[accuracy_map == 1] *= beta
        if verbose:
            print("Accuracy", best_acc)
            print("Best WC idx", best_idx + n)
            print("alpha", alpha)
            print("beta", beta)
            print("-" * 20)

        best_classifiers.append(best_clf)

    output = {"weak_classifiers": best_classifiers, "weights": alphas}
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    return output


def test(X, Y, strong_classifier, debug_image=None):
    scores = np.zeros(Y.shape)
    weak_classifiers = strong_classifier["weak_classifiers"]
    weights = strong_classifier["weights"]
    weights /= np.sum(weights)
    for i, wc in enumerate(weak_classifiers):
        weight = weights[i]
        scores += (wc.predict(X)).astype(np.int) * weight

    W = np.ones(Y.shape)
    W[Y == 1] = W[Y == 1] / np.sum(Y == 1)
    W[Y == -1] = W[Y == -1] / np.sum(Y == -1)
    total_pos = np.sum(W[Y == 1])
    total_neg = np.sum(W) - total_pos
    Y = Y.flatten()
    data = sorted(list(zip(scores, Y, W)), key=lambda x: x[0])
    theta, polarity, best_error = None, None, np.inf
    seen_pos_weight, seen_neg_weight, error = 0, 0, 0
    for score, label, weight in data:
        error = min(seen_pos_weight + total_neg - seen_neg_weight, seen_neg_weight + total_pos - seen_pos_weight)
        if error < best_error:
            theta = score
            best_error = error
            polarity = -1 if seen_pos_weight >= seen_neg_weight else 1

        if label > 0:
            seen_pos_weight += weight
        else:
            seen_neg_weight += weight
    theta = theta
    polarity = polarity

    true_label = np.ones(Y.shape)
    prediction = np.where(scores >= theta, true_label, true_label * -1)
    acc = np.sum(prediction == Y) / float(Y.shape[0])
    print(f"Test Accuracy {acc}")
    print("theta", theta)
    print("polarity", polarity)
    plt.hist(scores[Y == -1], label="no_face", alpha=0.5)
    plt.hist(scores[Y == 1], label="face", alpha=0.5)
    plt.legend()
    plt.show()


def convert_image_to_batch(image, x_offset, y_offset, downscale_factor, patch_height=24, patch_width=24):
    image_height, image_width = image.shape[0], image.shape[1]
    pad_ds_image = np.zeros((((image_height + patch_height) // patch_height) * patch_height,
                             ((image_width + patch_width) // patch_width) * patch_width))
    pad_ds_image[:image_height - y_offset, :image_width - x_offset] = image[y_offset:, x_offset:]

    start_x_coord, start_y_coord = np.arange(0, pad_ds_image.shape[1], patch_width) * downscale_factor, \
                                   np.arange(0, pad_ds_image.shape[0], patch_height) * downscale_factor
    start_x_coord = np.tile(start_x_coord.flatten(), reps=(len(start_y_coord), 1))
    start_y_coord = np.tile(start_y_coord.reshape((-1, 1)), reps=(1, start_x_coord.shape[1]))
    start_x_coord, start_y_coord = start_x_coord.flatten(), start_y_coord.flatten()
    end_x_coord, end_y_coord = np.arange(patch_width, pad_ds_image.shape[1] + patch_width, patch_width) * downscale_factor, \
                               np.arange(patch_height, pad_ds_image.shape[0] + patch_height, patch_height) * downscale_factor
    end_x_coord = np.tile(end_x_coord.flatten(), reps=(len(end_y_coord), 1))
    end_y_coord = np.tile(end_y_coord.reshape((-1, 1)), reps=(1, end_x_coord.shape[1]))
    end_x_coord, end_y_coord = end_x_coord.flatten(), end_y_coord.flatten()
    start_x_coord, end_x_coord = start_x_coord + x_offset * downscale_factor, end_x_coord + x_offset * downscale_factor
    start_y_coord, end_y_coord = start_y_coord + y_offset * downscale_factor, end_y_coord + y_offset * downscale_factor

    batch = pad_ds_image.reshape((pad_ds_image.shape[0] // patch_height, patch_height,
                                  pad_ds_image.shape[1] // patch_width, patch_width))
    batch = np.transpose(batch, axes=(0, 2, 1, 3))
    batch = batch.reshape((-1, 24, 24))
    return batch, start_x_coord, end_x_coord, start_y_coord, end_y_coord


def detect_face_in_image(sc, image, threshold=0.0):
    weights = sc["weights"]
    # Normalize the weights (not necessary)
    weights /= np.sum(weights)
    faces, downscale_factors = None, np.array([6, 8, 10, 14])
    weak_classifiers = sc["weak_classifiers"]
    patch_height, patch_width = 24, 24
    for downscale_factor in downscale_factors:
        height, width = image.shape
        ds_image = np.array(Image.fromarray(image).resize((width // downscale_factor, height // downscale_factor)))
        for y_offset in range(0, patch_height, 4):
            for x_offset in range(0, patch_width, 4):
                batch, start_x_coord, end_x_coord, start_y_coord, end_y_coord = convert_image_to_batch(ds_image, x_offset=x_offset, y_offset=y_offset, downscale_factor=downscale_factor)
                batch = normalize_image(batch)
                batch = integral_image(batch)
                scores = np.zeros(batch.shape[0])

                for i, wc in enumerate(weak_classifiers):
                    weight = weights[i]
                    scores += (wc.predict(batch)) * weight
                coordinates = np.hstack([start_x_coord.reshape((-1, 1)), start_y_coord.reshape((-1, 1)),
                                         end_x_coord.reshape((-1, 1)), end_y_coord.reshape((-1, 1))])

                detection_mask = scores >= threshold
                coordinates, scores = coordinates[detection_mask, :], scores[detection_mask]
                result = np.hstack([coordinates, scores.reshape((-1, 1))])
                if faces is None:
                    faces = result
                else:
                    faces = np.vstack([faces, result])
    return faces


def run_realtime_face_detection(strong_classifier, detection_threshold=0.5):
    cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    condition = True
    while condition:
        if frame is not None:
            cv2.imshow("preview", frame)

        rval, frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # normalize image
        im = np.divide(gray, 255.)
        faces = detect_face_in_image(strong_classifier, im, threshold=detection_threshold)
        for i in range(faces.shape[0]):
            start_point = (int(faces[i][0]), int(faces[i][1]))
            end_point = (int(faces[i][2]), int(faces[i][3]))
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(frame, start_point, end_point, color, thickness)
            cv2.putText(frame, "{:.2f}".format(faces[i][4]), (start_point[0], start_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            break
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    is_capture_data = False
    train_fd = False
    eval_fd = False
    test_fd = True
    if is_capture_data:
        capture_data(data_type="no_face", number_of_frames=1000)
    else:
        if train_fd:
            generate_filters(24, 24, 5000, output_path="../filters", min_height=3, max_height=24, min_width=3, max_width=24)
            filters = load_pickle("../pickle/filters.pkl")
            X, Y = load_train_database(cache=False)
            result = train(filters, X, Y, N=10, verbose=True, output_path="../pickle/strong_clf_faces_db.pkl")

        if eval_fd:
            X, Y = load_test_database()
            strong_classifier = load_pickle("../pickle/strong_clf_faces_db.pkl")
            test(X, Y,  strong_classifier)

        if test_fd:
            strong_classifier = load_pickle("../pickle/strong_clf_faces_db.pkl")
            run_realtime_face_detection(strong_classifier=strong_classifier, detection_threshold=0.45)
