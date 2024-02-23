import cv2 
import numpy as np
import os
import csv

# for a single file in the current folder
# files = ["2um_002.tif"]
# for all files in current folder
files = os.listdir("./")

# the order of the parameters is important
# the inverted and find circles parameters always execute last
processing_parameters = {
    "high confidence parameterization": {
        "test_image": True,
        "enabled": True,
        "contrast": {
            "enabled": True
        },
        "cutoff": {
            "enabled": True,
            "min": 0,
            "max": 255
        },
        "sobel": {
            "enabled": True,
            "contrasted": False,
            "inverted": False,
            "weight": 0.2
        },
        "inverted": True,
        "find circles": {
            "minArea": 400,
            "maxArea": 2400,
            "minCircularity": 0.1,
            "maxCircularity": 1,
            "minConvexity": 0.1,
            "maxConvexity": 1,
            "minInertiaRatio": 0.8,
            "maxInertiaRatio": 1
        }
    },
    "low confidence parameterization": {
        "enabled": False,
        "contrast": {
            "enabled": True
        },
        "cutoff": {
            "enabled": True,
            "min": 0,
            "max": 255
        },
        "sobel": {
            "enabled": True,
            "contrasted": False,
            "weight": 0.2
        },
        "inverted": True,
        "find circles": {
            "minArea": 400,
            "maxArea": 2400,
            "minCircularity": 0.1,
            "maxCircularity": 1,
            "minConvexity": 0.1,
            "maxConvexity": 1,
            "minInertiaRatio": 0.1,
            "maxInertiaRatio": 1
        }
    }
}

# image processing parameters
# min_cutoff_threshold        # between 0 and 255 (anything below this value will be set to black)
# max_cutoff_threshold        # between 0 and 255 (anything above this value will be set to white)
# sobel_weight                # between 0 and 1   (the higher the value, the more the sobel image will be taken into account) (the sobel image is the gradient of the image, it is used to find the edges of the image)
# contrast                    # True or False     (enables preprocessing of the image to increase contrast)

# circular blobs parameters
# minArea                     # circle minArea in pixels (not negative)
# maxArea                     # circle maxArea in pixels (not negative)
# minCircularity              # circle minCircularity (between 0 and 1)
# maxCircularity              # circle maxCircularity (between 0 and 1)
# minConvexity                # circle minConvexity (between 0 and 1)
# maxConvexity                # circle maxConvexity (between 0 and 1)
# minInertiaRatio             # ratio of how much the surface should be stable (approximately same color), the closer to 1 the more stable it is (between 0 and 1)
# maxInertiaRatio             # ratio of how much the surface should be stable (approximately same color), the closer to 1 the more stable it is (between 0 and 1)

# parameters that find more circles, but with less confidence
# minInertiaRatio: 0.1
# minCircularity:  0.1
# sobel_threshold: 0.1

### internal parameters (do not change) ###

# visual aid
margin = 1

#subfolders
generated_images_folder = "generated_images"
intermediary_images_folder = "intermediary_images"

# create subfolder for generated images

if not os.path.exists(generated_images_folder):
    os.makedirs(generated_images_folder)

# create subfolder for intermediary images

if not os.path.exists(intermediary_images_folder):
    os.makedirs(intermediary_images_folder)

base_output_file_name = "out"
output_file_name = base_output_file_name

new_file_counter = 1
while os.path.exists(generated_images_folder+"/" + output_file_name + ".txt"):
    new_file_counter += 1
    output_file_name = base_output_file_name + "_" + str(new_file_counter)

detection_output_img_id = 0

for file_name in files:
    if file_name.endswith("x.tif"):
        continue
    elif file_name.endswith(".tif"):
        img_name = file_name
    else:
        continue

    for parameters in processing_parameters:
        if not processing_parameters[parameters]["enabled"]:
            continue

        test_img_name = file_name.split(".tif")[0]
        test_img_name = test_img_name + "xxxx.tif"

        or_test_image = None

        # read the image
        or_image = cv2.imread(img_name)

        # remove the information at the bottom of the image which is after a line of white pixels

        for i in range(or_image.shape[0]):
            if (or_image[i] == [255, 255, 255]).all():
                or_image = or_image[0:i]
                break

        try:
            # Load image
            image = or_image.copy()

            # convert the image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Load image with expected circular blobs

            or_test_image = cv2.imread(test_img_name)

            # remove the information at the bottom of the image which is after a line of white pixels

            for i in range(or_test_image.shape[0]):
                if (or_test_image[i] == [255, 255, 255]).all():
                    or_test_image = or_test_image[0:i]
                    break

            test_image = or_test_image.copy()

            # extract the manually determined circular blobs using their diameter in the image

            # get the white pixels in the test image

            _, test_white = cv2.threshold(test_image, 254, 255, cv2.THRESH_BINARY)

            # save the image to a file
            #cv2.imwrite(intermediary_images_folder + "/" + test_img_name + "_white.png", test_white)

            image = or_image.copy()

            # get the white pixels in the img

            _, white = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)

            # save the image to a file
            #cv2.imwrite(intermediary_images_folder + "/" + img_name + "_white.png", white)

            # remove all the pixels from the image that are also in the test image

            image = cv2.subtract(test_white, white)

            # save the image to a file
            #cv2.imwrite(intermediary_images_folder + "/" + img_name + "_subtracted.png", image)

            # find the lines in the image

            # to do that, we just need to find the contours of the image

            # convert the image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # save the lines to a file

            bounding_rects_img = or_test_image.copy()

            bounding_rects = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_rects.append((x, y, w, h))
                bounding_rects_img = cv2.rectangle(bounding_rects_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            #cv2.imwrite(intermediary_images_folder + "/" + img_name + "_bounding_rects.png", bounding_rects_img)
                
            # since the lines are a diameter, we need to convert them to a radius
                
            labels = []

            for bounding_rect in bounding_rects:
                x, y, w, h = bounding_rect
                x = x + w / 2
                y = y + h / 2
                r = (w**2 + h**2)**0.5 / 2
                labels.append((x, y, r))

            manual_blobs_image = or_image.copy()

            # the comparison img should fill the circles with red color to calculate the intersection of the blobs and the manually determined circular blobs
            test_comparison_image = or_image.copy()

            # draw the circular blobs on the image
                
            for label in labels:
                x, y, r = label
                test_comparison_image = cv2.circle(test_comparison_image, (int(x), int(y)), int(r), (0, 0, 255), -1)
                r = r + margin
                manual_blobs_image = cv2.circle(manual_blobs_image, (int(x), int(y)), int(r), (0, 0, 255), 1)

            # save the image to a file

            cv2.imwrite(generated_images_folder + "/" + img_name + "_manual_circles.png", manual_blobs_image)
        except:
            pass

        # Load image
        image = or_image.copy()

        # convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        counter = 0
        for parameter in processing_parameters[parameters]:
            if parameter == "enabled":
                continue
            if parameter == "test_image":
                continue
            if parameter == "inverted":
                continue
            if "enabled" in processing_parameters[parameters][parameter] and not processing_parameters[parameters][parameter]["enabled"]:
                continue
            if parameter == "contrast":
                image = cv2.equalizeHist(image)
                cv2.imwrite(intermediary_images_folder + "/" + img_name + "_"+str(counter)+"_contrasted.png", image)
                counter += 1
            if parameter == "cutoff":
                min_cutoff_threshold = processing_parameters[parameters][parameter]["min"]
                max_cutoff_threshold = processing_parameters[parameters][parameter]["max"]

                # make a new image with only the pixels that are between the min and max cutoff thresholds

                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        if image[i][j] < min_cutoff_threshold:
                            image[i][j] = min_cutoff_threshold
                        elif image[i][j] > max_cutoff_threshold:
                            image[i][j] = max_cutoff_threshold

                # save the image to a file
                
                cv2.imwrite(intermediary_images_folder + "/" + img_name + "_"+str(counter)+"_cutoff.png", image)
                counter += 1
            if parameter == "sobel":
                sobel_weight = processing_parameters[parameters][parameter]["weight"]

                # get the borders of the image to itself to make it easier to find the circular blobs

                # find all the borders of the image by calculating the gradient of the image in both directions and then combining the results

                sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
                sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

                # convert the image to 8 bit

                sobel = cv2.convertScaleAbs(sobel)

                # if the pixel is lighter than gray, make it darker than gray in the same proportion

                if "inverted" in processing_parameters[parameters][parameter] and processing_parameters[parameters][parameter]["inverted"]:
                    for i in range(sobel.shape[0]):
                        for j in range(sobel.shape[1]):
                            if sobel[i][j] < 128:
                                sobel[i][j] = 255 - sobel[i][j]
                else:
                    for i in range(sobel.shape[0]):
                        for j in range(sobel.shape[1]):
                            if sobel[i][j] > 127:
                                sobel[i][j] = 255 - sobel[i][j]

                # save the image to a file

                cv2.imwrite(intermediary_images_folder + "/" + img_name + "_"+str(counter)+"_sobel.png", sobel)
                counter += 1

                if processing_parameters[parameters][parameter]["contrasted"]:
                    sobel = cv2.equalizeHist(sobel)

                    # save the image to a file

                    cv2.imwrite(intermediary_images_folder + "/" + img_name + "_"+str(counter)+"_contrasted_sobel.png", image)
                    counter += 1

                # add the borders of the image to itself to make it easier to find the circular blobs

                image = cv2.addWeighted(sobel, sobel_weight, image, 1 - sobel_weight, 0)

                # save the image to a file

                cv2.imwrite(intermediary_images_folder + "/" + img_name + "_"+str(counter)+"_sobel_added.png", image)
                counter += 1

        if "inverted" in processing_parameters[parameters] and processing_parameters[parameters]["inverted"]:
            # invert the image
            image = cv2.bitwise_not(image)

            # save the image to a file
            cv2.imwrite(intermediary_images_folder + "/" + img_name + "_"+str(counter)+"_inverted.png", image)
            counter += 1
        
        # Set our filtering parameters 
        # Initialize parameter setting using cv2.SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()

        # Set Area filtering parameters
        params.filterByArea = False
        # Set Circularity filtering parameters 
        params.filterByCircularity = False
        # Set Convexity filtering parameters 
        params.filterByConvexity = False
        # Set inertia filtering parameters 
        params.filterByInertia = False

        minArea = None
        maxArea = None
        minCircularity = None
        maxCircularity = None
        minConvexity = None
        maxConvexity = None
        minInertiaRatio = None
        maxInertiaRatio = None

        if processing_parameters[parameters]["find circles"]["minArea"] != None:
            minArea = processing_parameters[parameters]["find circles"]["minArea"]
            params.filterByArea = True
        if processing_parameters[parameters]["find circles"]["maxArea"] != None:
            maxArea = processing_parameters[parameters]["find circles"]["maxArea"]
            params.filterByArea = True
        if processing_parameters[parameters]["find circles"]["minCircularity"] != None:
            minCircularity = processing_parameters[parameters]["find circles"]["minCircularity"]
            params.filterByCircularity = True
        if processing_parameters[parameters]["find circles"]["maxCircularity"] != None:
            maxCircularity = processing_parameters[parameters]["find circles"]["maxCircularity"]
            params.filterByCircularity = True
        if processing_parameters[parameters]["find circles"]["minConvexity"] != None:
            minConvexity = processing_parameters[parameters]["find circles"]["minConvexity"]
            params.filterByConvexity = True
        if processing_parameters[parameters]["find circles"]["maxConvexity"] != None:
            maxConvexity = processing_parameters[parameters]["find circles"]["maxConvexity"]
            params.filterByConvexity = True
        if processing_parameters[parameters]["find circles"]["minInertiaRatio"] != None:
            minInertiaRatio = processing_parameters[parameters]["find circles"]["minInertiaRatio"]
            params.filterByInertia = True
        if processing_parameters[parameters]["find circles"]["maxInertiaRatio"] != None:
            maxInertiaRatio = processing_parameters[parameters]["find circles"]["maxInertiaRatio"]
            params.filterByInertia = True

        if minArea != None:
            params.minArea = minArea
        if maxArea != None:
            params.maxArea = maxArea
        if minCircularity != None:
            params.minCircularity = minCircularity
        if maxCircularity != None:
            params.maxCircularity = maxCircularity
        if minConvexity != None:
            params.minConvexity = minConvexity
        if maxConvexity != None:
            params.maxConvexity = maxConvexity
        if minInertiaRatio != None:
            params.minInertiaRatio = minInertiaRatio
        if maxInertiaRatio != None:
            params.maxInertiaRatio = maxInertiaRatio

        # Create a detector with the parameters
        try:
            detector = cv2.SimpleBlobDetector_create(params)
        except:
            raise Exception("Error creating the detector")
        
        # Detect blobs
        keypoints = detector.detect(image)
        number_of_blobs = len(keypoints)

        # reorder the keypoints by x (if the x is the same, order by y)

        keypoints = sorted(keypoints, key=lambda x: (x.pt[0], x.pt[1]))

        with open(generated_images_folder+"/"+output_file_name+".txt", "a") as file:
            file.write("minArea: "+str(minArea)+"\n"+"maxArea: "+str(maxArea)+"\n"+"minCircularity: "+str(minCircularity)+"\n"+"maxCircularity: "+str(maxCircularity)+"\n"+"minConvexity: "+str(minConvexity)+"\n"+"maxConvexity: "+str(maxConvexity)+"\n"+"minInertiaRatio: "+str(minInertiaRatio)+"\n"+"maxInertiaRatio: "+str(maxInertiaRatio)+"\n")
            file.write("img_name: "+img_name+"\n"+"number_of_blobs: " + str(number_of_blobs) + "\n\n")

        # Draw detected blobs location on our image with red circles
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(or_image, keypoints, blank, (0, 0, 255), 
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the image
        detection_output_img_id += 1
        detection_img_file_name = "detection_output_img_" + str(new_file_counter) + "_" + str(detection_output_img_id) + ".png"
        cv2.imwrite(generated_images_folder+"/" + detection_img_file_name, blobs)

        # for every keypoint, draw an identification number on top of the circle

        for i in range(len(keypoints)):
            x, y = keypoints[i].pt
            size = keypoints[i].size
            cv2.putText(blobs, str(i+2), (int(x-size/2), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save the image
        detection_img_file_name = "detection_output_img_labeled_" + str(new_file_counter) + "_" + str(detection_output_img_id) + ".png"
        cv2.imwrite(generated_images_folder+"/" + detection_img_file_name, blobs)

        # dump keypoint information to a csv file
        
        # if file does not exist, create it and write the header

        if not os.path.exists(generated_images_folder+"/" + output_file_name + ".csv"):
            with open(generated_images_folder+"/" + output_file_name + ".csv", "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["x", "y", "size"])
        
        with open(generated_images_folder+"/" + output_file_name + ".csv", "a", newline='') as file:
            writer = csv.writer(file)
            for keypoint in keypoints:
                x, y = keypoint.pt
                size = keypoint.size
                writer.writerow([x, y, size])
