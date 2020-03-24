import cv2 as cv
import numpy as np
import decode

def getEdges(input, cap, model, inpWidth, inpHeight, confThreshold, nmsThreshold) :
    net = cv.dnn.readNet(model)

    kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    emptyArray = []
    Edges = np.array(emptyArray)
    angles = np.array(emptyArray)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        net.setInput(blob)
        output = net.forward(outputLayers)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = decode.decode(scores, geometry, confThreshold)
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        for i in indices:
            vertices = cv.boxPoints(boxes[i[0]])
            angles = np.append(angles, [boxes[i[0]][2]])
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
                if j % 2 == 1 :
                    Edges = np.append(Edges, [p1[0], p2[1]])
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        Edges = Edges.reshape(int(Edges.size/4), 4)

        cv.imshow(kWinName,frame)
        cv.imwrite("output/out-{}".format(input),frame)
        cv.waitKey()

        return Edges, angles