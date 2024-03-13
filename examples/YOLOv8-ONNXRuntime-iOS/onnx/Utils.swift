//
//  Utils.swift
//  onnx
//
//  Created by Pradeep Banavara on 05/03/24.
//

import Foundation
import SwiftUI
import onnxruntime_objc
import UIKit

/**
    ### This function accepts an UIImage and renders the detected pose points on the said image.
 *  It is key to use the correct model for the said purpose. Use the [Model generation] (https://onnxruntime.ai/docs/tutorials/mobile/pose-detection.html)
 *  It is also key to register the customOps function using the BridgingHeader
 */
func plotPose(image: UIImage) -> UIImage{
    do {
        guard let modelPath = Bundle.main.path(forResource: "yolov8n-pose-pre", ofType: "onnx") else {
            fatalError("Model file not found")
        }
        let inputData = image.pngData()!
        let inputDataLength = inputData.count
        let inputShape = [NSNumber(integerLiteral: inputDataLength)]
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData), elementType:ORTTensorElementDataType.uInt8, shape:inputShape)
        let ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.info)
        let ortSessionOptions = try ORTSessionOptions()
        try ortSessionOptions.registerCustomOps(functionPointer: RegisterCustomOps) // Register the bridging-header in Build settings
        let ortSession = try ORTSession(
            env: ortEnv, modelPath: modelPath, sessionOptions: ortSessionOptions)
        let inputNames = try ortSession.inputNames()    // The input names should match the model input names. Visualize the model in Netron
        let outputNames = try ortSession.outputNames()  // Check the model outnames in Netron
        let outputs = try ortSession.run(
            withInputs: [inputNames[0]: inputTensor], outputNames: Set(outputNames), runOptions: nil)
        
        guard let outputTensor = outputs[outputNames[0]] else {
            fatalError("Failed to get model keypoint output from inference.")
        }
        return try convertOutputTensorToImage(opTensor: outputTensor, inputImageData: inputData)
        
    } catch {
        print(error)
        fatalError("Error in running the ONNX model")
    }
}

/**
    Helper function to convert the output tensor into an image with the bounding box and keypoint data.
 */
private func convertOutputTensorToImage(opTensor: ORTValue, inputImageData: Data) throws -> UIImage{
    
    let output = try opTensor.tensorData()
    var arr2 = Array<Float32>(repeating: 0, count: output.count/MemoryLayout<Float32>.stride)   // Do not change the datatype Float32
    _ = arr2.withUnsafeMutableBytes { output.copyBytes(to: $0) }
    
    var keypoints:[Float32] = Array()
    
    // 57 is hardcoded based on the keypoints returned from the model. Refer to the Netron visualisation for the output shape
    for i in stride(from: arr2.count-57, to: arr2.count, by: 1) {
        keypoints.append(arr2[i])
    }
    let box = keypoints[0..<4] // The first 4 points are the bounding box co-ords.
    // Refer yolov8_pose_e2e.py run_inference method under the https://onnxruntime.ai/docs/tutorials/mobile/pose-detection.html
    let half_w = box[2] / 2
    let half_h = box[3] / 2
    let x = Double(box[0] - half_w)
    let y = Double(box[1] - half_h)
    
    
    let rect = CGRect(x: x, y: y, width: Double(half_w * 2), height: Double(half_h * 2))
    NSLog("Rect is \(rect)")
    let image:UIImage = UIImage(data: inputImageData) ?? UIImage()
    let keypointsWithoutBoxes = Array(keypoints[6..<keypoints.count]) // Based on 17 keypoints and 3 entries per keypoint x,y,confidence
    return drawKeyPointsOnImage(image: image, rectangle: rect, keypoints: keypointsWithoutBoxes)
}

/**
    Helper function takes an input image and a boundding box CGRect along with the keypoints data to return a new image with the rect and keypoints drawn/
    TODO: // Optimize on generating a new image instead paint the data on the same image. iOS experts to chime in.
 
 */
private func drawKeyPointsOnImage(image: UIImage, rectangle:CGRect, keypoints: [Float32]) -> UIImage {
    let imageSize = image.size
    let scale: CGFloat = 0
    UIGraphicsBeginImageContextWithOptions(imageSize, false, scale)
    image.draw(at: CGPoint.zero)
    UIColor.red.setFill()
    UIColor.red.setStroke()
    UIRectFrame(rectangle)
    
    for i in stride(from: 0, through: keypoints.count-1, by: 3) {
        let kp_x = keypoints[i]
        let kp_y = keypoints[i+1]
        let confidence = keypoints[i+2]
        if (confidence < 0.5) { // Can potentially remove hardcoding and make the confidence configurable
            continue
        }
        let rect = CGRect(x: Double(kp_x), y: Double(kp_y), width: 20.0, height: 20.0)
        UIRectFill(rect)
        
    }
    
    let newImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return newImage!
}


