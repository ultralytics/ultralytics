//
//  ContentView.swift
//  onnx
//
//  Created by Pradeep Banavara on 05/03/24.
//  This is a demo app for pose detection on iOS using ONNX models.
//  This simple view is for selecting an image and the image is rendered with
//  detected pose points and the bounding box. Currently works for single images
//  even though the model is tested for multiple person classes.
//
//
import SwiftUI
import CoreData
import UIKit
import AVKit
import PhotosUI


struct ContentView: View {
    @State private var item: PhotosPickerItem?
    @State private var itemWithPose: Image?
    
    var body: some View {
        VStack (alignment: .trailing) {
            PhotosPicker("Select image", selection: $item, matching: .images)
            itemWithPose?.resizable().scaledToFit()
        }
        .onChange(of: item) {
            Task {
                if let loaded = try? await item?.loadTransferable(type: Image.self) {
                    itemWithPose = Image(uiImage: plotPose(image: loaded.render()!))
                    
                } else {
                    print("Failed to load image")
                }
            }
        }
    }
}
/**
 
 * This is an anti pattern as per Swift guidelines to convert Image which is a view to UIView which is a class.
 *   There's room for improvement here to follow the right pattern.
 */
extension View {
    /// Usually you would pass  `@Environment(\.displayScale) var displayScale`
    @MainActor func render(scale displayScale: CGFloat = 1.0) -> UIImage? {
        let renderer = ImageRenderer(content: self)

        renderer.scale = displayScale
        
        return renderer.uiImage
    }
    
}

#Preview {
    ContentView()
}
