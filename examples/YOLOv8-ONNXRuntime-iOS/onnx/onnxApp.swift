//
//  onnxApp.swift
//  onnx
//
//  Created by Pradeep Banavara on 05/03/24.
//

import SwiftUI

@main
struct onnxApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
