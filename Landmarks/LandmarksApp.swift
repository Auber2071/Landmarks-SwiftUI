//
//  LandmarksApp.swift
//  Landmarks
//
//  Created by ALPS on 2023/3/10.
//

import SwiftUI

@main
struct LandmarksApp: App {
    @StateObject private var modelData = ModelData()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelData)
        }
    }
}
