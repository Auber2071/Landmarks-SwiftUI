//
//  LandmarkList.swift
//  Landmarks
//
//  Created by ALPS on 2023/3/10.
//

import SwiftUI

struct LandmarkList: View {
    @EnvironmentObject var modelData: ModelData
    @State private var showFavoriteOnly = false
    
    var filteredLandmarks: [Landmark] {
        modelData.landmarks.filter { landmark in
            (!showFavoriteOnly || landmark.isFavorite)
        }
    }
    
    var body: some View {
        
        NavigationView {
            List {
                Toggle(isOn: $showFavoriteOnly) {
                    Text("Favorites only")
                }
                
                ForEach(filteredLandmarks) { landmark in
                    NavigationLink {
                        LandmarkDetail(landmark: landmark)
                    } label: {
                        LandmarkRow(landmark: landmark)
                    }
                }
            }
            .navigationTitle("Landmarks")
        }
    }
}

struct LandmarkList_Previews: PreviewProvider {
    static var previews: some View {
        LandmarkList()
        /*
         ForEach(["iPhone SE (2nd generation)", "iPhone XS Max"], id: \.self) { deviceName in
         LandmarkList()
         .previewDevice(PreviewDevice(rawValue: deviceName))
         .previewDisplayName(deviceName)
         }
         */
    }
}
