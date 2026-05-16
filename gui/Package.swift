// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DashStack",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "DashStack", targets: ["DashStack"]),
    ],
    targets: [
        .executableTarget(
            name: "DashStack",
            path: "Sources/DashStack"
        ),
    ]
)
