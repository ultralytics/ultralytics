// On-device SKU recognition: detect products in a shelf photo with the YOLO CoreML detector, embed each
// crop with the ReID CoreML model, and assign it to a folder-per-SKU gallery by a top-k cosine vote.
// This is the iOS/CoreML counterpart of the Python sku_recognition.py --source pipeline.
//
// Usage: swift sku_recognition.swift <detector.mlpackage> <reid.mlpackage> <gallery_dir> <shelf_image>
//   gallery_dir: each immediate subfolder is one SKU holding a few reference crops.
// Writes an annotated <name>_sku.jpg next to the shelf image and logs each detection.

import CoreGraphics
import CoreML
import CoreText
import Foundation
import ImageIO
import Vision

struct RunError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// Load a Core ML model on the Neural Engine + CPU, with the GPU excluded. This matches the official
/// Ultralytics yolo-ios-app default (which frees the GPU for camera preview) and sidesteps a current
/// coremltools 9.x issue where the .all path does not compile on the GPU, so keep .cpuAndNeuralEngine.
func loadModel(_ modelURL: URL) throws -> MLModel {
    let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: compiled, configuration: config)
}

/// Run a single-output model on one image with the given crop/scale option and return its output tensor.
func runVision(_ model: VNCoreMLModel, on image: CGImage, cropAndScale: VNImageCropAndScaleOption) throws -> MLMultiArray {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = cropAndScale
    try VNImageRequestHandler(cgImage: image).perform([request])
    guard let array = (request.results as? [VNCoreMLFeatureValueObservation])?.first?.featureValue.multiArrayValue
    else { throw RunError("model produced no output") }
    return array
}

func loadCGImage(_ url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { throw RunError("cannot read image \(url.lastPathComponent)") }
    return image
}

/// YOLO26 end2end detector: returns product boxes in original-image pixel coordinates.
final class SKUDetector {
    private let visionModel: VNCoreMLModel
    private let inputSize: CGFloat  // model input side, read from the model so it tracks the export imgsz

    init(modelURL: URL) throws {
        let model = try loadModel(modelURL)
        visionModel = try VNCoreMLModel(for: model)
        let imageInput = model.modelDescription.inputDescriptionsByName.values.first { $0.imageConstraint != nil }
        inputSize = CGFloat(imageInput?.imageConstraint?.pixelsWide ?? 640)  // square input, e.g. 640x640
    }

    func detect(_ image: CGImage, conf: Float) throws -> [CGRect] {
        let array = try runVision(visionModel, on: image, cropAndScale: .scaleFit)  // scaleFit letterboxes like Ultralytics

        // Reverse the letterbox: model coords live in a centered inputSize square, scaled by r.
        let w = CGFloat(image.width), h = CGFloat(image.height)
        let r = inputSize / max(w, h)
        let padX = (inputSize - w * r) / 2, padY = (inputSize - h * r) / 2
        let numDet = array.shape[1].intValue  // 300 candidate rows, [x1, y1, x2, y2, conf, class]

        var boxes = [CGRect]()
        for i in 0 ..< numDet {
            func field(_ j: Int) -> CGFloat { CGFloat(array[[0, i, j] as [NSNumber]].floatValue) }
            guard field(4) > CGFloat(conf) else { continue }
            let x1 = (field(0) - padX) / r, y1 = (field(1) - padY) / r
            let x2 = (field(2) - padX) / r, y2 = (field(3) - padY) / r
            boxes.append(CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1))
        }
        return boxes
    }
}

/// ReID model: turns an image (crop or reference) into an L2-normalized embedding.
final class ReIDEmbedder {
    private let visionModel: VNCoreMLModel

    init(modelURL: URL) throws { visionModel = try VNCoreMLModel(for: try loadModel(modelURL)) }

    /// Vision's centerCrop plus the model's baked-in 1/255 scale reproduce the Ultralytics reid transform
    /// (resize shortest edge, center crop, /255 RGB).
    func embed(_ image: CGImage) throws -> [Float] {
        let array = try runVision(visionModel, on: image, cropAndScale: .centerCrop)
        var vector = (0 ..< array.count).map { array[$0].floatValue }
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        if norm > 0 { vector = vector.map { $0 / norm } }
        return vector
    }
}

/// In-memory reference gallery with a top-k similarity-weighted vote, mirroring the Python assign().
struct SKUGallery {
    let labels: [String]
    let embeddings: [[Float]]

    func assign(_ query: [Float], topK: Int = 5, simThresh: Float = 0.5) -> (sku: String, confidence: Float) {
        let scores = embeddings.map { dot($0, query) }  // cosine, vectors are unit length
        let neighbors = scores.indices.sorted { scores[$0] > scores[$1] }.prefix(topK)
        var total = [String: Float](), count = [String: Int]()
        for i in neighbors {
            total[labels[i], default: 0] += scores[i]
            count[labels[i], default: 0] += 1
        }
        guard let best = total.max(by: { $0.value < $1.value })?.key else { return ("unknown", 0) }
        let confidence = total[best]! / Float(count[best]!)  // mean similarity of the winning SKU's neighbors
        return (confidence >= simThresh ? best : "unknown", confidence)
    }
}

func dot(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }

func crop(_ image: CGImage, to rect: CGRect) -> CGImage? {
    let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
    let box = rect.integral.intersection(bounds)
    guard !box.isNull, box.width >= 1, box.height >= 1 else { return nil }
    return image.cropping(to: box)
}

/// Draw each box and its "sku conf" label onto the shelf image and save it as a JPEG.
func saveAnnotated(_ image: CGImage, results: [(rect: CGRect, sku: String, conf: Float)], to url: URL) throws {
    let w = image.width, h = image.height
    guard let ctx = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { throw RunError("cannot create drawing context") }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))  // context origin is bottom-left
    ctx.setLineWidth(max(2, CGFloat(w) / 500))
    let fontSize = max(10, CGFloat(w) / 120)
    let font = CTFontCreateWithName("Helvetica-Bold" as CFString, fontSize, nil)
    let knownColor = CGColor(red: 0.2, green: 0.8, blue: 0.3, alpha: 1)
    let unknownColor = CGColor(red: 0.6, green: 0.6, blue: 0.6, alpha: 1)

    for res in results {
        let color = res.sku == "unknown" ? unknownColor : knownColor
        let flippedY = CGFloat(h) - res.rect.origin.y - res.rect.height  // top-left -> bottom-left
        ctx.setStrokeColor(color)
        ctx.stroke(CGRect(x: res.rect.origin.x, y: flippedY, width: res.rect.width, height: res.rect.height))

        let attrs = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: color] as CFDictionary
        let text = CFAttributedStringCreate(nil, "\(res.sku) \(String(format: "%.2f", res.conf))" as CFString, attrs)!
        ctx.textPosition = CGPoint(x: res.rect.origin.x, y: flippedY + res.rect.height + 2)
        CTLineDraw(CTLineCreateWithAttributedString(text), ctx)
    }
    guard let out = ctx.makeImage(),
        let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil)
    else { throw RunError("cannot write \(url.lastPathComponent)") }
    CGImageDestinationAddImage(dest, out, nil)
    guard CGImageDestinationFinalize(dest) else { throw RunError("failed to finalize \(url.lastPathComponent)") }
}

// MARK: - pipeline

let args = CommandLine.arguments
guard args.count == 5 else {
    FileHandle.standardError.write(Data("usage: swift sku_recognition.swift <detector.mlpackage> <reid.mlpackage> <gallery_dir> <shelf_image>\n".utf8))
    exit(2)
}
let detector = try SKUDetector(modelURL: URL(fileURLWithPath: args[1]))
let embedder = try ReIDEmbedder(modelURL: URL(fileURLWithPath: args[2]))
let galleryDir = URL(fileURLWithPath: args[3])
let shelfURL = URL(fileURLWithPath: args[4])

// Build the folder-per-SKU gallery: embed every reference image once.
let imageExts: Set<String> = ["jpg", "jpeg", "png", "webp", "bmp"]
var labels: [String] = [], embeddings: [[Float]] = []
for skuDir in try FileManager.default.contentsOfDirectory(at: galleryDir, includingPropertiesForKeys: [.isDirectoryKey]).sorted(by: { $0.path < $1.path }) {
    guard (try? skuDir.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
    for image in try FileManager.default.contentsOfDirectory(at: skuDir, includingPropertiesForKeys: nil)
        .filter({ imageExts.contains($0.pathExtension.lowercased()) }).sorted(by: { $0.path < $1.path }) {
        embeddings.append(try embedder.embed(try loadCGImage(image)))
        labels.append(skuDir.lastPathComponent)
    }
}
let gallery = SKUGallery(labels: labels, embeddings: embeddings)
print("gallery: \(labels.count) reference images across \(Set(labels).count) SKUs")

// Detect, embed each crop, assign against the gallery.
let shelf = try loadCGImage(shelfURL)
let boxes = try detector.detect(shelf, conf: 0.25)
var results: [(rect: CGRect, sku: String, conf: Float)] = []
for box in boxes {
    guard let patch = crop(shelf, to: box) else { continue }
    let (sku, conf) = gallery.assign(try embedder.embed(patch))
    results.append((box, sku, conf))
}

let outName = shelfURL.deletingPathExtension().lastPathComponent + "_sku.jpg"  // matches the Python output
let outURL = shelfURL.deletingLastPathComponent().appendingPathComponent(outName)
try saveAnnotated(shelf, results: results, to: outURL)
print("detected \(results.count) products, saved \(outURL.lastPathComponent)")
