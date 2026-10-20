// Sign reader, step 1 — on-device OCR over every street view (Apple Vision).
// Usage: swift scripts/ocr_signs.swift [data/imagery] [data/ocr_text.jsonl]
// One JSON line per image: {"image": "<vbo>_<heading>.jpg", "text": ["...", ...]}
// Resumable: images already present in the output file are skipped.
import Foundation
import Vision

let args = CommandLine.arguments
let imgDir = URL(fileURLWithPath: args.count > 1 ? args[1] : "data/imagery")
let outURL = URL(fileURLWithPath: args.count > 2 ? args[2] : "data/ocr_text.jsonl")

var done = Set<String>()
if let existing = try? String(contentsOf: outURL, encoding: .utf8) {
    for line in existing.split(separator: "\n") {
        if let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
           let name = obj["image"] as? String { done.insert(name) }
    }
}

let files = (try FileManager.default.contentsOfDirectory(atPath: imgDir.path))
    .filter { $0.hasSuffix(".jpg") && !done.contains($0) }
    .sorted()
FileManager.default.createFile(atPath: outURL.path, contents: nil, attributes: nil)
guard let out = try? FileHandle(forWritingTo: outURL) else { fatalError("cannot open \(outURL.path)") }
out.seekToEndOfFile()
let lock = NSLock()
var processed = 0
let start = Date()
print("OCR: \(files.count) to read, \(done.count) already done")

DispatchQueue.concurrentPerform(iterations: files.count) { i in
    let name = files[i]
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.recognitionLanguages = ["nl-NL", "en-US"]
    req.usesLanguageCorrection = false
    var text: [String] = []
    let handler = VNImageRequestHandler(url: imgDir.appendingPathComponent(name), options: [:])
    if (try? handler.perform([req])) != nil {
        text = (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    }
    let line = try! JSONSerialization.data(withJSONObject: ["image": name, "text": text])
    lock.lock()
    out.write(line); out.write(Data("\n".utf8))
    processed += 1
    if processed % 250 == 0 {
        print("  \(processed)/\(files.count)  (\(Int(Date().timeIntervalSince(start)))s)")
    }
    lock.unlock()
}
out.closeFile()
print("OCR done: \(processed) images in \(Int(Date().timeIntervalSince(start)))s → \(outURL.path)")
