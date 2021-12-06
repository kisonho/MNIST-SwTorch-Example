import PythonKit
import SwTorch

@main
struct MNISTTraining {
    public static func main() throws {
        // load core lib
        let sys = Python.import("sys")
        let os = Python.import("os")
        sys.path.insert(0, os.getcwd())
        let models = Python.import("lib.models")
        
        // load model
        let model = PyModule(models.CNN())!
        
        // load dataset
        let datasets = Python.import("lib.datasets")
        let mnist = datasets.loadMNIST()
        let trainingDataset = mnist.tuple2.0
        let valDataset = mnist.tuple2.1
        
        // load optimizer
        let optimizer = PyOptimizer(torch.optim.SGD(model.parameters, lr: 0.01, momentum: 0.9))!
        
        // initialize training manager
        var manager = TrainingManager(model, optimizer, loss: { yTrue, yPred in
            let lossFn = CrossEntropyLoss()
            return lossFn(yTrue: yTrue, yPred: yPred)
        }, metrics: { yTrue, yPred in
            let accFn = SparseCategoricalAccuracy()
            let acc = accFn(yTrue: yTrue, yPred: yPred)
            return ["accuracy": acc]
        }, device: .cuda, useMultiGPUs: false)
        let results = try manager.train(trainingDataset: trainingDataset, epochs: 10, validationDataset: valDataset)!
        print("Final results: loss=\(results["loss"]!), acc=\(results["accuracy"]!)")
    }
}
