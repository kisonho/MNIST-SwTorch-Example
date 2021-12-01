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
        
        // initialize training manager
        var manager = MNISTTrainingManager(model)
        let results = try manager.train(trainingDataset: trainingDataset, epochs: 10, validationDataset: valDataset)!
        print("Final results: loss=\(results["loss"]!), acc=\(results["accuracy"]!)")
    }
}

/// The main training manager
class MNISTTrainingManager: Training {
    
    typealias LrSchedulerType = ExponentionLr<OptimizerType>
    
    typealias OptimizerType = PyOptimizer
    
    typealias ModuleType = PyModule
    
    var device: Device
    
    /// The loss function
    var lossFn = CrossEntropyLoss()
    
    var lrScheduler: ExponentionLr<OptimizerType>? = nil
    
    var model: PyModule
    
    var optimizer: PyOptimizer
    
    var useMultiGPUs: Bool = false
    
    /// Constructor
    /// - Parameters:
    ///   - model: Target `PyModule`
    ///   - device: Target `Device` of module
    init(_ model: PyModule, device: Device = .cuda) {
        self.model = model
        self.model.to(device)
        self.device = device
        self.optimizer = PyOptimizer(torch.optim.SGD(self.model.parameters, lr: 0.01, momentum: 0.9))!
    }
    
    func calculateMetrics(yTrue: Tensor, yPred: Tensor) -> [String : Float] {
        let y = yPred.argmax(dim: 1)
        let acc = Float(y.equal(yTrue).to(dtype: .float32).mean())!
        return ["accuracy": acc]
    }
    
    func calculateLoss(yTrue: Tensor, yPred: Tensor) -> Tensor {
        return lossFn(yTrue: yTrue, yPred: yPred)
    }
    
    func onBatchEnd(batch: Int, result: [String : Float]) {
        return
    }
    
    func onEpochStart(epoch: Int, totalEpochs: Int) {
        print("Training \(epoch + 1)/\(totalEpochs)")
        model.train()
    }
    
    func onEpochEnd(epoch: Int, totalEpochs: Int, trainingResult: [String : Float], valResult: [String : Float]?) -> Bool {
        print("Epoch \(epoch + 1)/\(totalEpochs): loss=\(valResult!["loss"]!), acc=\(valResult!["accuracy"]!)")
        return true
    }
    
    func onValStart() {
        model.eval()
        return
    }
    
    func trainStep(_ xTrain: Tensor, _ yTrain: Tensor) -> [String : Float] {
        // forward pass
        let y = model(xTrain)
        let loss = calculateLoss(yTrue: yTrain, yPred: y)
        
        // backward pass
        optimizer.zeroGrad(setToNone: false)
        loss.backward()
        optimizer.step()
        
        // summarize
        var summary = calculateMetrics(yTrue: yTrain, yPred: y)
        summary["loss"] = Float(loss.mean())
        return summary
    }
    
    func valStep(_ xTest: Tensor, _ yTest: Tensor) -> [String : Float] {
        // forward pass
        let y = model(xTest)
        let loss = calculateLoss(yTrue: yTest, yPred: y)
        
        // summarize
        var summary = calculateMetrics(yTrue: yTest, yPred: y)
        summary["loss"] = Float(loss.mean())
        return summary
    }
}
