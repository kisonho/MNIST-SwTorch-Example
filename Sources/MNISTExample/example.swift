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
        let results = try manager.train(trainingDatasetLoader: trainingDataset, epochs: 10, validationDatasetLoader: valDataset)!
        print("Final results: loss=\(results["loss"]!), acc=\(results["accuracy"]!)")
    }
}

/// The main training manager
class MNISTTrainingManager: TrainingManager {
    typealias LrSchedulerType = ExponentionLr<OptimizerType>
    
    typealias OptimizerType = PyOptimizer
    
    typealias ModuleType = PyModule
    
    var device: Device
    
    /// The PyTorch loss function
    var lossFn = torch.nn.CrossEntropyLoss().to(torch.device("cuda"))
    
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
        let y = yPred.argmax(axis: 1)
        let acc = Float(torch.eq(y, yTrue).float().mean())!
        return ["accuracy": acc]
    }
    
    func calculateLoss(yTrue: Tensor, yPred: Tensor) -> Tensor {
        return Tensor(lossFn(yPred, yTrue))
    }
    
    func onBatchEnd(batch: Int, result: [String : Float]) {
        return
    }
    
    func onEpochStart(epoch: Int, totalEpochs: Int) {
        print("Training \(epoch + 1)/\(totalEpochs)")
    }
    
    func onEpochEnd(epoch: Int, totalEpochs: Int, trainingResult: [String : Float], valResult: [String : Float]?) -> Bool {
        print("Epoch \(epoch + 1)/\(totalEpochs): loss=\(valResult!["loss"]!), acc=\(valResult!["accuracy"]!)")
        return true
    }
}
