package model_structure

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.distribution.{Distribution, GaussianDistribution, NormalDistribution}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.{ConvolutionMode, MultiLayerConfiguration, NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

object AlexNet4test{

  val height:Int=101
  val width:Int=101
  val channels:Int=3
  val numLabels:Int=5
  val seed:Int=123


  val graph : MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .weightInit(new NormalDistribution(0.0, 0.1)) // initial weight
    .activation(Activation.RELU) //default activation
    .updater(new Adam(0.01))
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .miniBatch(false)
    .convolutionMode(ConvolutionMode.Same)
    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
    .list()
    .layer(0, convInit("cnn1", channels, 96, Array[Int](11, 11), Array[Int](4, 4), Array[Int](3, 3), 0))
    .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build)
    .layer(2, maxPool("maxpool1", Array[Int](3, 3)))
    .layer(3, conv5x5("cnn2", 256, Array[Int](1, 1), Array[Int](2, 2), 1))
    .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build)
    .layer(5, maxPool("maxpool2", Array[Int](3, 3)))
    .layer(6, conv3x3("cnn3", 384, 0))
    .layer(7, conv3x3("cnn4", 384, 1))
    .layer(8, conv3x3("cnn5", 256, 1))
    .layer(9, maxPool("maxpool3", Array[Int](3, 3)))
    .layer(10, fullyConnected("ffn1", 4096, 1, 0.5, new GaussianDistribution(0, 0.005)))
    .layer(11, fullyConnected("ffn2", 4096, 1, 0.5, new GaussianDistribution(0, 0.005)))
    .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .name("output")
      .nOut(numLabels)
      .activation(Activation.SOFTMAX)
      .build())
    .backprop(true).pretrain(false)
    .setInputType(InputType.convolutional(height, width, channels))
    .build()

  private def convInit(name: String, in: Int, out: Int, kernel: Array[Int], stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer = {
    new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build()
  }

  private def conv3x3(name: String, out: Int, bias: Double): ConvolutionLayer = {
    new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name(name).nOut(out).biasInit(bias).build()
  }

  private def conv5x5(name: String, out: Int, stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer = {
    new ConvolutionLayer.Builder(Array[Int](5, 5), stride, pad).name(name).nOut(out).biasInit(bias).build()
  }

  private def maxPool(name: String, kernel: Array[Int]): SubsamplingLayer = {
    new SubsamplingLayer.Builder(kernel, Array[Int](2, 2)).name(name).build()
  }

  private def fullyConnected(name: String, out: Int, bias: Double, dropOut: Double, dist: Distribution): DenseLayer = {
    new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build()
  }

  lazy val net = new MultiLayerNetwork(graph)

}
