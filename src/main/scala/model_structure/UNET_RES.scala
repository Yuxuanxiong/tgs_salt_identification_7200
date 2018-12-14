package model_structure

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.distribution._
import org.deeplearning4j.nn.conf.graph.{ElementWiseVertex, MergeVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, ConvolutionMode, NeuralNetConfiguration, WorkspaceMode, _}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config._
import org.nd4j.linalg.lossfunctions.LossFunctions



case class UNET_RES(seed:Long,
                channels:Int,
                height:Int,
                width:Int){

  import UNET_RES._

  def init_model: ComputationGraph = {

    val model = new ComputationGraph(init_conf)
    model.init()
    model
  }

  def init_conf: ComputationGraphConfiguration = {

    val graph : ComputationGraphConfiguration.GraphBuilder = graphBuilder(seed)
    graph.addInputs("input").setInputTypes(InputType.convolutional(height,width,channels))
    graph.build
  }

}


object UNET_RES {

  private val seed:Long=12345
  private val weightInit:WeightInit = WeightInit.RELU
  private val initFilterNum = 16
  private val dropOut = 0.4
  private val updater = new AdaDelta
  private val cacheMode = CacheMode.NONE
  private val workspaceMode = WorkspaceMode.ENABLED
  private val cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST

  def graphBuilder(seed:Long) : ComputationGraphConfiguration.GraphBuilder = {

    val builder: ComputationGraphConfiguration.GraphBuilder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .weightInit(weightInit)
      .dist(new TruncatedNormalDistribution(0.0, 0.5))
      .updater(updater)
      .l2(5e-5)
      .miniBatch(true)
      .cacheMode(cacheMode)
      .trainingWorkspaceMode(workspaceMode)
//      .inferenceWorkspaceMode(workspaceMode)
      .graphBuilder()



    var blockName:String = "sub1"

    //subsampling block 1   101->50
    builder
      .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, "input")

      //residual block 1
      .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_conv")
      .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
      .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res1_conv1")
      .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
      .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_conv", blockName+"_res1_conv2")

      //residual block 2
      .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_res1_skip")
      .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
      .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res2_conv1")
      .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
      .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_res1_skip", blockName+"_res2_conv2")
      .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res2_skip")

      //pooling
      .addLayer(blockName+"_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).build,
      blockName+"_res2_BN2")

      //dropout
      .addLayer(blockName+"_dropout", new DropoutLayer.Builder(dropOut/2).build, blockName+"_pool")


    blockName = "sub2"
    //subsampling block 2   50->25
    builder
      .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, "sub1_dropout")

      //residual block 1
      .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_conv")
      .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
      .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res1_conv1")
      .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
      .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_conv", blockName+"_res1_conv2")

      //residual block 2
      .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_res1_conv2")
      .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
      .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res2_conv1")
      .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
      .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_res1_skip", blockName+"_res2_conv2")
      .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res2_skip")

      //pooling
      .addLayer(blockName+"_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).build,
      blockName+"_res2_BN2")

      //dropout
      .addLayer(blockName+"_dropout", new DropoutLayer.Builder(dropOut).build, blockName+"_pool")


    blockName = "sub3"
    //subsampling block 3   25->12
    builder
      .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, "sub2_dropout")

      //residual block 1
      .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_conv")
      .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
      .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res1_conv1")
      .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
      .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_conv", blockName+"_res1_conv2")

      //residual block 2
      .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_res1_conv2")
      .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
      .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res2_conv1")
      .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
      .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_res1_skip", blockName+"_res2_conv2")
      .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res2_skip")

      //pooling
      .addLayer(blockName+"_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).build,
      blockName+"_res2_BN2")

      //dropout
      .addLayer(blockName+"_dropout", new DropoutLayer.Builder(dropOut).build, blockName+"_pool")



    blockName = "sub4"
    //subsampling block 4   12->6
    builder
      .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, "sub3_dropout")

      //residual block 1
      .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_conv")
      .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
      .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res1_conv1")
      .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
      .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_conv", blockName+"_res1_conv2")

      //residual block 2
      .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_res1_conv2")
      .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
      .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res2_conv1")
      .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
      .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_res1_skip", blockName+"_res2_conv2")
      .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res2_skip")

      //pooling
      .addLayer(blockName+"_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).build,
      blockName+"_res2_BN2")

      //dropout
      .addLayer(blockName+"_dropout", new DropoutLayer.Builder(dropOut).build, blockName+"_pool")


    blockName = "middle"
    //middle block
    builder
      .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, "sub4_dropout")

      //residual block 1
      .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_conv")
      .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
      .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res1_conv1")
      .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
      .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_conv", blockName+"_res1_conv2")

      //residual block 2
      .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_res1_skip")
      .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
      .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res2_conv1")
      .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
      .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_res1_skip", blockName+"_res2_conv2")
      .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res2_skip")




    blockName = "up4"
    //upSampling block 1, 6->12
    //The upSampling blocks indexes will be reversed to correspond with subSampling blocks
      builder
        //transpose convolution
        .addLayer(blockName+"_deconv", new Deconvolution2D.Builder(3,3).stride(2,2).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, "middle_res2_BN2")
        //Concatenate
        .addVertex(blockName+"_uconv", new MergeVertex, "up4_deconv", "sub4_res2_BN2")
        //dropout
        .addLayer(blockName+"_dropout", new DropoutLayer.Builder(dropOut).build, blockName+"_uconv")
        //conv
        .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, blockName+"_dropout")

        //residual block 1
        .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_conv")
        .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
        .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
          .activation(Activation.RELU).build,blockName+"_res1_conv1")
        .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
        .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
          blockName+"_conv", blockName+"_res1_conv2")

        //residual block 2
        .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res1_conv2")
        .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
        .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
          .activation(Activation.RELU).build,blockName+"_res2_conv1")
        .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
        .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
          blockName+"_res1_skip", blockName+"_res2_conv2")
        .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
          .activation(Activation.RELU).build, blockName+"_res2_skip")


    blockName = "up3"
    //upSampling block 2,  12->25
    builder
      //transpose convolution
      .addLayer(blockName+"_deconv", new Deconvolution2D.Builder(3,3).stride(2,2).nOut(initFilterNum*4)
      .convolutionMode(ConvolutionMode.Truncate).build, "up4_res2_BN2")
      //Concatenate
      .addVertex(blockName+"_uconv", new MergeVertex, "up3_deconv", "sub3_res2_BN2")
      //dropout
      .addLayer(blockName+"_dropout", new DropoutLayer.Builder(dropOut).build, blockName+"_uconv")
      //conv
      .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*4)
      .convolutionMode(ConvolutionMode.Same).build, blockName+"_dropout")

      //residual block 1
      .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_conv")
      .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
      .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res1_conv1")
      .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
      .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_conv", blockName+"_res1_conv2")

      //residual block 2
      .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_res1_conv2")
      .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
      .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res2_conv1")
      .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
      .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_res1_skip", blockName+"_res2_conv2")
      .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res2_skip")



    blockName = "up2"
    //upSampling block 3,  25->50
    builder
      //transpose convolution
      .addLayer(blockName+"_deconv", new Deconvolution2D.Builder(3,3).stride(2,2).nOut(initFilterNum*2)
      .convolutionMode(ConvolutionMode.Same).build, "up3_res2_BN2")
      //Concatenate
      .addVertex(blockName+"_uconv", new MergeVertex, "up2_deconv", "sub2_res2_BN2")
      //dropout
      .addLayer(blockName+"_dropout", new DropoutLayer.Builder(dropOut).build, blockName+"_uconv")
      //conv
      .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*2)
      .convolutionMode(ConvolutionMode.Same).build, blockName+"_dropout")

      //residual block 1
      .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_conv")
      .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
      .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res1_conv1")
      .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
      .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_conv", blockName+"_res1_conv2")

      //residual block 2
      .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_res1_conv2")
      .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
      .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res2_conv1")
      .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
      .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_res1_skip", blockName+"_res2_conv2")
      .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res2_skip")


    blockName = "up1"
    //upSampling block 4, 50->101
    builder
      //transpose convolution
      .addLayer(blockName+"_deconv", new Deconvolution2D.Builder(3,3).stride(2,2).nOut(initFilterNum)
      .convolutionMode(ConvolutionMode.Truncate).build, "up2_res2_BN2")
      //Concatenate
      .addVertex(blockName+"_uconv", new MergeVertex, "up1_deconv", "sub1_res2_BN2")
      //dropout
      .addLayer(blockName+"_dropout", new DropoutLayer.Builder(dropOut).build, blockName+"_uconv")
      //conv
      .addLayer(blockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum)
      .convolutionMode(ConvolutionMode.Same).build, blockName+"_dropout")

      //residual block 1
      .addLayer(blockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_conv")
      .addLayer(blockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN")
      .addLayer(blockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res1_conv1")
      .addLayer(blockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res1_BN1")
      .addVertex(blockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_conv", blockName+"_res1_conv2")

      //residual block 2
      .addLayer(blockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, blockName+"_res1_conv2")
      .addLayer(blockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN")
      .addLayer(blockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,blockName+"_res2_conv1")
      .addLayer(blockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, blockName+"_res2_BN1")
      .addVertex(blockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        blockName+"_res1_skip", blockName+"_res2_conv2")
      .addLayer(blockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, blockName+"_res2_skip")


    //output
        .addLayer("final_conv_sigmoid", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1)
          .convolutionMode(ConvolutionMode.Same).activation(Activation.SIGMOID).build, "up1_res2_BN2")
        .addLayer("outputs", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
          .build, "final_conv_sigmoid")
        .setOutputs("outputs")


    builder
  }


}
