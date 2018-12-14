import java.io.File
import java.util.Random

import model_structure.AlexNet4test
import org.apache.log4j.{BasicConfigurator, Logger}
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform._
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}

object AlexNetTestTraining extends App {
  protected val logger:Logger = Logger.getLogger(AlexNetTestTraining.getClass)
  BasicConfigurator.configure()

  val height:Int=101
  val width:Int=101
  val channel:Int=3
  val seed1:Long=123
  val seed2:Long=42
  val rng1=new Random(seed1)
  val rng2=new Random(seed2)
  val splitRate:Double=0.7
  val batchSize:Int=32
  val epochs:Int=50
  val numLabels:Int=5

  logger.info("Loading data...")

  val labelMaker = new ParentPathLabelGenerator
  val mainPath:String = System.getProperty("user.dir")
//  val path = new File(System.getProperty("user.dir"), "Dataset/images/images")
  val path = new File("/Dataset/train/images")
//  val instanceNum :Int = Option(path.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)
  val numImages:Int = path.listFiles().filter(_.isDirectory).map(_.list).foldLeft(0)((totalCount, curDir)=>totalCount+curDir.filter(_.endsWith(".png")).size)

  logger.info(s"$numImages images found.")

  val fileSplit = new FileSplit(path, NativeImageLoader.ALLOWED_FORMATS, rng1)
  val pathFilter = new BalancedPathFilter(rng1, labelMaker, numImages, numLabels, batchSize)

  val inputSplit = fileSplit.sample(pathFilter, numImages*(1+splitRate), numImages*(1-splitRate))
  val trainData = inputSplit(0)
  val testData = inputSplit(1)

  logger.info("Transforming inputs...")

  val resizeTransform: ImageTransform = new ResizeImageTransform(101,101)
  val flipTransform1: ImageTransform = new FlipImageTransform(rng1)
  val flipTransform2: ImageTransform = new FlipImageTransform(rng2)
  val warpTransform: ImageTransform = new WarpImageTransform(rng1,42)
  val transPipe :ImageTransform = new PipelineImageTransform(resizeTransform, flipTransform1, flipTransform2, warpTransform)

  val scaler:DataNormalization = new ImagePreProcessingScaler(0,1)

  logger.info("Building model...")

  AlexNet4test.net.init()
  println(AlexNet4test.net.summary())
  AlexNet4test.net.setListeners(new ScoreIterationListener(1))


  val recordReader: ImageRecordReader = new ImageRecordReader(height, width, channel, labelMaker)
  var dataIter: DataSetIterator = null

  logger.info("Training start....")

  recordReader.initialize(trainData, transPipe)
  dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
  println(dataIter.getLabels)
  println("it wont be long")
  scaler.fit(dataIter)
  dataIter.setPreProcessor(scaler)
  AlexNet4test.net.fit(dataIter, epochs)


  logger.info("Evaluating model....")

  recordReader.initialize(testData, resizeTransform)
  dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
  scaler.fit(dataIter)
  dataIter.setPreProcessor(scaler)
  val eval: Evaluation = AlexNet4test.net.evaluate(dataIter)

  logger.info(eval.stats(true))

  logger.info("Saving model....")

  val modelDir = new File(mainPath,"saved_models/")
  if(!modelDir.exists()) modelDir.mkdir()

  val saveTo :File = new File(mainPath, "saved_models/AlexNet4Test.zip")
  saveTo.createNewFile()
  ModelSerializer.writeModel(AlexNet4test.net, saveTo, false)

  dataIter.reset()
  val testExample: DataSet = dataIter.next()

  val expectedResult = testExample.getFeatures()
  val prediction = AlexNet4test.net.predict(testExample)
  val modelResult = prediction.get(0)
  print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n")
}
