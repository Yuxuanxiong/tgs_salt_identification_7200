import java.io.File
import java.util.Random

import org.apache.log4j.{BasicConfigurator, Logger}
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{ImageTransform, PipelineImageTransform, ResizeImageTransform, ShowImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}

object load_model_test extends App{

  protected val logger: Logger = Logger.getLogger(load_model_test.getClass)

  BasicConfigurator.configure();

  val height: Int = 101
  val width: Int = 101
  val channel: Int = 3
  val seed1: Long = 123
  val seed2: Long = 42
  val rng1 = new Random(seed1)
  val rng2 = new Random(seed2)
  val splitRate: Double = 0.7
  val batchSize: Int = 32
  val epochs: Int = 50
  val numLabels: Int = 5

  new load_model_test().run()
}


class load_model_test {

  import load_model_test._

  def run() {
    logger.info("Loading data...")

    val labelMaker = new ParentPathLabelGenerator
    val mainPath: String = System.getProperty("user.dir")
    //  val path = new File(System.getProperty("user.dir"), "Dataset/images/images")
    val path = new File("/Users/mindachen/Downloads/flowers/")
    //  val instanceNum :Int = Option(path.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)
    val numImages: Int =
      path.listFiles().filter(_.isDirectory).map(_.list).foldLeft(0)((totalCount, curDir) => totalCount + curDir.filter(_.endsWith(".jpg")).size)

    logger.info(s"$numImages images found.")

    val fileSplit = new FileSplit(path, NativeImageLoader.ALLOWED_FORMATS, rng1)
    val pathFilter = new BalancedPathFilter(rng1, NativeImageLoader.ALLOWED_FORMATS, labelMaker)

    val inputSplit = fileSplit.sample(pathFilter, splitRate, 1 - splitRate)
//    val trainData = inputSplit(0)
    val testData = inputSplit(1)

    logger.info("Transforming inputs...")

    val resizeTransform: ImageTransform = new ResizeImageTransform(101, 101)
    //  val flipTransform1: ImageTransform = new FlipImageTransform(rng1)
    //  val flipTransform2: ImageTransform = new FlipImageTransform(rng2)
    //  val warpTransform: ImageTransform = new WarpImageTransform(rng1,42)
    //  val transPipe = List[ImageTransform](resizeTransform, flipTransform1,flipTransform2,warpTransform).asJava

    val showBefore: ImageTransform = new ShowImageTransform("Display - before ")

    val transPipe :ImageTransform = new PipelineImageTransform(resizeTransform, showBefore)

    val scaler: DataNormalization = new ImagePreProcessingScaler(0, 1)

    val recordReader: ImageRecordReader = new ImageRecordReader(height, width, channel, labelMaker)
    recordReader.initialize(testData, transPipe)
    println(recordReader.nextRecord.getMetaData)


    logger.info("Loading model....")

    println(recordReader.getLabels)
    val dataIter = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
        .classification(1, numLabels)
        .preProcessor(scaler)
        .build

    val testExample: DataSet = dataIter.next()
    println(testExample.getLabels)
    val expectedResult = testExample.getLabelNamesList()
    println(expectedResult.toString)
    val model = ModelSerializer.restoreMultiLayerNetwork(mainPath + "/saved_models/AlexNet4Test.zip")
    val prediction = model.predict(testExample)
    val modelResult = prediction.get(0)
    print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n")
  }

}
