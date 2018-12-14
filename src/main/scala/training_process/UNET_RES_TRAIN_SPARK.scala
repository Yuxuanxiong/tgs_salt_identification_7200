package training_process

import java.io.File

import model_structure.UNET_RES
import org.apache.log4j.{BasicConfigurator, Logger}
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.datavec.image.loader.{Java2DNativeImageLoader, NativeImageLoader}
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.FileStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode

import scala.util.{Failure, Success, Try}

object UNET_RES_TRAIN_SPARK{

  def main(args:Array[String]):Unit = {

    val log: Logger = Logger.getLogger(UNET_RES_TRAIN_SPARK.getClass)
    BasicConfigurator.configure()

    val sparkConf: SparkConf = new SparkConf()
      .setAppName("7200_Salt_Identification")
      .setMaster("local[4]")
      .set("spark.ui.port", "18080")


    val sc = SparkContext.getOrCreate(sparkConf)

    val height: Int = 101
    val width: Int = 101
    val channels: Int = 1
    val seed: Long = 12345
    val splitRate: Double = 0.7


    implicit def OStoInt(os:Option[String]):Option[Int]=os.flatMap(s=>Try(s.toInt).toOption)

    //Get args using Try and Option
    def getArgs(args:Array[String], index:Int):Option[String]={
      Try(args(index)) match {
        case Success(s)=>Some(s)
        case Failure(_)=>None
      }
    }
//
//    val mainPath: Option[String] = getArgs(args,0)
//    val epochs: Option[Int] = getArgs(args,1)
//    val batchSize: Option[Int] = getArgs(args,2)
//    val trainMonitorLog: Option[String] = getArgs(args,3)
//    val modelSavingPath: Option[String] = getArgs(args,4)

    val mainPath: String = "Dataset"
    val epochs: Option[Int] = Some(5)
    val batchSize:  Option[Int] = Some(16)
    val trainMonitorLog: Option[String] = Some("")
    val modelSavingPath: Option[String] = Some("/saved_model")


    val imgDir = mainPath + "/train/images/"
    val maskDir = mainPath + "/train/masks/"

    val unet: UNET_RES = UNET_RES(seed, channels, height, width)
    val model = Try(unet.init_model)


    val imgs: RDD[(String, PortableDataStream)] = sc.binaryFiles(imgDir,4)
    val masks: RDD[(String, PortableDataStream)] = sc.binaryFiles(maskDir,4)


    val features_RDD: RDD[(String, INDArray)] = imgs.map(
      v=>(v._1.split("/").last,
        (new Java2DNativeImageLoader(height, width, channels) : NativeImageLoader).asMatrix(v._2.open()).div(255))
    )

    val masks_RDD: RDD[(String, INDArray)] = imgs.map(
      v=>(v._1.split("/").last,
        (new Java2DNativeImageLoader(height, width, channels) : NativeImageLoader).asMatrix(v._2.open()).div(65535))
    )

    val dataset_RDD = new JavaRDD(features_RDD.join(masks_RDD).map[DataSet](x => new DataSet(x._2._1, x._2._2)))


    // Configure distributed training required for gradient sharing implementation
    val conf = VoidConfiguration.builder()
      .unicastPort(40123)
      .networkMask("10.0.0.0/16")
      .controllerAddress("10.0.2.4")
      .meshBuildMode(MeshBuildMode.MESH)
      .build()

    //Create the TrainingMaster instance
    val trainingMaster = new SharedTrainingMaster.Builder(conf, dataset_RDD.count().toInt)
      .batchSizePerWorker(batchSize.getOrElse(16)) //Batch size for training, default 16
      .thresholdAlgorithm(new AdaptiveThresholdAlgorithm(1e-4))
      .workersPerNode(4)
      .collectTrainingStats(true)
      .build()

    val jsc = new JavaSparkContext(sc)

    //Create the SparkDl4jComputationGraph instance
    val UNET_RES_SPARK: SparkComputationGraph = new SparkComputationGraph(jsc, model.get, trainingMaster)
    val remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9001")
    val statsStorage :Try[StatsStorage] = Try(new FileStatsStorage(new File("")))

    statsStorage match {
      case Success(ss) => UNET_RES_SPARK.setListeners(new StatsListener(ss))
      case Failure(e)=> log.info("Fail to create statsStorage, not using it", e)
    }

    epochs match {
      case Some(epoch) => {
        for (i <- 1 until epoch) {
          log.info(s"Epoch $i start")
          UNET_RES_SPARK.fit(dataset_RDD)
        }
      }
      case None => {
        log.info(s"Epoch number not given, using default 20")
        for (i <- 1 until 20) {
          log.info(s"Epoch $i start")
          UNET_RES_SPARK.fit(dataset_RDD)
        }
      }
    }

      modelSavingPath match {
        case Some(s) => ModelSerializer.writeModel(UNET_RES_SPARK.getNetwork, s, false)
        case None => log.info(s"Saving path not given, fail to save model.")
      }
    }



}
