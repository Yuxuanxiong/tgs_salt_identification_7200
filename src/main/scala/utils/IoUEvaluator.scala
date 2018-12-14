package utils

import org.apache.log4j.{BasicConfigurator, Logger}
import org.nd4j.linalg.api.ndarray.INDArray

import scala.util.{Failure, Success, Try}

case class IoUEvaluator(ys :Seq[INDArray],
                        y_hats :Seq[INDArray]) {

  //logger
  val log: Logger = Logger.getLogger(this.getClass)
  BasicConfigurator.configure()

  //try to zip input sequences
  lazy val pairs:Seq[(INDArray,INDArray)]= {
    if(ys.length!=y_hats.length) throw new IllegalArgumentException("Input sequences must have same size!")
    else ys.zip(y_hats)
  }

  // calculate IoU for 2 INDArrays
  def eval(y:INDArray, y_hat:INDArray) : Double = {
    if(!y.equalShapes(y_hat)) throw new IllegalArgumentException("Inputs must have same shape!")
    y.eps(y_hat).sumNumber().doubleValue()/y.length
  }

  // calculate average IoU for all INDArrays
  def eval(): Option[Double] = Try(pairs) match {
    case Success(pairs:Seq[(INDArray,INDArray)]) => {
      // create an Seq[Option[Double]], flatten it and calculate average
      // use option to simply drop those INDArrays that throw exceptions during evaluation
      val os:Seq[Double]= pairs.flatMap(x => Try(eval(x._1, x._2)) match {
          case Success(v) => Some(v)
          case Failure(e) => log.info(s"Fail to eval Arrays at index ${pairs.indexOf(x)}", e); None
      })
      Some(os.sum/os.length)
    }
    case Failure(e) => log.info("Fail to zip values", e); None
  }

}
