
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import utils.IoUEvaluator

class IoUEvaluatorSpec extends FlatSpec with Matchers{

  behavior of "IoUEval"
  "IoUEvaluator" should "0.75" in {
    val y = Nd4j.create(Array[Double](1, 0, 1, 1))
    val y_hat = Nd4j.create(Array[Double](1, 0, 1, 0))

    val eval: IoUEvaluator = IoUEvaluator(Seq(y, y, y), Seq(y_hat, y_hat, y_hat))
    assert(eval.eval() == Some(0.75))
  }



}
