package SVM

import scala.collection.mutable.HashMap
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import SVMLabel._

import breeze.linalg.{ DenseMatrix => BDM }
import breeze.linalg.{ DenseVector => BDV }
import breeze.linalg.{ Matrix => BM }
import breeze.linalg.{ Vector => BV }
import breeze.linalg.{ axpy => brzAxpy }
import scala.collection.mutable.ArrayBuffer

case class PredictSVM(pre: BDM[Double],err: Double) extends Serializable

class SvmModel(
  val config: SVMConfig,
  val alpha: BDM[Double],
  val initb: Double) extends Serializable{ 
  /**
   * 返回预测结果
   *  返回格式：(label, feature,  predict_label, error)
   */
  def predict(data: RDD[(breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double])]): RDD[PredictSVM] = {
    val sc = data.sparkContext
//    val svm_W = sc.broadcast(weights)
    val svm_A = sc.broadcast(alpha)
    val b = sc.broadcast(initb)
    val svmconfig = sc.broadcast(config)
    val train_svm = SVMO.SVM(data, svmconfig, svm_A, b)
    val predict = train_svm.map{ f => 
      val pre = f._1.pre
      val err = f._1.err
      
      PredictSVM(pre,err)
    }
    predict
  }

//  /**
//   * 计算输出误差
//   * 平均误差;
//   */
//  def Loss(predict: RDD[PredictNNLabel]): Double = {
//    val predict1 = predict.map(f => f.error)
//    // error and loss
//    // 输出误差计算
//    val loss1 = predict1
//    val (loss2, counte) = loss1.treeAggregate((0.0, 0L))(
//      seqOp = (c, v) => {
//        // c: (e, count), v: (m)
//        val e1 = c._1
//        val e2 = (v :* v).sum
//        val esum = e1 + e2
//        (esum, c._2 + 1)
//      },
//      combOp = (c1, c2) => {
//        // c: (e, count)
//        val e1 = c1._1
//        val e2 = c2._1
//        val esum = e1 + e2
//        (esum, c1._2 + c2._2)
//      })
//    val Loss = loss2 / counte.toDouble
//    Loss * 0.5
//  }
}